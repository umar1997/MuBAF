import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
from getEmbeddings import get_glove_embedding, get_one_hot_vector, get_elmo_embedding, get_elmo_embedder

def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class FullyConnetedNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

class HighwayNetwork(nn.Module):
    
    def __init__(self, input_dim, num_layers=2):
        # input_dim = 256
        super().__init__()        
        self.num_layers = num_layers
        
        self.flow_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])       
    def forward(self, x):
        for i in range(self.num_layers):
            
            flow_value = F.relu(self.flow_layer[i](x))
            gate_value = torch.sigmoid(self.gate_layer[i](x))
            
            x = gate_value * flow_value + (1-gate_value) * x
        return x


class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        # input_dim = output_emb_dim*2 = 256
        # hidden_dim = 128
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)       
        self.highway_net = HighwayNetwork(input_dim)
        
    def forward(self, x):
        # x = [bs, seq_len, input_dim] = [bs, seq_len, output_emb_dim*2]       
        highway_out = self.highway_net(x)
        # highway_out = [bs, seq_len, input_dim]     
        outputs, _ = self.lstm(highway_out)
        # outputs = [bs, seq_len, output_emb_dim*2]
        
        return outputs

class MuBAF(nn.Module):
    def __init__(self, device, elmo_emb_dim, glove_emb_dim, output_emb_dim, pos_dict, ner_dict):
        super().__init__()
        
        self.device = device
        
        self.pos_dict = pos_dict
        self.ner_dict = ner_dict
        
        self.elmo_emb_dim = elmo_emb_dim
        self.glove_emb_dim = glove_emb_dim
        self.output_emb_dim = output_emb_dim
        self.hidden_dim = output_emb_dim
    
        self.elmo = get_elmo_embedder()
        self.gloveEmbeddingFunc = get_glove_embedding()
        
        self.contextual_embedding = ContextualEmbeddingLayer(output_emb_dim*2, self.hidden_dim).to(self.device) # (256, 128)
        
        self.similarity_weight = nn.Linear(output_emb_dim*6, 1, bias=False).to(self.device)
          
        self.modeling_lstm = nn.LSTM(output_emb_dim*8, output_emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.2).to(self.device)
        
        self.output_start = nn.Linear(output_emb_dim*10, 1, bias=False).to(self.device)
        self.output_end = nn.Linear(output_emb_dim*10, 1, bias=False).to(self.device)
        self.end_lstm = nn.LSTM(output_emb_dim*2, output_emb_dim, bidirectional=True, batch_first=True).to(self.device)
        
    def compute_glove_embedding(self, context, pos_words, ner_words):
        # pos = get_one_hot_vector(self.pos_dict, pos_words)
        # ner  = get_one_hot_vector(self.ner_dict, ner_words)
        # tag_ohv = torch.Tensor(np.concatenate((pos, ner), axis = 2))
        # tag_ohv = tag_ohv.to(self.device)

        glove_context = self.gloveEmbeddingFunc(context)
        glove_context = glove_context.to(self.device)
        # embedding_glove_tag = torch.cat([glove_context,tag_ohv],dim=2)
        # return embedding_glove_tag
        return glove_context
    
    def compute_elmo_embedding(self, small_words):
        embedding_elmo = get_elmo_embedding(self.elmo, small_words)
        elmo_embedding = embedding_elmo.to(self.device)
        return elmo_embedding
        
    def forward(self, context, question, small_words, small_questions, pos_words, ner_words, pos_qsts, ner_qsts):
        
        # context = [bs, ctx_len]
        # question = [bs, ques_len]
        
        ctx_len = context.shape[1]  
        ques_len = question.shape[1]
        
        ###################### EMBEDDING LAYER
        start_time = time.time()
        # Get Glove and ELMo Embeddings
        glove_embedding = self.compute_glove_embedding(context, pos_words, ner_words)
        elmo_embedding = self.compute_elmo_embedding(small_words) 
        glove_embedding_qst = self.compute_glove_embedding(question, pos_qsts, ner_qsts)
        elmo_embedding_qst = self.compute_elmo_embedding(small_questions)
        # elmo_embedding  = [bs, ctx_len, elmo_emb_dim] elmo_emb_dim = 256
        # glove_embedding  = [bs, ctx_len, glove_emb_dim] glove_emb_dim = 154 -> 100 + 49 + 5
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Embedding | Time: {epoch_mins}m {epoch_secs}s")
        
        start_time = time.time()
        # Run both Embeddings through a FC layer to get same output dimension
        fc_elmo = FullyConnetedNetwork(self.elmo_emb_dim, self.output_emb_dim).to(self.device)
        # fc_glove = FullyConnetedNetwork(self.glove_emb_dim, self.output_emb_dim)
        fc_elmo_qst = FullyConnetedNetwork(self.elmo_emb_dim, self.output_emb_dim).to(self.device)
        # fc_glove_qst = FullyConnetedNetwork(self.glove_emb_dim, self.output_emb_dim)


        # glove_embedding = fc_glove(glove_embedding)
        elmo_embedding = fc_elmo(elmo_embedding)
        # glove_embedding_qst = fc_glove_qst(glove_embedding_qst)
        elmo_embedding_qst = fc_elmo_qst(elmo_embedding_qst)
        glove_embedding, elmo_embedding, glove_embedding_qst, elmo_embedding_qst = glove_embedding.to(self.device), elmo_embedding.to(self.device), glove_embedding_qst.to(self.device), elmo_embedding_qst.to(self.device)
        # emb_dim = 128
        # elmo_embedding  = [bs, ctx_len, emb_dim] 
        # glove_embedding  = [bs, ctx_len, emb_dim]
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Fully Connected Embedding | Time: {epoch_mins}m {epoch_secs}s")
        

        start_time = time.time()
        #  Run through Contextual Embedding layer
        ctx_contextual_inp = torch.cat([elmo_embedding, glove_embedding],dim=2)
        ques_contextual_inp = torch.cat([elmo_embedding_qst, glove_embedding_qst],dim=2)      
        
        ctx_contextual_emb = self.contextual_embedding(ctx_contextual_inp)
        ques_contextual_emb = self.contextual_embedding(ques_contextual_inp)
        ctx_contextual_emb, ques_contextual_emb = ctx_contextual_emb.to(self.device), ques_contextual_emb.to(self.device)
        # [bs, ctx_len, emb_dim*2]
        # [bs, ques_len, emb_dim*2]
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Contextual Embedding | Time: {epoch_mins}m {epoch_secs}s")
        ###################### ATTENTION FLOW LAYER
        
        start_time = time.time()
        # Create similarity matrix
        ctx_ = ctx_contextual_emb.unsqueeze(2).repeat(1,1,ques_len,1)   
        ques_ = ques_contextual_emb.unsqueeze(1).repeat(1,ctx_len,1,1)
        # [bs, ctx_len, 1, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        # [bs, 1, ques_len, emb_dim*2] => [bs, ctx_len, ques_len, emb_dim*2]
        elementwise_prod = torch.mul(ctx_, ques_)
        # [bs, ctx_len, ques_len, emb_dim*2]
        alpha = torch.cat([ctx_, ques_, elementwise_prod], dim=3)
        # [bs, ctx_len, ques_len, emb_dim*6]
        similarity_matrix = self.similarity_weight(alpha).view(-1, ctx_len, ques_len)
        # [bs, ctx_len, ques_len]
        
        
        # Calculate Context2Query Attention
        a = F.softmax(similarity_matrix, dim=-1)
        # [bs, ctx_len, ques_len]  
        c2q = torch.bmm(a, ques_contextual_emb)
        # Batch matrix multiplication
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        # [bs] ([ctx_len, ques_len] X [ques_len, emb_dim*2]) => [bs, ctx_len, emb_dim*2]
        
        
        # Calculate Query2Context Attention
        b = F.softmax(torch.max(similarity_matrix,2)[0], dim=-1)
        # [bs, ctx_len]     
        b = b.unsqueeze(1)
        # [bs, 1, ctx_len]
        q2c = torch.bmm(b, ctx_contextual_emb)
        # [bs] ([bs, 1, ctx_len] X [bs, ctx_len, emb_dim*2]) => [bs, 1, emb_dim*2]
        q2c = q2c.repeat(1, ctx_len, 1)
        # [bs, ctx_len, emb_dim*2]
        
        
        ## Query Aware Representation
        # [bs, ctx_len, emb_dim*8]
        G = torch.cat([ctx_contextual_emb, c2q, 
                       torch.mul(ctx_contextual_emb,c2q), 
                       torch.mul(ctx_contextual_emb, q2c)], dim=2)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Attention | Time: {epoch_mins}m {epoch_secs}s")
        ###################### MODELLING LAYER
        
        start_time = time.time()
        M, _ = self.modeling_lstm(G)
        # [bs, ctx_len, emb_dim*2]
        
        ###################### OUTPUT LAYER
        M2, _ = self.end_lstm(M)
        
        p1 = self.output_start(torch.cat([G,M], dim=2))
        p1 = p1.squeeze()
        # [bs, ctx_len, 1]
        # [bs, ctx_len]
        
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        # [bs, ctx_len, 1] => [bs, ctx_len]
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Modelling + Output | Time: {epoch_mins}m {epoch_secs}s")
        return p1, p2


if __name__ == '__main__':

# This model failed because it was taking too long to compute
# the ELMo embeddings and then for some reason was also giving 
# a very low EM and F1 score

# Also commented out the POS Tagging and NER One Hot Vetor Implementation

    pos_dict = ner_dict = None

    ELMO_EMB_DIM = 256
    # GLOVE_EMB_DIM = 154
    GLOVE_EMB_DIM = 100
    OUT_EMB_DIM = 128
    # OUT_EMB_DIM = 100
    POS_DICT = pos_dict
    NER_DICT = ner_dict
    device = torch.device('cuda')

    model = MuBAF(device,
                ELMO_EMB_DIM, 
                GLOVE_EMB_DIM,
                OUT_EMB_DIM,
                POS_DICT,
                NER_DICT
                ).to(device)
        
        