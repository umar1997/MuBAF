import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from getEmbeddings import get_glove_embedding


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


class OutputNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super().__init__()
        self.fc1 = nn.Linear(input_dim, int(input_dim*0.8))
        self.fc2 = nn.Linear(int(input_dim*0.8), int(input_dim*0.4))
        self.fc3 = nn.Linear(int(input_dim*0.4), int(input_dim*0.1))
        self.fc4 = nn.Linear(int(input_dim*0.1), int(input_dim*0.01))
        self.fc5 = nn.Linear(int(input_dim*0.01), output_dim)

        # self.fcx = nn.Linear(input_dim, int(input_dim*0.7))
        # self.fcy = nn.Linear(int(input_dim*0.7), int(input_dim*0.3))
        # self.fcz = nn.Linear(int(input_dim*0.3), output_dim)
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)

        # x = self.fcx(x)
        # x = F.relu(x)
        # x = self.fcy(x)
        # x = F.relu(x)
        # x = self.fcz(x)
        
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

class ContextualEmbedding(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        # input_dim = output_emb_dim = 100
        # hidden_dim = 100
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = FullyConnetedNetwork(input_dim*2, input_dim)
    def forward(self, x):
        # x = [bs, seq_len, input_dim] = [bs, seq_len, output_emb_dim]    
        outputs, _ = self.lstm(x)
        # outputs = [bs, seq_len, output_emb_dim*2]
        outputs = self.fc(outputs)
        return outputs


class ContextualEmbeddingLayer(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        # input_dim = output_emb_dim*2 = 200
        # hidden_dim = 100
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
    def __init__(self, device, glove_emb_dim, output_emb_dim, pos_dict, ner_dict):
        super().__init__()
        
        self.device = device
        
        self.pos_dict = pos_dict
        self.ner_dict = ner_dict
        
        self.glove_emb_dim = glove_emb_dim # 100
        self.output_emb_dim = output_emb_dim # 100
        self.hidden_dim = output_emb_dim # 100
    
        self.gloveEmbeddingFunc = get_glove_embedding()
        
        self.contextual_embedding = ContextualEmbedding(output_emb_dim, self.hidden_dim).to(self.device) # (100, 100)
        self.contextual_embedding_layer = ContextualEmbeddingLayer(output_emb_dim*2, self.hidden_dim).to(self.device) # (200, 100)
        
        self.similarity_weight = nn.Linear(output_emb_dim*6, 1).to(self.device)
        # self.similarity_weight = OutputNetwork(output_emb_dim*6, 1).to(self.device)
        self.layer_norm = nn.LayerNorm(output_emb_dim*8)
        self.dropout = nn.Dropout(0.1)

        self.multi_head_attention = MultiHeadAttentionLayer(hid_dim=output_emb_dim*8, n_heads=4, dropout=0.1, device=self.device)  
        # self.self_attn_layer_norm = nn.LayerNorm(output_emb_dim*8)

        self.modeling_lstm = nn.LSTM(output_emb_dim*8, output_emb_dim, bidirectional=True, num_layers=2, batch_first=True, dropout=0.1).to(self.device)
        
        # self.output_start = OutputNetwork(output_emb_dim*2, 1).to(self.device)
        # self.output_end = OutputNetwork(output_emb_dim*2, 1).to(self.device)
        # self.output_start = nn.Linear(output_emb_dim*10, 1).to(self.device)
        # self.output_end = nn.Linear(output_emb_dim*10).to(self.device)
        self.output_start = OutputNetwork(output_emb_dim*10, 1).to(self.device)
        self.output_end = OutputNetwork(output_emb_dim*10, 1).to(self.device)
        self.end_lstm = nn.LSTM(output_emb_dim*2, output_emb_dim, bidirectional=True, batch_first=True).to(self.device)
        
    def compute_glove_embedding(self, context, pos_words, ner_words):

        glove_context = self.gloveEmbeddingFunc(context)
        glove_context = glove_context.to(self.device)

        return glove_context, glove_context
    
        
    def forward(self, context, question, small_words, small_questions, pos_words, ner_words, pos_qsts, ner_qsts):
        
        # context = [bs, ctx_len]
        # question = [bs, ques_len]
        
        ctx_len = context.shape[1]  
        ques_len = question.shape[1]
        
        ###################### EMBEDDING LAYER
        # Get Glove and Contextual EMbeddings
        glove_embedding, glove_embedding_contextual = self.compute_glove_embedding(context, pos_words, ner_words)
        glove_embedding_qst, glove_embedding_qst_contextual = self.compute_glove_embedding(question, pos_qsts, ner_qsts)
        # glove_embedding  = [bs, ctx_len, glove_emb_dim]
        
        
        glove_embedding_contextual = self.contextual_embedding(glove_embedding_contextual)
        glove_embedding_qst_contextual = self.contextual_embedding(glove_embedding_qst_contextual)

        #  Run through Contextual Embedding layer
        ctx_contextual_inp = torch.cat([glove_embedding, glove_embedding_contextual],dim=2)
        ques_contextual_inp = torch.cat([glove_embedding_qst, glove_embedding_qst_contextual],dim=2)    
        
        ctx_contextual_emb = self.contextual_embedding_layer(ctx_contextual_inp)
        ques_contextual_emb = self.contextual_embedding_layer(ques_contextual_inp)
        ctx_contextual_emb, ques_contextual_emb = ctx_contextual_emb.to(self.device), ques_contextual_emb.to(self.device)
        # [bs, ctx_len, emb_dim*2]
        # [bs, ques_len, emb_dim*2]
        
        ###################### ATTENTION FLOW LAYER
        

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

        # Running Multi Head Attention
        
        # G_, _ = self.multi_head_attention(G, G, G)
        # G = self.self_attn_layer_norm(G + self.dropout(G_))
        # G = G_
        # self.layer_norm(self.dropout(G))
        ###################### MODELLING LAYER

        M, _ = self.modeling_lstm(G)
        # [bs, ctx_len, emb_dim*2]
        
        ###################### OUTPUT LAYER
        M2, _ = self.end_lstm(M)
        # p1 = self.output_start(M).squeeze()
        p1 = self.output_start(torch.cat([G,M], dim=2)).squeeze()
        # p1 = F.softmax(p1, dim=-1)
        # [bs, ctx_len, 1]
        # [bs, ctx_len]
        
        # p2 = self.output_start(M2).squeeze()
        p2 = self.output_end(torch.cat([G, M2], dim=2)).squeeze()
        # p2 = F.softmax(p2, dim=-1)
        # [bs, ctx_len, 1] => [bs, ctx_len]

        return p1, p2

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)   
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()  
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        #x = [batch size, query len, hid dim]
        
        return x, attention