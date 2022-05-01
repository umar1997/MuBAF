import time
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import MuBAF
from datasetGenerator import SquadDataset
from utils import evaluate, epoch_time
# from buildVocab import Vocab
# from dataPreprocessing import DataPreparation, getTagDF
# from getEmbeddings import get_glove_embedding, get_one_hot_vector, get_elmo_embedding, get_elmo_embedder


# Issue: So I saved my preprocessed data in pickle files
# Allennlp 0.9.0
# Allennlp requires Conllu 1.3.1 to run the Elmo embeddings
# Allennlp rquires overrides 3.1.0
# flair 0.11.1
# flair requires Conllu 4.4.1 for tokenization
# flair requires overrides 6.1 (I think)
# flair 0.11.1 requires conllu>=4.0, but you have conllu 1.3.1 which is incompatible.


def train(model, train_dataset):
    print("Starting training ........")

    train_loss = 0.
    batch_count = 1
    model.train()
    for batch in tqdm(train_dataset):

        optimizer.zero_grad()

#         if batch_count % 5 == 0:
#             break
#         if batch_count % 800 == 0:
#             print(f"Starting batch: {batch_count}")
#         batch_count += 1

        context, question, pos_words, pos_qsts, ner_words, ner_qsts, label, ctx_text, ans, ids, small_words, small_questions = batch
        context, question, label = context.to(device), question.to(device), label.to(device)

        preds = model(context, question, small_words, small_questions, pos_words, ner_words, pos_qsts, ner_qsts)

        start_pred, end_pred = preds
        s_idx, e_idx = label[:,0], label[:,1]
        
        loss = F.cross_entropy(start_pred, s_idx) + F.cross_entropy(end_pred, e_idx)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss/len(train_dataset)


def valid(model, valid_dataset, valid_df):   
    print("Starting validation .........")
   
    valid_loss = 0.
    batch_count = 1
    f1, em = 0., 0.
    predictions = {}
    
    model.eval()
    for batch in tqdm(valid_dataset):

#         if batch_count % 2 == 0:
#             break
#         if batch_count % 800 == 0:
#             print(f"Starting batch {batch_count}")
#         batch_count += 1

        context, question, pos_words, pos_qsts, ner_words, ner_qsts, label, ctx_text, ans, ids, small_words, small_questions = batch
        context, question, label = context.to(device), question.to(device), label.to(device)

        
        with torch.no_grad():

            preds = model(context, question, small_words, small_questions, pos_words, ner_words, pos_qsts, ner_qsts)

            p1, p2 = preds
            s_idx, e_idx = label[:,0], label[:,1]

            loss = F.cross_entropy(p1, s_idx) + F.cross_entropy(p2, e_idx)
            valid_loss += loss.item()

            
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            # Lower triangle matrix to ensure end index chosen after start
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            # [bs, c_len, c_len]
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
           
            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i]+1]
                pred = [idx2word[idx.item()] for idx in pred]
                predictions[id] = pred
            

    
    em, f1 = evaluate(valid_df, predictions)
    return valid_loss/len(valid_dataset), em, f1, predictions







if __name__ == '__main__':

    TRAIN_SQUAD_FILE = './data/Squad/train-v1.1.json'
    TRAIN_CSV_FILE = './tags/TagsTrain.csv'

    VALID_SQUAD_FILE = './data/Squad/dev-v1.1.json'
    VALID_CSV_FILE = './tags/TagsValid.csv'

    device = torch.device('cuda')

    with open('./tags/pos_dict.json', 'r', encoding='utf-8') as f:
        pos_dict = json.loads(f.read())
        
    with open('./tags/ner_dict.json', 'r', encoding='utf-8') as f:
        ner_dict = json.loads(f.read())
########################################################

    # print('1. Importing and Preparing Data')
    # dataPrep = DataPreparation(TRAIN_SQUAD_FILE)
    # tr_df = dataPrep.run()
    # tagTrain = getTagDF(TRAIN_CSV_FILE)
    # train_df = pd.merge(tr_df, tagTrain, how='inner', on = 'id')

    # dataPrep = DataPreparation(VALID_SQUAD_FILE)
    # v_df = dataPrep.run()
    # tagValid = getTagDF(VALID_CSV_FILE)
    # valid_df = pd.merge(v_df, tagValid, left_index=True, right_index=True)

    # print('Train: Length of OG Data: {} and Length of Tagged Data: {} and Length of Merged Data: {}'.format(len(tr_df), len(tagTrain), len(train_df)))
    # print('Valid: Length of OG Data: {} and Length of Tagged Data: {} and Length of Merged Data: {}'.format(len(v_df), len(tagValid), len(valid_df)))

########################################################

    # print('2. Formatting Dataframe and Building Vocabulary')
    # vocab = Vocab(train_df, valid_df)
    # train_df, valid_df, word2idx, idx2word, word_vocab = vocab.run()

    # print('Train: Length of Formatted Data: {}'.format(len(train_df)))
    # print('Valid: Length of Formatted Data: {}'.format(len(valid_df)))

    # print('3. Saving to Pickle File')
    # train_df.to_pickle('./pickle_files/train_df.pkl')
    # valid_df.to_pickle('./pickle_files/valid_df.pkl')

    # with open('./pickle_files/w2idx.pickle','wb') as p:
    #     pickle.dump(word2idx, p)

# YOU CAN COMMENTS THE ABOVE STEPS TO AVOID PREPROCESSING
#################################################

    print('1. Loading from Pickle File')
    train_df = pd.read_pickle('./pickle_files/train_df.pkl')
    valid_df = pd.read_pickle('./pickle_files/valid_df.pkl')

    print('Train: Length of Formatted Data: {}'.format(len(train_df)))
    print('Valid: Length of Formatted Data: {}'.format(len(valid_df)))

    with open('./pickle_files/w2idx.pickle','rb') as p:
        word2idx = pickle.load(p)

    idx2word = {v:k for k,v in word2idx.items()}


    print('2. Getting Dataset/DataLoaders')
    train_dataset = SquadDataset(train_df, 32)
    valid_dataset = SquadDataset(valid_df, 32)

    print('3. Initialize Model')
    GLOVE_EMB_DIM = 100
    OUT_EMB_DIM = 100
    POS_DICT = pos_dict
    NER_DICT = ner_dict
    device = torch.device('cuda')

    model = MuBAF(device, 
                GLOVE_EMB_DIM,
                OUT_EMB_DIM,
                POS_DICT,
                NER_DICT
                ).to(device)

    optimizer = optim.Adadelta(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-08)

    print('4. Training and Validation')
    train_losses = []
    valid_losses = []
    ems = []
    f1s = []
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        start_time = time.time()
        
        train_loss = train(model, train_dataset)
        valid_loss, em, f1, predictions = valid(model, valid_dataset, valid_df)
        ems.append(em)
        f1s.append(f1)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'em':em,
                'f1':f1,
                }, './models/mubaf_run_{}.pth'.format(epoch))
        
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch train loss : {train_loss}| Time: {epoch_mins}m {epoch_secs}s")
        print(f"Epoch valid loss: {valid_loss}")
        print(f"Epoch EM: {em}")
        print(f"Epoch F1: {f1}")
        print("=============================================================================")