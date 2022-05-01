import torch
from torch import nn

import allennlp
from allennlp.commands.elmo import ElmoEmbedder

import numpy as np

device = torch.device('cuda')





def get_elmo_embedder():
    elmo = ElmoEmbedder(
        options_file = './data/ELMo/options_128.json',
        weight_file = './data/ELMo/elmo_weights_128.hdf5'
    #     options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json', 
    #     weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
    )
    return elmo

def get_elmo_embedding(elmo, tokens):
    max_len = max([len(p) for p in tokens])
    batch_embeddings = []
    for token in tokens:
        emb = elmo.embed_sentence(token)[2]
        
        if len(token) != max_len:
            unk = np.zeros(((max_len - len(token)), emb.shape[1]))
            emb = np.concatenate([emb,unk],axis=0)
        
        batch_embeddings.append(emb)
    batch_embeddings = torch.Tensor(np.array(batch_embeddings))
    return batch_embeddings


def get_glove_embedding():
        
        weights_matrix = np.load('./data/Glove/glove_embeddings.npy')
        num_embeddings, embedding_dim = weights_matrix.shape
        embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(device),freeze=True)

        return embedding

def get_one_hot_vector(dictionary, listOfTags):
    batch_tags = []
    for tagSent in listOfTags:
        max_len_words = len(tagSent)
        listOfIndx = [int(dictionary[i]) if i!= 100 else 100 for i in tagSent]
        try:
            ind = listOfIndx.index(100)
        except:
            ind = max_len_words
        
        a = np.array(listOfIndx[:ind])
        b = np.zeros((ind, len(dictionary)))
        b[np.arange(ind),a] = 1
        
        if ind == max_len_words:
            batch_tags.append(b)
        else:
            ind = max_len_words - ind
            c = np.zeros((ind, len(dictionary)))
            b = np.concatenate([b,c],axis=0)
            batch_tags.append(b)
            
    return np.array(batch_tags)


