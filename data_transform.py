
# coding: utf-8

# In[1]:


import numpy as np
from collections import defaultdict
import pickle
import os
import sys


# In[11]:


def hasDigits(input_str):
    return any(char.isdigit() for char in input_str)


# In[12]:


def create_vocab(data,unk_thres=0):
    #print(len(data))
    aux = defaultdict(int)
    for i in range(len(data)):
        for j in range(1,len(data[i])):
            if hasDigits(data[i][j]):
                break
            aux[data[i][j]] += 1
    #print(aux)
    vocab = []
    unk_list = []
    for i in aux:
        if aux[i] < unk_thres:
            if not unk_list:
                vocab.append('UNK')
            unk_list.append(i)
        else:
            vocab.append(i)
    #print(vocab)
    #print(unk_list)
    return vocab, unk_list


# In[13]:


def create_dictionaries(vocab):
    word2idx = defaultdict(int)
    idx2word = defaultdict(int)
    k = 0
    for i in range(len(vocab)):
        word2idx[vocab[i]] = k
        idx2word[k] = vocab[i]
        k += 1
    
    with open('variables/word2idx','wb') as handle:
        pickle.dump(word2idx,handle,protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('variables/idx2word','wb') as handle:
        pickle.dump(idx2word,handle,protocol=pickle.HIGHEST_PROTOCOL)


# In[14]:


def transform_data_BOW(data,vocab,unk_list,word2idx):
    N = len(vocab)
    dat_trans = np.zeros((len(data),N+1))
    for i in range(len(data)):
        if hasDigits(data[i][-1]):
            digits = 0
            j = len(data[i])-1
            while hasDigits(data[i][j]):
                digits += 1
                j -= 1
#             print(digits)
            dat_trans[i,N] = word2idx[data[i][len(data[i])-digits-1]]
            for j in range(1,len(data[i])-digits-1):
                if data[i][j] in unk_list:
                    dat_trans[i,word2idx['UNK']] += 1
                else:
                    dat_trans[i,word2idx[data[i][j]]] += 1
        else:
            if data[i][0] == '1':
                dat_trans[i,N] = -2
            else:
                dat_trans[i,N] = -1
            for j in range(1,len(data[i])):
                #print(data[i][j],unk_list)
                if data[i][j] in unk_list:
                    dat_trans[i,word2idx['UNK']] += 1
                else:
                    dat_trans[i,word2idx[data[i][j]]] += 1
    
    return dat_trans

def transform_data_pe(data,vocab,unk_list,word2idx):
    N = len(vocab)
    dat_trans = []
    for i in range(len(data)):
        aux = []
        if hasDigits(data[i][-1]):
            digits = 0
            j = len(data[i])-1
            while hasDigits(data[i][j]):
                digits += 1
                j -= 1
#             print(digits)
            #dat_trans[i,N] = word2idx[data[i][len(data[i])-digits-1]]
            for j in range(1,len(data[i])-digits-1):
                if data[i][j] in unk_list:
                    aux.append(int(word2idx['UNK']))
                else:
                    aux.append(int(word2idx[data[i][j]]))
            aux.append(word2idx[data[i][len(data[i])-digits-1]])
        else:
            for j in range(1,len(data[i])):
                #print(data[i][j],unk_list)
                if data[i][j] in unk_list:
                    aux.append(int(word2idx['UNK']))
                else:
                    aux.append(int(word2idx[data[i][j]]))
            if data[i][0] == '1':
                aux.append(-2)
            else:
                aux.append(-1)
        dat_trans.append(np.array(aux).reshape(1,-1))
    return dat_trans


# In[15]:


def get_data(train_fname, valid_fname, test_fname, unk_thres=0):
    train_dat_aux = []
    valid_dat_aux = []
    test_dat_aux = []
    punctuations = ['.',',','?']
    for l in open(train_fname+'.txt'):
        temp = ''.join(ch for ch in l if ch not in punctuations)
        train_dat_aux.append(temp.strip().split())

    for l in open(valid_fname+'.txt'):
        temp = ''.join(ch for ch in l if ch not in punctuations)
        valid_dat_aux.append(temp.strip().split())

    for l in open(test_fname+'.txt'):
        temp = ''.join(ch for ch in l if ch not in punctuations)
        test_dat_aux.append(temp.strip().split())

    print('Train Data Size : ',len(train_dat_aux))
    print('Valid Data Size : ',len(valid_dat_aux))
    print('Test Data Size : ',len(test_dat_aux))
    
    vocab, unk_list = create_vocab(train_dat_aux,unk_thres)
    if not os.path.exists('variables'):
        os.makedirs('variables')
    create_dictionaries(vocab)
    #create_dictionaries(vocab)
    
    with open('variables/word2idx','rb') as handle:
        word2idx = pickle.load(handle)

    with open('variables/idx2word','rb') as handle:
        idx2word = pickle.load(handle)
    train_data_BOW = transform_data_BOW(train_dat_aux,vocab,unk_list,word2idx)
    valid_data_BOW = transform_data_BOW(valid_dat_aux,vocab,unk_list,word2idx)
    test_data_BOW = transform_data_BOW(test_dat_aux,vocab,unk_list,word2idx)
    train_data_pe = transform_data_pe(train_dat_aux,vocab,unk_list,word2idx)
    valid_data_pe = transform_data_pe(valid_dat_aux,vocab,unk_list,word2idx)
    test_data_pe = transform_data_pe(test_dat_aux,vocab,unk_list,word2idx)
    #print(train_dat_aux[:5])
    return train_data_BOW, valid_data_BOW, test_data_BOW,train_data_pe, valid_data_pe, test_data_pe, vocab

