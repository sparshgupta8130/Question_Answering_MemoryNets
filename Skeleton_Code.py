
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from collections import defaultdict
import pickle
from torch.autograd import Variable
import torch.optim as optim
import sys
import data_transform
import model_funcs_GPU
import model_funcs
import model_funcs_pt2_GPU
import model_funcs_pt2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_foldername = 'en-valid-10k'
train_filename = 'qa7_train'
train_fname = 'bAbI_Data/'+str(train_foldername)+'/'+str(train_filename)

valid_foldername = 'en-valid-10k'
valid_filename = 'qa7_valid'
valid_fname = 'bAbI_Data/'+str(valid_foldername)+'/'+str(valid_filename)

test_foldername = 'en-valid-10k'
test_filename = 'qa7_test'
test_fname = 'bAbI_Data/'+str(test_foldername)+'/'+str(test_filename)

vec_fname = 'bAbI_Data/model.vec'

unk_thres = 0
pre_embed = False

train_data_BOW, valid_data_BOW, test_data_BOW, train_data_pe, valid_data_pe, test_data_pe, vocab = data_transform.get_data(
    train_fname, valid_fname, test_fname, vec_fname=vec_fname, unk_thres = unk_thres, pre_embed=pre_embed)
print(train_data_BOW.shape)
print(valid_data_BOW.shape)
print(test_data_BOW.shape)
print(len(vocab))
# print(train_data_pe[0:5])


# In[ ]:


model_identity = 'comb_lstm0_5_3hops'
if pre_embed == True:
    embed_wts = data_transform.get_embeddings(vec_fname)
else:
    embed_wts = None
if pre_embed == False:
    embedding_dim = 10
else:
    embedding_dim = embed_wts.shape[1]
vocab_size = len(vocab)
num_hops = 3
max_mem_size = 40
epochs = 50
eta = 0.00003
LS = 0
ls_thres = 0.001
temporal = True
positional = False
same = 1
GPU = True
pyTorch2 = False
dropout = 0.5
visualize = True


# In[ ]:


if pyTorch2 == False:
    if GPU == True:
        print('Using GPU...')
        model = model_funcs_GPU.QuesAnsModel(embedding_dim, vocab_size, num_hops = num_hops, max_mem_size = max_mem_size,
                                         temporal=temporal, same=same, positional=positional, dropout=dropout,
                                                                     pre_embed=pre_embed, embed_wts=embed_wts)
        l_tr, accuracy_tr, l_vd, accuracy_vd = model_funcs_GPU.train(model, train_data_BOW, valid_data_BOW,
                                                                 train_data_pe, valid_data_pe,
                                                                 epochs=epochs,eta=eta,opt=optim.Adam,LS=LS,ls_thres=ls_thres,
                                                                     model_name=model_identity,visualize=visualize)
    else:
        print('Using CPU...')
        model = model_funcs.QuesAnsModel(embedding_dim, vocab_size, num_hops = num_hops, max_mem_size = max_mem_size,
                                         temporal=temporal, same=same, positional=positional, dropout=dropout,
                                                                     pre_embed=pre_embed, embed_wts=embed_wts)
        l_tr, accuracy_tr, l_vd, accuracy_vd = model_funcs.train(model, train_data_BOW, valid_data_BOW,
                                                                 train_data_pe, valid_data_pe,
                                                                 epochs=epochs,eta=eta,opt=optim.Adam,LS=LS,ls_thres=ls_thres,
                                                                 model_name=model_identity,visualize=visualize)
else:
    if GPU == True:
        print('Using GPU...') 
        model = model_funcs_pt2_GPU.QuesAnsModel(embedding_dim, vocab_size, num_hops = num_hops, max_mem_size = max_mem_size,
                                         temporal=temporal, same=same, positional=positional, dropout=dropout,
                                                                     pre_embed=pre_embed, embed_wts=embed_wts)
        l_tr, accuracy_tr, l_vd, accuracy_vd = model_funcs_pt2_GPU.train(model, train_data_BOW, valid_data_BOW,
                                                                 train_data_pe, valid_data_pe,
                                                                 epochs=epochs,eta=eta,opt=optim.Adam,LS=LS,ls_thres=ls_thres,
                                                                         model_name=model_identity,visualize=visualize)
    else:
        print('Using CPU...')
        model = model_funcs_pt2.QuesAnsModel(embedding_dim, vocab_size, num_hops = num_hops, max_mem_size = max_mem_size,
                                         temporal=temporal, same=same, positional=positional, dropout=dropout,
                                                                     pre_embed=pre_embed, embed_wts=embed_wts)
        l_tr, accuracy_tr, l_vd, accuracy_vd = model_funcs_pt2.train(model, train_data_BOW, valid_data_BOW,
                                                                 train_data_pe, valid_data_pe,
                                                                 epochs=epochs,eta=eta,opt=optim.Adam,LS=LS,ls_thres=ls_thres,
                                                                     model_name=model_identity,visualize=visualize)


# In[ ]:


if pyTorch2 == False:
    if GPU == True:
        acc = model_funcs_GPU.test(model,test_data_BOW,test_data_pe)
    else:
        acc = model_funcs.test(model,test_data_BOW,test_data_pe,get_probs=True, num_words=5)
else:
    if GPU == True:
        acc = model_funcs_pt2_GPU.test(model,test_data_BOW,test_data_pe)
    else:
        acc = model_funcs_pt2.test(model,test_data_BOW,test_data_pe)

