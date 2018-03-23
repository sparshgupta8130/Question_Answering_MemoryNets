
# coding: utf-8

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
#get_ipython().run_line_magic('matplotlib', 'inline')


def get_data(folder_name,qa_name,unk_thres=0,pre_embed=False):
    train_foldername = folder_name
    train_filename = qa_name + '_train'
    train_fname = 'bAbI_Data/'+str(train_foldername)+'/'+str(train_filename)

    valid_foldername = folder_name
    valid_filename = qa_name + '_valid'
    valid_fname = 'bAbI_Data/'+str(valid_foldername)+'/'+str(valid_filename)

    test_foldername = folder_name
    test_filename = qa_name + '_test'
    test_fname = 'bAbI_Data/'+str(test_foldername)+'/'+str(test_filename)

    vec_fname = 'bAbI_Data/model.vec'

    unk_thres = 0
    pre_embed = pre_embed

    train_data_BOW, valid_data_BOW, test_data_BOW, train_data_pe, valid_data_pe, test_data_pe, vocab = data_transform.get_data(
        train_fname, valid_fname, test_fname, vec_fname=vec_fname, unk_thres = unk_thres, pre_embed=pre_embed)
    print('Train data size : ',train_data_BOW.shape)
    print('Valid data size : ',valid_data_BOW.shape)
    print('Test data size : ',test_data_BOW.shape)
    print('Vocab size : ',len(vocab))
    return train_data_BOW, valid_data_BOW, test_data_BOW, train_data_pe, valid_data_pe, test_data_pe, vocab


def train(folder_name='en-valid-10k',qa_name='qa1',unk_thres=0,pre_embed=False,model_identity='a_model_has_no_name',
               embedding_dim=10,num_hops = 1,max_mem_size=15,epochs=10,eta=0.0001,LS=0,ls_thres=0.001,temporal=False,
         positional=False,same=0,dropout=0,visualize=True,GPU=False,pyTorch2=False,test=True):
    vec_fname = 'bAbI_Data/model.vec'
    if pre_embed == True:
        embed_wts = data_transform.get_embeddings(vec_fname)
    else:
        embed_wts = None
    
    if pre_embed == False:
        embedding_dim = embedding_dim
    else:
        embedding_dim = embed_wts.shape[1]
    
     
    print('Folder : ',folder_name)
    print('Task : ',qa_name)
    
    train_data_BOW, valid_data_BOW, test_data_BOW, train_data_pe, valid_data_pe, test_data_pe, vocab = get_data(folder_name,qa_name,
                                                                                                               unk_thres,pre_embed)
    vocab_size = len(vocab)
    
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
    
    if test == True:
        test(model,test_data_BOW,test_data_pe,GPU=GPU,pyTorch2=pyTorch2)
    
    return model


def test(model,test_data_BOW,test_data_pe,GPU=False,pyTorch2=False):
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
    print('Test Accuracy : ',acc)
