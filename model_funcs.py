
# coding: utf-8

# In[3]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from collections import defaultdict
import pickle
from torch.autograd import Variable
import torch.optim as optim
import os
import sys
from time import gmtime, strftime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


def smax(x):
    y = torch.div(torch.exp(x),torch.sum(torch.exp(x)))
    return y


# In[2]:


def comp(out,target):
    if (target.data[0] == np.argmax(smax(out.data))):
        return 1
    else:
        return 0


# In[4]:

class QuesAnsModel(torch.nn.Module):
    def __init__(self,embedding_dim, vocab_size, num_hops = 1, max_mem_size=15,temporal=False,same=0,positional=False,dropout=0,pre_embed=False,embed_wts=None):
        super(QuesAnsModel,self).__init__()
        self.max_mem_size = max_mem_size
        self.vocab_size = vocab_size
        self.num_hops = num_hops
        self.embedding_dim = embedding_dim
        self.current_mem_size = 0
        self.temporal = temporal
        self.positional = positional
        self.same = same
        self.num_layers = 1
        self.dropout = dropout
        if self.positional == True:
            self.memory = []
        else:
            self.memory = self.init_memory()
        self.embedding_A = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
        self.embedding_C = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
        self.embedding_B = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
        torch.nn.init.xavier_normal(self.embedding_A.weight)
        torch.nn.init.xavier_normal(self.embedding_C.weight)
        torch.nn.init.xavier_normal(self.embedding_B.weight)
        
        if pre_embed == True:
            self.embedding_A.weight = torch.nn.Parameter(torch.from_numpy(embed_wts).float().t(),requires_grad=False)
            self.embedding_C.weight = torch.nn.Parameter(torch.from_numpy(embed_wts).float().t(),requires_grad=False)
            self.embedding_B.weight = torch.nn.Parameter(torch.from_numpy(embed_wts).float().t(),requires_grad=False)
            self.same = 1
        
        self.W = torch.nn.Linear(self.embedding_dim,self.vocab_size,bias=False)
        torch.nn.init.xavier_normal(self.W.weight)
        self.temporal_A = torch.nn.Parameter(torch.randn(self.max_mem_size,self.embedding_dim).float())
        self.temporal_C = torch.nn.Parameter(torch.randn(self.max_mem_size,self.embedding_dim).float())
        self.lstm_A = torch.nn.LSTM(self.embedding_dim,self.embedding_dim,self.num_layers,dropout=self.dropout)
        self.lstm_B = torch.nn.LSTM(self.embedding_dim,self.embedding_dim,self.num_layers,dropout=self.dropout)
        self.lstm_C = torch.nn.LSTM(self.embedding_dim,self.embedding_dim,self.num_layers,dropout=self.dropout)
#         self.hidden_A = self.init_hidden()
#         self.hidden_B = self.init_hidden()
#         self.hidden_C = self.init_hidden()
        self.softmax = torch.nn.Softmax(dim=0)
    
    def init_memory(self):
        aux = torch.zeros((self.max_mem_size, self.vocab_size)).float()
#         for i in range(aux.shape[0]):
#             for j in range(aux.shape[1]):
#                 aux[i,j] = -10000000000
        return Variable(aux,requires_grad=False)
    
    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(self.num_layers,batch_size,self.embedding_dim)),
               Variable(torch.zeros(self.num_layers,batch_size,self.embedding_dim)))

    def forward(self, seq, seq_pe, tag, LS = 0):
#         if tag == 's':
        if tag in ['s','f']:
            if self.positional == False:
                if self.current_mem_size < self.max_mem_size:
                    self.memory[self.current_mem_size] = Variable(torch.from_numpy(seq).float()).view(1,-1)
                    self.current_mem_size += 1
                else:
                    aux1 = self.memory.data[1:,:].numpy()
                    aux1 = np.vstack((aux1,seq))
                    self.memory = Variable(torch.from_numpy(aux1).float())
            else:
                if self.current_mem_size < self.max_mem_size:
                    self.memory.append(Variable(torch.from_numpy(seq_pe).float()).view(1,-1))
                    self.current_mem_size += 1
                else:
                    del self.memory[0]
                    self.memory.append(Variable(torch.from_numpy(seq_pe).float()).view(1,-1))
                    
            return True
        
#         elif tag == 'f':    
#             del self.memory
#             if self.positional == False:
#                 self.memory = self.init_memory()
#                 self.current_mem_size = 1
#                 self.memory[0] = Variable(torch.from_numpy(seq).float()).view(1,-1)
#             else:
#                 self.memory = []
#                 self.current_mem_size = 1
#                 self.memory.append(Variable(torch.from_numpy(seq_pe).float()).view(1,-1))
#             return True
        
        else:
            if self.same == 0:
                if self.positional == True:
                    self.question = Variable(torch.from_numpy(seq_pe).float()).view(1,-1)

                    maxlen = 0
                    for i in range(len(self.memory)):
                        J = self.memory[i].data.shape[1]
                        if J > maxlen:
                            maxlen = J
                    
                    input_mem = torch.zeros((len(self.memory),maxlen,self.vocab_size))
                    for i in range(len(self.memory)):
                        J = self.memory[i].data.shape[1]
                        for j in range(J):
                            input_mem[i,j,int(self.memory[i].data[0,j])] = 1
                    auxa = self.embedding_A(Variable(input_mem.view(-1,self.vocab_size)))
                    auxa = auxa.view(-1,maxlen,self.embedding_dim)
                    auxc = self.embedding_C(Variable(input_mem.view(-1,self.vocab_size)))
                    auxc = auxc.view(-1,maxlen,self.embedding_dim)
                    
                    self.hidden_A = self.init_hidden(len(self.memory))
                    self.hidden_C = self.init_hidden(len(self.memory))
                    out, _ = self.lstm_A(auxa.view(auxa.data.shape[1],auxa.data.shape[0],-1),self.hidden_A)
                    current_A = torch.squeeze(torch.index_select(out,0,Variable(torch.LongTensor([maxlen-1]))))
                    out, _ = self.lstm_C(auxc.view(auxc.data.shape[1],auxc.data.shape[0],-1),self.hidden_C)
                    current_C = torch.squeeze(torch.index_select(out,0,Variable(torch.LongTensor([maxlen-1]))))
                    
                    self.hidden_B = self.init_hidden(1)
                    J = self.question.data.shape[1]
                    addr_list = []
                    for j in range(J):
                        addr_list.append(int(self.question.data[0,j]))
                    aux = torch.index_select(self.embedding_B.weight.t(),0,Variable(torch.LongTensor(addr_list)))
                    out, _ = self.lstm_B(aux.view(aux.data.shape[0],1,-1),self.hidden_B)
                    out = out.view(out.data.shape[0],out.data.shape[2])
                    ques_d = torch.index_select(out,0,Variable(torch.LongTensor([J-1])))
                    
                    curr_len = current_A.data.shape[0]
                    if self.max_mem_size != curr_len:
                        app_mat = Variable(torch.zeros((self.max_mem_size-curr_len,self.embedding_dim)))
                        current_A = torch.cat((current_A,app_mat),0)
                        current_C = torch.cat((current_C,app_mat),0)
                    
                    if self.temporal == True:
                        current_A = current_A + self.temporal_A
                        current_C = current_C + self.temporal_C
            
                else:
                    self.question = Variable(torch.from_numpy(seq).float()).view(1,-1)
                    ques_d = self.embedding_B(self.question)
                    if self.temporal == True:
        #                 temp_mem = np.flipud(np.array(self.memory.data))
        #                 self.memory = Variable(torch.from_numpy(temp_mem.copy())).float()
                        current_A = self.embedding_A(self.memory) + self.temporal_A
                        current_C = self.embedding_C(self.memory) + self.temporal_C
                    else:
                        current_A = self.embedding_A(self.memory)
                        current_C = self.embedding_C(self.memory)

                for i in range(self.num_hops):
                    aux = torch.mm(ques_d, current_A.t()).t()
                    if LS == 0:
                        P = self.softmax(aux)
                    else:
                        P = aux
                    o = torch.mm(P.t(),current_C) + ques_d
                    ques_d = o
                output = self.W(o)
                return output
            
            else:
                if self.positional == True:
                    self.question = Variable(torch.from_numpy(seq_pe).float()).view(1,-1)

                    maxlen = 0
                    for i in range(len(self.memory)):
                        J = self.memory[i].data.shape[1]
                        if J > maxlen:
                            maxlen = J
                    
                    input_mem = torch.zeros((len(self.memory),maxlen,self.vocab_size))
                    for i in range(len(self.memory)):
                        J = self.memory[i].data.shape[1]
                        for j in range(J):
                            input_mem[i,j,int(self.memory[i].data[0,j])] = 1
                    auxa = self.embedding_A(Variable(input_mem.view(-1,self.vocab_size)))
                    auxa = auxa.view(-1,maxlen,self.embedding_dim)
                    
                    self.hidden_A = self.init_hidden(len(self.memory))
                    out, _ = self.lstm_A(auxa.view(auxa.data.shape[1],auxa.data.shape[0],-1),self.hidden_A)
                    current_A = torch.squeeze(torch.index_select(out,0,Variable(torch.LongTensor([maxlen-1]))))
                    
                    self.hidden_A = self.init_hidden(1)
                    J = self.question.data.shape[1]
                    addr_list = []
                    for j in range(J):
                        addr_list.append(int(self.question.data[0,j]))
                    aux = torch.index_select(self.embedding_A.weight.t(),0,Variable(torch.LongTensor(addr_list)))
                    out, _ = self.lstm_A(aux.view(aux.data.shape[0],1,-1),self.hidden_A)
                    out = out.view(out.data.shape[0],out.data.shape[2])
                    ques_d = torch.index_select(out,0,Variable(torch.LongTensor([J-1])))
                    
                    curr_len = current_A.data.shape[0]
                    if self.max_mem_size != curr_len:
                        app_mat = Variable(torch.zeros((self.max_mem_size-curr_len,self.embedding_dim)))
                        current_A = torch.cat((current_A,app_mat),0)
                    
                    if self.temporal == True:
                        current_A = current_A + self.temporal_A
            
                else:
                    self.question = Variable(torch.from_numpy(seq).float()).view(1,-1)
                    ques_d = self.embedding_A(self.question)
                    if self.temporal == True:
        #                 temp_mem = np.flipud(np.array(self.memory.data))
        #                 self.memory = Variable(torch.from_numpy(temp_mem.copy())).float()
                        current_A = self.embedding_A(self.memory) + self.temporal_A
                        #current_C = self.embedding_C(self.memory) + self.temporal_C
                    else:
                        current_A = self.embedding_A(self.memory)
                        #current_C = self.embedding_C(self.memory)

                for i in range(self.num_hops):
                    aux = torch.mm(ques_d, current_A.t()).t()
                    if LS == 0:
                        P = self.softmax(aux)
                    else:
                        P = aux
                    o = torch.mm(P.t(),current_A) + ques_d
                    ques_d = o
                output = self.W(o)
                return output


# In[5]:


def train(model,tr_dt_bow,vd_dt_bow,tr_dt_pe, vd_dt_pe,opt=optim.Adam,epochs=10,eta=0.0001,LS=0,ls_thres=0.001,isLSTM=True, model_name ='a_model_has_no_name'):
    optimizer = opt(filter(lambda p: p.requires_grad, model.parameters()),lr=eta)
    loss = torch.nn.CrossEntropyLoss()
    print(optimizer)
    tr_shape = tr_dt_bow.shape
    vd_shape = vd_dt_bow.shape
    eps = []
    l_tr = []
    l_vd = []
    accuracy_tr = []
    accuracy_vd = []
    
    if LS == 1:
        ls = 1
        ls_only = 0
    elif LS == 0:
        ls = 0
        ls_only = 0
    else:
        ls = 1
        ls_only = 1
    
    file_path = "./observations/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = "./saved_models/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ts = strftime("%Y-%m-%d__%Hh%Mm%Ss_" + model_name, gmtime())
    
    for epoch in range(epochs):
        count=0
        l = open('./observations/QAmem_trial_'+str(ts)+'.txt','a+')
        ################################# Training
        n_corr = 0
        l_temp = 0
        for i in range(tr_shape[0]):
            tag = 'q'
            if(tr_dt_bow[i,-1]==-1):
                tag = 's'
                model(tr_dt_bow[i,:-1],tr_dt_pe[i][0,:-1],tag)
            elif(tr_dt_bow[i,-1]==-2):
                tag = 'f'
                model(tr_dt_bow[i,:-1],tr_dt_pe[i][0,:-1],tag)
            else:
                count+=1
                out = model(tr_dt_bow[i,:-1],tr_dt_pe[i][0,:-1],tag,LS=ls)
                target = Variable(torch.from_numpy(np.array([tr_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([tr_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
                optimizer.zero_grad()
                loss_tr = loss(out,target)
                loss_tr.backward(retain_graph=True)
                optimizer.step()
                l_temp += loss_tr.data[0]
                n_corr += comp(out,target)
        acc_tr = n_corr/count*100
        l_tr.append(l_temp/tr_shape[0])
        accuracy_tr.append(acc_tr)
#         print(model.embedding_B.weight[0:2,2:9])
        
        ############################# Validation
        n_corr = 0
        count = 0
        l_temp = 0
        for i in range(vd_shape[0]):
            tag = 'q'
            if(vd_dt_bow[i,-1]==-1):
                tag = 's'
                model(vd_dt_bow[i,:-1],vd_dt_pe[i][0,:-1],tag)
            elif(vd_dt_bow[i,-1]==-2):
                tag = 'f'
                model(vd_dt_bow[i,:-1],vd_dt_pe[i][0,:-1],tag)
            else:
                count+=1
                out = model(vd_dt_bow[i,:-1],vd_dt_pe[i][0,:-1],tag)
                target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
                optimizer.zero_grad()
                loss_vd = loss(out,target)
                l_temp += loss_vd.data[0]
                n_corr += comp(out,target)
        acc_vd = n_corr/count*100
        l_vd.append(l_temp/vd_shape[0])
        accuracy_vd.append(acc_vd)
        if not ls_only:
            n = len(l_vd)
            if n >= 3:
                if abs(l_vd[-3] - l_vd[-1]) < ls_thres and ls == 1:
                    print('Inserting Softmax...')
                    ls = 0
            elif n > 1:
                if abs(l_vd[0] - l_vd[-1]) < ls_thres and ls == 1:
                    print('Inserting Softmax...')
                    ls = 0
        
        eps.append(epoch)
        print(epoch,'Training Loss : ',l_tr[-1],' , Training Acc : ',accuracy_tr[-1])
        print(epoch,'Validation Loss : ',l_vd[-1],' , Validation Acc : ',accuracy_vd[-1])
        l.write(str(epoch)+' T '+str(l_tr[-1])+' V '+str(l_vd[-1])+' TA '+str(accuracy_tr[-1])+' VA '+str(accuracy_vd[-1])+'\n')
        l.close()
        if os.path.exists('saved_models/model' + str(ts) + '.pt'):
            os.remove('saved_models/model' + str(ts) + '.pt')
        torch.save(model,'saved_models/model' + str(ts) + '.pt')    
    print('end')
    
    fig = plt.figure()
    plt.plot(eps,l_tr)
    plt.plot(eps,l_vd)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss','Validation Loss'])
    fig.savefig('observations/LossPlot_' + str(ts) + '.png', dpi=fig.dpi)
    plt.show()
    
    fig = plt.figure()
    plt.plot(eps,accuracy_tr)
    plt.plot(eps,accuracy_vd)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy%')
    plt.legend(['Training Accuracy','Validation Accuracy'],loc=4)
    plt.show()
    fig.savefig('observations/AccuracyPlot_' + str(ts) + '.png', dpi=fig.dpi)
    
    return l_tr, accuracy_tr, l_vd, accuracy_vd

def test_visualize(model,test_dt_bow,test_dt_pe, get_probs, num_words):
    test_shape = test_dt_bow.shape
    n_corr = 0;
    count = 0;
    with open('variables/word2idx','rb') as handle:
        word2idx = pickle.load(handle)

    with open('variables/idx2word','rb') as handle:
        idx2word = pickle.load(handle)
    print(idx2word)
    for i in range(test_shape[0]):
        l_temp = 0
        tag = 'q'
        if(test_dt_bow[i,-1]==-1):
            tag = 's'
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            print(" ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]]))
        elif(test_dt_bow[i,-1]==-2):
            tag = 'f'
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            print(" ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]]))
        else:
            count+=1
            out = model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            target = Variable(torch.from_numpy(np.array([test_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
            n_corr += comp(out,target)
            print('QQ: '," ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]]))
            print('target: ', idx2word[target.data[0]])
            print('out: ', idx2word[np.argmax(out.data.numpy())])
            if get_probs:
                probs_list = [(out.data.numpy()[0,j],idx2word[j]) for j in range(out.data.numpy().shape[1])]
                print(sorted(probs_list,reverse=True)[:num_words])
            probs = [pr[0] for pr in sorted(probs_list,reverse=True)[:num_words]]
            plt.bar(np.arange(len(probs)), np.divide(np.exp(probs),np.sum(np.exp(probs)))) #
            plt.xticks(np.arange(len(probs)), [pr[1] for pr in sorted(probs_list,reverse=True)[:num_words]])
            plt.ylabel('Probability Distribution')
            plt.show()
    accuracy = n_corr/count*100
    print(accuracy)
    return accuracy

# In[6]:

def test(model,test_dt_bow,test_dt_pe):
    test_shape = test_dt_bow.shape
    n_corr = 0;
    count = 0;
    for i in range(test_shape[0]):
        l_temp = 0
        tag = 'q'
        if(test_dt_bow[i,-1]==-1):
            tag = 's'
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
        elif(test_dt_bow[i,-1]==-2):
            tag = 'f'
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
        else:
            count+=1
            out = model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            target = Variable(torch.from_numpy(np.array([test_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
            n_corr += comp(out,target)
    accuracy = n_corr/count*100
    print(accuracy)
    return accuracy

