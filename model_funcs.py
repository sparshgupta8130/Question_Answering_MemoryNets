
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
    def __init__(self,embedding_dim, vocab_size, num_hops = 1, max_mem_size=15,temporal=False):
        super(QuesAnsModel,self).__init__()
        self.max_mem_size = max_mem_size
        self.vocab_size = vocab_size
        self.num_hops = num_hops
        self.embedding_dim = embedding_dim
        self.memory = self.init_memory()
        self.current_mem_size = 0
        self.temporal = temporal
        for i in range(self.num_hops):
            self.embedding_A = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
            self.embedding_C = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
            torch.nn.init.xavier_normal(self.embedding_A.weight)
            torch.nn.init.xavier_normal(self.embedding_C.weight)
        self.embedding_B = torch.nn.Linear(self.vocab_size,self.embedding_dim,bias=False)
        
        self.W = torch.nn.Linear(self.embedding_dim,self.vocab_size,bias=False)
        
        self.temporal_A = torch.nn.Parameter(torch.randn(self.max_mem_size,self.embedding_dim).float())
        self.temporal_C = torch.nn.Parameter(torch.randn(self.max_mem_size,self.embedding_dim).float())
        torch.nn.init.xavier_normal(self.embedding_B.weight)
        torch.nn.init.xavier_normal(self.W.weight)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def init_memory(self):
        aux = torch.zeros((self.max_mem_size, self.vocab_size)).float()
#         for i in range(aux.shape[0]):
#             for j in range(aux.shape[1]):
#                 aux[i,j] = -10000000000
        return Variable(aux,requires_grad=False)

    def forward(self, seq, tag):
        if tag == 's':
            if self.current_mem_size < self.max_mem_size:
                self.memory[self.current_mem_size] = Variable(torch.from_numpy(seq).float()).view(1,-1)
                self.current_mem_size += 1
            else:
                aux1 = self.memory.data[1:,:].numpy()
                aux1 = np.vstack((aux1,seq))
                self.memory = Variable(torch.from_numpy(aux1).float())
            return True
        
        elif tag == 'f':    
            del self.memory
            self.memory = self.init_memory()
            self.current_mem_size = 1
            self.memory[0] = Variable(torch.from_numpy(seq).float()).view(1,-1)
            return True
        
        else:
            self.question = Variable(torch.from_numpy(seq).float()).view(1,-1)
#             self.question = Variable(torch.from_numpy(seq).float().cuda()).view(1,-1)
            ques_d = self.embedding_B(self.question)
            for i in range(self.num_hops):
                if self.temporal == True:
    #                 temp_mem = np.flipud(np.array(self.memory.data))
    #                 self.memory = Variable(torch.from_numpy(temp_mem.copy())).float()
                    current_A = self.embedding_A(self.memory) + self.temporal_A
                    current_C = self.embedding_C(self.memory) + self.temporal_C
                else:
                    current_A = self.embedding_A(self.memory)
                    current_C = self.embedding_C(self.memory)
                P = self.softmax(torch.mm(ques_d, current_A.t()).t())
                o = torch.mm(P.t(),current_C) + ques_d
                ques_d = o
            output = self.W(o)
            return output


# In[5]:


def train(model,tr_dt_bow,vd_dt_bow,opt=optim.Adam,epochs=10,eta=0.0001):
    optimizer = opt(model.parameters(),lr=eta)
    loss = torch.nn.CrossEntropyLoss()
    print(optimizer)
    tr_shape = tr_dt_bow.shape
    vd_shape = vd_dt_bow.shape
    eps = []
    l_tr = []
    l_vd = []
    accuracy_tr = []
    accuracy_vd = []
    
    for epoch in range(epochs):
        count=0;
        ################################# Training
        n_corr = 0;
        for i in range(tr_shape[0]):
            l_temp = 0
            tag = 'q'
            if(tr_dt_bow[i,-1]==-1):
                tag = 's'
                model(tr_dt_bow[i,:-1],tag)
            elif(tr_dt_bow[i,-1]==-2):
                tag = 'f'
                model(tr_dt_bow[i,:-1],tag)
            else:
                count+=1
                out = model(tr_dt_bow[i,:-1],tag)
                target = Variable(torch.from_numpy(np.array([tr_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([tr_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
                optimizer.zero_grad()
                loss_tr = loss(out,target)
                loss_tr.backward(retain_graph=True)
                optimizer.step()
                l_temp += loss_tr.data[0]
                n_corr += comp(out,target)
        acc_tr = n_corr/count*100
        l_tr.append(l_temp)
        accuracy_tr.append(acc_tr)
#         print(model.embedding_B.weight[0:2,2:9])
        
        ############################# Validation
        n_corr = 0;
        count = 0;
        for i in range(vd_shape[0]):
            l_temp = 0
            tag = 'q'
            if(vd_dt_bow[i,-1]==-1):
                tag = 's'
                model(vd_dt_bow[i,:-1],tag)
            elif(vd_dt_bow[i,-1]==-2):
                tag = 'f'
                model(vd_dt_bow[i,:-1],tag)
            else:
                count+=1
                out = model(vd_dt_bow[i,:-1],tag)
                target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
                optimizer.zero_grad()
                loss_vd = loss(out,target)
                l_temp += loss_vd.data[0]
                n_corr += comp(out,target)
        acc_vd = n_corr/count*100
        l_vd.append(l_temp)
        accuracy_vd.append(acc_vd)
        
        eps.append(epoch)
        print(epoch,'Training Loss : ',l_tr[-1],' , Training Acc : ',accuracy_tr[-1])
        print(epoch,'Validation Loss : ',l_vd[-1],' , Validation Acc : ',accuracy_vd[-1])
    print('end')
    file_path = "./observations/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('dfddf')
    file_path = "./saved_models/"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ts = strftime("%Y-%m-%d__%Hh%Mm%Ss_", gmtime())
    if os.path.exists('saved_models/model' + str(ts) + '.pt'):
        os.remove('saved_models/model' + str(ts) + '.pt')
    
    torch.save(model,'saved_models/model' + str(ts) + '.pt')    
    
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


# In[6]:


def test(model,test_dt):
    test_shape = test_dt.shape
    n_corr = 0;
    count = 0;
    for i in range(test_shape[0]):
        l_temp = 0
        tag = 'q'
        if(test_dt[i,-1]==-1):
            tag = 's'
            model(test_dt[i,:-1],tag)
        elif(test_dt[i,-1]==-2):
            tag = 'f'
            model(test_dt[i,:-1],tag)
        else:
            count+=1
            out = model(test_dt[i,:-1],tag)
            target = Variable(torch.from_numpy(np.array([test_dt[i,-1]])).type(torch.LongTensor))
#                 target = Variable(torch.from_numpy(np.array([vd_dt_bow[i,-1]])).type(torch.LongTensor).cuda())
            n_corr += comp(out,target)
    accuracy = n_corr/count*100
    print(accuracy)
    return accuracy

