
# coding: utf-8

# In[22]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from collections import defaultdict
import pickle
from torch.autograd import Variable as Var
import torch.optim as optim
import sys
import data_transform
import model_funcs_GPU
import model_funcs
import model_funcs_pt2_GPU
import model_funcs_pt2
import random
from datetime import datetime
from tkinter import *


# In[23]:


dataset = "qa1"
model_name = "best_model.pt"


# In[28]:


train_foldername = 'en-valid-10k'
train_filename = dataset + '_train'
train_fname = 'bAbI_Data/'+str(train_foldername)+'/'+str(train_filename)

valid_foldername = 'en-valid-10k'
valid_filename = dataset + '_valid'
valid_fname = 'bAbI_Data/'+str(valid_foldername)+'/'+str(valid_filename)

test_foldername = 'en-valid-10k'
test_filename = dataset + '_test'
test_fname = 'bAbI_Data/'+str(test_foldername)+'/'+str(test_filename)

vec_fname = 'bAbI_Data/model.vec'

unk_thres = 0
pre_embed = False

train_data_BOW, valid_data_BOW, test_data_BOW, train_data_pe, valid_data_pe, test_data_pe, vocab = data_transform.get_data(
    train_fname, valid_fname, test_fname, vec_fname=vec_fname, unk_thres = unk_thres, pre_embed=pre_embed)
# print(train_data_pe[0:5])


# In[25]:


databag = []
model = torch.load('saved_models/' + model_name)
def test_visualize(model,test_dt_bow,test_dt_pe):
    this_story = []
    test_shape = test_dt_bow.shape
    n_corr = 0;
    count = 0;
    lim = 2000
    with open('variables/word2idx','rb') as handle:
        word2idx = pickle.load(handle)

    with open('variables/idx2word','rb') as handle:
        idx2word = pickle.load(handle)
#     print(idx2word)
    for i in range(test_shape[0]):
        l_temp = 0
        tag = 'q'
        if(test_dt_bow[i,-1]==-1):
            tag = 's'
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            if i < lim:
                this_story.append(('s'," ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]])))
        elif(test_dt_bow[i,-1]==-2):
            tag = 'f'
            if len(this_story) > 0:
                databag.append(this_story)
            this_story=[]
            model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
            if i < lim:
                this_story.append(('s'," ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]])))
        else:
            count+=1
            out = model(test_dt_bow[i,:-1],test_dt_pe[i][0,:-1],tag)
#             print(int(torch.from_numpy(np.array([test_dt_bow[i,-1]]))))
            target = Var(torch.from_numpy(np.array([test_dt_bow[i,-1]])).type(torch.LongTensor))
            n_corr += model_funcs.comp(out,target)
            if i < lim:
                this_story.append(('q'," ".join([idx2word[statement] for statement in test_dt_pe[i][0,:-1]]), 
                                   idx2word[np.argmax(out.data.numpy())], idx2word[target.data[0]]))
    accuracy = n_corr/count*100
    return accuracy

test_acc = test_visualize(model,test_data_BOW,test_data_pe)


# In[26]:


random.seed(datetime.now())
random.shuffle(databag)
# print(databag[0])


# In[27]:


class Application(Frame):
    def say_hi(self):
        print("hi there, everyone!")

    def createWidgets(self):
        self.NEXT = Button(self)
        self.NEXT["text"] = "NEXT STORY"
        self.NEXT["fg"]   = "red"
        self.NEXT["command"] =  self.next

        self.NEXT.pack({"side": "left"})

        self.QUIT = Button(self)
        self.QUIT["text"] = "QUIT"
        self.QUIT["fg"]   = "red"
        self.QUIT["command"] =  self.quit

        self.QUIT.pack({"side": "left"})
        
    def clear_screen(self):
        self.text.delete('1.0', END)
        
    def next(self):
        self.clear_screen()
        self.text.insert(INSERT, "STORY " + str(self.K+1) + "\n\n")
        indx_counter = 3
        for i in range(len(self.databag[self.K])):
            if (self.databag[self.K][i][0] == 'q'):
                self.text.insert(INSERT, "\n" + self.databag[self.K][i][1] + "?\n")
                self.text.insert(INSERT, "Model's Output: " + str(self.databag[self.K][i][2]) + "\n")
                self.text.insert(INSERT, "Target: " + str(self.databag[self.K][i][3]) + "\n\n")
                if self.databag[self.K][i][2] == self.databag[self.K][i][3]:
                    self.text.tag_add("out_correct", str(indx_counter+2) + ".16", str(indx_counter+2) + ".120")
                else:
                    self.text.tag_add("out_incorrect", str(indx_counter+2) + ".16", str(indx_counter+2) + ".120")
                self.text.tag_add("ques", str(indx_counter+1) + ".0", str(indx_counter+1) + ".120")
                self.text.tag_add("tgt", str(indx_counter+3) + ".8", str(indx_counter+3) + ".120")
                indx_counter += 5
            else:
                self.text.insert(INSERT, self.databag[self.K][i][1] + "\n")
                indx_counter += 1
        self.K = (self.K+1)%len(self.databag)
        self.text.tag_add("story", "1.0", "1.120")
        
    def __init__(self, databag, master=None, text=None):
        Frame.__init__(self, master)
        self.pack()
        self.text = text
        self.createWidgets()
        self.text.tag_config("story", underline=True, font="arial 15 bold")
        self.text.tag_config("ques", foreground="red")
        self.text.tag_config("out_correct", foreground="green", font="arial 13 bold")
        self.text.tag_config("out_incorrect", foreground="red", font="arial 13 bold")
        self.text.tag_config("tgt", foreground="black", font="arial 13 bold")
        self.databag = databag
        self.K=0
        self.next()
        

root = Tk()
root.configure(background='white')
headText = Text(root,width=70,height=3,font='Calibri 13 bold', padx=30,borderwidth=0)
headText.tag_configure("center", justify='center')
headText.pack()
headText.insert(INSERT, "Test Filename: " + test_filename + "\n" + "Test Accuracy on our model: " + str(test_acc) + "\n")
headText.tag_add("center", "1.0", "end")

text = Text(root,width=70,height=35,font='Calibri 13',padx=30, borderwidth=0)
text.pack()
        
app = Application(master=root, databag=databag, text=text)
app.mainloop()
root.destroy()

