#!/usr/bin/env python
# coding: utf-8

# In[22]:


import torch, torch.nn as nn, torch.optim as opt, numpy as np
import sys,os


# In[23]:


model=nn.Sequential(nn.Linear(10,2000),nn.ReLU(),nn.Linear(2000,4))
model.load_state_dict(torch.load('model.pth'))
file=open(sys.argv[2],'r')
tmp=file.read().split('\n')
inp=[]
file1=open("Software1.txt",'w')
file2=open("Software2.txt",'w')
def fizzbuzz(i):
    if i%15==0: return 'fizzbuzz'
    if i%3==0: return 'fizz'
    if i%5==0: return 'buzz'
    return str(i)
for i in tmp:
    if i.isdigit():
        inp.append(int(i))
    else:
        break
for i in inp:
    file1.write(fizzbuzz(i))
    file1.write('\n')
n=len(inp)
# inp2=[[for char in '{:010b}'.format(i)] for i in inp]
inp2=torch.tensor([([float(j) for j in (list('{0:010b}'.format(i)))]) for i in inp])
# inp2=torch.tensor(inp2)


# In[24]:


output=model(inp2)
for i in range(n):
    res=torch.argmax(output[i])
    if res==0: file2.write(str(inp[i]))
    if res==1: file2.write('fizz')
    if res==2: file2.write('buzz')
    if res==3: file2.write('fizzbuzz')
    file2.write('\n')

