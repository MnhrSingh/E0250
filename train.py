#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch, torch.nn as nn, torch.optim as opt, numpy as np
import os


# In[13]:


def onehot(num):
    if num%3==0:
        if num%15==0: return [0,0,0,1]
        return [0,1,0,0]
    if num%5==0: return [0,0,1,0]
    return [1,0,0,0]


# In[14]:


x_tmp=['{:010b}'.format(i) for i in range(101,1001)]
# print(x_tmp[0])
x_tmp=[[int(j) for j in i] for i in x_tmp]
# print(x_tmp[0])
x_train=torch.tensor(x_tmp,dtype=torch.float32)
# print(x_train[0])
y_train=torch.tensor([onehot(i) for i in range(101,1001)],dtype=torch.float32)


# In[15]:


model=nn.Sequential(nn.Linear(10,2000),nn.ReLU(),nn.Linear(2000,4))
lossfx=nn.MSELoss()#MSELoss, reduction=sum
optimizer=opt.Adam(model.parameters())
# print(os.getcwd())


# In[23]:


epochs=2000
for e in range(epochs):
    # if e%100==0: print(e)
    optimizer.zero_grad()
    output=model(x_train)
    loss=lossfx(output,y_train)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), 'Model/model.pth')


# In[ ]:


