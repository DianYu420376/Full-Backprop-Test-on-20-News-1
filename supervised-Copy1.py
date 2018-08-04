
# coding: utf-8

# In[88]:


'''
Training a fully supervised one layer NMF on 20 news group dataset
'''
# define some global variables
save_PATH = 'saved_data/'
save_filename = 'supervised_one_layer_pinv_run2'


# In[89]:


# import package
import sys
import torch
from torch.autograd import Variable
import Ipynb_importer
from deep_nmf import Deep_NMF, Energy_Loss_Func, Fro_Norm
from writer import Writer
from matplotlib import pyplot as plt
import numpy as np
from auxillary_functions import *
from pinv import PinvF


# In[90]:


# load the dataset for twenty news
from twenty_news_group_data_loading import data, Y, target,L20, L50, L90, sparsedata_cr_entr, sparsedata_L2


# In[164]:


# Define the network 
m = data.shape[1]
k = 20
c = 20
lambd = 1e-4
net = Deep_NMF([m, k])
loss_func = Energy_Loss_Func(lambd = lambd, classification_type = 'L2')
data_input = data*1000
dataset = sparsedata_L2(data_input, 1000*Y)
criterion = Fro_Norm()
pinv = PinvF.apply


# In[165]:


# try initializing the network with the unsupervised version
#A = np.load(save_PATH + '20_news_group_A.npy')
#net.lsqnonneglst[0].A.data = torch.from_numpy(A)


# In[166]:


# Training process!
import time
# setting training parameters
batchsize = 100
epoch = 8
lr = 5000
lr_nmf = 5000
lr_cl = 5000
loss_lst = []
# train!
for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    total_loss = 0
    for (i, (inputs, label)) in enumerate(dataloader):
        t1 = time.time()
        inputs = inputs.view([inputs.shape[0], inputs.shape[2]])
        label = label.view([label.shape[0], -1])
        inputs, label = Variable(inputs), Variable(label)
       #train the lsqnonneg layers
        net.zero_grad()
        S_lst = net(inputs)
        S = S_lst[-1]
        B = pinv(S)
        B = torch.mm(pinv(S),label)
        pred = torch.mm(S,B)
        loss = loss_func(inputs, S_lst,list(net.lsqnonneglst.parameters()),pred,label)
        loss.backward()
        loss_lst.append(loss.data)
        total_loss += loss.data
        print('current at batch:', i+1, loss.data)
        sys.stdout.flush()
        t2 = time.time()
        print(t2 - t1)
        for A in net.parameters():
            A.data = A.data.sub_(lr*A.grad.data)
        for A in net.lsqnonneglst.parameters():
            A.data = A.data.clamp(min = 0)
    print('epoch = ', epo, '\n', total_loss)


# In[96]:


# save the data
np.savez(save_PATH + save_filename,
         param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[167]:


# plot the loss curve
# plt.plot(loss_lst)
# plt.show()


# In[168]:


history = Writer()
for A in net.parameters():
    A.requires_grad = False
# should be changed
n = 18846
for i in range(n):
    if (i+1)%100 == 0:
        print('current at batch:', i+1)
    total_data = dataset[i]
    inputs,label = total_data
    S_lst = net(inputs)
    S = S_lst[-1] 
    history.add_tensor('S', S)


# In[169]:


S = history.get('S')


# In[170]:


S1 = torch.cat(S,0)
print(S1.shape)


# In[171]:


S_np = S1.numpy()
inv_S = np.linalg.pinv(S_np)
Y_sub = Y[0:n,:]
Y_pred = S_np@(inv_S@Y_sub)


# In[172]:


print(np.sum(np.argmax(Y_pred,1) == np.argmax(Y_sub,1))/n)

