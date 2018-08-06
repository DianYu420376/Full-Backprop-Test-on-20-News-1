
# coding: utf-8

# In[1]:


'''
Training a semi_50 one layer NMF on 20 news group dataset
'''
# define some global variables
save_PATH = 'saved_data/'
save_filename = 'semi_50_one_layer_pinv'


# In[2]:


import sys
package_dir = '../full_backprop_package'
sys.path.append(package_dir)
import torch
from torch.autograd import Variable
from deep_nmf import Deep_NMF, Energy_Loss_Func, Fro_Norm
from writer import Writer
from matplotlib import pyplot as plt
import numpy as np
from auxillary_functions import *
from pinv import PinvF


# In[3]:


from twenty_news_group_data_loading import data, Y, target,L20, L50, L90, sparsedata_cr_entr, sparsedata_L2


# In[40]:


m = data.shape[1]
k = 20
c = 20
lambd = 1e-4
net = Deep_NMF([m, k])
loss_func = Energy_Loss_Func(lambd = lambd, classification_type = 'L2')
data_input = data*1000
dataset = sparsedata_L2(data_input, 1000*Y, L50)
criterion = Fro_Norm()
pinv = PinvF.apply


# In[17]:


# Training process!
import time
# setting training parameters
batchsize = 150
epoch = 10
lr = 5000
lr_nmf = 5000
lr_cl = 5000
loss_lst = []
# train!
for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    total_loss = 0
    for (i, (inputs, label, L_batch)) in enumerate(dataloader):
        t1 = time.time()
        inputs = inputs.view([inputs.shape[0], inputs.shape[2]])
        label = label.view([label.shape[0], -1])
        L_batch = L_batch.view([L_batch.shape[0], -1])
        inputs, label = Variable(inputs), Variable(label)
       #train the lsqnonneg layers
        net.zero_grad()
        S_lst = net(inputs)
        S = S_lst[-1]
        l_batch = L_batch[:,0]
        observed = l_batch == 1
        S_observed = S[observed,:]
        label_observed = label[observed,:]
        #B = pinv(S_observed)
        B = torch.mm(pinv(S_observed),label_observed)
        pred = torch.mm(S,B)
        loss = loss_func(inputs, S_lst,list(net.lsqnonneglst.parameters()),pred,label,L_batch)
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


# In[37]:


# save the data
np.savez(save_PATH + save_filename,
         param_lst = [A.data.numpy() for A in net.parameters()], loss_lst = loss_lst)


# In[24]:


history = Writer()
for A in net.parameters():
    A.requires_grad = False
# should be changed
n = 18846
for i in range(n):
    if (i+1)%100 == 0:
        print('current at batch:', i+1)
    total_data = dataset[i]
    inputs,label,L_batch = total_data
    S_lst = net(inputs)
    S = S_lst[-1] 
    history.add_tensor('S', S)
    #history.add_tensor('L', L_batch)


# In[22]:


S = history.get('S')
S1 = torch.cat(S,0)
print(S1.shape)


# In[34]:


S_np = S1.numpy()
l = L50[0:n,0]
S_observed = S_np[l==1,:]
inv_S = np.linalg.pinv(S_observed)
Y_sub = Y[0:n,:]
Y_observed = Y_sub[l == 1,:]
Y_pred = S_np@(inv_S@Y_observed)
pred = np.argmax(Y_pred,1)
print(np.sum(pred == np.argmax(Y_sub,1))/n)


# In[39]:


np.savez(save_PATH + save_filename,
         param_lst = [A.data.numpy() for A in net.parameters()] , loss_lst = loss_lst, S = S_np, pred = pred)

