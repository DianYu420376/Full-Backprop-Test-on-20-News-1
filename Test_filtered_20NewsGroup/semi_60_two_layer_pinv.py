
# coding: utf-8

# In[1]:


'''
Test on filtered 20NewsGroup dataset
semi_60 two layer
'''
save_PATH = 'saved_data/'
save_filename = 'semi_60_two_layer_pinv'


# In[2]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
package_dir = '../full_backprop_package/'
sys.path.append(package_dir)
from deep_nmf import Deep_NMF, Fro_Norm
from pinv import PinvF
from writer import Writer


# In[3]:


from data_loading import sparsedata_L2, data, Y1, L1, Y2, L2


# In[7]:


n = data.shape[0]
m = data.shape[1]
k1 = 20
k2 = 6
net = Deep_NMF([m, k1, k2])
criterion = Fro_Norm()
pinv = PinvF.apply
dataset = sparsedata_L2(1000*data, 1000*Y2, L2)


# In[8]:


lr = 5000
batchsize = 100
lambd = 1e-4
epoch = 11
loss_lst = []
for epo in range(epoch):
    total_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = False)
    for (i,(inputs, label,l_batch)) in enumerate(dataloader):
        net.zero_grad()
        
        inputs = torch.reshape(inputs, [inputs.shape[0], inputs.shape[2]])
        label = torch.reshape(label, [label.shape[0], label.shape[2]])
        l_batch = torch.reshape(l_batch,[l_batch.shape[0], l_batch.shape[2]])
        inputs, label = Variable(inputs), Variable(label.double())
        
        S_lst = net(inputs)
        S1 = S_lst[0]
        S2 = S_lst[1]
        l_batch = l_batch[:,0]
        observed = l_batch == 1
        S_observed = S2[observed,:]
        label_observed = label[observed,:]
        B = torch.mm(pinv(S_observed),label_observed)
        pred = torch.mm(S2,B)
        
        loss1 = criterion(torch.mm(S1, net.lsqnonneglst[0].A), inputs)
        loss2 = criterion(torch.mm(S2, net.lsqnonneglst[1].A), S1)
        loss3 = criterion(pred, label)
        loss = loss1 + loss2 + lambd*loss3
        print('epoch = ', epo, 'batch = ',i)
        print(loss.data, loss1.data, loss2.data, lambd*loss3.data)
        sys.stdout.flush()
        loss_lst.append(loss.data)
        total_loss += loss.data
        
        loss.backward()
        for A in net.parameters():
            A.data = A.data.sub_(lr*A.grad.data)
            A.data = A.data.clamp(min = 0)

    print('epoch = ', epo)
    print('total_loss = ', total_loss)


# In[7]:


np.savez(save_PATH+save_filename, param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[10]:


history = Writer()
n = 18846
for A in net.parameters():
    A.requires_grad = False
for i in range(n):
    inputs, label1, label2 = dataset[i]
    S_lst = net(inputs)
    S1 = S_lst[0]
    S2 = S_lst[-1]
    history.add_tensor('S1',S1)
    history.add_tensor('S2',S2)
    if (i+1)% 100 == 0:
        print('current at batch:', i+1)


# In[11]:


S1_lst = history.get('S1')
S2_lst = history.get('S2')
S1 = torch.cat(S1_lst,0)
S2 = torch.cat(S2_lst,0)
print(S1.shape)
print(S2.shape)


# In[12]:


S1_np = S1.numpy()
S2_np = S2.numpy()
inv_S1 = np.linalg.pinv(S1_np)
Y_sub1 = Y1[0:n,:]
Y_pred1 = S1_np@(inv_S1@Y_sub1)
inv_S2 = np.linalg.pinv(S2_np)
Y_sub2 = Y2[0:n,:]
Y_pred2 = S2_np@(inv_S2@Y_sub2)
print(np.sum(np.argmax(Y_pred1,1) == np.argmax(Y_sub1,1))/n)
print(np.sum(np.argmax(Y_pred2,1) == np.argmax(Y_sub2,1))/n)


# In[13]:


np.savez(save_PATH+save_filename,
         param_lst = [A.data.numpy() for A in net.parameters()], S1 = S1_np, S2 = S2_np,pred1 = Y_pred1,pred2 = Y_pred2, loss_lst = loss_lst)

