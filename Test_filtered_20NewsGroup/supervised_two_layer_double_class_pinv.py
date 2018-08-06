
# coding: utf-8

# In[1]:


'''
Test on filtered 20NewsGroup dataset
supervised one layer
'''
save_PATH = 'saved_data/'
save_filename = 'supervised_two_layer_double_class_pinv'


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


# In[5]:


from data_loading import sparsedata_L2, data, Y1, Y2


# In[10]:


n = data.shape[0]
m = data.shape[1]
k1 = 20
k2 = 6
net = Deep_NMF([m, k1,k2])
criterion = Fro_Norm()
pinv = PinvF.apply
dataset = sparsedata_L2(1000*data, 1000*Y1, 1000*Y2)


# In[14]:


lr = 10000
batchsize = 100
lambd1 = 1e-4
lambd2 = 1e-4
epoch = 10
loss_lst = []
for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    total_loss = 0
    for (i,(inputs, label1, label2)) in enumerate(dataloader):
        net.zero_grad()
        
        inputs = torch.reshape(inputs, [inputs.shape[0], inputs.shape[2]])
        label1 = torch.reshape(label1, [label1.shape[0], label1.shape[2]])
        label2 = torch.reshape(label2, [label2.shape[0], label2.shape[2]])
        inputs, label1, label2 = Variable(inputs), Variable(label1), Variable(label2)
        
        S_lst = net(inputs)
        S1 = S_lst[0]
        S2 = S_lst[-1]
        B1 = torch.mm(pinv(S1), label1)
        pred1 = torch.mm(S1,B1)
        B2 = torch.mm(pinv(S2), label2)
        pred2 = torch.mm(S2,B2)
        
        loss1 = criterion(torch.mm(S1, net.lsqnonneglst[0].A), inputs)
        loss2 = criterion(torch.mm(S2, net.lsqnonneglst[1].A), S1)
        loss3 = criterion(pred1, label1)
        loss4 = criterion(pred2, label2)
        loss = loss1 + loss2 + lambd1*loss3 + lambd2*loss4
        print('epoch = ', epo, 'batch = ',i)
        print(loss.data, loss1.data, loss2.data, lambd1*loss3.data, lambd2*loss4.data)
        sys.stdout.flush()
        total_loss += loss.data
        loss_lst.append(loss.data)
        
        loss.backward()
        for A in net.parameters():
            A.data = A.data.sub_(lr*A.grad.data)
            A.data = A.data.clamp(min = 0)
    print('epoch = ', epo)
    print('total_loss = ', total_loss)


# In[15]:


np.savez(save_PATH+save_filename, param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[17]:


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


# In[18]:


S1_lst = history.get('S1')
S2_lst = history.get('S2')
S1 = torch.cat(S1_lst,0)
S2 = torch.cat(S2_lst,0)
print(S1.shape)
print(S2.shape)


# In[19]:


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


# In[20]:

np.savez(save_PATH+save_filename,
         param_lst = [A.data.numpy() for A in net.parameters()], S1 = S1_np, S2 = S2_np,pred1 = Y_pred1,pred2 = Y_pred2, loss_lst = loss_lst)
