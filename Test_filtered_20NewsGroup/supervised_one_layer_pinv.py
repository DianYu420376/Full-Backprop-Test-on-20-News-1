
# coding: utf-8

# In[1]:


'''
Test on filtered 20NewsGroup dataset
supervised one layer
'''
save_PATH = 'saved_data/'
save_filename = 'supervised_one_layer_pinv'


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


from data_loading import sparsedata_L2, data, Y1


# In[4]:


n = data.shape[0]
m = data.shape[1]
k1 = 20
k2 = 6
net = Deep_NMF([m, k1])
criterion = Fro_Norm()
pinv = PinvF.apply
# should be changed
dataset = sparsedata_L2(1000*data[0:220,:], 1000*Y1[0:220,:])


# In[6]:


lr = 5000
batchsize = 100
lambd = 1e-4
epoch = 7

for epo in range(epoch):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    for (i,(inputs, label)) in enumerate(dataloader):
        net.zero_grad()
        
        inputs = torch.reshape(inputs, [inputs.shape[0], inputs.shape[2]])
        label = torch.reshape(label, [label.shape[0], label.shape[2]])
        inputs, label = Variable(inputs), Variable(label.double())
        
        S_lst = net(inputs)
        S = S_lst[-1]
        B = torch.mm(pinv(S), label)
        pred = torch.mm(S,B)
        
        loss1 = criterion(torch.mm(S, net.lsqnonneglst[0].A), inputs)
        loss2 = criterion(pred, label)
        loss = loss1 + lambd*loss2
        print('epoch = ', epo, 'batch = ',i)
        print(loss.data, loss1.data, lambd*loss2.data)
        
        loss.backward()
        for A in net.parameters():
            A.data = A.data.sub_(lr*A.grad.data)
            A.data = A.data.clamp(min = 0)


# In[ ]:


np.savez(save_PATH+save_filename, param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[8]:


history = Writer()
# should be changed
n = 220
for A in net.parameters():
    A.requires_grad = False
for i in range(n):
    inputs, label = dataset[i]
    S_lst = net(inputs)
    S = S_lst[-1]
    history.add_tensor('S',S)
    if (i+1)% 100 == 0:
        print('current at batch:', i+1)


# In[10]:


S_lst = history.get('S')
S = torch.cat(S_lst,0)
print(S.shape)


# In[12]:


S_np = S.numpy()
inv_S = np.linalg.pinv(S_np)
Y_sub = Y1[0:n,:]
Y_pred = S_np@(inv_S@Y_sub)
print(np.sum(np.argmax(Y_pred,1) == np.argmax(Y_sub,1))/n)

