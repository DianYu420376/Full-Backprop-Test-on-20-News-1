
# coding: utf-8

# In[18]:


'''
Test on filtered 20NewsGroup dataset
semi_60 one layer
'''
save_PATH = 'saved_data/'
save_filename = 'semi_60_one_layer_pinv'


# In[19]:


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


# In[20]:


from data_loading import sparsedata_L2, data, Y1, L1


# In[21]:


n = data.shape[0]
m = data.shape[1]
k1 = 20
k2 = 6
net = Deep_NMF([m, k1])
criterion = Fro_Norm()
pinv = PinvF.apply
dataset = sparsedata_L2(1000*data, 1000*Y1, L1)


# In[22]:


lr = 5000
batchsize = 200
lambd = 1e-4
epoch = 11
loss_lst = []
for epo in range(epoch):
    total_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = True)
    for (i,(inputs, label, l_batch)) in enumerate(dataloader):
        net.zero_grad()
        
        inputs = torch.reshape(inputs, [inputs.shape[0], inputs.shape[2]])
        label = torch.reshape(label, [label.shape[0], label.shape[2]])
        l_batch = torch.reshape(l_batch,[l_batch.shape[0], l_batch.shape[2]])
        inputs, label = Variable(inputs), Variable(label.double())
        
        S_lst = net(inputs)
        S = S_lst[-1]
        l_batch = l_batch[:,0]
        observed = l_batch == 1
        S_observed = S[observed,:]
        label_observed = label[observed,:]
        B = torch.mm(pinv(S_observed),label_observed)
        pred = torch.mm(S,B)
        
        loss1 = criterion(torch.mm(S, net.lsqnonneglst[0].A), inputs)
        loss2 = criterion(pred, label)
        loss = loss1 + lambd*loss2
        print('epoch = ', epo, 'batch = ',i)
        print(loss.data, loss1.data, lambd*loss2.data)
        sys.stdout.flush()
        loss_lst.append(loss.data)
        total_loss += loss.data
        
        loss.backward()
        for A in net.parameters():
            A.data = A.data.sub_(lr*A.grad.data)
            A.data = A.data.clamp(min = 0)

    print('epoch = ', epo)
    print('total_loss = ', total_loss)


# In[30]:


from matplotlib import pyplot as plt
plt.plot(loss_lst[200:300])
plt.show()


# In[23]:


np.savez(save_PATH+save_filename, param_lst = list(net.parameters()), loss_lst = loss_lst)


# In[37]:


history = Writer()
n = 18846
for A in net.parameters():
    A.requires_grad = False
for i in range(n):
    inputs, label, l_batch = dataset[i]
    S_lst = net(inputs)
    S = S_lst[-1]
    history.add_tensor('S',S)
    if (i+1)% 100 == 0:
        print('current at batch:', i+1)


# In[38]:


S_lst = history.get('S')
S = torch.cat(S_lst,0)
print(S.shape)


# In[39]:


S_np = S.numpy()
inv_S = np.linalg.pinv(S_np)
Y_sub = Y1[0:n,:]
Y_pred = S_np@(inv_S@Y_sub)
print(np.sum(np.argmax(Y_pred,1) == np.argmax(Y_sub,1))/n)

np.savez(save_PATH + save_filename, 
param_lst = [A.data.numpy() for A in net.parameters()] , loss_lst = loss_lst, S = S_np, pred=Y_pred)

