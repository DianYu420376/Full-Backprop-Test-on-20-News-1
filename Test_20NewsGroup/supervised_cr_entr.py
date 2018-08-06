
# coding: utf-8

# In[131]:


'''
Trying out cross entropy version
'''
save_PATH = 'saved_data/'
save_filename = 'supervised_one_layer_cr_entr'


# In[134]:


import sys
package_dir = '../full_backprop_package/'
sys.path.append(package_dir)


# In[135]:


from twenty_news_group_data_loading import data, Y, target,sparsedata_cr_entr
from lsqnonneg_module import LsqNonneg
from deep_nmf import Deep_NMF, Fro_Norm
from writer import Writer


# In[136]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


# In[137]:


m = data.shape[1]
k = 20
c = 20
net = LsqNonneg(m, k)
W = Variable(torch.randn(k,c).double(), requires_grad = True)
dataset = sparsedata_cr_entr(1000*data, target)


# In[138]:


cr1 = Fro_Norm()
cr2 = nn.CrossEntropyLoss()

lambd = 1
lr = 1000
epoch = 20
loss_lst = []
for epo in range(epoch):
    total_loss = 0
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size = 2000)
    for (i,(inputs, label)) in enumerate(dataloader):
        net.zero_grad()
        inputs = inputs.view(inputs.shape[0], -1)
        label = label.view(-1)
        inputs, label = Variable(inputs), Variable(label)

        S = net(inputs)
        pred = torch.mm(S, W)

        #loss1 = cr1(inputs, torch.mm(S, net.A))
        loss2 = cr2(pred, label)
        loss = lambd*loss2# + loss1
        print(loss.data)
        sys.stdout.flush()
        total_loss += loss.data
        loss_lst.append(loss.data)
        loss.backward()
        net.A.data = net.A.data.sub_(lr*net.A.grad.data)
        net.A.data = net.A.data.clamp(min = 0)

        W.data = W.data.sub_(lr*W.grad.data)
        W.grad.data.zero_()
    print('epoch = ', epo, total_loss)

# In[127]:


n = 18846
history = Writer()
net.A.requires_grad = False
W.requires_grad = False
for i in range(n):
    inputs, label = dataset[i]
    S = net(inputs)
    pred = torch.mm(S, W)
    history.add_tensor('S', S)
    history.add_tensor('pred', pred)
    if (i+1)%100 == 0:
        print('current at batch:', i+1)


# In[128]:


S_lst = history.get('S')
pred_lst = history.get('pred')
S = torch.cat(S_lst,0)
pred = torch.cat(pred_lst, 0)


# In[ ]:


np.savez(save_PATH + save_filename,  S = S.data, pred = pred.data, A = net.A.data, W = W.data, loss_lst = loss_lst)


# In[129]:


print(torch.sum((torch.argmax(pred,1) == torch.from_numpy(target)).float())/n)

