
# coding: utf-8

# In[11]:


# debugging the supervised version of deep NMF
import sys
package_dir = '../full_backprop_package/'
sys.path.append(package_dir)
import torch
from torch.autograd import Variable
from auxillary_functions import *
from pinv import PinvF
import numpy as np
from writer import Writer
from matplotlib import pyplot as plt
from lsqnonneg_module import LsqNonneg, LsqNonnegF
from deep_nmf import Fro_Norm
from pinv import PinvF
import torch.nn as nn
from twenty_news_group_data_loading import Y, target


# In[38]:


# Testing on random generated dataset to see if it is the misuse of pytorch that caused the problem

# Interesting Fact : L2 norm and crossentropy are completely different! For crossentropy it can get the 400 sample right,
# but the L2 just get around 310 at most

# The loss curve for crossentropy is super weird... still looking into what has happened

# Doing it stochastically doesn't seem to affect the accuracy.
m = 100
n = 400
k = 20
net = LsqNonneg(m,k)
cr1 = Fro_Norm()
cr2 = Fro_Norm()
cr3 = nn.CrossEntropyLoss()

data = torch.abs(torch.randn(n, m)).double()
label = torch.from_numpy(Y[0:n,:]).double()
data, label = Variable(data), Variable(label)

label = torch.from_numpy(target[0:n]).long()
#label = torch.from_numpy(Y[0:n,:]).double()

W = Variable(torch.randn(k,k).double(),requires_grad = True)
epoch = 1000
lr = 10000
loss_lst = []
grad_lst = []
for epo in range(epoch):
    net.zero_grad()
    randperm = torch.randperm(n)
    data_shuffle = data[randperm, :]
    label_shuffle = label[randperm]
    for i in range(1):
        inputs = data_shuffle[i*n:(i+1)*n,:]
        #label_ = label[i*400:(i+1)*400,:]
        label_ = label_shuffle[i*n:(i+1)*n]
        net.zero_grad()
        S = net(inputs)
        #pred = torch.mm(S,torch.mm(f(S),label_))
        pred = torch.mm(S,W)
        classification = cr3(pred, label_)
        loss = classification
        loss.backward()
        print(epo, i, loss.data)
        loss_lst.append(loss.data)
        
        if epo > 500 and loss.data > 1:
            # check gradient
            grad_true_A = net.A.grad.data
            grad_A = torch.zeros(grad_true_A.shape)
            delta = 1e-6
            f = LsqNonnegF.apply
            for i in range(grad_A.shape[0]):
                for j in range(grad_A.shape[1]):
                    A = net.A.data
                    A_delta = A.clone()
                    A_delta[i,j] += delta
                    S_delta = f(inputs.data, A_delta)
                    pred_delta = torch.mm(S_delta, W.data)
                    loss_delta = cr3(pred_delta, label_.data)
                    grad_A[i,j] = (loss_delta - loss.data)/delta
            grad_error = torch.norm(grad_A - grad_true_A.float())
            print('the error betweem grad and numeric grad:')
            print(grad_error)
            grad_lst.append(grad_error)
            np.savez('saved_data/check_gradient',loss_lst = loss_lst, grad_lst = grad_lst)
        
        net.A.data = net.A.data.sub_(lr*net.A.grad.data)
        net.A.data = net.A.data.clamp(min = 0)
        W.data = W.data.sub_(lr*W.grad.data)
        W.grad.zero_()



# In[39]:


# S = net(data)
# #pred =  torch.mm(S,torch.mm(f(S),label))
# pred = torch.mm(S,W)
# #torch.sum(torch.argmax(label,1)== torch.argmax(pred,1))
# torch.sum(label == torch.argmax(pred,1))
# print(torch.sum(label == torch.argmax(pred,1)))
# plt.plot(loss_lst)
# plt.show()

