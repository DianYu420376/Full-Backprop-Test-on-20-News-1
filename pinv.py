
# coding: utf-8

# In[ ]:


'''
Date: 2018.08.01
Author: Runyu Zhang
Creating a torch autograd function and nn.Module subclass for the pseudo inverse operation
'''


# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import once_differentiable


# In[62]:


class PinvF(torch.autograd.Function):
    '''
    Define the forward and backward process for pseudo inverse calculation
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output_np = np.linalg.pinv((input.data.numpy()))
        output = torch.from_numpy(output_np).double()
        ctx.intermediate = output
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        output = ctx.intermediate
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = calc_grad_pinv(input.data,output, grad_output)
        return grad_input
        


# In[63]:


def calc_grad_pinv(A, pinvA, G):
    part1 = torch.mm(torch.mm(pinvA.t(), G), pinvA.t())
    part2 = torch.mm(G,torch.mm(pinvA.t(), pinvA))
    part3 = torch.mm(A.t(), torch.mm(pinvA.t(),part2))
    return -part1 + part2.t() - part3.t()


# In[65]:


# from torch.autograd import gradcheck
# input = torch.randn(20,100).double()
# input = Variable(torch.abs(input), requires_grad = True)
# a = [input]
# print(len(a))
# test = gradcheck(PinvF().apply, a, eps = 1e-6, atol = 0)

