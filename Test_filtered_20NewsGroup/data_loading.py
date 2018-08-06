
# coding: utf-8

# In[14]:


# import package
import sys
package_dir = '../full_backprop_package/'
sys.path.append(package_dir)
import torch
from torch.autograd import Variable
from deep_nmf import Deep_NMF, Energy_Loss_Func
from writer import Writer
import scipy.io as sio
import numpy as np
import torch.utils.data


# In[19]:


filename = 'Twenty_newgroups_formatted_full'
X = sio.loadmat(filename)
data = X['X'].T
Y1 = np.double(X['Ysub'].T)
L1 = np.double(np.zeros(Y1.shape))
observed_num = 11314
L1[0:observed_num,:] = 1
Y2 = np.double(X['Ysuper'].T)
L2 = np.double(np.zeros(Y2.shape))
L2[0:observed_num,:] = 1


# In[25]:


class sparsedata_L2(torch.utils.data.Dataset):
    def __init__(self, *args, transform = None):
        '''
        Note: The data must be the first element in *args
        '''
        self.inputs_lst = list(args)
        self.len = self.inputs_lst[0].shape[0]
        self.feature_num_lst = [self.inputs_lst[i].shape[1] for i in range(len(self.inputs_lst))]
        self.transform = transform
    def __getitem__(self, index):
        inputs = self.inputs_lst[0][index,:]
        if self.transform is not None:
            inputs = self.transform(inputs)
        inputs = inputs.todense()
        inputs = torch.from_numpy(inputs)
        inputs_lst = [inputs]
        for i in range(1,len(self.inputs_lst)):
            inputs = self.inputs_lst[i][index,:]
            inputs = torch.from_numpy(inputs)
            inputs = torch.reshape(inputs, (-1, self.feature_num_lst[i]))
            inputs_lst.append(inputs)
        return tuple(inputs_lst)
        
        
    def __len__(self):
        return self.len

