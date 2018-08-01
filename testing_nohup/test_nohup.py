print('hello')
print('testing how this is working')
import torch
a = torch.Tensor([1,2,3,4])
print(a)
from time import time
t1 = time()
t2 = time()
while t2 - t1 < 200:
    print(t2)
    t2 = time()
