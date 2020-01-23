import torch
import sys


x = torch.zeros(4, 4)
x[0] = 1.0

y = sys.float_info.max

x[x == 0.0] = float('-inf')
print(x)