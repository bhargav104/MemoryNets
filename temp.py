import numpy as np
import torch

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[[1,1], [1,1]], [[1,1], [1,1]], [[1,1], [1,1]]])
print(torch.mul(a.unsqueeze(2).expand_as(b), b))