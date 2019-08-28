import torch
import torch.nn as nn
from common import modrelu, henaff_init

verbose = False

class RNN(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=True, cuda=False, r_initializer=None,
                 i_initializer=nn.init.xavier_normal_):
        super(RNN, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        self.params = []
        self.orthogonal_params = []

        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
            self.params.append(self.nonlinearity.b)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.V = nn.Linear(hid_size, hid_size, bias=False)
        self.params.append(self.U.weight)
        if bias:
            self.params.append(self.U.bias)
        self.i_initializer = i_initializer
        self.r_initializer = r_initializer

        self.reset_parameters()

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if not isinstance(type(self.r_initializer),type(torch.nn.init.kaiming_normal_)) and not isinstance(type(self.r_initializer),type(torch.nn.init.xavier_normal_)):
            self.V.data = torch.as_tensor(self.r_initializer(self.hidden_size))
        else:
            self.r_initializer(self.V.weight.data)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size,requires_grad=True)

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h = self.nonlinearity(h)
        return h