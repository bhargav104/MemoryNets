import torch
import torch.nn as nn
from common import modrelu, henaff_init
from exp_numpy import expm
verbose = False

class RNN(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=True, cuda=False, r_initializer=None,
                 i_initializer=nn.init.xavier_normal_):
        super(RNN, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size

        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.V = nn.Linear(hid_size, hid_size, bias=False)
        self.i_initializer = i_initializer
        self.r_initializer = r_initializer

        self.reset_parameters()

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)
        if not isinstance(self.r_initializer,type(torch.nn.init.kaiming_normal_)) and not \
                isinstance(self.r_initializer,type(torch.nn.init.xavier_normal_)):
            self.V.weight.data = torch.as_tensor(self.r_initializer(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        else:

            self.V.weight.data = torch.as_tensor(self.r_initializer(self.hidden_size))


    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size,requires_grad=True)

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h = self.nonlinearity(h)
        return h

class MemRNN(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=True, cuda=False, r_initializer=None,
                 i_initializer=nn.init.xavier_normal_):
        super(MemRNN, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size

        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.V = nn.Linear(hid_size, hid_size, bias=False)
        self.Ua = nn.Linear(hid_size, hid_size, bias=False)
        self.Va = nn.Linear(hid_size, hid_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(1,hid_size))
        nn.init.xavier_normal_(self.v.data)

        self.i_initializer = i_initializer
        self.r_initializer = r_initializer

        self.reset_parameters()

    def reset_parameters(self):
        self.i_initializer(self.U.weight.data)

        if not isinstance(self.r_initializer, type(torch.nn.init.kaiming_normal_)) and not \
                isinstance(self.r_initializer, type(torch.nn.init.xavier_normal_)):
            self.V.weight.data = torch.as_tensor(self.r_initializer(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        else:
            self.r_initializer(self.V.weight.data)

    def forward(self, x, hidden=None):
        if hidden is None:
            self.count = 0
            hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=True)
            self.memory = []
            h = self.U(x) + self.V(hidden)
            self.st = h

        else:
            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)

            es = torch.matmul(self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs), self.v.unsqueeze(2)).squeeze(2)
            alphas = self.softmax(es)
            all_hs = torch.stack(self.memory,0)
            ct = torch.sum(torch.mul(alphas.unsqueeze(2).expand_as(all_hs),all_hs),dim=0)
            self.st = all_hs[-1] + ct
            h = self.U(x) + self.V(self.st)

        if self.nonlinearity:
            h = self.nonlinearity(h)
        h.retain_grad()
        self.memory.append(h)
        return h