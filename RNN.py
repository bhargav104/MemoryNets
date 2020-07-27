import torch
import torch.nn as nn
from common import modrelu, henaff_init,cayley_init,random_orthogonal_init
from exp_numpy import expm
import sys
import time
import numpy as np
verbose = False

class LSTM(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin):
        super(LSTM, self).__init__()

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

        self.r_initializer = r_initializer
        self.i_initializer = i_initializer

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)
        self.V = nn.Linear(hid_size, hid_size, bias=False)
        self.i_initializer = i_initializer
        self.r_initializer = r_initializer
        self.memory = []
        self.app = 1

        self.reset_parameters()

    def reset_parameters(self):
        if self.r_initializer == "cayley":
            self.V.weight.data = torch.as_tensor(cayley_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "henaff":
            self.V.weight.data = torch.as_tensor(henaff_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "random":
            self.V.weight.data = torch.as_tensor(random_orthogonal_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == 'xavier':
            nn.init.xavier_normal_(self.V.weight.data)
        elif self.r_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.V.weight.data)
        elif self.r_initializer == 'identity':
            self.V.weight.data = torch.tensor(np.identity(self.hidden_size), dtype=torch.float)
        if self.i_initializer == "xavier":
            nn.init.xavier_normal_(self.U.weight.data)
        elif self.i_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.U.weight.data)


    def forward(self, x, hidden, reset=False):
        if hidden is None or reset:
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size,requires_grad=True)                
            self.memory = []

        h = self.U(x) + self.V(hidden)
        if self.nonlinearity:
            h = self.nonlinearity(h)
        self.memory.append(h)
        return h, (None, None), None

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
        self.ctr = 0
        self.app = 1

        self.reset_parameters()

    def reset_parameters(self):
        if self.r_initializer == "cayley":
            self.V.weight.data = torch.as_tensor(cayley_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "henaff":
            self.V.weight.data = torch.as_tensor(henaff_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "random":
            self.V.weight.data = torch.as_tensor(random_orthogonal_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == 'xavier':
            nn.init.xavier_normal_(self.V.weight.data)
        elif self.r_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.V.weight.data)
        if self.i_initializer == "xavier":
            nn.init.xavier_normal_(self.U.weight.data)
        elif self.i_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.U.weight.data)

    def forward(self, x, hidden, attn=1.0, reset=False):
        if hidden is None or reset:
            self.count = 0
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=False)
            # Initialize memory
            self.memory = []
            h = self.U(x) + self.V(hidden)
            self.st = h

        else:
            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)

            es = torch.matmul(self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs), self.v.unsqueeze(2)).squeeze(2)
            alphas = self.softmax(es)
            all_hs = torch.stack(self.memory,0)
            ct = torch.sum(torch.mul(alphas.unsqueeze(2).expand_as(all_hs), all_hs), dim=0)
            self.st = 0.5 * (all_hs[-1] + ct * attn)
            h = self.U(x) + self.V(self.st)

        if self.nonlinearity:
            h = self.nonlinearity(h)
        h.retain_grad()
        if self.app == 1:
            self.memory.append(h)
        
        if self.count == 0:
            self.count = 1
            return h, (None, None), None
        else:
            return h, (es, alphas), None

class RelMemRNN(nn.Module):
    def __init__(self, inp_size, hid_size, last_k, rsize, nonlin, bias=True, cuda=False, r_initializer=None,
                 i_initializer=nn.init.xavier_normal_):
        super(RelMemRNN, self).__init__()
        self.cudafy = cuda
        self.hidden_size = hid_size
        self.last_k = last_k
        self.rsize = rsize
        self.T = 0
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
        self.ctr = 0
        self.app = 1
        self.cutoff = 0.0
        self.reset_parameters()

    def reset_parameters(self):
        if self.r_initializer == "cayley":
            self.V.weight.data = torch.as_tensor(cayley_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "henaff":
            self.V.weight.data = torch.as_tensor(henaff_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == "random":
            self.V.weight.data = torch.as_tensor(random_orthogonal_init(self.hidden_size))
            A = self.V.weight.data.triu(diagonal=1)
            A = A - A.t()
            self.V.weight.data = expm(A)
        elif self.r_initializer == 'xavier':
            nn.init.xavier_normal_(self.V.weight.data)
        elif self.r_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.V.weight.data)
        if self.i_initializer == "xavier":
            nn.init.xavier_normal_(self.U.weight.data)
        elif self.i_initializer == 'kaiming':
            nn.init.kaiming_normal_(self.U.weight.data)

    def forward(self, x, hidden, attn=1.0, reset=False):
        if hidden is None or reset:
            self.count = 0
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=False)
            self.tcnt = -1
            self.long_scores = torch.zeros(self.T, x.shape[0], requires_grad=False).cuda()
            self.long_ctrs = torch.zeros(x.shape[0], requires_grad=False).cuda()
            self.long_mask = float('-inf') * torch.ones(self.rsize, x.shape[0], requires_grad=False).cuda()
            self.long_mem = [[] for i in range(x.shape[0])]
            self.long_ids = torch.ones(x.shape[0], self.rsize) * -1.0
            self.buck_scores = torch.zeros(x.shape[0], self.rsize)
            # Initialize short-term memory
            self.memory = []
            h = self.U(x) + self.V(hidden)
            self.st = h

        else:
            # Create the Relevant set.
            lm_list = []
            for i in range(self.rsize):
                temp_l = []
                for j in range(x.shape[0]):
                    if i < len(self.long_mem[j]):
                        temp_l.append(self.long_mem[j][i])
                    else:
                        temp_l.append(torch.zeros(self.hidden_size, requires_grad=False).cuda())
                lm_list.append(torch.stack(temp_l))
            
            long_h = torch.stack(lm_list)
            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)
            long_uas = self.Ua(long_h)
            es = torch.matmul(self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs), self.v.unsqueeze(2)).squeeze(2)
            es_long = torch.matmul(self.tanh(self.Va(self.st).expand_as(long_uas) + long_uas), self.v.unsqueeze(2)).squeeze(2)
            es_long = es_long + self.long_mask
            es_comb = torch.cat((es, es_long), dim=0)
            alphas = self.softmax(es_comb)

            det_a = alphas.detach()
            lv = max(0, self.tcnt - self.last_k + 1)
            # keep track of attention scores to see if it can transfer to relevant set
            self.long_scores[lv:(self.tcnt+1), :] += det_a[:-self.rsize, :]
            comb_hs = torch.cat((all_hs, long_h), dim=0)

            ct = torch.sum(torch.mul(alphas.unsqueeze(2).expand_as(comb_hs), comb_hs), dim=0)
            self.st = 0.5 * (comb_hs[-1-self.rsize] + ct * attn)
            h = self.U(x) + self.V(self.st)

        if self.nonlinearity:
            h = self.nonlinearity(h)
        h.retain_grad()
        self.tcnt += 1
        ret_pos = torch.zeros(x.shape[0], self.rsize)
        ret_pos.copy_(self.long_ids)
        # Decide if time step leaving short-term memory should transfer to relevant set.
        if self.tcnt >= self.last_k:
            new_mask = torch.zeros(self.rsize, x.shape[0], requires_grad=False).cuda()
            new_mask.copy_(self.long_mask)
            minp = torch.argmin(self.buck_scores, dim=1)
            for i in range(x.shape[0]):
                addpos = -1
                
                # if space is there is relevant set add it
                if len(self.long_mem[i]) < self.rsize:
                    addpos = len(self.long_mem[i])
                    self.long_mem[i].append(self.memory[0][i])
                    new_mask[addpos][i] = 0.0
                    self.buck_scores[i][addpos] = self.long_scores[self.tcnt-self.last_k][i]
                    self.long_ids[i][addpos] = self.tcnt - self.last_k
                
                # if relevant set is full, check and see if you can replace the time step with lowest relevance score
                elif self.long_scores[self.tcnt-self.last_k][i].item() > self.buck_scores[i][minp[i].item()].item():
                    addpos = minp[i].item()
                    self.long_mem[i][addpos] = self.memory[0][i]
                    new_mask[addpos][i] = 0.0
                    self.buck_scores[i][addpos] = self.long_scores[self.tcnt-self.last_k][i]
                    self.long_ids[i][addpos] = 1.0
                    self.long_ids[i][addpos] = self.tcnt - self.last_k
            
            self.long_mask = new_mask

        if self.app == 1:
            self.memory.append(h)
            if len(self.memory) > self.last_k:
                del self.memory[0]
        else:
            self.tcnt -= 1
            
        if self.count == 0:
            self.count = 1
            return h, (None, None), None
        else:
            return h, (es_comb, alphas), ret_pos