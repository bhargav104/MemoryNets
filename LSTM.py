import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda=True):
        super(LSTM, self).__init__()
        self.CUDA = cuda
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        torch.nn.init.xavier_normal_(self.Wf.weight)
        torch.nn.init.xavier_normal_(self.Wi.weight)
        torch.nn.init.xavier_normal_(self.Wo.weight)
        torch.nn.init.xavier_normal_(self.Wg.weight)

    def init_states(self, batch_size):

        self.ct = torch.zeros((batch_size, self.hidden_size))
        if self.CUDA:
            self.ct = self.ct.cuda()

    def forward(self, x, hidden=None, reset=False):
        if hidden is None or reset:
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size)
                self.init_states(x.shape[0])
            else:
                hidden = hidden.detach()
                self.ct = self.ct[:x.shape[0]].detach()

        inp = torch.cat((hidden, x), 1)
        ft = self.sigmoid(self.Wf(inp))
        it = self.sigmoid(self.Wi(inp))
        ot = self.sigmoid(self.Wo(inp))
        gt = self.tanh(self.Wg(inp))
        self.ft = ft
        self.it = it
        self.gt = gt
        self.ot = ot
        self.ct = torch.mul(ft, self.ct) + torch.mul(it, gt)
        hidden = torch.mul(ot, self.tanh(self.ct))
        return hidden, (None, None), None