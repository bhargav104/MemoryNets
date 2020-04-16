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

    def forward(self, x, hidden, reset=False):
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
        #self.V = None
        self.V = torch.stack([self.Wf.weight, self.Wi.weight, self.Wo.weight, self.Wg.weight])
        return hidden, (None, None), None

class RelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, last_k, rsize, cuda=True):
        super(RelLSTM, self).__init__()
        self.CUDA = cuda
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.T = 0
        self.last_k = last_k
        self.rsize = rsize
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(1,hidden_size))
        nn.init.xavier_normal_(self.v.data)

        torch.nn.init.xavier_normal_(self.Wf.weight)
        torch.nn.init.xavier_normal_(self.Wi.weight)
        torch.nn.init.xavier_normal_(self.Wo.weight)
        torch.nn.init.xavier_normal_(self.Wg.weight)

    def init_states(self, batch_size):

        self.ct = torch.zeros((batch_size, self.hidden_size))
        if self.CUDA:
            self.ct = self.ct.cuda()

    def forward(self, x, hidden, reset=False):
        if hidden is None or reset:
            self.count = 0
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size)
                self.init_states(x.shape[0])
            else:
                hidden = hidden.detach()
                self.ct = self.ct[:x.shape[0]].detach()
            self.tcnt = -1
            self.long_scores = torch.zeros(self.T, x.shape[0], requires_grad=False).cuda()
            self.long_ctrs = torch.zeros(x.shape[0], requires_grad=False).cuda()
            self.long_mask = float('-inf') * torch.ones(self.rsize, x.shape[0], requires_grad=False).cuda()
            self.long_mem = [[] for i in range(x.shape[0])]
            self.long_ids = torch.ones(x.shape[0], self.rsize) * -1.0
            self.buck_scores = torch.zeros(x.shape[0], self.rsize)
            self.memory = []
            self.st = hidden

        else:
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
            self.long_scores[lv:(self.tcnt+1), :] += det_a[:-self.rsize, :]
            comb_hs = torch.cat((all_hs, long_h), dim=0)
            ct = torch.sum(torch.mul(alphas.unsqueeze(2).expand_as(comb_hs), comb_hs), dim=0)
            self.st = 0.5 * (comb_hs[-1-self.rsize] + ct)

        inp = torch.cat((self.st, x), 1)
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

        self.tcnt += 1
        ret_pos = torch.zeros(x.shape[0], self.rsize)
        ret_pos.copy_(self.long_ids)

        if self.tcnt >= self.last_k:
            new_mask = torch.zeros(self.rsize, x.shape[0], requires_grad=False).cuda()
            new_mask.copy_(self.long_mask)
            minp = torch.argmin(self.buck_scores, dim=1)
            for i in range(x.shape[0]):
                addpos = -1
                
                if len(self.long_mem[i]) < self.rsize:
                    addpos = len(self.long_mem[i])
                    self.long_mem[i].append(self.memory[0][i])
                    new_mask[addpos][i] = 0.0
                    self.buck_scores[i][addpos] = self.long_scores[self.tcnt-self.last_k][i]
                    self.long_ids[i][addpos] = self.tcnt - self.last_k
                
                elif self.long_scores[self.tcnt-self.last_k][i].item() > self.buck_scores[i][minp[i].item()].item():
                    addpos = minp[i].item()
                    self.long_mem[i][addpos] = self.memory[0][i]
                    new_mask[addpos][i] = 0.0
                    self.buck_scores[i][addpos] = self.long_scores[self.tcnt-self.last_k][i]
                    self.long_ids[i][addpos] = 1.0
                    self.long_ids[i][addpos] = self.tcnt - self.last_k
            
            self.long_mask = new_mask

        self.memory.append(hidden)
        if len(self.memory) > self.last_k:
            del self.memory[0]

        if self.count == 0:
            self.count = 1
            return hidden, (None, None), None
        else:
            return hidden, (es_comb, alphas), ret_pos
