class MemLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda=True):
        super(MemLSTM, self).__init__()
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
        self.app = 1
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(1, hidden_size))
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
        if reset:
            self.count = 0
            '''
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.hidden_size)
                self.init_states(x.shape[0])
            else:
                hidden = hidden.detach()
                self.ct = self.ct[:x.shape[0]].detach()
            '''
            self.ct = hidden[1]
            hidden = hidden[0]
            self.tcnt = -1

            self.memory = []
            #self.memory.append(hidden)
            self.st = hidden


        else:
            h_t, self.ct = hidden
            #self.memory.append(hidden)

            all_hs = torch.stack(self.memory)
            Uahs = self.Ua(all_hs)
            es = torch.matmul(self.tanh(self.Va(self.st).expand_as(Uahs) + Uahs), self.v.unsqueeze(2)).squeeze(2)
            alphas = self.softmax(es)
            #det_a = alphas.detach()

            ct = torch.sum(torch.mul(alphas.unsqueeze(2).expand_as(all_hs), all_hs), dim=0)
            self.st = 0.5 * (h_t + ct)

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
        self.V = torch.stack([self.Wf.weight, self.Wi.weight, self.Wo.weight, self.Wg.weight])

        if self.app == 1:
            self.memory.append(hidden)
        else:
            self.tcnt -= 1

        if self.count == 0:
            self.count = 1
            return (hidden, self.ct)
        else:
            return (hidden, self.ct)