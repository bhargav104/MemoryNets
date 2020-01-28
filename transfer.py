import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
from tensorboardX import SummaryWriter
import time
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from common import henaff_init, cayley_init, random_orthogonal_init
from utils import str2bool, select_network

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', choices=['RNN', 'MemRNN', 'RelMemRNN'], help='options: RNN, MemRNN, RelMemRNN')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=300, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--labels', type=int, default=8, help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=10, help='sequence length')
parser.add_argument('--nonlin', type=str, default='modrelu',
                    choices=['none', 'relu', 'tanh', 'modrelu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--vari', type=str2bool, default=False, help='variable length')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier", help='input weight matrix initialization')
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--onehot', type=str2bool, default=True)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--adam', action='store_true', default=False, help='Use adam')
parser.add_argument('--name', type=str, default='default', help='save name')
parser.add_argument('--log', action='store_true', default=False, help='Use tensorboardX')
parser.add_argument('--load', action='store_true', default=False, help='load, dont train')
parser.add_argument('--lastk', type=int, default=10, help='Size of short term bucket')
parser.add_argument('--rsize', type=int, default=15, help='Size of long term bucket')
parser.add_argument('--cutoff', type=float, default=0.0, help='Cutoff for long term bucket')
parser.add_argument('--clip', type=float, default=1.0, help='Clip norms value')

args = parser.parse_args()

def generate_copying_sequence(T, labels, c_length):
    items = [1, 2, 3, 4, 5, 6, 7, 8, 0, 9]
    x = []
    y = []

    ind = np.random.randint(labels, size=c_length)
    for i in range(c_length):
        x.append([items[ind[i]]])
    for i in range(T - 1):
        x.append([items[8]])
    x.append([items[9]])
    for i in range(c_length):
        x.append([items[8]])

    for i in range(T + c_length):
        y.append([items[8]])
    for i in range(c_length):
        y.append([items[ind[i]]])

    x = np.array(x)
    y = np.array(y)

    return torch.FloatTensor([x]), torch.LongTensor([y])


def create_dataset(size, T, c_length=10):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_copying_sequence(T, 8, c_length)
        sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)  #

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y


def onehot(inp):
    # print(inp.shape)
    onehot_x = inp.new_zeros(inp.shape[0], args.labels + 2)
    return onehot_x.scatter_(1, inp.long(), 1)


class Model(nn.Module):
    def __init__(self, hidden_size, rec_net):
        super(Model, self).__init__()
        self.rnn = rec_net

        self.lin = nn.Linear(hidden_size, args.labels + 1)
        self.hidden_size = hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y):

        va = []
        hidden = None
        hiddens = []
        loss = 0
        accuracy = 0
        attn = 1.0
        self.rnn.app = 1
        rlist = []
        for i in range(len(x)):
            #if i >= 11:
            #    self.rnn.app = 0
            if args.onehot:
                inp = onehot(x[i])
                hidden, vals, rpos = self.rnn.forward(inp, hidden, attn)
            else:
                hidden, vals, rpos = self.rnn.forward(x[i], hidden, attn)
            rlist.append(rpos)
            va.append(vals)
            hidden.retain_grad()
            hiddens.append(hidden)
            out = self.lin(hidden)
            loss += self.loss_func(out, y[i].squeeze(1))

            if i >= T + args.c_length:
                preds = torch.argmax(out, dim=1)
                actual = y[i].squeeze(1)
                correct = preds == actual

                accuracy += correct.sum().item()

        accuracy /= (args.c_length * x.shape[1])
        loss /= (x.shape[0])
        return loss, accuracy, hiddens, va, None

nonlins = ['relu', 'tanh', 'sigmoid', 'modrelu']
nonlin = args.nonlin.lower()

random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
decay = args.weight_decay
hidden_size = args.nhid

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

inp_size = 1
T = args.T
batch_size = args.batch
out_size = args.labels + 1
if args.onehot:
    inp_size = args.labels + 2
rnn = select_network(NET_TYPE, inp_size, hidden_size, nonlin, args.rinit, args.iinit, CUDA, args.lastk, args.rsize)
net = Model(hidden_size, rnn)

net.load_state_dict(torch.load('relcopylogs/' + args.name + '.pt'))

net.rnn.T = args.T + 20
net.rnn.cutoff = args.cutoff
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()

print('Copy task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
print(nonlin)
print(hidden_size)
avg = 0
for i in range(100):
	x, y = create_dataset(100, T, args.c_length)
	if CUDA:
		x = x.cuda()
		y = y.cuda()
	x = x.transpose(0, 1)
	y = y.transpose(0, 1)

	with torch.no_grad():
		_, acc, _, _, _ = net.forward(x, y)
	avg += acc
	print(i+1, acc)

avg = avg / 100.0

print('final acc', avg)