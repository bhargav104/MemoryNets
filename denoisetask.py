import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pickle
import argparse
from tensorboardX import SummaryWriter
import time
import glob
from common import henaff_init, cayley_init
from utils import str2bool, select_network
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN',
                    help='options: RNN, MemRNN, RelMemRNN, LSTM, RelLSTM')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=200, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=100, help='random seed')
parser.add_argument('--labels', type=int, default=9, help='number of labels in the output and input')
parser.add_argument('--c-length', type=int, default=10, help='sequence length')
parser.add_argument('--nonlin', type=str, default='modrelu', help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--vari', type=str2bool, default=False, help='variable length')
parser.add_argument('--lr', type=float, default=1e-3) #0.0001
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming", help='input weight matrix initialization')
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--onehot', type=str2bool, default=True)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--log', action='store_true', default=False, help='Use tensorboardX')
parser.add_argument('--name', type=str, default='default', help='save name')
parser.add_argument('--adam', action='store_true', default=False, help='Use adam')
parser.add_argument('--load', action='store_true', default=False, help='load, dont train')
parser.add_argument('--lastk', type=int, default=15, help='Size of short term bucket')
parser.add_argument('--rsize', type=int, default=15, help='Size of long term bucket')
parser.add_argument('--cutoff', type=float, default=0.0, help='Cutoff for long term bucket')
parser.add_argument('--clip', type=float, default=1.0, help='Clip norms value')

args = parser.parse_args()


def onehot(inp):
    onehot_x = inp.new_zeros(inp.shape[0], args.labels + 2)
    return onehot_x.scatter_(1, inp.long(), 1)


def create_dataset(batch_size, T, n_sequence):
    seq = np.random.randint(1, high=args.labels, size=(batch_size, n_sequence))
    zeros1 = np.zeros((batch_size, T + n_sequence - 1))

    for i in range(batch_size):
        ind = np.random.choice(T + n_sequence - 1, n_sequence, replace=False)
        ind.sort()
        zeros1[i][ind] = seq[i]

    zeros2 = np.zeros((batch_size, T + n_sequence))
    marker = 10 * np.ones((batch_size, 1))
    zeros3 = np.zeros((batch_size, n_sequence))

    x = np.concatenate((zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros2, seq), axis=1).astype('int64')

    return torch.Tensor(x).unsqueeze(2), torch.LongTensor(y).unsqueeze(2)


class Model(nn.Module):
    def __init__(self, hidden_size, rec_net):
        super(Model, self).__init__()
        self.rnn = rec_net
        self.lin = nn.Linear(hidden_size, args.labels + 1)
        self.hidden_size = hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, x, y):
        hidden = None
        outs = []
        loss = 0
        loss2 =0
        accuracy = 0
        va = []
        rlist = []
        hiddens = []
        for i in range(len(x)):
            if args.onehot:
                inp = onehot(x[i])
            else:
                inp = x[i]
            hidden, vals, rpos = self.rnn.forward(inp, hidden)
            hidden.retain_grad()
            hiddens.append(hidden)
            va.append(vals)
            if rpos is not None:
                rlist.append(rpos)
            out = self.lin(hidden)
            loss += self.loss_func(out, y[i].squeeze(1))

            if i >= T + args.c_length:
                preds = torch.argmax(out, dim=1)
                actual = y[i].squeeze(1)
                loss2 += self.loss_func(out, y[i].squeeze(1))
                correct = preds == actual

                accuracy += correct.sum().item()
        accuracy /= (args.c_length * x.shape[1])
        loss /= (x.shape[0])
        loss2 /= (x.shape[0])

        if len(rlist) > 0:
            rlist = torch.stack(rlist)
        return loss, accuracy, va, hiddens, loss2

    def loss(self, logits, y):
        print(logits.shape)
        print(y.shape)
        print(logits.view(-1, 9))
        return self.loss_func(logits.view(-1, 9), y.view(-1))

    def accuracy(self, logits, y):
        preds = torch.argmax(logits, dim=2)[:, T + args.c_length:]

        return torch.eq(preds, y[:, T + args.c_length:]).float().mean()


def train_model(net, optimizer, batch_size, T):
    save_norms = []
    accs = []
    losses = []
    lc = 0
    tx, ty = create_dataset(1, T, args.c_length)
    xx = tx[0].squeeze(1)
    if CUDA:
        tx = tx.cuda()
        ty = ty.cuda()
    tx = tx.transpose(0, 1)
    ty = ty.transpose(0, 1)
    n_steps = 200000
    W_grads = []
    for i in range(n_steps):

        s_t = time.time()
        if args.vari:
            T = np.random.randint(1, args.T)
        x, y = create_dataset(batch_size, T, args.c_length)

        if CUDA:
            x = x.cuda()
            y = y.cuda()
        x = x.transpose(0, 1)

        y = y.transpose(0, 1)
        optimizer.zero_grad()

        loss, accuracy, vals, hidden_states, loss2 = net.forward(x, y)
        loss2.backward(retain_graph=True)
        if i % 50 == 0 or i == n_steps / 2 or i == n_steps - 1:
            plt.clf()
            plt.plot(range(len(hidden_states)),
                     [torch.norm(i.grad) for i in hidden_states])
            plt.savefig(os.path.join(SAVEDIR,
                                     'denoise_dLdh_t_{}_{}.png'.format(NET_TYPE,
                                                                    i)))
            pickle.dump(
                    [torch.norm(i.grad) for i in hidden_states], 
                    open(os.path.join(
                        SAVEDIR, 
                        'denoise_dLdh_t_{}_{}_data.pkl'.format(NET_TYPE,i)), 'wb')
                    )

        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        save_norms.append(norm)
        #writer.add_scalar('Grad Norms', norm, i)

        losses.append(loss.item())
        if orthog_optimizer:
            net.rnn.orthogonal_step(orthog_optimizer)

        optimizer.step()
        accs.append(accuracy)

        if args.log and len(accs) == 10:
            v1 = sum(accs) / len(accs)
            v2 = sum(losses) / len(losses)
            v3 = sum(save_norms) / len(save_norms)
            writer.add_scalar('Loss', v2, lc)
            writer.add_scalar('Accuracy', v1, lc)
            writer.add_scalar('dL/dw norm', v3, lc)
            lc += 1
            accs, losses, save_norms = [], [], []
            torch.save(net.state_dict(), './denoiselogs/' + args.name + '.pt')
        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'.format(i + 1, time.time() - s_t,
                                                                                       loss.item(), accuracy))
    plt.clf()
    plt.plot(range(len(W_grads)), W_grads)
    plt.savefig(os.path.join(SAVEDIR, 'denoise_dLdW_{}.png'.format(NET_TYPE)))
    pickle.dump(W_grads,
            open(os.path.join(SAVEDIR,'denoise_dLdW_{}_data.pkl'.format(NET_TYPE)), 'wb'))


    torch.save(net.state_dict(), './denoiselogs/' + args.name + '.pt')
    return


def load_model(net, optimizer, fname):
    if fname == 'l':
        print(SAVEDIR)
        list_of_files = glob.glob(SAVEDIR + '*')
        print(list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        print('Loading {}'.format(latest_file))

        check = torch.load(latest_file)
        net.load_state_dict(check['state_dict'])
        optimizer.load_state_dict(check['optimizer'])

    else:
        check = torch.load(fname)
        net.load_state_dict(check['state_dict'])
        
        optimizer.load_state_dict(check['optimizer'])
    epoch = check['epoch']
    return net, optimizer, epoch

def load_function():
    
    net.load_state_dict(torch.load('denoiselogs/' + args.name + '.pt'))
    x, y = create_dataset(1, T, args.c_length)
    xx = x.squeeze(2).squeeze(0)
    if CUDA:
        x = x.cuda()
        y = y.cuda()
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    a1 = []
    for i in range(11):
        a1.append([])
    _, acc,  vals = net.forward(x, y)
    av = [0]
    for i in range(120):
        if xx[i].item() != 0 and xx[i].item() != 10:
            av.append(i)

    ctr = 1

    for (a, b) in vals:
        if a is None:
            continue
        
        mv = torch.argmax(a.squeeze(1)).item()
        ctr += 1
        for i in range(11):
            if a.size(0) > av[i]:
                a1[i].append(b[av[i]][0].item())

    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'pink', 'orange', 'purple']
    legs = []
    for i in range(11):
        legs.append(str(av[i]))
    for i in range(11):
        plt.plot(range(av[i]+1, 120), a1[i], clrs[i], label=legs[i])
    plt.legend()
    plt.savefig('fig.png')


def save_checkpoint(state, fname):
    filename = SAVEDIR + fname
    torch.save(state, filename)


nonlins = ['relu', 'tanh', 'sigmoid', 'modrelu']
nonlin = args.nonlin.lower()
if nonlin not in nonlins:
    nonlin = 'none'
    print('Non lin not found, using no nonlinearity')

random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
decay = args.weight_decay
udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}'.format(args.nhid, nonlin, args.lr, args.batch,
                                                                   args.rinit, args.iinit, decay)
if args.onehot:
    udir = 'onehot/' + udir
LOGDIR = './logs/denoisetask/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)
SAVEDIR = './saves/denoisetask/gradtest/{}/{}/{}/{}'.format(NET_TYPE, udir, random_seed, args.name)

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
with open(SAVEDIR + 'hparams.txt', 'w') as fp:
    for key, val in args.__dict__.items():
        fp.write(('{}: {}'.format(key, val)))

if args.log:
    writer = SummaryWriter('./denoiselogs/' + args.name + '/')

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

if args.onehot:
    inp_size = args.labels + 2
else:
    inp_size = 1
hid_size = args.nhid
T = args.T
batch_size = args.batch
out_size = args.labels + 1

rnn = select_network(NET_TYPE, inp_size, hid_size, nonlin, args.rinit, args.iinit, CUDA, args.lastk, args.rsize)
net = Model(hid_size, rnn)
net.rnn.T = args.T + 20
net.rnn.cutoff = args.cutoff
if CUDA:
    net = net.cuda()
    net.rnn = net.rnn.cuda()

print('Denoise task')
print(NET_TYPE)
print('Cuda: {}'.format(CUDA))
print(nonlin)

l2_norm_crit = nn.MSELoss()

orthog_optimizer = None

if args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=args.alpha)

if args.load:
    load_function()
    sys.exit(0)

train_model(net, optimizer, batch_size, T)
