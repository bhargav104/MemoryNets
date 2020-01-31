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

parser.add_argument('--net-type', type=str, default='RNN', choices=['RNN', 'MemRNN', 'RelMemRNN'], help='options: RNN, MemRNN, RelMemRNN')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=200, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
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
parser.add_argument('--lastk', type=int, default=10, help='Size of short term bucket')
parser.add_argument('--rsize', type=int, default=15, help='Size of long term bucket')
parser.add_argument('--cutoff', type=float, default=0.0, help='Cutoff for long term bucket')
parser.add_argument('--clip', type=float, default=1.0, help='Clip norms value')

args = parser.parse_args()


def onehot(inp):
    # print(inp.shape)
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
        accuracy = 0
        va = []
        rlist = []
        for i in range(len(x)):
            if args.onehot:
                inp = onehot(x[i])
            else:
                inp = x[i]
            hidden, vals, rpos = self.rnn.forward(inp, hidden)
            va.append(vals)
            rlist.append(rpos)
            out = self.lin(hidden)
            loss += self.loss_func(out, y[i].squeeze(1))

            if i >= T + args.c_length:
                preds = torch.argmax(out, dim=1)
                actual = y[i].squeeze(1)

                correct = preds == actual

                accuracy += correct.sum().item()
        accuracy /= (args.c_length * x.shape[1])
        loss /= (x.shape[0])
        return loss, accuracy, va, torch.stack(rlist)

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

    for i in range(100000):

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

        loss, accuracy, vals, _ = net.forward(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        save_norms.append(norm)
        #writer.add_scalar('Grad Norms', norm, i)

        losses.append(loss.item())
        if orthog_optimizer:
            net.rnn.orthogonal_step(orthog_optimizer)

        optimizer.step()
        accs.append(accuracy)
        '''
        if i % 250 == 0:
            tl, ta, vals, rrs = net.forward(tx, ty)
            hist = net.rnn.long_scores.squeeze(1).detach().cpu().numpy()
            fig, ax = plt.subplots()
            plt.bar(np.arange(120), hist)
            av = []
            title = ''
            for j in range(120):
                if xx[j].item() != 0 and xx[j].item() != 10:
                    av.append(j)
                    title = title + str(j) + ' '
            mat = np.zeros((120, 120))
            for j in range(120):
                if vals[j][0] is None:
                    continue
                #avg = torch.sum(vals[j][1], dim=1) / vals[j][1].size(1)
                for k in range(vals[j][1].size(0)):
                    #mat[j][k] = vals[j][1][k][0]
                    adv = 0
                    if j > 10:
                        adv = j - 10
                    mat[j][k+adv] = vals[j][1][k][0]
                    adv = 0
                    if j > args.lastk:
                        adv = j - args.lastk
                    if k < args.lastk:
                        #print(j, k+adv, vals[j][1][k][0])
                        #time.sleep(0.1)
                        mat[j][k+adv] = vals[j][1][k][0]
                    elif rrs[j][0][k-args.lastk].item() != -1.0:
                        mat[j][int(rrs[j][0][k-args.lastk].item())] = vals[j][1][k][0]
            fig, ax = plt.subplots(figsize=(15,10))
            ax = sns.heatmap(mat, cmap='Greys')
            ax.set_title(title)
            name = 'step_' + str(i)  + '_acc_' + str(ta) +'_norm_' + str(norm)  
            plt.savefig('heatmaps_denoise/' + name + '.png')
            plt.close(fig)
        '''
        if args.log and len(accs) == 200:
            v1 = sum(accs) / len(accs)
            v2 = sum(losses) / len(losses)
            writer.add_scalar('Loss', v2, lc)
            writer.add_scalar('Accuracy', v1, lc)
            lc += 1
            accs, losses = [], []
            torch.save(net.state_dict(), './reldenoiselogs/' + args.name + '.pt')
        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'.format(i + 1, time.time() - s_t,
                                                                                       loss.item(), accuracy))

    '''
    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(losses, fp)

    with open(SAVEDIR + '{}_Train_Accuracy'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(accs, fp)

    with open(SAVEDIR + '{}_Grad_Norms'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(save_norms, fp)

    save_checkpoint({
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_step': i
    },
        '{}_{}.pth.tar'.format(NET_TYPE, i)
    )
    '''
    torch.save(net.state_dict(), './reldenoiselogs/' + args.name + '.pt')
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
    #print(x.squeeze(2).squeeze(0))
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
    #print(acc)
    #sys.exit(0)
    av = [0]
    for i in range(120):
        if xx[i].item() != 0 and xx[i].item() != 10:
            av.append(i)
    '''
    deltas = []
    for i in range(1, 120):
        diff = net.rnn.memory[i] - net.rnn.memory[i-1]
        val = torch.sum(diff ** 2).item()
        deltas.append(val)
    plt.plot(range(1, 120), deltas)
    plt.scatter(np.array(av)[1:], np.zeros(10), c='orange')
    plt.title('Change in hidden state Denoise task')
    plt.xlabel('t')
    plt.ylabel('delta h')
    plt.savefig('denoiselogs/delta.png')
    sys.exit(0)
    '''
    ctr = 1

    for (a, b) in vals:
        if a is None:
            continue
        
        mv = torch.argmax(a.squeeze(1)).item()
        #print(ctr, mv, xx[mv].item())
        ctr += 1
        for i in range(11):
            if a.size(0) > av[i]:
                a1[i].append(b[av[i]][0].item())
        #a2.append(b[9][0].item())

    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'pink', 'orange', 'purple']
    #legs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    legs = []
    for i in range(11):
        legs.append(str(av[i]))
    for i in range(11):
        plt.plot(range(av[i]+1, 120), a1[i], clrs[i], label=legs[i])
    plt.legend()
    #plt.plot(a1)
    #plt.plot(a2)
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
SAVEDIR = './saves/denoisetask/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)
'''
if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
with open(SAVEDIR + 'hparams.txt', 'w') as fp:
    for key, val in args.__dict__.items():
        fp.write(('{}: {}'.format(key, val)))
'''
if args.log:
    writer = SummaryWriter('./reldenoiselogs/' + args.name + '/')

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
