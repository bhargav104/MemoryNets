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
import wandb
from common import henaff_init, cayley_init, random_orthogonal_init
from utils import str2bool, select_network


wandb.init(project="memnet")
parser = argparse.ArgumentParser(description='auglang parameters')

parser.add_argument('--net-type', type=str, default='RNN', choices=['RNN', 'MemRNN', 'RelMemRNN', 'LSTM', 'RelLSTM'], help='options: RNN, MemRNN, RelMemRNN, LSTM, RelLSTM')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=300, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=100, help='random seed')
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
        hlist = []
        loss2 = 0
        for i in range(len(x)):
            #if i >= 11:
            #    self.rnn.app = 0
            if args.onehot:
                inp = onehot(x[i])
                hidden, vals, rpos = self.rnn.forward(inp, hidden)
            else:
                hidden, vals, rpos = self.rnn.forward(x[i], hidden)
            hlist.append(hidden)
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
                loss2 += self.loss_func(out, y[i].squeeze(1))
                accuracy += correct.sum().item()
            
        accuracy /= (args.c_length * x.shape[1])
        loss /= (x.shape[0])
        loss2 /= (x.shape[0])
        return loss, accuracy, hiddens, va, loss2

def train_model(net, optimizer, batch_size, T, n_steps):
    save_norms = []
    accs = []
    losses = []
    lc = 0
    tx, ty = create_dataset(1, T, args.c_length)
    W_grads = []
    if CUDA:
        tx = tx.cuda()
        ty = ty.cuda()
    tx = tx.transpose(0, 1)
    ty = ty.transpose(0, 1)

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
        loss, accuracy, hidden_states, vals, loss2 = net.forward(x, y)
        loss_act = loss
        loss2.backward(retain_graph=True)

        if i % 500 == 0 or i == n_steps / 2 or i == n_steps - 1:
            plt.clf()
            plt.plot(range(len(hidden_states)),
                     [torch.norm(i.grad) for i in hidden_states])
            wandb.log({"dLdh_{}".format(i): plt})
            plt.savefig(os.path.join(SAVEDIR,
                                     'copy_dLdh_t_{}_{}.png'.format(NET_TYPE,
                                                                    i)))

            pickle.dump([torch.norm(i.grad).cpu().numpy() for i in hidden_states], 
                    open(os.path.join(SAVEDIR,'copy_cldght_{}_{}_data.pickle'.format(NET_TYPE, i)), 'wb'))
        if NET_TYPE != 'LSTM':
            W_grads.append(torch.norm(net.rnn.V.weight.grad).cpu().numpy())
            wandb.log({"W grads": torch.norm(net.rnn.V.weight.grad).cpu().numpy()})
        else:
            V = torch.cat([net.rnn.Wo.weight.grad, net.rnn.Wi.weight.grad, net.rnn.Wf.weight.grad, net.rnn.Wg.weight.grad])
            W_grads.append(torch.norm(V).cpu().numpy())
            wandb.log({"W grads": torch.norm(V).cpu().numpy()})
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        #print(norm)
        save_norms.append(norm)

        losses.append(loss_act.item())

        optimizer.step()
        accs.append(accuracy)
        '''
        if i % 250 == 0:
            tl, ta, th, vals, rrs = net.forward(tx, ty)
            #print(rrs.size())
            #sys.exit(0)
            hist = net.rnn.long_scores.squeeze(1).detach().cpu().numpy()
            fig, ax = plt.subplots()
            plt.bar(np.arange(120), hist)
            mat = np.zeros((120, 120))
            for j in range(120):
                if vals[j][0] is None:
                    continue
                #avg = torch.sum(vals[j][1], dim=1) / vals[j][1].size(1)
                for k in range(vals[j][1].size(0)):
                    #mat[j][k] = vals[j][1][k][0]
                    adv = 0
                    if vals[j][1][k][0] == 0.0:
                        continue
                    if j > args.lastk:
                        adv = j - args.lastk
                    if k < 10:
                        mat[j][k+adv] = vals[j][1][k][0]
                    elif j > 10:
                        mat[j][k-10] = vals[j][1][k][0]
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
            name = 'step_' + str(i)  + '_acc_' + str(ta) +'_norm_' + str(norm)  
            plt.savefig('heatmaps_copy/' + name + '.png')
            plt.close(fig)
        '''
        if args.log and len(accs) == 100:
            v1 = sum(accs) / len(accs)
            v2 = sum(losses) / len(losses)
            writer.add_scalar('Loss', v2, lc)
            writer.add_scalar('Accuracy', v1, lc)
            lc += 1
            accs, losses = [], []
            torch.save(net.state_dict(), './relcopylogs/' + args.name + '.pt')
            #writer.add_scalar('Grad Norms', norm, i)
        
        print('Update {}, Time for Update: {} , Average Loss: {}, Accuracy: {}'.format(i + 1, time.time() - s_t,
                                                                                       loss_act.item(), accuracy))
    plt.clf()
    plt.plot(range(len(W_grads)), W_grads)
    plt.savefig(os.path.join(SAVEDIR, 'copy_dLdW_{}.png'.format(NET_TYPE)))
    pickle.dump(W_grads,
              open(os.path.join(SAVEDIR,'copy_dldW_{}_data.pickle'.format(NET_TYPE)), 'wb'))
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
        'time step': i
    },
        '{}_{}.pth.tar'.format(NET_TYPE, i)
    )
    '''
    torch.save(net.state_dict(), './relcopylogs/' + args.name + '.pt')
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
    
    net.load_state_dict(torch.load('copylogs/' + args.name + '.pt'))
    x, y = create_dataset(1, T, args.c_length)
    if CUDA:
        x = x.cuda()
        y = y.cuda()
    x = x.transpose(0, 1)
    y = y.transpose(0, 1)

    a1 = []
    for i in range(11):
        a1.append([])
    _, acc, _, vals = net.forward(x, y)
    print(acc)
    #sys.exit(0)

    '''
    deltas = []
    for i in range(1, 120):
        diff = net.rnn.memory[i] - net.rnn.memory[i-1]
        val = torch.sum(diff ** 2).item()
        deltas.append(val)
    plt.plot(range(1, 120), deltas)
    #plt.scatter(np.array(av)[1:], np.zeros(10), c='orange')
    plt.title('Change in hidden state Copy task')
    plt.xlabel('t')
    plt.ylabel('delta h')
    plt.savefig('copylogs/delta_h.png')
    sys.exit(0)
    '''
    ctr = 1
    for (a, b) in vals:
        if a is None:
            continue
        print(ctr, torch.argmax(a.squeeze(1)).item())
        ctr += 1
        for i in range(min(a.size(0), 11)):
            a1[i].append(b[i][0].item())
        #a2.append(b[9][0].item())

    clrs = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'brown', 'pink', 'orange', 'grey']
    legs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for i in range(11):
        plt.plot(range(i+1, 120), a1[i], clrs[i], label=legs[i])
    plt.legend()
    #plt.plot(a1)
    #plt.plot(a2)
    plt.savefig('fig.png')


def save_checkpoint(state, fname):
    filename = SAVEDIR + fname
    torch.save(state, filename)


nonlins = ['relu', 'tanh', 'sigmoid', 'modrelu']
nonlin = args.nonlin.lower()

random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
decay = args.weight_decay

hidden_size = args.nhid
udir = 'HS_{}_NL_{}_lr_{}_BS_{}_rinit_{}_iinit_{}_decay_{}_alpha_{}'.format(hidden_size, nonlin, args.lr, args.batch,
                                                                            args.rinit, args.iinit, decay, args.alpha)
if args.onehot:
    udir = 'onehot/' + udir

if not args.vari:
    n_steps = 20000 #100000
    LOGDIR = './logs/copytask/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)
    SAVEDIR = './saves/copytask/gradtest/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)
    print(SAVEDIR)
else:
    n_steps = 10000
    LOGDIR = './logs/varicopytask/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)
    SAVEDIR = './saves/varicopytask/{}/{}/{}/'.format(NET_TYPE, udir, random_seed)

if args.log:
    writer = SummaryWriter('./relcopylogs/' + args.name + '/')

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

if not os.path.exists(SAVEDIR):
    os.makedirs(SAVEDIR)
if not args.adam:
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=args.alpha, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
#with open(SAVEDIR + 'hparams.txt', 'w') as fp:
#    for key, val in args.__dict__.items():
#        fp.write(('{}: {}'.format(key, val)))

if args.load:
    load_function()
    sys.exit(0)

train_model(net, optimizer, batch_size, T, n_steps)
