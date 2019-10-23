import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import select_network, str2bool
from common import cayley_init,henaff_init,random_orthogonal_init
import pickle
import argparse
from tensorboardX import SummaryWriter
import shutil

parser = argparse.ArgumentParser(description='add task parameters')

parser.add_argument('--net-type', type=str, default='RNN', choices=['RNN', 'MemRNN'], help='options: RNN, MemRNN')
parser.add_argument('--nhid', type=int, default=128, help='hidden size of recurrent net')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--T', type=int, default=300, help='delay between sequence lengths')
parser.add_argument('--random-seed', type=int, default=400, help='random seed')
parser.add_argument('--nonlin', type=str, default='modrelu',
                    choices=['none', 'relu', 'tanh', 'modrelu', 'sigmoid'],
                    help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="xavier", help='input weight matrix initialization')
parser.add_argument('--batch', type=int, default=10)
parser.add_argument('--n-steps', type=int, default=20000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.99)

args = parser.parse_args()

def generate_adding_sequence(T):
    x = []
    sq = np.random.uniform(size=T)

    x = np.zeros((2 * T, 1))

    x[:T, 0] = sq
    fv = np.random.randint(0, T // 2, 1)[0]
    sv = np.random.randint(T // 2, T, 1)[0]

    x[T + fv] = 1.0
    x[T + sv] = 1.0

    y = torch.FloatTensor(np.array(sq[fv] + sq[sv]))
    x = torch.FloatTensor(x)
    return x, y


def create_dataset(size, T):
    d_x = []
    d_y = []
    for i in range(size):
        sq_x, sq_y = generate_adding_sequence(T)
        # sq_x, sq_y = sq_x[0], sq_y[0]
        d_x.append(sq_x)
        d_y.append(sq_y)

    d_x = torch.stack(d_x)
    d_y = torch.stack(d_y)
    return d_x, d_y


class Net(nn.Module):
    def __init__(self, hidden_size, output_size, rec_net):
        super(Net, self).__init__()
        self.rec_net = rec_net
        self.ol = nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.ol.weight.data)

    def forward(self, x, y):
        hidden = None
        hiddens = []
        for i in range(len(x)):
            hidden = self.rec_net.forward(x[i], hidden)
            hidden.retain_grad()
            hiddens.append(hidden)
        out = self.ol(hidden)
        loss = MSE_crit(out,y)
        return loss, hiddens

def train_model(net, optimizer, batch_size, T, num_steps):
    losses = []
    save_norms = []
    for i in range(num_steps):
        x,y = create_dataset(batch_size,T)

        if CUDA:
            x = x.cuda()
            y = y.cuda()
        x = x.transpose(0, 1)
        net.zero_grad()
        loss, hidden_states = net.forward(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 'inf')
        save_norms.append(norm)
        losses.append(loss.item())
        optimizer.step()

        print('Train step {}, Loss: {}'.format(i + 1, loss.item()))
    '''
    with open(SAVEDIR + '{}_losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(losses, fp)
        with open(SAVEDIR + '{}_norms'.format(NET_TYPE), 'wb') as fp:
            pickle.dump(save_norms, fp)

        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        },
            False,
            '{}_{}.pth.tar'.format(NET_TYPE,i)
        )
    '''

def save_checkpoint(state, is_best, fname):
    filename = SAVEDIR + fname
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, SAVEDIR + 'model_best.pth.tar')


random_seed = args.random_seed
NET_TYPE = args.net_type
CUDA = args.cuda
SAVEDIR = './saves/addtask/'+ str(random_seed) + '/'

weight_decay = args.weight_decay

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

inp_size = 1

hid_size = args.nhid
T = args.T
out_size = 1

MSE_crit = nn.MSELoss()
rec_net = select_network(NET_TYPE, inp_size, hid_size, args.nonlin, args.rinit, args.iinit, args.cuda)

net = Net(hid_size, out_size, rec_net)
if CUDA:
    net = net.cuda()

print(CUDA)
print(NET_TYPE)

optimizer = optim.RMSprop(net.parameters(), lr=args.lr ,alpha =args.alpha, weight_decay=args.weight_decay)

train_model(net, optimizer, args.batch, T, args.n_steps)
