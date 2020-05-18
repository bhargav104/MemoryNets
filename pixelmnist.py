import torch
import math
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision as T
import argparse
import os
import glob
from LSTM import LSTM, RelLSTM

parser = argparse.ArgumentParser(description='sequential MNIST parameters')
parser.add_argument('--full', action='store_true', default=False, help='Use full BPTT')
parser.add_argument('--trunc', type=int, default=5, help='size of H truncations')
parser.add_argument('--p-full', type=float, default=0.0, help='probability of opening bracket')
parser.add_argument('--p-detach', type=float, default=1.0, help='probability of detaching each timestep')
parser.add_argument('--permute', action='store_true', default=False, help='pMNIST or normal MNIST')
parser.add_argument('--save-dir', type=str, default='default', help='save directory')
parser.add_argument('--cos', action='store_true', default=False, help='print cosine between consecutive updates')
parser.add_argument('--norms', action='store_true', default=False, help='Print gradient norms')
parser.add_argument('--vhv', action='store_true', default=False, help='print ghg values')
parser.add_argument('--lstm-size', type=int, default=100, help='width of LSTM')
parser.add_argument('--seed', type=int, default=100, help='seed value')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for adam')
parser.add_argument('--dist', action='store_true', default=False, help='distance from initialization')
parser.add_argument('--pnorm', action='store_true', default=False, help='parameter norms')
parser.add_argument('--adam', action='store_true', default=False, help='Use SGD optimizer')
parser.add_argument('--clipval', type=str, default='1.0', help='gradient clipping value')
parser.add_argument('--algo', type=str, default='LSTM', help='LSTM, RelLSTM')
parser.add_argument('--lastk', type=int, default=10, help='Size of short term bucket')
parser.add_argument('--rsize', type=int, default=10, help='Size of long term bucket')
parser.add_argument('--k', type=int, default=1, help='Attend ever k timesteps')

args = parser.parse_args()

log_dir = './newImageLogs/' + args.save_dir + '/'
best_acc = 0
try:
	status = torch.load(log_dir + 'status.pt')
	best_acc = status['best_val_acc']
	print('resumed')
	print('start epoch', status['start_epoch'], 'best val acc', status['best_val_acc'])
except OSError:
	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)
	status = {'start_epoch': 0}

writer = SummaryWriter(log_dir=log_dir)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
tensor = torch.FloatTensor

train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000))
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(50000, 60000))

trainset = T.datasets.MNIST(root='./MNIST', train=True, download=True, transform=T.transforms.ToTensor())
testset = T.datasets.MNIST(root='./MNIST', train=False, download=True, transform=T.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, sampler=train_sampler, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, sampler=valid_sampler, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)
trainloader_1 = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False, num_workers=2)

n_epochs = 200
T = 784
batch_size = 100
inp_size = 1
hid_size = args.lstm_size
out_size = 10
lr = args.lr
train_size = 60000
test_size = 10000
update_fq = 40
ktrunc = args.trunc
clipval = float(args.clipval)

class Net(nn.Module):
	
	def __init__(self, inp_size, hid_size, out_size, algo, lastk, rsize):
		super().__init__()
		if algo == 'LSTM':
			self.lstm = LSTM(inp_size, hid_size)
		elif algo == 'RelLSTM':
			self.lstm = RelLSTM(inp_size, hid_size, lastk, rsize)
			self.lstm.T = 784
		self.fc1 = nn.Linear(hid_size, out_size)

	def forward(self, x, state):
		hidden, (_, _), _ = self.lstm(x, state)
		x = self.fc1(hidden)
		return x, hidden

def test_model(model, loader, criterion, order):
	
	accuracy = 0
	loss = 0
	with torch.no_grad():
		for i, data in enumerate(loader, 1):
			test_x, test_y = data
			test_x = test_x.view(-1, 784, 1)
			test_x, test_y = test_x.to(device), test_y.to(device)
			test_x.transpose_(0, 1)
			h = None
			#c = torch.zeros(batch_size, hid_size).to(device)
			model.lstm.app = 0
			time_c = 0
			for j in order:
				if time_c % args.k == 0:
					model.lstm.app = 1
				else:
					model.lstm.app = 0
				time_c += 1

				output, h = model(test_x[j], h)

			loss += criterion(output, test_y).item()
			preds = torch.argmax(output, dim=1)
			correct = preds == test_y
			accuracy += correct.sum().item()

	accuracy /= 100.0
	loss /= 100.0

	print('test loss ' + str(loss) + ' accuracy ' + str(accuracy))
	return loss, accuracy

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

def get_flat_grads(model):
	ret = []
	for param in model.parameters():
		ret.append(param.grad.data.view(-1))
	ret = torch.cat(ret, dim=0)
	return ret

def get_flat_params(model):
	ret = []
	for param in model.parameters():
		ret.append(param.data.clone().view(-1))
	ret = torch.cat(ret, dim=0)
	return ret

def train_model(model, start_epoch, epochs, criterion, optimizer):

	global best_acc
	#best_acc = 0.0
	ctr = 0	
	global lr
	if args.permute:
		rand = np.random.RandomState(100)
		order = rand.permutation(T)
	else:
		order = np.arange(T)
	
	cc = 0
	nc = 0
	vc = 0
	old_grads = None
	test_acc = 0
	for epoch in range(start_epoch, epochs):
		#print('epoch ' + str(epoch + 1))
		epoch_loss = 0

		'''
		if epoch % update_fq == update_fq - 1 and args.sgd:
			lr = lr / 2.0
			optimizer.lr = lr
		'''
		for z, data in enumerate(trainloader, 0):
			inp_x, inp_y = data
			inp_x = inp_x.view(-1, 28*28, 1)
			inp_x, inp_y = inp_x.to(device), inp_y.to(device)
			inp_x.transpose_(0, 1)
			h = None
			#c = torch.zeros(batch_size, hid_size).to(device)
			sq_len = T
			loss = 0
			zone = 0.15
			model.lstm.app = 0
			time_c = 0
			for i in order:
				if time_c % args.k == 0:
					model.lstm.app = 1
				else:
					model.lstm.app = 0
				time_c += 1
				'''
				if args.p_detach != 1.0 and not args.full:
					val = np.random.random(size=1)[0]
					if val <= args.p_detach:
						h = h.detach()
				'''
				output, h = model(inp_x[i], h)

			loss += criterion(output, inp_y)

			model.zero_grad()
			loss.backward()
			norms = nn.utils.clip_grad_norm_(model.parameters(), clipval)
			
			optimizer.step()
			loss_val = loss.item()
			#print(z, loss_val)
			writer.add_scalar('/MNIST', loss_val, ctr)
			ctr += 1

		t_loss, accuracy = test_model(model, validloader, criterion, order)
		if best_acc < accuracy:
			best_acc = accuracy
			_, test_acc = test_model(model, testloader, criterion, order)
			torch.save(model.state_dict(), log_dir + 'best_model.pt')

		print('epoch', epoch+1, 'test accuracy ' + str(test_acc))
		writer.add_scalar('/accMNIST', accuracy, epoch)
		writer.add_scalar('/testaccMNIST', test_acc, epoch)
		status = {'start_epoch': epoch+1, 'best_val_acc': best_acc, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}
		torch.save(status, log_dir + 'status.pt')
		print('model checkpoint saved')

device = torch.device('cuda')
net = Net(inp_size, hid_size, out_size, args.algo, args.lastk, args.rsize).to(device)
if 'model_state' in status:
	net.load_state_dict(status['model_state'])
	print('model restored')
#init_param = get_flat_params(net)
#net_1 = Net(inp_size, hid_size, out_size).to(device)
criterion = nn.CrossEntropyLoss()
if not args.adam:
	optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=0.99)
else:
	optimizer = optim.Adam(net.parameters(), lr=lr)

if 'optimizer_state' in status:
	optimizer.load_state_dict(status['optimizer_state'])
	print('optimizer restored')

start_epoch = status['start_epoch']

train_model(net, start_epoch, n_epochs, criterion, optimizer)
writer.close()

'''
MNISTstoch - full, p-detach = 0.9, 0.75, 0.5, 0.25, 0.1, no forget - full, trunc 20, p-detach = 0.05, 0.01, 0.4
pMNIST - same as above

python pixelmnist.py --full --permute --lr=0.001 --k=20 --algo=RelLSTM --save-dir=pm_rl_0.001lr_20k_rms
'''
