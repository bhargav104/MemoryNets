import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from RNN import RNN, MemRNN
from utils import str2bool, select_network
import numpy as np
import os
import pickle
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn, ntoken, ninp, nhid, tie_weights=False):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = rnn
        self.decoder = nn.Linear(nhid, ntoken)
        #self.params = rnn.params + [self.encoder.weight, self.decoder.weight, self.decoder.bias]
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):

        emb = self.encoder(input)
        hs = []
        e_s = []
        alpha_s = []
        for i in range(emb.shape[0]):
            hidden, (es,alphas) = self.rnn(emb[i], hidden, 1.0, i==0)
            e_s.append(es)
            alpha_s.append(alphas)
            hs.append(hidden)
        output = torch.stack(hs)

        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, (e_s, alpha_s)


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--capacity', type=int, default=2,
                    help='unitary matrix capacity')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=150,
                    help='sequence length')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=400,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, default=True, help='use cuda')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--net-type', type=str, default='NRNN2',
                    help='rnn net type')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_orth', type=float, default=2e-5)
parser.add_argument('--rinit', type=str, default="henaff", help='recurrent weight matrix initialization')
parser.add_argument('--iinit', type=str, default="kaiming", help='input weight matrix initialization')
parser.add_argument('--ostep_method', type=str, default='exp', help='if learnable P, which way, exp or cayley')
parser.add_argument('--alam', type=float, default=0.0001, help='alpha values lamda for ARORNN and ARORNN2')
parser.add_argument('--nonlin', type=str, default='modrelu', help='non linearity none, relu, tanh, sigmoid')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.99)
parser.add_argument('--adam', action='store_true', default=False, help='Use adam')


args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = Corpus('./data/pennchar/')


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
NET_TYPE = args.net_type
inp_size = args.emsize
hid_size = args.nhid  # calc_hidden_size_PTB(NET_TYPE,2150000,50,args.emsize)

alam = args.alam
CUDA = args.cuda
nonlin = args.nonlin

#rnn = select_network(NET_TYPE, inp_size, hid_size, nonlin, args.rinit, args.iinit, CUDA)

#model = RNNModel(rnn, ntokens, inp_size, hid_size, args.tied)
model = torch.load('./PTBNet/model.pt', map_location=torch.device("cpu"))
#if args.cuda:
#    model.cuda()
print('Language Task')
print(NET_TYPE)
print(args)

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = None
    correct = 0
    processed = 0
    for i in range(0, data_source.size(0) - 1, args.bptt):

        data, targets = get_batch(data_source, i, evaluation=True)
        if i == 0 and NET_TYPE == 'LSTM':
            model.rnn.init_states(data.shape[1])
        output, hidden, (es,alphas) = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).item()
        correct += torch.eq(torch.argmax(output_flat, dim=1), targets).sum().item()
        processed += targets.shape[0]
        hidden = hidden.detach()
        if NET_TYPE == 'LSTM':
            model.rnn.ct = model.rnn.ct.detach()
    return total_loss / len(data_source), correct / processed


def train(optimizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = None
    losses = []
    bpcs = []
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if hidden is not None:
            hidden = hidden.detach()
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            losses.append(cur_loss)
            bpcs.append(cur_loss / math.log(2))

            total_loss = 0
            start_time = time.time()
    return np.mean(losses)


# Loop over epochs.
lr = args.lr
decay = args.weight_decay
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
if not args.adam:
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.StepLR(optimizer,1,gamma=0.5)

try:

    udir = 'HS_{}_lr_{}_rinit_{}_iinit_{}_decay_{}'.format(hid_size, lr, args.rinit, args.iinit, decay)
    SAVEDIR = './saves/PTBTask/{}/{}/{}/{}/'.format(NET_TYPE, args.bptt, udir, args.seed)
    LOGDIR = SAVEDIR
    writer = SummaryWriter(LOGDIR)

    if not os.path.exists(SAVEDIR):
        os.makedirs(SAVEDIR)
    with open(SAVEDIR + 'hparams.txt', 'w') as fp:
        for key, val in args.__dict__.items():
            fp.write(('{}: {}'.format(key, val)))
    tr_losses = []
    v_losses = []
    v_accs = []
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        #loss = train(optimizer)
        #tr_losses.append(loss)

        val_loss, val_acc = evaluate(val_data)
        v_losses.append(val_loss)
        v_accs.append(val_acc)
        #writer.add_scalar('train_loss', loss)
        writer.add_scalar('valid_accuracy', val_acc)
        writer.add_scalar('valid_bpc', val_loss / math.log(2))
        writer.add_scalar('valid_bpc', val_loss)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f} | valid bpc {:8.3f} | valid accuracy {:5.2f} '.format(epoch,
                                                                                       (time.time() - epoch_start_time),
                                                                                       val_loss, math.exp(val_loss),
                                                                                       val_loss / math.log(2), val_acc))
        print('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(SAVEDIR + args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            scheduler.step()

    with open(SAVEDIR + '{}_Train_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(tr_losses, fp)

    with open(SAVEDIR + '{}_Val_Losses'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(v_losses, fp)

    with open(SAVEDIR + '{}_Val_Accs'.format(NET_TYPE), 'wb') as fp:
        pickle.dump(v_accs, fp)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(SAVEDIR + args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, test_accuracy = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}, test bpc {:8.2f} | test acc {:8.2f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2), test_accuracy))
print('=' * 89)

with open(SAVEDIR + 'testdat.txt', 'w') as fp:
    fp.write('Test loss: {} Test Accuracy: {}'.format(test_loss, test_accuracy))
