###############################################################################
#
# Train the network
#
###############################################################################

import argparse
import time
import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank LSTM Language Model')
parser.add_argument('--data', type=str, default='../datasets/penn/unknown',
                    help='location of the data corpus')
parser.add_argument('--nhid', type=int, default=1000,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=15,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model',
                    help='path to save results (in output directory)')
parser.add_argument('--char_vs_word', type=str, default='char',
                    help='Using character, word or subword based model (char/word/subword)')
parser.add_argument('--resume_training', action='store_true',
                    help='Resume training from files')
parser.add_argument('--test_on_orig', action='store_true',
                    help='Test word model on full text (doesnt really work the way we hoped)')

args = parser.parse_args()

## set to true if you want to see plots during training
PLOTTING = False

##logging variables
loss_train = []
loss_val = []
loss_test = []

bpc_train = []
bpc_val = []
bpc_test = []

ppl_train = []
ppl_val = []
ppl_test = []

learning_rates = []

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        print('Using Cuda')

###############################################################################
# Load data
###############################################################################

if args.char_vs_word == 'char':
    corpus = data.Corpus_char(args.data)
elif args.char_vs_word == 'word':
    corpus = data.Corpus_word(args.data)
elif args.char_vs_word == 'subword':
    corpus = data.Corpus_subword(args.data)

if args.test_on_orig:
    corpus_unk = data.Corpus_word('../datasets/TR/unknown')

ntokens = len(corpus.dictionary)
print(ntokens, 'different symbols/tokens.')

print('Sequence length used for learning: ', args.bptt)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

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

eval_batch_size = 20
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

if args.resume_training: #resume training from checkpoint files
    param_file = np.load('output/' + args.save + '.npz')
    loss_train = param_file['loss_train'].tolist()
    loss_val = param_file['loss_val'].tolist()
    loss_test = param_file['loss_test'].tolist()
    bpc_train = param_file['bpc_train'].tolist()
    bpc_val = param_file['bpc_val'].tolist()
    bpc_test = param_file['bpc_test'].tolist()
    ppl_train = param_file['ppl_train'].tolist()
    ppl_val = param_file['ppl_val'].tolist()
    ppl_test = param_file['ppl_test'].tolist()
    train_avg_len = param_file['train_avg_len'].tolist()
    valid_avg_len = param_file['valid_avg_len'].tolist()
    test_avg_len = param_file['test_avg_len'].tolist()
    num_param = param_file['num_param'].tolist()
    ntokens_resume = param_file['ntokens'].tolist()
    learning_rates = param_file['learning_rates'].tolist()

    args.lr = learning_rates[-1]

    print('Trying to open model')

    with open('output/' + args.save + '.pt', 'rb') as f:
        model = torch.load(f)
else:
    model = model.OurModel(ntokens, nhid=args.nhid)

print(model)

print('Total number of parameters in the model: ', model.get_num_param())

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)

    new_input = source[i:i+seq_len]
    new_target = source[i+1:i+1+seq_len]

    if args.test_on_orig: #doesnt work the way we hoped
        tmp_input = source[i:i+seq_len]
        tmp_target = source[i+1:i+1+seq_len]

        new_input = torch.LongTensor(tmp_input.size())
        new_target = torch.LongTensor(tmp_target.size())

        for num1, row in enumerate(tmp_input):
            for num2, elem in enumerate(row):
                word = corpus.dictionary.idx2word[elem]
                try:
                    new_ind = corpus_unk.dictionary.word2idx[word]
                except:
                    new_ind = corpus_unk.dictionary.word2idx['<unk>']
                new_input[num1, num2] = new_ind

        for num1, row in enumerate(tmp_target):
            for num2, elem in enumerate(row):
                word = corpus.dictionary.idx2word[elem]
                try:
                    new_ind = corpus_unk.dictionary.word2idx[word]
                except:
                    new_ind = corpus_unk.dictionary.word2idx['umarim'] #use a really infrequent word that would most likely give a high error
                new_target[num1, num2] = new_ind


    ids = torch.unsqueeze(new_input, 2)
    y_onehot = torch.FloatTensor(ids.size(0), ids.size(1), ntokens).zero_()
    y_onehot.scatter_(2, ids.cpu(), 1)

    data = Variable(y_onehot.cuda(), volatile=evaluation)
    target = Variable(new_target.cuda().view(-1))

    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout if present.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        if args.test_on_orig:
            output_flat = output.view(-1, len(corpus_unk.dictionary))
        else:
            output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(data_source):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    report_loss = 0
    counter = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    batch = list(range(0, data_source.size(0) - 1, args.bptt))
    #random.shuffle(batch) # randomization yields worse results
    for batch, i in enumerate(batch):
        data, targets = get_batch(data_source, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data
        report_loss += loss.data * len(data)
        counter += 1

        if batch % args.log_interval == 0 and batch > 0 and args.epochs > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | bpc {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.log(math.exp(cur_loss),2), math.exp(((corpus.train_avg_len) * cur_loss))))

            total_loss = 0
            start_time = time.time()

    return report_loss[0] / len(data_source)

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    plt.axis([0, args.epochs + 1, 0, 2])
    if PLOTTING:
        plt.ion()

    if args.epochs > 0:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train(train_data)
            val_loss = evaluate(val_data)

            ##logging
            loss_train.append(train_loss)
            bpc_train.append(math.log(math.exp(train_loss),2))
            ppl_train.append(math.exp(((corpus.train_avg_len) * train_loss)))

            loss_val.append(val_loss)
            bpc_val.append(math.log(math.exp(val_loss),2))
            ppl_val.append(math.exp(((corpus.valid_avg_len) * val_loss)))

            learning_rates.append(lr)

            ##plotting
            plt.scatter(epoch, train_loss, color='red')
            plt.scatter(epoch, val_loss, color='blue')

            if PLOTTING:
                plt.pause(0.05)

            ##printing
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid bpc {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.log(math.exp(val_loss),2), math.exp(((corpus.valid_avg_len) * val_loss))))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open('output/' + args.save + '.pt', 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

        if PLOTTING:
           plt.show(block=True)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

#save the learning curve plot
plt.savefig('output/' + args.save + '.png')

#dynamically evaluate using different learning rates
test_learning_rates = [0, 0.01, 0.1, 1, 2, 5, 10]
for _lr in test_learning_rates:

    # Load the best saved model.
    # reload every time we test, so we do not 'train' on the test set 
    with open('output/' + args.save + '.pt', 'rb') as f:
        model = torch.load(f)

    # Run on test data. Use dynamic evaluation (how can this be legal?)
    #test_loss = evaluate(test_data)
    lr = _lr
    test_loss = train(test_data)
    print('=' * 89)

    print('Learning rate', lr)
    print('| End of training | test loss {:5.2f} | test bpc {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.log(math.exp(test_loss),2),  math.exp(((corpus.test_avg_len) * test_loss))))

    ppl_test.append(math.exp(((corpus.test_avg_len) * test_loss)))
    loss_test.append(test_loss)
    bpc_test.append(math.log(math.exp(test_loss), 2))


np.savez('output/' + args.save + '.npz', loss_train=loss_train, loss_val=loss_val, loss_test=loss_test,
         bpc_train=bpc_train, bpc_val=bpc_val, bpc_test=bpc_test,
         ppl_train=ppl_train, ppl_val=ppl_val, ppl_test=ppl_test,
         train_avg_len=corpus.train_avg_len, valid_avg_len=corpus.valid_avg_len, test_avg_len=corpus.test_avg_len,
         num_param = model.get_num_param(), ntokens=ntokens, learning_rates = learning_rates)

print('=' * 89)
