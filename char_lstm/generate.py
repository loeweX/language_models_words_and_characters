###############################################################################
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='../datasets/EN/unknown',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./output/outENUnkWord.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='output/generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--char_vs_word', type=str, default='word',
                    help='Using character or word based model (char/word)') ##not implemented for subwords
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        print('Using Cuda')

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

if args.char_vs_word == 'char':
    corpus = data.Corpus_char(args.data)
elif args.char_vs_word == 'word':
    corpus = data.Corpus_word(args.data)

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)


#generate random first token and get the one-hot vector
new_input = torch.rand(1, 1).mul(ntokens).long()
ids = torch.unsqueeze(new_input, 2)
y_onehot = torch.FloatTensor(ids.size(0), ids.size(1), ntokens).zero_()
y_onehot.scatter_(2, ids.cpu(), 1)

if args.cuda:
    input = Variable(y_onehot.cuda(), volatile=True)
else:
    input = Variable(y_onehot, volatile=True)

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]

        y_onehot = torch.FloatTensor(ids.size(0), ids.size(1), ntokens).zero_()
        y_onehot[0,0,word_idx] = 1

        if args.cuda:
            input = Variable(y_onehot.cuda(), volatile=True)
        else:
            input = Variable(y_onehot, volatile=True)

        if args.char_vs_word == 'char':
            word = corpus.dictionary.idx2word[word_idx]
            if word == '§':
                outf.write('. \n')
            else:
                outf.write(word)
        elif args.char_vs_word == 'word':
            word = corpus.dictionary.idx2word[word_idx]

            if word == '<eos>':
                outf.write('. \n')
            else:
                outf.write(word + ' ')

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
