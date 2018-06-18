###############################################################################
#
# set up the model structure
#
###############################################################################


import torch.nn as nn
from torch.autograd import Variable
import torch

class OurModel(nn.Module):

    def __init__(self, ntoken, nhid=1000):
        super(OurModel, self).__init__()
        #ntoken - input dimension / output dimension
        #self.encoder = nn.Embedding(ntoken, ntoken) # dirty hack so we don't need to change the input from the original code,
                                                     # should also improve performance compared to one-hot-vectors
                                                     # but also increases the number of parameters in the word model by 100.000.000 (ooops)
        self.rnn = nn.LSTM(ntoken, nhid, 1)
        self.decoder = nn.Linear(nhid, ntoken)
        self.ntoken = ntoken

        self.init_weights()
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        #emb = self.encoder(input)
        output, hidden = self.rnn(input, hidden) #input = emb
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.nhid).zero_()),
                Variable(weight.new(1, bsz, self.nhid).zero_()))

    def get_num_param(self):
        num_param = 0
        for parameter in self.parameters():
            if parameter.dim() > 1:
                num_param += parameter.size(0) * parameter.size(1)
            else:
                num_param += parameter.size(0)
        return num_param

