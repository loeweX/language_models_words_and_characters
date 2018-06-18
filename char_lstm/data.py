###############################################################################
#
# loading the datasets
#
###############################################################################

import os
import torch
import nltk
import math
import numpy as np

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


class Corpus_word(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.train_avg_len = 1
        self.valid_avg_len = 1
        self.test_avg_len = 1

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




class Corpus_char(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, self.train_avg_len = self.tokenize_char(os.path.join(path, 'train.txt'))
        self.valid, self.valid_avg_len = self.tokenize_char(os.path.join(path, 'valid.txt'))
        self.test, self.test_avg_len = self.tokenize_char(os.path.join(path, 'test.txt'))
        print('Average length of words in the train set: ', self.train_avg_len)
        print('Average length of words in the validation set: ',self.valid_avg_len)
        print('Average length of words in the test set: ',self.test_avg_len)

    def tokenize_char(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add chars to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                chars = list(line.strip() + '§')
                tokens += len(chars)
                for char in chars:
                    self.dictionary.add_word(char)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                chars = list(line.strip() + '§')
                for char in chars:
                    ids[token] = self.dictionary.word2idx[char]
                    token += 1

        with open(path, 'r') as f:
            tokens = 0
            avg_len = 0
            for line in f:
                words = line.split()
                tokens += len(words)
                for word in words:
                    avg_len += len(word)

        return ids, (avg_len/tokens)+1 ##plus one for the space belonging to each single word



class Corpus_subword(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, self.train_avg_len = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid, self.valid_avg_len = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, self.test_avg_len = self.tokenize(os.path.join(path, 'test.txt'))

        print('Average number of subwords a word is devided in in the train set: ', self.train_avg_len)
        print('Average number of subwords a word is devided in in the validation set: ',self.valid_avg_len)
        print('Average number of subwords a word is devided in in the test set: ',self.test_avg_len)

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

        num_subwords = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split()
                sub_num = 1
                for word in words:
                    if word[-1] == '@':
                        sub_num += 1
                    else:
                        num_subwords.append(sub_num)
                        sub_num = 1

        num_subwords = np.array(num_subwords)

        return ids, np.mean(num_subwords)
