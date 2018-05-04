import os
from collections import Counter
import numpy as np
import torch


PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

EXTRA_VOCAB = ['_PAD', '_SOS', '_EOS', '_UNK']


class Corpus(object):
    def __init__(self, datadir, min_n=2, bow_vocab_size=None, max_vocab_size=None, max_length=None):
        self.min_n = min_n
        self.bow_vocab_size = bow_vocab_size
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        filenames = ['train.txt', 'valid.txt', 'test.txt']
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        self._construct_vocab()
        self.train_data = Data(self.datapaths[0], self.word2idx, bow_vocab_size, max_length)
        self.valid_data = Data(self.datapaths[1], self.word2idx, bow_vocab_size, max_length)
        self.test_data = Data(self.datapaths[2], self.word2idx, bow_vocab_size, max_length)

    def _construct_vocab(self):
        self._vocab = Counter()
        for datapath in self.datapaths:
            with open(datapath) as f:
                # parse data files to construct vocabulary            
                for line in f:
                    if line == '\n':
                        continue
                    self._vocab.update(line.strip().lower().split())
        vocab_size = len([x for x in self._vocab if self._vocab[x] >= self.min_n])
        self.idx2word = EXTRA_VOCAB + list(next(zip(*self._vocab.most_common(self.max_vocab_size)))[:vocab_size])
        self.word2idx = dict((w, i) for (i, w) in enumerate(self.idx2word))


class Data(object):
    def __init__(self, datapath, vocab, bow_vocab_size, max_length):
        self.vocab_size = len(vocab)
        self.bow_vocab_size = bow_vocab_size
        data = []
        counts = []
        with open(datapath) as f:
            for line in f:
                if line == '\n\n':
                    continue
                words = line.strip().lower().split()[:max_length]
                indices = [vocab.get(x, UNK_ID) for x in words]
                data.append([SOS_ID] + indices + [EOS_ID])
                word_count = {}
                for idx in indices:
                    if bow_vocab_size is not None:
                        idx = idx if idx < bow_vocab_size else UNK_ID
                    word_count[idx] = word_count.get(idx, 0) + 1
                counts.append(word_count)
        self.data = np.array(data)
        self.word_counts = np.array(counts)

    @property
    def size(self):
        return len(self.data)
    
    def get_batch(self, start_id, batch_size, eps=1e-4):
        batch_data = self.data[start_id:(start_id+batch_size)]
        batch_counts = self.word_counts[start_id:(start_id+batch_size)]
        lengths = np.array([len(x) - 1 for x in batch_data])    # actual length is 1 less
        # sort by length in order to use packed sequence
        idx = np.argsort(lengths)[::-1]
        batch_data = batch_data[idx]
        batch_counts = batch_counts[idx]
        lengths = list(lengths[idx])
        
        max_len = int(lengths[0] + 1)
        data_tensor = torch.LongTensor(batch_size, max_len).fill_(PAD_ID)
        bow_tensor = torch.FloatTensor(batch_size, self.bow_vocab_size).fill_(eps)
        for i, x in enumerate(batch_data):
            n = len(x)
            data_tensor[i][:n] = torch.from_numpy(np.array(x))
        for i, c in enumerate(batch_counts):
            for k, v in c.items():
                bow_tensor[i][k] += v
        inputs = data_tensor[:, :-1].clone()
        targets = data_tensor[:, 1:].clone()
        bow_tensor = bow_tensor / bow_tensor.sum(-1, keepdim=True)
        return inputs, targets, bow_tensor, lengths
            

