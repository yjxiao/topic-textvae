import os
from collections import Counter
import pickle
import numpy as np
import torch


PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

EXTRA_VOCAB = ['_PAD', '_SOS', '_EOS', '_UNK']


class Corpus(object):
    def __init__(self, datadir, num_topics, min_n=2, max_vocab_size=None, max_length=None, with_label=False):
        self.min_n = min_n
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.with_label = with_label
        filenames = ['train.txt', 'valid.txt', 'test.txt']            
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        self._construct_vocab()
        self.train, self.valid, self.test = [
            Data(dp, (self.word2idx, self.label2idx), num_topics, max_length, with_label) \
            for dp in self.datapaths]

    def _construct_vocab(self):
        self._vocab = Counter()
        labels = []
        for datapath in self.datapaths:
            with open(datapath) as f:
                # parse data files to construct vocabulary            
                for line in f:
                    if self.with_label:
                        label, text = line.strip().split('\t')
                        if label not in labels:
                            labels.append(label)
                    else:
                        text = line.strip()
                    self._vocab.update(text.lower().split())
        vocab_size = len([x for x in self._vocab if self._vocab[x] >= self.min_n])
        self.idx2word = EXTRA_VOCAB + list(next(zip(*self._vocab.most_common(self.max_vocab_size)))[:vocab_size])
        self.word2idx = dict((w, i) for (i, w) in enumerate(self.idx2word))
        self.idx2label = sorted(labels)
        self.label2idx = dict((w, i) for (i, w) in enumerate(self.idx2label))


class Data(object):
    def __init__(self, datapath, vocabs, num_topics, max_length=None, with_label=False):
        word2idx, label2idx = vocabs
        self.vocab_size = len(word2idx)
        texts = []
        labels = []
        with open(datapath) as f:
            for line in f:
                if with_label:
                    label, text = line.strip().split('\t')
                else:
                    label, text = None, line.strip()
                words = text.lower().split()
                if max_length is not None:
                    words = words[:max_length]
                indices = [word2idx.get(x, UNK_ID) for x in words]
                texts.append([SOS_ID] + indices + [EOS_ID])
                if with_label:
                    labels.append(label2idx[label])
        self.texts = np.array(texts)
        self.labels = np.array(labels, dtype=np.int32) if with_label else None
        topic_datapath = datapath + '.{0:d}.tpcs'.format(num_topics)
        with open(topic_datapath, 'rb') as f:
            self.topics = pickle.load(f).astype(np.float32)
        self.has_label = with_label
            
    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.texts = self.texts[perm]
        self.labels = None if self.labels is None else self.labels[perm]
        self.topics = self.topics[perm]
        
    @property
    def size(self):
        return len(self.texts)
    
    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_texts = self.texts[batch_idx]
        batch_labels = self.labels[batch_idx] if self.has_label else None
        batch_topics = self.topics[batch_idx]
        lengths = np.array([len(x) - 1 for x in batch_texts])    # length used in training is 1 less
        # sort by length in order to use packed sequence
        idx = np.argsort(lengths)[::-1]
        batch_texts = batch_texts[idx]
        batch_labels = batch_labels[idx] if self.has_label else None
        batch_topics = batch_topics[idx]
        lengths = list(lengths[idx])
        max_len = int(lengths[0] + 1)
        text_tensor = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        for i, x in enumerate(batch_texts):
            n = len(x)
            text_tensor[i][:n] = torch.from_numpy(np.array(x))
        label_tensor = torch.from_numpy(batch_labels) if self.has_label else None
        topic_tensor = torch.from_numpy(batch_topics)
        return text_tensor, label_tensor, topic_tensor, lengths, np.argsort(idx)
