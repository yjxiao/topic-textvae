import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.dirichlet import Dirichlet
from data import UNK_ID


class DropWord(nn.Module):
    def __init__(self, dropout, unk_id):
        super(DropWord, self).__init__()
        self.dropout = dropout
        self.unk_id = unk_id

    def forward(self, inputs):
        if not self.training or self.dropout == 0:
            return inputs
        else:
            dropmask = torch.bernoulli(
                inputs.data.new(inputs.size()).float().fill_(self.dropout)
            ).byte()
            
            inputs = inputs.clone()
            inputs[dropmask] = self.unk_id
            return inputs
                                                                                                    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size, dropout):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fcmu = nn.Linear(hidden_size, code_size)
        self.fclogvar = nn.Linear(hidden_size, code_size)

    def forward(self, inputs, lengths):
        inputs = self.drop(inputs)
        inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        _, (hn, _) = self.rnn(inputs)
        return self.fcmu(hn), self.fclogvar(hn)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fcz = nn.Linear(code_size, hidden_size * 2)
        
    def forward(self, inputs, z, bow_code, lengths=None, init_hidden=None):
        # inputs size: batch_size x sequence_length x embed_size
        # bow size: batch_size x hidden_size
        
        # Options: bow can be fused with z to initialize the decoder state
        #          or feed into each step of input e.g.
        
        #   inputs = torch.cat([inputs, bow.unsqueeze(1).expand(-1, inputs.size(1), -1)], dim=2)
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        if init_hidden is None:
            latent_code = self.fcz(z)
            init_hidden = [x.contiguous() for x in torch.chunk(F.tanh(latent_code + bow_code), 2, 2)]
            #init_hidden = [x.contiguous() for x in torch.chunk(F.tanh(self.fcz(z)), 2, 2)]
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class TextVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, code_size, dropout, dropword):
        super(TextVAE, self).__init__()
        self.dropword = DropWord(dropword, UNK_ID)
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size, code_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size, code_size, dropout)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)
        # transform bag-of-words distribution to a lower dimension
        self.fcbow = nn.Linear(vocab_size, hidden_size * 2)
        self.bow_prior = BoWPrior(vocab_size, hidden_size, code_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, bow, lengths):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(self.dropword(inputs))
        mu, logvar = self.encoder(enc_emb, lengths)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        alphas = self.bow_prior(z)
        bow_code = self.fcbow(bow)
        #if not self.training:
        #    dist = Dirichlet(alphas.cpu())
        #    bow_samples = dist.sample().cuda()
        #    bow_code = self.fcbow(bow_samples)
        outputs, _ = self.decoder(dec_emb, z, bow_code, lengths=lengths)
        outputs = self.fcout(outputs)
        return outputs, mu, logvar, alphas

    def reconstruct(self, inputs, bow, lengths, max_length, num_samples, sos_id, eos_id):
        enc_emb = self.lookup(inputs)
        mu, logvar = self.encoder(enc_emb, lengths)
        mu = mu.repeat(1, num_samples, 1)
        logvar = logvar.repeat(1, num_samples, 1)
        # z size: 1 x (num_samples*batch_size) x code_size
        z = self.reparameterize(mu, logvar)
        if bow is None:
            alphas = self.bow_prior(z)
            if alphas.device.type == 'cuda':
                dist = Dirichlet(alphas.cpu())
                bow = dist.sample().cuda()
            else:
                dist = Dirichlet(alphas)
                bow = dist.sample()
        else:
            bow = bow.repeat(num_samples, 1)
        bow_code = self.fcbow(bow)
        batch_size = z.size(1)
        generated = inputs.data.new(batch_size, max_length)
        dec_inputs = inputs.data.new(batch_size, 1).fill_(sos_id)
        hidden = None
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, z, bow_code, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs.data[:, 0].clone()
        return generated

    def sample(self, z, max_length, num_samples, sos_id, eos_id):
        pass

class BoWPrior(nn.Module):
    def __init__(self, vocab_size, hidden_size, code_size, eps=1e-4):
        super(BoWPrior, self).__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.eps = eps

    def forward(self, inputs):
        """
        Inputs:  latent code of size 1 x batch_size x code_size or batch_size x code_size
        Outputs: predicted alphas of dirichlet distribution

        """
        if inputs.dim() == 3:
            inputs = inputs.squeeze(0)
        # alphas are positive reals
        alphas = F.relu(self.fc2(self.activation(self.fc1(inputs)))) + self.eps
        return alphas


        
