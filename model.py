import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, inputs, lengths):
        inputs = self.drop(inputs)
        inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        _, hn = self.rnn(inputs)
        return hn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, code_size, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fccode = nn.Linear(code_size, hidden_size * 2)
        
    def forward(self, inputs, code, lengths=None, init_hidden=None):
        # inputs size: batch_size x sequence_length x embed_size
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
        if init_hidden is None:
            init_hidden = [x.contiguous() for x in torch.chunk(F.tanh(self.fccode(code)), 2, 2)]
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class TopicPrior(nn.Module):
    def __init__(self, num_topics, hidden_size, code_size):
        super(TopicPrior, self).__init__()
        self.fc1 = nn.Linear(code_size, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_topics)

    def forward(self, inputs):
        """
        Inputs:  latent code of size 1 x batch_size x code_size or batch_size x code_size
        Outputs: predicted alphas of dirichlet distribution

        """
        # alphas are positive reals
        alphas = self.fc2(self.activation(self.fc1(inputs))).exp()
        return alphas


class TextVAE(nn.Module):
    def __init__(self, vocab_size, num_topics, embed_size, hidden_size, code_size, dropout):
        super(TextVAE, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size, code_size + num_topics, dropout)
        self.fcmu = nn.Linear(hidden_size, code_size)
        self.fclogvar = nn.Linear(hidden_size, code_size)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.topic_prior = TopicPrior(num_topics, hidden_size, code_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, topics, lengths, sample_topics=False):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(inputs)
        hn, _ = self.encoder(enc_emb, lengths)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        alphas = self.topic_prior(z)
        topics = topics.unsqueeze(0)
        if sample_topics:
            device = bow.device
            dist = Dirichlet((topics * alphas.sum(2, keepdim=True)).cpu())
            topics = dist.rsample().to(device)
        code = torch.cat((z, topics), dim=2)
        outputs, _ = self.decoder(dec_emb, code, lengths=lengths)
        outputs = self.fcout(outputs)
        return outputs, mu, logvar, alphas

    def reconstruct(self, inputs, topics, lengths, max_length, sos_id):
        enc_emb = self.lookup(inputs)
        hn, _ = self.encoder(enc_emb, lengths)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        z = self.reparameterize(mu, logvar)
        return self.generate(z, topics.unsqueeze(0), max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id, device):
        """Randomly sample latent code to sample texts. 
        Note that num_samples should not be too large. 

        """
        z_size = self.fcmu.out_features
        z = torch.randn(1, num_samples, z_size, device=device)
        alphas = self.topic_prior(z)
        dist = Dirichlet(alphas.cpu())
        topics = dist.sample().to(device)
        return self.generate(z, topics, max_length, sos_id)

    def interpolate(self, input_pairs, topic_pairs, length_pairs, max_length, sos_id, num_pts=4):
        z_pairs = []
        for inputs, topics, lengths in zip(input_pairs, topic_pairs, length_pairs):
            enc_emb = self.lookup(inputs)
            hn, _ = self.encoder(enc_emb, lengths)
            z_pairs.append(self.fcmu(hn))
        generated = []
        for i in range(num_pts+2):
            z = _interpolate(z_pairs, i, num_pts+2)
            topics = _interpolate(topic_pairs, i, num_pts+2)
            generated.append(self.generate(z, topics.unsqueeze(0), max_length, sos_id))
        return generated

    def generate(self, z, topics, max_length, sos_id):
        batch_size = z.size(1)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = None
        code = torch.cat((z, topics), dim=2)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, code, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated
        

def _interpolate(pairs, i, n):
    x1, x2 = [x.clone() for x in pairs]
    return x1 * (n - 1 - i) / (n - 1) + x2 * i / (n - 1)


class ZPrior(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ZPrior, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)
        self.act = nn.ReLU()

    def forward(self, inputs):
        # make sure input dimension is 1 x batch_size x input_size
        return torch.chunk(self.fc2(self.act(self.fc1(inputs))), chunks=2, dim=2)

        
class TextCVAE(nn.Module):
    def __init__(self, vocab_size, num_topics, num_classes, embed_size,
                 label_embed_size, hidden_size, code_size, dropout):
        super(TextCVAE, self).__init__()
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.label_lookup = nn.Embedding(num_classes, label_embed_size)
        self.encoder = Encoder(embed_size, hidden_size, dropout)
        self.decoder = Decoder(embed_size, hidden_size,
                               code_size + num_topics + label_embed_size, dropout)
        self.fcmu = nn.Linear(hidden_size + label_embed_size, code_size)
        self.fclogvar = nn.Linear(hidden_size + label_embed_size, code_size)
        self.z_prior = ZPrior(label_embed_size, hidden_size, code_size)
        # output layer
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.topic_prior = TopicPrior(num_topics, hidden_size, code_size + label_embed_size)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, inputs, labels, topics, lengths, sample_topics=False):
        enc_emb = self.lookup(inputs)
        dec_emb = self.lookup(inputs)
        lab_emb = self.label_lookup(labels).unsqueeze(0)    # to match with shape of z
        # prior of z
        mu_pr, logvar_pr = self.z_prior(lab_emb)
        h, _ = self.encoder(enc_emb, lengths)
        hn = torch.cat([h, lab_emb], dim=2)
        # posterior of z
        mu_po, logvar_po = self.fcmu(hn), self.fclogvar(hn)
        if self.training:
            z = self.reparameterize(mu_po, logvar_po)
        else:
            z = mu_po
        alphas = self.topic_prior(torch.cat([z, lab_emb], dim=2))
        topics = topics.unsqueeze(0)
        if sample_topics:
            dist = Dirichlet((topics * topics.size(2)).cpu())
            topics = dist.rsample().to(alphas.device)
        code = torch.cat([z, topics, lab_emb], dim=2)
        outputs, _ = self.decoder(dec_emb, code, lengths=lengths)
        outputs = self.fcout(outputs)
        return outputs, (mu_pr, mu_po), (logvar_pr, logvar_po), alphas

    def reconstruct(self, inputs, labels, topics, lengths, max_length, sos_id):
        enc_emb = self.lookup(inputs)
        lab_emb = self.label_lookup(labels).unsqueeze(0)
        h, _ = self.encoder(enc_emb, lengths)
        hn = torch.cat([h, lab_emb], dim=2)
        mu, logvar = self.fcmu(hn), self.fclogvar(hn)
        z = self.reparameterize(mu, logvar)
        return self.generate(z, topics, lab_emb, max_length, sos_id)

    def sample(self, labels, max_length, sos_id):
        lab_emb = self.label_lookup(labels).unsqueeze(0)
        mu, logvar = self.z_prior(lab_emb)
        z = self.reparameterize(mu, logvar)
        alphas = self.topic_prior(torch.cat([z, lab_emb], dim=2))
        dist = Dirichlet(alphas.cpu())
        topics = dist.sample().to(alphas.device)
        return self.generate(z, topics, lab_emb, max_length, sos_id)
    
    def generate(self, z, topics, lab_emb, max_length, sos_id):
        batch_size = z.size(1)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = None
        code = torch.cat([z, topics.unsqueeze(0), lab_emb], dim=2)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, code, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated
