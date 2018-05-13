import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

from model import TextVAE, TextCVAE
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--num_topics', type=int, default=32,
                    help="number of topics to use for the topic modeling input")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--label_embed_size', type=int, default=8,
                    help="size of the word embedding")
parser.add_argument('--hidden_size', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="number of hidden units for RNN")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--sample_topics', action='store_true',
                    help='sample bow from posterior during training')
parser.add_argument('--kla', action='store_true',
                    help='do kl annealing')
parser.add_argument('--bow', action='store_true',
                    help='add bow loss')
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--log_every', type=int, default=2000,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
args = parser.parse_args()

torch.manual_seed(args.seed)


    
def loss_function(targets, outputs, mu, logvar, alphas, topics, bow=None):
    """

    Inputs:
        targets: target tokens
        outputs: predicted tokens
        mu:      latent mean
        logvar:  log of the latent variance
        alphas:  parameters of the dirichlet prior p(w|z) given latent code
        topics:  actual distribution of topics q(w|x,z) i.e. posterior given x

    Outputs:
        ce_loss: cross entropy loss of the tokens
        kld:     D(q(z|x)||p(z))
        kld_tpc: D(q(w|x,z)||p(w|z))
    
    """
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    if bow is None:
        bow_loss = torch.tensor(0., device=outputs.device)
    else:
        bow = bow.unsqueeze(1).repeat(1, outputs.size(1), 1).contiguous()
        bow_loss = F.cross_entropy(bow.view(bow.size(0) * bow.size(1), bow.size(2)),
                                   targets.view(-1),
                                   size_average=False,
                                   ignore_index=PAD_ID)
    if type(mu) == torch.Tensor:
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        kld = -0.5 * torch.sum(1 + logvar[1] - logvar[0] - 
                               ((mu[1] - mu[0]).pow(2) + logvar[1].exp()) / logvar[0].exp())
    prior = Dirichlet(alphas)
    alphas2 = topics * topics.size(1)
    posterior = Dirichlet(alphas2)
    kld_tpc = kl_divergence(posterior, prior).sum()
    return ce_loss, kld, kld_tpc, bow_loss


def evaluate(data_source, model, device):
    model.eval()
    total_ce = 0.0
    total_kld = 0.0
    total_kld_tpc = 0.0
    total_words = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, labels, topics, lengths, _ = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        targets = texts[:, 1:].clone().to(device)
        topics = topics.to(device)
        labels = labels.to(device) if data_source.has_label else None
        if data_source.has_label:
            outputs, mu, logvar, alphas, _ = model(inputs, labels, topics, lengths)
        else:
            outputs, mu, logvar, alphas, _ = model(inputs, topics, lengths)
        ce, kld, kld_tpc, _ = loss_function(targets, outputs, mu, logvar, alphas, topics)
        total_ce += ce.item()
        total_kld += kld.item()
        total_kld_tpc += kld_tpc.item()
        total_words += sum(lengths)
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_kld_tpc / data_source.size, ppl)


def train(data_source, model, optimizer, device, epoch):
    model.train()
    total_ce = 0.0
    total_kld = 0.0
    total_kld_tpc = 0.0
    total_bow = 0.0
    total_words = 0
    for i in range(args.log_every):
        texts, labels, topics, lengths, _ = data_source.get_batch(args.batch_size)
        inputs = texts[:, :-1].clone().to(device)
        targets = texts[:, 1:].clone().to(device)
        topics = topics.to(device)
        labels = labels.to(device) if data_source.has_label else None
        if data_source.has_label:
            outputs, mu, logvar, alphas, bow = model(inputs, labels, topics, lengths)
        else:
            outputs, mu, logvar, alphas, bow = model(inputs, topics, lengths)
        if not args.bow:
            bow = None
        ce, kld, kld_tpc, bow_loss = loss_function(
            targets, outputs, mu, logvar, alphas, topics, bow)
        total_ce += ce.item()
        total_kld += kld.item()
        total_kld_tpc += kld_tpc.item()
        total_bow = bow_loss.item()
        total_words += sum(lengths)
        if args.kla:
            kld_weight = weight_schedule(args.log_every * (epoch - 1) + i)
        else:
            kld_weight = 1.
        optimizer.zero_grad()
        loss = ce + kld_weight * kld + kld_tpc + bow_loss
        loss.backward()
        optimizer.step()
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_kld_tpc / data_source.size, ppl, total_bow / data_source.size)


def interpolate(i, start, duration):
    return max(min((i - start) / duration, 1), 0)


def weight_schedule(t):
    """Scheduling of the KLD annealing weight. """
    return interpolate(t, 2000, 40000)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    path = './saves/z{0:d}.tpc{1:d}{2}{3}{4}{5}.{6}.pt'.format(
        args.code_size, args.num_topics, '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        '.sampletpc' if args.sample_topics else '', '.kla' if args.kla else '',
        '.bow' if args.bow else '', dataset)
    return path


def main(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    print("Loading {} data".format(dataset))
    if dataset in ['yahoo', 'yelp']:
        with_label = True
    else:
        with_label = False
    corpus = data.Corpus(args.data, args.num_topics, max_vocab_size=args.max_vocab,
                         max_length=args.max_length, with_label=with_label)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train.size)
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    if with_label:
        model = TextCVAE(vocab_size, args.num_topics, corpus.num_classes,
                         args.embed_size, args.label_embed_size, args.hidden_size,
                         args.code_size, args.dropout).to(device)
    else:
        model = TextVAE(vocab_size, args.num_topics, args.embed_size, args.hidden_size,
                        args.code_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    print("\nStart training")
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_ce, train_kld, train_tpc, train_ppl, train_bow = train(
                corpus.train, model, optimizer, device, epoch)
            valid_ce, valid_kld, valid_tpc, valid_ppl = evaluate(corpus.valid, model, device)
            print('-' * 90)
            print("| epoch {:2d} | time {:5.2f}s | train loss {:5.2f} ({:5.2f}, {:.2f}) "
                  "| train ppl {:5.2f} | bow loss {:5.2f}".format(
                      epoch, time.time()-epoch_start_time, train_ce, train_kld,
                      train_tpc, train_ppl, train_bow))
            print("|                         | valid loss {:5.2f} ({:5.2f}, {:.2f}) "
                  "| valid ppl {:5.2f}".format(
                      valid_ce, valid_kld, valid_tpc, valid_ppl), flush=True)
            if best_loss is None or valid_ce + valid_kld + valid_tpc < best_loss:
                best_loss = valid_ce + valid_kld + valid_tpc
                with open(get_savepath(args), 'wb') as f:
                    torch.save(model, f)

    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')

    with open(get_savepath(args), 'rb') as f:
        model = torch.load(f)
    test_ce, test_kld, test_tpc, test_ppl = evaluate(corpus.test, model, device)
    print('=' * 90)
    print("| End of training | test loss {:5.2f} ({:5.2f}, {:.2f}) | test ppl {:5.2f}".format(
        test_ce, test_kld, test_tpc, test_ppl))
    print('=' * 90)


if __name__ == '__main__':
    main(args)
