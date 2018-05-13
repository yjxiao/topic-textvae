import argparse
import time
import math
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

from model import TextCVAE
import data
from data import PAD_ID


parser = argparse.ArgumentParser(description='Text CVAE')
parser.add_argument('--data', type=str, default='./data/yelp',
                    help="location of the data folder")
parser.add_argument('--bow_vocab', type=int, default=10000,
                    help="vocabulary used in calculating bow KLD")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--lb_embed_size', type=int, default=16,
                    help="size of the label embedding")
parser.add_argument('--hidden_size', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=16,
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
parser.add_argument('--kla', action='store_true',
                    help='do kl annealing')
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--log_every', type=int, default=3000,
                    help="random seed")
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
parser.add_argument('--save', type=str,  default='./saves/model.pt',
                    help="path to save the final model")
args = parser.parse_args()

torch.manual_seed(args.seed)


    
def loss_function(targets, outputs, mus, logvars, alphas, bow_posterior):
    """

    Inputs:
        targets: target tokens
        outputs: predicted tokens
        mu:      latent mean
        logvar:  log of the latent variance
        alphas:  parameters of the dirichlet prior p(w|z) given latent code
        bow_posterior: actual distribution of words q(w|x,z) i.e. posterior given x

    Outputs:
        ce_loss: cross entropy loss of the tokens
        kld:     D(q(z|x)||p(z))
        kld_bow: D(q(w|x,z)||p(w|z))
    
    """
    ce_loss = F.cross_entropy(outputs.view(outputs.size(0)*outputs.size(1),
                                           outputs.size(2)),
                              targets.view(-1),
                              size_average=False,
                              ignore_index=PAD_ID)
    kld = -0.5 * torch.sum(1 + logvars[0] - logvars[1] - ((mus[0] - mus[1]).pow(2) + logvars[0].exp()) / logvars[1].exp())
    prior = Dirichlet(alphas)
    alphas2 = bow_posterior * alphas.sum(1, keepdim=True)
    posterior = Dirichlet(alphas2)
    kld_bow = kl_divergence(posterior, prior).sum()
    return ce_loss, kld, kld_bow


def evaluate(data_source, model):
    model.eval()
    total_ce = 0.0
    total_kld = 0.0
    total_bow = 0.0
    total_words = 0
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, targets, bows, lengths, labels, _ = data_source.get_batch(batch_size, i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        bows = bows.to(device)
        labels = labels.to(device)
        outputs, mus, logvars, alphas = model(inputs, labels, bows, lengths)
        ce, kld, kld_bow = loss_function(targets, outputs, mus, logvars, alphas, bows)
        total_ce += ce.item()
        total_kld += kld.item()
        total_bow += kld_bow.item()
        total_words += sum(lengths)
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_bow / data_source.size, ppl)


def train(data_source, model, optimizer, epoch):
    model.train()
    total_ce = 0.0
    total_kld = 0.0
    total_bow = 0.0
    total_words = 0
    for i in range(args.log_every):
        inputs, targets, bows, lengths, labels, _ = data_source.get_batch(args.batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        bows = bows.to(device)
        labels = labels.to(device)
        outputs, mus, logvars, alphas = model(inputs, labels, bows, lengths)
        ce, kld, kld_bow = loss_function(targets, outputs, mus, logvars, alphas, bows)
        total_ce += ce.item()
        total_kld += kld.item()
        total_bow += kld_bow.item()
        total_words += sum(lengths)
        optimizer.zero_grad()
        loss = ce + kld + kld_bow
        loss.backward()
        optimizer.step()
            
    ppl = math.exp(total_ce / total_words)
    return (total_ce / data_source.size, total_kld / data_source.size,
            total_bow / data_source.size, ppl)


print("Loading data")
corpus = data.LabeledCorpus(args.data, None, bow_vocab_size=args.bow_vocab,
                            max_vocab_size=args.max_vocab, max_length=args.max_length)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
print("Constructing model")
print(args)
device = torch.device('cpu' if args.nocuda else 'cuda')
model = TextCVAE(vocab_size, corpus.num_classes, args.bow_vocab, args.embed_size,
                 args.lb_embed_size, args.hidden_size, args.code_size, args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
best_loss = None

print("\nStart training")
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_ce, train_kld, train_bow, train_ppl = train(corpus.train_data, model, optimizer, epoch)
        valid_ce, valid_kld, valid_bow, valid_ppl = evaluate(corpus.valid_data, model)
        print('-' * 90)
        print("| epoch {:2d} | time {:5.2f}s | train loss {:5.2f} ({:5.2f}, {:.2f}) "
              "| train ppl {:5.2f}".format(
                  epoch, time.time()-epoch_start_time, train_ce, train_kld,
                  train_bow, train_ppl))
        print("|                         | valid loss {:5.2f} ({:5.2f}, {:.2f}) "
              "| valid ppl {:5.2f}".format(
                  valid_ce, valid_kld, valid_bow, valid_ppl), flush=True)
        if best_loss is None or valid_ce + valid_kld < best_loss:
            best_loss = valid_ce + valid_kld
            with open(args.save, 'wb') as f:
                torch.save(model, f)

except KeyboardInterrupt:
    print('-' * 90)
    print('Exiting from training early')


with open(args.save, 'rb') as f:
    model = torch.load(f)

test_ce, test_kld, test_bow, test_ppl = evaluate(corpus.test_data, model)
print('=' * 90)
print("| End of training | test loss {:5.2f} ({:5.2f}, {:.2f}) | test ppl {:5.2f}".format(
    test_ce, test_kld, test_bow, test_ppl))
print('=' * 90)
