import argparse

import torch
import data
from data import SOS_ID, EOS_ID

parser = argparse.ArgumentParser(description='Text VAE generate')
parser.add_argument('--data', type=str, default='./data/books',
                    help='location of the corpus (same as training)')
parser.add_argument('--checkpoint', type=str, default='./saves/model.std.books.pt',
                    help='location of the model file')
parser.add_argument('--task', type=str, default='reconstruct',
                    help='task to perform: [reconstruct, interpolate, sample]')
parser.add_argument('--interpolate_type', type=str, default='both',
                    help='how to interpolate [z, bow, both]')
parser.add_argument('--bow_vocab', type=int, default=10000,
                    help="vocabulary used in calculating bow KLD")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--input_file', type=str, default='./data/books/valid.txt',
                    help='location of the input texts for reconstruct task')
parser.add_argument('--output_file', type=str, default='./outputs.txt',
                    help='output file to write reconstructed texts')
parser.add_argument('--max_length', type=int, default=20,
                    help='maximum generation length')
parser.add_argument('--num_samples', type=int, default=1,
                    help='number of samples per reconstruction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size used in generation')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
args = parser.parse_args()

torch.manual_seed(args.seed)


def indices_to_sentence(indices, idx2word):
    words = []
    for i in indices:
        if i == EOS_ID:
            break
        else:
            words.append(idx2word[i])
    return ' '.join(words)
    
    
def reconstruct(data_source, model, idx2word):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, _, bow, lengths, idx = data_source.get_batch(batch_size, i)
        inputs = inputs.to(device)
        bow = bow.to(device)
        samples = model.reconstruct(inputs, bow, lengths, args.max_length, args.num_samples, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results


def sample(model, idx2word):
    results = []
    samples = model.sample(args.num_samples, args.max_length, SOS_ID, device)
    for sample in samples.cpu().numpy():
        results.append(indices_to_sentence(sample, idx2word))
    return results


def interpolate(data_source, model, idx2word, type):
    samples = []
    for i in range(args.num_samples):
        inputs1, _, bow1, lengths1, _ = data_source.get_batch(args.batch_size)
        inputs2, _, bow2, lengths2, _ = data_source.get_batch(args.batch_size)
        if type == 'z':
            input_pairs = (inputs1.to(device), inputs2.to(device))
            bow_pairs = (bow1.to(device), bow1.to(device))
            length_pairs = (lengths1, lengths2)
        elif type == 'bow':
            input_pairs = (inputs1.to(device), inputs1.to(device))
            bow_pairs = (bow1.to(device), bow2.to(device))
            length_pairs = (lengths1, lengths1)
        elif type == 'both':
            input_pairs = (inputs1.to(device), inputs2.to(device))
            bow_pairs = (bow1.to(device), bow2.to(device))
            length_pairs = (lengths1, lengths2)
        samples.append(model.interpolate(input_pairs, bow_pairs, length_pairs,
                                         args.max_length, SOS_ID))
    results = []
    for x in zip(*samples):
        sentences = []
        # each x is a list of batch_size x max_length, i.e. batch of generated sentences
        for sample in torch.cat(x).cpu().numpy():
            sentences.append(indices_to_sentence(sample, idx2word))
        results.append(sentences)
    return results

        
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
device = torch.device('cuda' if args.cuda else 'cpu')
model.to(device)

print("Loading data")
corpus = data.Corpus(args.data, bow_vocab_size=args.bow_vocab, max_vocab_size=args.max_vocab)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
# data to be reconstructed
input_data = data.Data(args.input_file, corpus.word2idx, args.bow_vocab)

if args.task == 'reconstruct':
    results = reconstruct(input_data, model, corpus.idx2word)
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(results))
elif args.task == 'sample':
    results = sample(model, corpus.idx2word)
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(results))
elif args.task == 'interpolate':
    results = interpolate(input_data, model, corpus.idx2word, args.interpolate_type)
    for i, x in enumerate(results):
        with open('{0:d}.txt'.format(i), 'w') as f:
            f.write('\n'.join(x))
