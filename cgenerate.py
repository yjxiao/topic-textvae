import argparse

import torch
import data
from data import SOS_ID, EOS_ID

parser = argparse.ArgumentParser(description='Text CVAE generate')
parser.add_argument('--data', type=str, default='./data/yelp',
                    help='location of the corpus (same as training)')
parser.add_argument('--checkpoint', type=str, default='./saves/model.std.cvae.yelp.pt',
                    help='location of the model file')
parser.add_argument('--task', type=str, default='reconstruct',
                    help='task to perform: [reconstruct, change_label, sample]')
parser.add_argument('--bow_vocab', type=int, default=10000,
                    help="vocabulary used in calculating bow KLD")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--input_file', type=str, default='./data/yelp/valid_with_label.txt',
                    help='location of the input texts for reconstruct task')
parser.add_argument('--output_file', type=str, default='./outputs.txt',
                    help='output file to write reconstructed texts')
parser.add_argument('--max_length', type=int, default=64,
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
        inputs, _, bow, lengths, labels, idx = data_source.get_batch(batch_size, i)
        inputs = inputs.to(device)
        bow = bow.to(device)
        labels = labels.to(device)
        samples = model.reconstruct(inputs, labels, bow, lengths, args.max_length, args.num_samples, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results


def change_label(data_source, model, new_label, idx2word):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, _, bow, lengths, labels, idx = data_source.get_batch(batch_size, i)
        inputs = inputs.to(device)
        bow = bow.to(device)
        labels = labels.to(device)
        new_labels = torch.empty_like(labels).fill_(new_label).to(device)
        samples = model.change_label(inputs, labels, new_labels, bow, lengths, args.max_length, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results

        
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
device = torch.device('cuda' if args.cuda else 'cpu')
model.to(device)

print("Loading data")
corpus = data.LabeledCorpus(args.data, None, bow_vocab_size=args.bow_vocab,
                            max_vocab_size=args.max_vocab)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
# data to be reconstructed
input_data = data.LabeledData(args.input_file, corpus.word2idx, args.bow_vocab, corpus.num_classes)

if args.task == 'reconstruct':
    results = reconstruct(input_data, model, corpus.idx2word)
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(results))
elif args.task == 'change_label':
    for k in range(corpus.num_classes):
        results = change_label(input_data, model, k, corpus.idx2word)
        with open(args.output_file+'.{0:d}'.format(k), 'w') as f:
            f.write('\n'.join(results))
