import argparse

import torch
from torch.autograd import Variable

import data
from data import SOS_ID, EOS_ID

parser = argparse.ArgumentParser(description='Text VAE translate')
parser.add_argument('--data', type=str, default='./data/fami',
                    help='location of the corpus (same as training)')
parser.add_argument('--checkpoint', type=str, default='./saves/model.pt',
                    help='location of the model file')
parser.add_argument('--input_file', type=str, default='./data/pos_samples.txt',
                    help='location of the input texts')
parser.add_argument('--output_file', type=str, default='./data/pos_samples.txt.out',
                    help='output file to write reconstructed texts')
parser.add_argument('--max_length', type=int, default=36,
                    help='maximum generation length')
parser.add_argument('--num_samples', type=int, default=4,
                    help='number of samples per reconstruction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size used in generation')
parser.add_argument('--sample_bow', action='store_true',
                    help='use sampled bow instead of ground truth to reconstruct')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)


def translate_sentence(indices, idx2word):
    words = []
    for i in indices:
        if i == EOS_ID:
            break
        else:
            words.append(idx2word[i])
    return ' '.join(words)
    
    
def predict(data_source, model, idx2word):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        inputs, _, bows, lengths = data_source.get_batch(i, batch_size)
        inputs = inputs.to(device)
        if args.sample_bow:
            bows = None
        else:
            bows = bows.to(device)
        samples = model.reconstruct(inputs, bows, lengths, args.max_length, args.num_samples, SOS_ID, EOS_ID)
        for sample in samples.cpu().view(-1, args.max_length).numpy():
            results.append(translate_sentence(sample, idx2word))
    return results


with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
device = torch.device('cuda' if args.cuda else 'cpu')
model.to(device)

print("Loading data")
corpus = data.Corpus(args.data)
vocab_size = len(corpus.word2idx)
print("\ttraining data size: ", corpus.train_data.size)
print("\tvocabulary size: ", vocab_size)
# data to be reconstructed
input_data = data.Data(args.input_file, corpus.word2idx)
results = predict(input_data, model, corpus.idx2word)
with open(args.output_file, 'w') as f:
    f.write('\n'.join(results))
