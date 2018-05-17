import argparse
import os
import torch
import data
from data import SOS_ID, EOS_ID

parser = argparse.ArgumentParser(description='Text VAE generate')
parser.add_argument('--data', type=str, default='./data/books',
                    help='location of the corpus (same as training)')
parser.add_argument('--ckpt', type=str, default='./saves/model.pt',
                    help='location of the trained model')
parser.add_argument('--num_topics', type=int, default=32,
                    help="number of topics for the model")
parser.add_argument('--task', type=str, default='reconstruct',
                    help='task to perform: [reconstruct, interpolate, sample, change_label]')
parser.add_argument('--interpolate_type', type=str, default='topics',
                    help='how to interpolate [z, topics, both]')
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--input_file', type=str, default='valid.txt',
                    help='location of the input texts for reconstruct task')
parser.add_argument('--output_file', type=str, default='outputs.txt',
                    help='output file to write reconstructed texts')
parser.add_argument('--max_length', type=int, default=20,
                    help='maximum generation length')
parser.add_argument('--num_samples', type=int, default=1,
                    help='number of samples per reconstruction')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size used in generation')
parser.add_argument('--bow', action='store_true',
                    help='using model trained with bow loss')
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
    
    
def reconstruct(data_source, model, idx2word, device):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, labels, topics, lengths, idx = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        topics = topics.to(device)
        if data_source.has_label:
            labels = labels.to(device)
            samples = model.reconstruct(inputs, labels, topics, lengths, args.max_length, SOS_ID)
        else:
            samples = model.reconstruct(inputs, topics, lengths, args.max_length, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results


def sample(data_source, model, label, idx2word, device):
    results = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_size = min(args.num_samples - i, args.batch_size)
        if data_source.has_label:
            labels = torch.full((batch_size,), label, dtype=torch.long, device=device)
            samples = model.sample(labels, args.max_length, SOS_ID)
        else:
            samples = model.sample(batch_size, args.max_length, SOS_ID, device)
        for i, sample in enumerate(samples.cpu().numpy()):
            if data_source.has_label:
                label = labels[i].item()
                prefix = '{0:d}\t'.format(label)
            else:
                prefix = ''
            results.append(prefix + indices_to_sentence(sample, idx2word))
    return results


def interpolate(data_source, model, idx2word, type, device):
    samples = []
    for i in range(args.num_samples):
        texts1, _, topics1, lengths1, _ = data_source.get_batch(args.batch_size)
        texts2, _, topics2, lengths2, _ = data_source.get_batch(args.batch_size)
        if type == 'z':
            input_pairs = (texts1[:, :-1].clone().to(device),
                           texts2[:, :-1].clone().to(device))
            topic_pairs = (topics1.to(device), topics1.to(device))
            length_pairs = (lengths1, lengths2)
        elif type == 'topics':
            input_pairs = (texts1[:, :-1].clone().to(device),
                           texts1[:, :-1].clone().to(device))
            topic_pairs = (topics1.to(device), topics2.to(device))
            length_pairs = (lengths1, lengths1)
        elif type == 'both':
            input_pairs = (texts1[:, :-1].clone().to(device),
                           texts2[:, :-1].clone().to(device))
            topic_pairs = (topics1.to(device), topics2.to(device))
            length_pairs = (lengths1, lengths2)
        samples.append(model.interpolate(input_pairs, topic_pairs, length_pairs,
                                         args.max_length, SOS_ID))
    results = []
    for x in zip(*samples):
        sentences = []
        # each x is a list of batch_size x max_length, i.e. batch of generated sentences
        for sample in torch.cat(x).cpu().numpy():
            sentences.append(indices_to_sentence(sample, idx2word))
        results.append(sentences)
    return results


def change_label(data_source, model, new_label, idx2word, device):
    results = []
    for i in range(0, data_source.size, args.batch_size):
        batch_size = min(data_source.size-i, args.batch_size)
        texts, labels, topics, lengths, idx = data_source.get_batch(batch_size, i)
        inputs = texts[:, :-1].clone().to(device)
        topics = topics.to(device)
        new_labels = torch.full_like(labels, new_label).to(device)
        samples = model.reconstruct(inputs, new_labels, topics, lengths, args.max_length, SOS_ID)
        for sample in samples.cpu().numpy()[idx]:
            results.append(indices_to_sentence(sample, idx2word))
    return results


def main(args):
    with open(args.ckpt, 'rb') as f:
        model = torch.load(f)
    model.eval()
    device = torch.device('cuda' if args.cuda else 'cpu')
    model.to(device)
    dataset = args.data.rstrip('/').split('/')[-1]
    with_label = True if dataset in ['yahoo', 'yelp'] else False
    print("Loading data")
    corpus = data.Corpus(args.data, args.num_topics, max_vocab_size=args.max_vocab,
                         max_length=args.max_length, with_label=with_label)
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", corpus.train.size)
    print("\tvocabulary size: ", vocab_size)
    # data to be reconstructed
    input_path = os.path.join(args.data, args.input_file)
    output_path = os.path.join(args.data, args.output_file)
    input_data = data.Data(input_path, (corpus.word2idx, corpus.label2idx),
                           args.num_topics, args.max_length, with_label=with_label)
    if args.task == 'reconstruct':
        for i in range(args.num_samples):
            results = reconstruct(input_data, model, corpus.idx2word, device)
            with open('{0}.{1:d}.rec'.format(output_path, i), 'w') as f:
                f.write('\n'.join(results))
    elif args.task == 'sample':
        if with_label:
            for label in range(corpus.num_classes):
                results = sample(input_data, model, label, corpus.idx2word, device)
                with open('{0}.{1:d}.samp'.format(output_path, label), 'w') as f:
                    f.write('\n'.join(results))
        else:
            results = sample(input_data, model, None, corpus.idx2word, device)
            with open('{0}.samp'.format(output_path), 'w') as f:
                f.write('\n'.join(results))
    elif args.task == 'interpolate':
        if with_label:
            raise ValueError("interpolate option not supported for labeled data")
        results = interpolate(input_data, model, corpus.idx2word, args.interpolate_type, device)
        for i, x in enumerate(results):
            with open('{0}.{1:d}.{2}.int'.format(output_path, i, args.interpolate_type), 'w') as f:
                f.write('\n'.join(x))
    elif args.task == 'change_label':
        if not with_label:
            raise ValueError("change_label not supported for unlabeled data")
        for k in range(corpus.num_classes):
            results = change_label(input_data, model, k, corpus.idx2word, device)
            with open('{0}.{1:d}.cl'.format(output_path, k), 'w') as f:
                f.write('\n'.join(results))
        

if __name__ == '__main__':
    main(args)
