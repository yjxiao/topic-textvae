import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()


def arrange_and_print(data):
    size = data.shape[1]
    # get maximum length for each column
    max_len = np.zeros(size)
    for i, ztext, btext in zip(range(size), data[0], data[2]):
        max_len[i] = max(len(ztext), len(btext))
    results = []
    linebase = '\t'.join(['{:' + str(int(max_len[i])) + '}' for i in range(size)])
    results.append(linebase.format(*data[0]))
    for i in range(1, size):
        items = np.full(size, '', dtype=object)
        items[0] = data[1][i]
        items[i] = data[2][i]
        results.append(linebase.format(*items))
    return '\n'.join(results)


def main(args):
    namebase = 'outputs.txt.{0:d}.{1}.int'
    with open(os.path.join(args.data, namebase.format(0, 'z'))) as f:
        num_samples = len(f.read().strip().split('\n'))
    contents = np.empty((3, 6, num_samples), dtype=object)
    for i, tp in enumerate(['z', 't', 'both']):
        for j in range(6):
            filepath = os.path.join(args.data, namebase.format(j, tp))
            with open(filepath) as f:
                contents[i, j] = np.array(f.read().strip().split('\n'))

    for k in range(num_samples):
        with open('{0}.txt'.format(str(k).zfill(2)), 'w') as f:
            f.write(arrange_and_print(contents[:, :, k]))


if __name__ == '__main__':
    main(args)
