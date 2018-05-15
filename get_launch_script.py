import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data/ptb',
                    help="location of the data folder")
parser.add_argument('--num_topics', type=int, default=8,
                    help="number of topics to use for the training")
parser.add_argument('--joint', action='store_true',
                    help="use joint model, otherwise use marginal model")
parser.add_argument('--bow', action='store_true',
                    help="add bow loss during training")
args = parser.parse_args()


command_base = 'python main.py --data={0} --num_topics={1:d} --wd={2:.0e} --kla{3}{4}'
dataset = args.data.rstrip('/').split('/')[-1]
logpath_base = 'logs/{0}/std.tpc{1:d}.wd{2:.0e}.kla{3}{4}.{0}.log'
lines = []
for wd in [0.001, 0.003, 0.01]:
    command = command_base.format(args.data, args.num_topics, wd,
                                  ' --bow' if args.bow else '',
                                  ' --joint' if args.joint else '')
    logpath = logpath_base.format(dataset, args.num_topics, wd,
                                  '.bow' if args.bow else '',
                                  '.jt' if args.joint else '')
    lines.append(command + ' > ' + logpath)

script_name = 'std.{0}.tpc{1:d}{2}{3}.sh'.format(dataset, args.num_topics,
                                                 '.bow' if args.bow else '',
                                                 '.jt' if args.joint else '')    
with open(script_name, 'w') as f:
    f.write('\n'.join(lines))
    
