import argparse
import datetime
import os
import subprocess
import sys

from src.utils import filesys


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):

        self.parser.add_argument('--use_gpu', type=int, default=1,
                                 help='Whether to use gpu (1) or cpu (0)')
        # Input params
        self.parser.add_argument('--dataset', type=str, default='gteagazeplus',
                                 help='dataset to use among\
                                 (uciego|gtea|gteagazeplus|smthgsmthg)')
        self.parser.add_argument('--threads', type=int, default=4,
                                 help='number of threads used for data\
                                 loading')
        self.parser.add_argument('--use_flow', type=int, default=0,
                                 help='Whether to use flow or RGB')
        self.parser.add_argument('--batch_size', type=int, default=2,
                                 help='input mini-batch size')
        # Save params
        self.parser.add_argument('--checkpoint_dir', type=str,
                                 default='./checkpoints',
                                 help='where to save models')
        self.parser.add_argument('--exp_id', type=str, default='experiment',
                                 help='name of experiment, determines where\
                                 to store experiment data')

        # Averaging params
        self.parser.add_argument('--frame_nb', type=int,
                                 default=10, help='number of frames to average\
                                 at test time')

        # Display params
        self.parser.add_argument('--visualize', type=int,
                                 default=1, help='0 to disable visdom plots')

        # GTEAGaze+ specific option
        self.parser.add_argument('--leave_out', type=int, default=0,
                                 help="Index of sequence item to leave out\
                                 for validation")

    def parse(self, arguments=None):
        self.opt = self.parser.parse_args(arguments)

        # Print  options
        args = vars(self.opt)

        print('---- Options ----')
        for k, v in sorted(args.items()):
            print('{option}: {value}'.format(option=k, value=v))

        # Save options
        exp_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp_id)
        filesys.mkdir(exp_dir)
        opt_path = os.path.join(exp_dir, 'opt.txt')
        with open(opt_path, 'a') as opt_file:
            opt_file.write('====== Options ======\n')
            for k, v in sorted(args.items()):
                opt_file.write('{option}: {value}\n'.format(
                    option=str(k), value=str(v)))
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            opt_file.write('git hash: {}\n'.format(git_hash.strip()))
            opt_file.write('launched {} at {}\n'.format(
                str(sys.argv[0]),
                str(datetime.datetime.now())))

        return self.opt
