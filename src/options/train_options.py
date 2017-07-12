import argparse
import os

from src.utils import filesys


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Train params
        self.parser.add_argument('--dataset', type=str, default='gtea',
                                 help='dataset to use (uciego|gtea|gteagazeplus)')
        self.parser.add_argument('--epochs', type=int,
                                 default=1, help='number of training epochs')
        self.parser.add_argument(
            '--batch_size', type=int, default=2, help='input mini-batch size')
        self.parser.add_argument('--criterion', type=str, default='MSE',
                                 help='(MSE for mean square | CE for cross-entropy)')
        self.parser.add_argument(
            '--use_gpu', type=int, default=1, help='1 to use gpu, 0 for cpu')
        # Save params
        self.parser.add_argument('--checkpoint_dir', type=str,
                                 default='./checkpoints', help='where to save models')
        self.parser.add_argument('--exp_id', type=str, default='experiment',
                                 help='name of experiment, determines where to store experiment data')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # Print  options
        args = vars(self.opt)

        print('---- Options ----')
        for k, v in sorted(args.items()):
            print('{option}: {value}'.format(option=k, value=v))

        # Save options
        exp_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp_id)
        filesys.mkdir(exp_dir)
        opt_path = os.path.join(exp_dir, 'opt.txt')
        with open(opt_path, 'wt') as opt_file:
            for k, v in sorted(args.items()):
                opt_file.write('{option}: {value}\n'.format(
                    option=str(k), value=str(v)))
        return self.opt
