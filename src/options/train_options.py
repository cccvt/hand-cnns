import argparse
import os
import subprocess

from src.utils import filesys


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Input params
        self.parser.add_argument('--dataset', type=str, default='gtea',
                                 help='dataset to use among\
                                 (uciego|gtea|gteagazeplus)')
        self.parser.add_argument('--normalize', type=int, default=1,
                                 help='use imageNet normalization values\
                                 for input during training')
        self.parser.add_argument('--threads', type=int, default=4,
                                 help='number of threads used for data\
                                 loading')

        # Train params
        self.parser.add_argument('--train', type=int, default=1,
                                 help='1 for training, 2 for testing')
        self.parser.add_argument('--epochs', type=int, default=10,
                                 help='number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=2,
                                 help='input mini-batch size')
        self.parser.add_argument('--use_gpu', type=int, default=1,
                                 help='1 to use gpu, 0 for cpu')

        # Valid params
        self.parser.add_argument('--leave_out', type=int, default=0,
                                 help="Index of sequence item to leave out\
                                 for validation")

        # Optim params
        self.parser.add_argument('--lr', type=float, default=0.001,
                                 help='Base learning rate for training')
        self.parser.add_argument('--new_lr', type=float, default=0.01,
                                 help='Learning rate for new (not pretrained) layers\
                                 typically, lr < new_lr')
        self.parser.add_argument('--momentum', type=float, default=0.9,
                                 help='Base learning rate for training')
        self.parser.add_argument('--criterion', type=str, default='MSE',
                                 help='(MSE for mean square |\
                                 CE for cross-entropy)')
        # Save params
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                                 help='where to save models')
        self.parser.add_argument('--exp_id', type=str, default='experiment',
                                 help='name of experiment, determines where\
                                 to store experiment data')
        self.parser.add_argument('--save-freq', type=int, default=5,
                                 help='Frequency at which to save the \
                                 network weights')
        self.parser.add_argument('--save-latest', type=int, default=1,
                                 help='Whether to save the latest computed weights \
                                 at each epoch')

        # Load params
        self.parser.add_argument('--continue_training', type=int, default=0,
                                 help='1 to continue training from saved weights')
        self.parser.add_argument('--continue_epoch', type=int, default=0,
                                 help='Epoch to load for trianing continuation \
                                 latest if 0')

        # Display params
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='number of iters between displays of results\
                                 in visdom')


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
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            opt_file.write('git hash: {}\n'.format(git_hash.strip()))

        return self.opt
