import argparse
import datetime
import os
import subprocess
import sys

from netraining.utils import filesys
from netraining.utils import options


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):

        # GPU params
        self.parser.add_argument(
            '--use_gpu',
            type=int,
            default=1,
            help='Whether to use gpu (1) or cpu (0)')
        self.parser.add_argument(
            '--gpu_parallel',
            action='store_true',
            help='Whether to use several gpus to parallelize in the batch dim')
        self.parser.add_argument(
            '--gpu_nb',
            type=int,
            default=1,
            help='Number of GPUs to use if parallelizing')

        # Input params
        self.parser.add_argument(
            '--dataset',
            type=str,
            default='gteagazeplus',
            help='dataset to use among\
                                 (uciego|gtea|gteagazeplus|gteagazeplus_tres|smthg)'
        )
        self.parser.add_argument(
            '--threads',
            type=int,
            default=4,
            help='number of threads used for data\
                                 loading')
        self.parser.add_argument(
            '--modality',
            default='rgb',
            help='Can be [rgb|flow|melspec|objectness|heatmaps|..]')
        self.parser.add_argument(
            '--flow_type', type=str, default='tvl1', help='in [farn|tvl1]')
        self.parser.add_argument(
            '--rescale_flows',
            action='store_true',
            help='activate to scale to [min, max], otherwise to stay in [0,255]'
        )
        self.parser.add_argument(
            '--heatmap_nb',
            type=int,
            default='21',
            help='Number of heatmap channels')
        self.parser.add_argument(
            '--batch_size', type=int, default=2, help='input mini-batch size')

        # Save params
        self.parser.add_argument(
            '--checkpoint_dir',
            type=str,
            default='./checkpoints',
            help='where to save models')
        self.parser.add_argument(
            '--exp_id',
            type=str,
            default='experiment',
            help='name of experiment, determines where\
                                 to store experiment data')

        # Averaging params
        self.parser.add_argument(
            '--frame_nb',
            type=int,
            default=10,
            help='number of frames to average\
                                 at test time')

        # Display params
        self.parser.add_argument(
            '--visualize',
            type=int,
            default=1,
            help='0 to disable visdom plots')
        self.parser.add_argument(
            '--display_port',
            type=int,
            default=8097,
            help='Visdom display port')

        # GTEAGaze+ specific option
        self.parser.add_argument(
            '--leave_out',
            type=int,
            default=0,
            help="Index of sequence item to leave out\
                                 for validation")

    def parse(self, arguments=None):
        self.opt = self.parser.parse_args(arguments)

        # Print and save options
        exp_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp_id)
        options.process_args(self.opt, exp_dir)

        return self.opt
