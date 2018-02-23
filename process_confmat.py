import argparse
from collections import defaultdict
from operator import itemgetter
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from actiondatasets.gteagazeplus import GTEAGazePlus
from src.post_utils.display import plot_epoch_conf_mat


def get_confmat(path):
    """Processes logs in format "(somehting),loss_1:0.1234,loss_2:0.3"
    """
    with open(path, 'rb') as confmat_file:
        conf_mat = pickle.load(confmat_file)
    return conf_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        type=str,
        help='path to checkpoints folders')
    parser.add_argument(
        '--vis', action='store_true', help='Whether to plot the log curves')
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Each row of the confmat sums to 1')
    parser.add_argument(
        '--epoch', type=int, help='What itertation use to average results')
    parser.add_argument(
        '--prefixes',
        nargs='+',
        type=str,
        help='Descriptions of run for labels, one per checkpoint')
    parser.add_argument('--gteagazeplus', action='store_true')
    opt = parser.parse_args()

    if opt.prefixes is not None:
        assert len(opt.prefixes) == len(opt.checkpoints), \
                'Should have as many prefixes as checkpoints but '\
                'got {} and {}'.format(opt.prefixes, opt.checkpoints)

    if opt.gteagazeplus:
        all_subjects = [
            'Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh'
        ]
        for checkpoint in opt.checkpoints:
            dataset = GTEAGazePlus(
                root_folder='data/GTEAGazePlusdata2',
                original_labels=True,
                seqs=all_subjects)

            train_conf_template = os.path.join(
                checkpoint, 'gtea_lo_{}/train_conf_mat.pickle')
            val_conf_template = os.path.join(checkpoint,
                                             'gtea_lo_{}/val_conf_mat.pickle')
            for leave_out in range(6):
                print(leave_out)
                train_conf_path = train_conf_template.format(str(leave_out))
                val_conf_path = val_conf_template.format(str(leave_out))
                train_confmat = get_confmat(train_conf_path)
                val_confmat = get_confmat(val_conf_path)
                if opt.vis:
                    plot_epoch_conf_mat(
                        train_confmat,
                        title='Train conf mat',
                        epoch=opt.epoch,
                        labels=dataset.classes,
                        normalize=opt.normalize)
                    plot_epoch_conf_mat(
                        val_confmat,
                        title='Val conf mat',
                        epoch=opt.epoch,
                        labels=dataset.classes,
                        normalize=opt.normalize)
