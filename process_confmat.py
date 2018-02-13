import argparse
from collections import defaultdict
from operator import itemgetter
import pickle
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from actiondatasets.gteagazeplus import GTEAGazePlus


def normalize_rows(mat):
    mat = np.copy(mat)
    for i, row in enumerate(mat):
        norm_row = row.sum() or 1
        mat[i] = row / norm_row
    return mat


def get_confmat(path):
    """Processes logs in format "(somehting),loss_1:0.1234,loss_2:0.3"
    """
    with open(path, 'rb') as confmat_file:
        conf_mat = pickle.load(confmat_file)
    return conf_mat


def plot_conf_mat(confmat,
                  title=None,
                  labels=None,
                  epoch=None,
                  normalize=False):
    """
    Args:
        score_type (str): label for current curve, [valid|train|aggreg]
    """
    if epoch is None:
        mat = confmat[-1]
    else:
        if epoch > confmat.shape[0]:
            raise ValueError(
                'Epoch {} should be below {}'.format(epoch, confmat.shape[0]))
        mat = confmat[epoch]
        # Plot confmat
    if normalize:
        mat = normalize_rows(mat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(mat, cmap='viridis')
    fig.colorbar(cax)
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        str_labels = [stringify(label) for label in labels]
        ax.set_xticklabels(str_labels, rotation=90)
        ax.set_xticks(range(len(str_labels)))
        ax.set_yticklabels(str_labels)
        ax.set_yticks(range(len(str_labels)))
    plt.tight_layout()
    plt.show()


def stringify(nested):
    if isinstance(nested, str):
        return nested
    if (isinstance(nested, tuple)
            or isinstance(nested, list)) and len(nested) == 1:
        return stringify(nested[0])
    else:
        return stringify(nested[0]) + '_' + stringify(nested[1:])


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
                    plot_conf_mat(
                        train_confmat,
                        title='Train conf mat',
                        epoch=opt.epoch,
                        labels=dataset.classes,
                        normalize=opt.normalize)
                    plot_conf_mat(
                        val_confmat,
                        title='Val conf mat',
                        epoch=opt.epoch,
                        labels=dataset.classes,
                        normalize=opt.normalize)
