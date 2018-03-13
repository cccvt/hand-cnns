import argparse
from collections import defaultdict
from operator import itemgetter
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from actiondatasets.gteagazeplus import GTEAGazePlus
from actiondatasets.smthgv2 import SmthgV2
import argutils
from src.post_utils.display import plot_epoch_conf_mat, plot_confmat, normalize_rows


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
    args = parser.parse_args()
    argutils.print_args(args)

    if args.prefixes is not None:
        assert len(args.prefixes) == len(args.checkpoints), \
                'Should have as many prefixes as checkpoints but '\
                'got {} and {}'.format(args.prefixes, args.checkpoints)

    if args.gteagazeplus:
        all_subjects = [
            'Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh'
        ]
        for checkpoint in args.checkpoints:
            dataset = GTEAGazePlus(
                root_folder='data/GTEAGazePlusdata2',
                original_labels=True,
                seqs=all_subjects)
            import pdb
            pdb.set_trace()

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
                if args.vis:
                    plot_epoch_conf_mat(
                        train_confmat,
                        title='Train conf mat',
                        epoch=args.epoch,
                        labels=dataset.classes,
                        normalize=args.normalize)
                    plot_epoch_conf_mat(
                        val_confmat,
                        title='Val conf mat',
                        epoch=args.epoch,
                        labels=dataset.classes,
                        normalize=args.normalize)
    else:
        for checkpoint in args.checkpoints:
            train_conf_path = os.path.join(checkpoint, 'train_conf_mat.pickle')
            train_confmat = get_confmat(train_conf_path)
            val_conf_path = os.path.join(checkpoint, 'val_conf_mat.pickle')
            val_confmat = get_confmat(val_conf_path)
            dataset = SmthgV2()
            plot_epoch_conf_mat(
                train_confmat,
                title='Train conf mat',
                epoch=args.epoch,
                labels=dataset.classes,
                normalize=args.normalize,
                both_labels=True,
                display=args.vis)
            val_fig, val_ax = plot_epoch_conf_mat(
                val_confmat,
                title='Val conf mat',
                epoch=args.epoch,
                labels=dataset.classes,
                normalize=args.normalize,
                both_labels=True,
                display=args.vis)
            take_put = True
            if take_put is True:
                take_put_idx = [
                    idx for idx, class_name in enumerate(dataset.classes)
                    if class_name.startswith('Tak')
                    or class_name.startswith('Put')
                ]
                take_put_classes = [
                    class_name
                    for idx, class_name in enumerate(dataset.classes)
                    if class_name.startswith('Tak')
                    or class_name.startswith('Put')
                ]
                epoch_confmat = val_confmat[args.epoch]
                sub_confmat = epoch_confmat[np.ix_(take_put_idx, take_put_idx)]
                if args.normalize:
                    sub_confmat = normalize_rows(sub_confmat) * 100
                plot_confmat(
                    sub_confmat,
                    display=True,
                    labels=take_put_classes,
                    normalize=False,
                    annotate=True,
                    both_labels=False)
            val_fig.set_size_inches(30, 30)
            results_path = os.path.join(checkpoint, 'results')
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            pdf_path = os.path.join(results_path,
                                    'conf_mat_epoch_{}.pdf'.format(args.epoch))
            val_fig.savefig(pdf_path)
            print('Saved to {}'.format(pdf_path))
