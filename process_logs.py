import argparse
from collections import defaultdict
from operator import itemgetter
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def get_logs(path):
    """Processes logs in format "(somehting),loss_1:0.1234,loss_2:0.3"
    """
    logs = defaultdict(list)
    with open(path, 'r') as log_file:
        for line in log_file:
            if line[0] != '=':
                results = line.strip().split(')')[1].split(',')
                for score in results:
                    if ':' in score:
                        score_name, score_value = score.split(':')
                        logs[score_name].append(float(score_value))
    return logs


def plot_logs(logs, score_name='top1', y_max=1, prefix=None):

    # Plot all losses
    scores = logs[score_name]
    if prefix is None:
        label = score_name
    else:
        label = prefix + score_name
    plt.plot(scores, label=label)
    if score_name == 'top1':
        # Set maximum for y axis
        plt.minorticks_on()
        x1, x2, y1, y2 = plt.axis()
        axes = plt.gca()
        axes.yaxis.set_minor_locator(MultipleLocator(0.02))
        plt.axis((x1, x2, 0, y_max))
        plt.grid(b=True, which='minor', color='k', alpha=0.2, linestyle='-')
        plt.grid(b=True, which='major', color='k', linestyle='-')


def process_logs(logs, plot_metric='top1', score_iter=20):
    iter_scores = []
    for score_name, score_values in logs.items():
        assert_message = 'index {} out of range for score_values of len {}'.format(
            score_iter, len(score_values))
        assert len(score_values) > score_iter, assert_message

        iter_scores.append((score_name, score_values[score_iter]))
    return sorted(iter_scores, key=itemgetter(0))


def display_logs(log_file,
                 score_type,
                 score_iter=10,
                 plot_metric='top1',
                 vis=True):
    """Process logs, prints the results for the given score_iter
    and plots the matching curves
    """
    logs = get_logs(log_file)
    iter_scores = process_logs(
        logs, score_iter=score_iter, plot_metric=plot_metric)
    # Plot all losses
    if vis:
        plot_logs(logs, score_name=plot_metric)

        print('==== {} scores ===='.format(score_type))
        for loss, val in iter_scores:
            print('{val}: {loss}'.format(val=val, loss=loss))


def print_iter_scores(all_iter_scores, iter_nb):
    """
    Args:
    all_iter_scores (list): in format
    [{score_name, [fold1_score, fold2_score,...]}, /..]
    """
    # Display iter scores
    print(all_iter_scores)
    for loss, value in all_iter_scores.items():
        print('{loss} : {m} at iter {it} ({values})'.format(
            m=np.mean(value), it=iter_nb, loss=loss, values=value))


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
        '--plot_metric',
        default='top1',
        help='Metric to display in plot in [top1|top5|loss]')
    parser.add_argument(
        '--score_iter',
        type=int,
        default=10,
        help='What itertation use to average results')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--aggreg', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gteagazeplus', action='store_true')
    opt = parser.parse_args()

    if opt.gteagazeplus:
        template_aggreg = os.path.join(opt.checkpoint,
                                       'gtea_lo_{}/valid_aggreg.txt')
        template_valid = os.path.join(opt.checkpoint,
                                      'gtea_lo_{}/valid_log.txt')
        template_train = os.path.join(opt.checkpoint,
                                      'gtea_lo_{}/train_log.txt')

        if opt.aggreg:

            # Dict to collect the score from the score_iter iteration for
            # all folds
            all_iter_scores = defaultdict(list)
            for leave_out in range(6):
                print(leave_out)
                log_file = template_aggreg.format(str(leave_out))
                logs = get_logs(log_file)
                leave_out_iter_scores = process_logs(
                    logs,
                    score_iter=opt.score_iter,
                    vis=opt.vis,
                    plot_metric=opt.plot_metric)
                # Plot all losses
                if opt.vis:
                    plot_logs(logs, score_name=opt.plot_metric)

                for loss, value in leave_out_iter_scores:
                    all_iter_scores[loss].append(value)

        # Display iter scores
        print('==== Aggreg scores ====')
        print_iter_scores(all_iter_scores, opt.score_iter)

        if opt.valid:
            all_iter_scores = defaultdict(list)
            for leave_out in range(6):
                print(leave_out)
                log_file = template_valid.format(str(leave_out))
                leave_out_iter_scores = process_logs(
                    log_file,
                    score_iter=opt.score_iter,
                    vis=opt.vis,
                    plot_metric=opt.plot_metric)
                # Plot all losses
                if opt.vis:
                    plot_logs(logs, score_name=opt.plot_metric)
                for loss, value in leave_out_iter_scores:
                    all_iter_scores[loss].append(value)

            # Display iter scores
            print('==== Valid scores ====')
            print_iter_scores(all_iter_scores, opt.score_iter)

        if opt.train:
            all_iter_scores = defaultdict(list)
            for leave_out in range(6):
                print(leave_out)
                log_file = template_train.format(leave_out)
                leave_out_iter_scores = process_logs(
                    log_file,
                    score_iter=opt.score_iter,
                    vis=opt.vis,
                    plot_metric=opt.plot_metric)
                # Plot all losses
                if opt.vis:
                    plot_logs(logs, score_name=opt.plot_metric)

                for loss, value in leave_out_iter_scores:
                    all_iter_scores[loss].append(value)

                    # Display iter scores
                    print('==== Train scores ====')
                    print_iter_scores(all_iter_scores, opt.score_iter)

    # If not GTEA gaze dataset
    else:
        for checkpoint in opt.checkpoints:
            print('==== Scores for checkpoint {} ===='.format(checkpoint))
            aggreg_file = os.path.join(checkpoint, 'valid_aggreg.txt')
            valid_file = os.path.join(checkpoint, 'valid_log.txt')
            train_file = os.path.join(checkpoint, 'train_log.txt')
            if opt.aggreg:
                display_logs(
                    aggreg_file,
                    score_type='Aggreg',
                    score_iter=opt.score_iter,
                    plot_metric=opt.plot_metric,
                    vis=opt.vis)

            if opt.valid:
                display_logs(
                    valid_file,
                    score_type='Valid',
                    score_iter=opt.score_iter,
                    plot_metric=opt.plot_metric,
                    vis=opt.vis)

            if opt.train:
                display_logs(
                    train_file,
                    score_type='Train',
                    score_iter=opt.score_iter,
                    plot_metric=opt.plot_metric,
                    vis=opt.vis)
        if opt.vis:
            plt.legend()
            plt.show()
