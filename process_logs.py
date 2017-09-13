import argparse
from collections import defaultdict
from operator import itemgetter
import os

import matplotlib.pyplot as plt
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


def plot_logs(logs, exclude=None):
    for score_name, scores in logs.items():
        plt.plot(scores, label=score_name)

    plt.legend()
    plt.show()


def process_logs(log_path, vis=True,
                 score_iter=20,
                 pop_loss=None):
    logs = get_logs(log_path)

    # Pop loss to display separately
    if pop_loss is not None:
        pop_value = logs.pop(pop_loss)
        pop_logs = {pop_loss: pop_value}
        if vis:
            plot_logs(pop_logs)

    # Plot all (remaining) losses
    if vis:
        plot_logs(logs)
    iter_scores = []
    for score_name, score_values in logs.items():
        iter_scores.append((score_name, score_values[score_iter]))
    return sorted(iter_scores, key=itemgetter(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        help='path to checkpoint folder')
    parser.add_argument('--vis', action='store_true',
                        help='Whether to plot the log curves')
    parser.add_argument('--score_iter', type=int, default=10,
                        help='What itertation use to average results')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--aggreg', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gteagazeplus', action='store_true')
    opt = parser.parse_args()

    if opt.gteagazeplus:
        template_aggreg = os.path.join(
            opt.checkpoint, 'gtea_lo_{}/valid_aggreg.txt')
        template_valid = os.path.join(
            opt.checkpoint, 'gtea_lo_{}/valid_log.txt')
        template_train = os.path.join(
            opt.checkpoint, 'gtea_lo_{}/train_log.txt')

        if opt.aggreg:

            # Dict to collect the score from the score_iter iteration for
            # all folds
            all_iter_scores = defaultdict(list)
            for leave_out in range(6):
                print(leave_out)
                log_file = template_aggreg.format(str(leave_out))
                leave_out_iter_scores = process_logs(log_file,
                                                     score_iter=opt.score_iter,
                                                     vis=opt.vis)
                for loss, value in leave_out_iter_scores:
                    all_iter_scores[loss].append(value)

            # Display iter scores
            for loss, value in all_iter_scores.items():
                print(loss)
                print('mean {m} for loss {loss} \
                        ({values})'.format(m=np.mean(value),
                                           loss=loss, values=value))

        if opt.valid:
            valid_scores = []
            for leave_out in range(6):
                print(leave_out)
                log_file = template_valid.format(str(leave_out))
                logs = get_logs(log_file)
                loss_name = 'loss'
                loss_values = logs.pop(loss_name)
                loss_log = {loss_name: loss_values}
                if opt.vis:
                    plot_logs(logs)
                    plot_logs(loss_log)
                valid_scores.append(logs['top1'][opt.score_iter])
            print(valid_scores)
            print('mean of valid scores : {}'.format(np.mean(valid_scores)))
        if opt.train:
            train_scores = []
            for leave_out in range(6):
                print(leave_out)
                log_file = template_train.format(str(leave_out))
                logs = get_logs(log_file)
                loss_name = 'loss'
                loss_values = logs.pop(loss_name)
                loss_log = {loss_name: loss_values}
                if opt.vis:
                    plot_logs(logs)
                    plot_logs(loss_log)
                train_scores.append(logs['top1'][opt.score_iter])
            print(train_scores)
            print('mean of train scores: {}'.format(np.mean(train_scores)))

    else:
        aggreg_file = os.path.join(opt.checkpoint, 'valid_aggreg.txt')
        valid_file = os.path.join(opt.checkpoint, 'valid_log.txt')
        train_file = os.path.join(opt.checkpoint, 'train_log.txt')
        if opt.aggreg:
            aggreg_logs = get_logs(aggreg_file)
            plot_logs(aggreg_logs)
        if opt.valid:
            valid_logs = get_logs(valid_file)
            plot_logs(valid_logs)
        if opt.train:
            train_logs = get_logs(train_file)
            plot_logs(train_logs)
