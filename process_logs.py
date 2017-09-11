import argparse
from collections import defaultdict
import os

import matplotlib.pyplot as plt

def process_logs(path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
            help='path to checkpoint folder')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--aggreg', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--gteagazeplus', action='store_true')
    opt = parser.parse_args()

    if opt.gteagazeplus:
        template_aggreg = os.path.join(opt.checkpoint, 'gtea_lo_{}/valid_aggreg.txt')
        template_valid = os.path.join(opt.checkpoint, 'gtea_lo_{}/valid_log.txt')
        template_train = os.path.join(opt.checkpoint, 'gtea_lo_{}/train_log.txt')
        if opt.aggreg:
            for leave_out in range(6):
                print(leave_out)
                log_file = template_aggreg.format(str(leave_out))
                logs = process_logs(log_file)
                plot_logs(logs)

        if opt.valid:
            for leave_out in range(6):
                print(leave_out)
                log_file = template_valid.format(str(leave_out))
                logs = process_logs(log_file)
                loss_name = 'loss'
                loss_values = logs.pop(loss_name)
                loss_log = {loss_name: loss_values}
                plot_logs(logs)
                plot_logs(loss_log)
        if opt.train:
            for leave_out in range(6):
                print(leave_out)
                log_file = template_train.format(str(leave_out))
                logs = process_logs(log_file)
                loss_name = 'loss'
                loss_values = logs.pop(loss_name)
                loss_log = {loss_name: loss_values}
                plot_logs(logs)
                plot_logs(loss_log)

    else:
        aggreg_file = os.path.join(opt.checkpoint, 'valid_aggreg.txt')
        valid_file = os.path.join(opt.checkpoint, 'valid_log.txt')
        train_file = os.path.join(opt.checkpoint, 'train_log.txt')
        if opt.aggreg:
            aggreg_logs = process_logs(aggreg_file)
            plot_logs(aggreg_logs)
        if opt.valid:
            valid_logs = process_logs(valid_file)
            plot_logs(valid_logs)
        if opt.train:
            train_logs = process_logs(train_file)
            plot_logs(train_logs)

   
