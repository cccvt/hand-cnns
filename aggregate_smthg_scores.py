import argparse
from collections import OrderedDict
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import torch

from src.datasets.smthg import Smthg
from src.netscripts.test import save_preds
from src.utils import options

parser = argparse.ArgumentParser()
parser.add_argument(
    'score_paths',
    type=str,
    nargs='+',
    help='Path to score files (as many as you want)')
parser.add_argument(
    '--split',
    type=str,
    default='test',
    help='Name of the split (int [test|valid])')
parser.add_argument(
    '--destination_folder',
    type=str,
    default='checkpoints/test/debug_aggreg',
    help='')
parser.add_argument(
    '--display_hist', action='store_true', help='Display class histograms')
args = parser.parse_args()

save_folder = os.path.join(args.destination_folder, args.split)
options.process_args(args, save_folder)

if args.split == 'test':
    evaluate = False
else:
    evaluate = True

# softmax = torch.nn.Softmax()
softmax = None

dataset = Smthg(split=args.split)

all_scores = []
for path in args.score_paths:
    with open(path, 'rb') as f:
        scores = pickle.load(f)
        print("Got {} scores from {}".format(len(scores), path))
        all_scores.append(scores)

sanity = [len(score) == len(dataset) for score in all_scores]

mean_preds = {}

if evaluate:
    conf_mat = np.zeros((dataset.class_nb, dataset.class_nb))
for clip_id, label, max_frame in dataset.sample_list:
    clip_scores = []
    for scores in all_scores:
        clip_score = scores[clip_id]
        if softmax is not None:
            clip_score = softmax(clip_score.unsqueeze(0))
        else:
            clip_score = clip_score.unsqueeze(0)
        clip_scores.append(clip_score)
    mean_clip_score = sum(clip_scores) / len(clip_scores)

    max_val, best_idx = torch.max(mean_clip_score, 1)
    if softmax is None:
        best_idx = best_idx[0]
    else:
        best_idx = best_idx.data[0]
    best_class = dataset.classes[best_idx]
    mean_preds[clip_id] = best_class
    if evaluate:
        class_idx = dataset.classes.index(label)
        conf_mat[class_idx, best_idx] += 1

if evaluate:
    acc = conf_mat.trace() / conf_mat.sum()
    result_path = os.path.join(save_folder, 'result.txt')
    message = 'mean accuracy :{}'.format(acc)
    print(message)
    print(result_path)
    with open(result_path, "a") as file:
        file.write(message + '\n')

prediction_file = os.path.join(
    save_folder, 'predictions_{split}.csv'.format(split=args.split))

save_preds(mean_preds, prediction_file)

if args.display_hist:
    # Print accuracy plots
    class_accs = {}
    class_freqs = {}
    # Plot labels scores
    for class_idx, class_preds in enumerate(conf_mat):
        class_acc = class_preds[class_idx] / np.sum(class_preds)
        class_accs[dataset.classes[class_idx]] = class_acc
        class_freqs[dataset.classes[class_idx]] = np.sum(class_preds) / np.sum(
            conf_mat)

    class_accs = OrderedDict(sorted(class_accs.items(), key=lambda t: t[1]))
    class_freqs = OrderedDict(sorted(class_freqs.items(), key=lambda t: t[1]))

    def plot_dict(ordered_dic):
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(ordered_dic)), ordered_dic.values(), color='r')
        ax.set_ylim([0, 0.02])
        plt.xticks(np.arange(len(ordered_dic)))
        labels = ordered_dic.keys()

        ax.set_xticklabels(labels, rotation=45, ha='right')
        fig.tight_layout()
        plt.show()

    choose_nb = 10
    best = {
        k: class_freqs[k]
        for i, (k, v) in enumerate(class_accs.items())
        if i > len(class_accs) - choose_nb
    }
    worst = {
        k: class_freqs[k]
        for i, (k, v) in enumerate(class_accs.items()) if i < choose_nb
    }

    plot_dict(best)
    plot_dict(worst)
    plot_dict(class_freqs)
