import argparse
from collections import OrderedDict
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import torch

from src.datasets.gteagazeplusimage import GTEAGazePlusImage
from src.netscripts.test import save_preds
from src.utils import options

parser = argparse.ArgumentParser()
parser.add_argument(
    'score_paths',
    type=str,
    nargs='+',
    help='Path to the folders containing the leave out folders')
parser.add_argument(
    '--leave_out',
    type=int,
    default=None,
    help='Idx of subject to evaluate in [0,5]')
parser.add_argument(
    '--destination_folder',
    type=str,
    default='checkpoints/test/gtea_gaze_plus',
    help='')
parser.add_argument(
    '--clip_sizes',
    type=int,
    nargs='+',
    default=[],
    help='Size of extracted clips for each prediction\
    in smae order as score_paths')
args = parser.parse_args()

save_folder = os.path.join(args.destination_folder)
options.process_args(args, save_folder)

assert len(args.clip_sizes) == len(args.score_paths), 'Should have {}\
    clip_sizes for {} score_paths'.format(
    len(args.clip_sizes), len(args.score_paths))

# softmax = torch.nn.Softmax()
softmax = None

all_subjects = ['Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh']

# If leave_out is provided compute only matching score, else compute for
# all subjects
if args.leave_out is not None:
    leave_out_idxs = [args.leave_out]
else:
    leave_out_idxs = list(range(0, 6))

final_scores = []
conf_mats = []
for leave_out_idx in leave_out_idxs:
    seqs = [
        all_subjects[leave_out_idx],
    ]

    dataset = GTEAGazePlusImage(seqs=seqs)

    all_scores = []
    for folder_path in args.score_paths:
        score_path = os.path.join(folder_path, 'gtea_lo_' + str(leave_out_idx),
                                  'prediction_scores.pickle')
        with open(score_path, 'rb') as f:
            scores = pickle.load(f)
            print("Got {} scores from {}".format(len(scores), score_path))
            all_scores.append(scores)

    # Extract indexes for which scores have been produced
    idx_lists = []
    for i in range(len(args.score_paths)):
        if args.clip_sizes:
            scored_idxs = [
                idx
                for idx, (action, obj, subj, rec, beg,
                          end) in enumerate(dataset.action_clips)
                if end - beg >= args.clip_sizes[i]
            ]
            idx_lists.append(scored_idxs)
            assert len(scored_idxs) == len(all_scores[i]),\
                'Received {} predictions for {} samples'.format(
                    len(all_scores[i]),
                    len(len(scored_idxs)))
        else:
            assert len(scored_idxs) == len(all_scores[i]),\
                'Received {} predictions for {} samples'.format(
                    len(all_scores[i]),
                    len(dataset.action_clips))

    mean_preds = {}

    # Compute scores
    conf_mat = np.zeros((dataset.class_nb, dataset.class_nb))
    common_idxs = set(idx_lists[0]).intersection(*idx_lists)
    for idx in range(len(dataset.action_clips)):
        # Extract label idx
        action, objects, subject, recipe, beg, end = dataset.action_clips[idx]
        class_idx = dataset.classes.index((action, objects))
        clip_scores = []
        for score_idx, scores in enumerate(all_scores):
            if idx in idx_lists[score_idx]:
                clip_score = scores[idx_lists[score_idx].index(idx)]
                if softmax is not None:
                    clip_score = softmax(clip_score.unsqueeze(0))
                else:
                    clip_score = clip_score.unsqueeze(0)
                clip_scores.append(clip_score)
            else:
                continue
        if len(clip_scores):
            mean_clip_score = sum(clip_scores) / len(clip_scores)
        else:
            continue

        max_val, best_idx = torch.max(mean_clip_score, 1)
        if softmax is None:
            best_idx = best_idx[0]
        else:
            best_idx = best_idx.data[0]
        best_class = dataset.classes[best_idx]
        mean_preds[idx] = best_class
        conf_mat[class_idx, best_idx] += 1

    acc = conf_mat.trace() / conf_mat.sum()
    print("accuracy {}".format(acc))
    final_scores.append(acc)
    conf_mats.append(conf_mat)

class_accs = {}
class_freqs = {}

sum_conf_mat = conf_mats[0]
for mat in conf_mats[1:]:
    sum_conf_mat = sum_conf_mat + mat
for class_idx, class_preds in enumerate(sum_conf_mat):
    class_acc = class_preds[class_idx] / np.sum(class_preds)
    class_accs[dataset.classes[class_idx]] = class_acc
    class_freqs[dataset.classes[class_idx]] = np.sum(class_preds) / np.sum(
        sum_conf_mat)

class_accs = OrderedDict(sorted(class_accs.items(), key=lambda t: t[1]))
class_freqs = OrderedDict(sorted(class_freqs.items(), key=lambda t: t[1]))


def plot_dict(ordered_dic):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(ordered_dic)), ordered_dic.values())
    plt.xticks(np.arange(len(ordered_dic)))
    labels = [
        dataset.get_class_str(act, obj) for act, obj in ordered_dic.keys()
    ]
    ax.set_xticklabels(labels, rotation=90)
    fig.tight_layout()
    plt.show()


# plot_dict(class_accs)
plot_dict(class_freqs)

fig, ax = plt.subplots()
bar_locations = np.arange(len(class_accs))
bar_width = 0.4
spatio = ax.bar(np.arange(len(class_accs)), class_accs.values(), bar_width)
rearranged_class_freqs = OrderedDict()
for class_key, class_value in class_accs.items():
    rearranged_class_freqs[class_key] = class_freqs[class_key]
rand = ax.bar(
    np.arange(len(class_freqs)) - bar_width,
    rearranged_class_freqs.values(),
    bar_width,
    color='r')
ax.legend((spatio[0], rand[0]), ('our model', 'frequency'))
plt.xticks(bar_locations)
labels = [dataset.get_class_str(act, obj) for act, obj in class_accs.keys()]
ax.set_xticklabels(labels, rotation=90)
fig.tight_layout()
plt.show()
print('Retrieved scores {}'.format(final_scores))
print('mean : {}'.format(sum(final_scores) / len(final_scores)))

import pdb
pdb.set_trace()
