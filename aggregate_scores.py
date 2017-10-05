import argparse
import os
import pickle

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
    help='Name of the split (int [test|valid])')
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
    print("accuracy {}".format(acc))

prediction_file = os.path.join(
    save_folder, 'predictions_{split}.csv'.format(split=args.split))

save_preds(mean_preds, prediction_file)
