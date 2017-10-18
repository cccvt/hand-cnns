import argparse
import os
import pickle

import numpy as np
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
    leave_out_idxs = list(range(len(all_subjects)))

final_scores = []
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
print('Retrieved scores {}'.format(final_scores))
print('mean : {}'.format(sum(final_scores) / len(final_scores)))
