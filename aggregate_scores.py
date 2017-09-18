import pickle
import os

import numpy as np
import torch

from src.datasets.smthg import Smthg
from src.netscripts.test import save_preds

split = 'test'
# split = 'valid'

if split == 'test':
    evaluate = False
else:
    evaluate = True

# softmax = torch.nn.Softmax()
softmax = None

dataset = Smthg(split=split)

if split == 'test':
    c3d_scores_path = 'checkpoints/test/c3d_rgb_smthgsmthg_epoch25/prediction_scores.pickle'
    stacked_flow_scores_path = 'checkpoints/test/res_stack_flow_smthg_epoch60/prediction_test_scores.pickle'
else:
    c3d_scores_path = 'checkpoints/test/c3d_rgb_smthgsmthg_valid_epoch25/prediction_scores.pickle'
    stacked_flow_scores_path = 'checkpoints/test/res_stack_flow_smthg_epoch60/prediction_scores.pickle'

with open(c3d_scores_path, 'rb') as c3d_file:
    c3d_scores = pickle.load(c3d_file)
    print("Got {} scores from {}".format(len(c3d_scores), c3d_scores_path))

with open(stacked_flow_scores_path, 'rb') as stack_file:
    stacked_flow_scores = pickle.load(stack_file)
    print("Got {} scores from {}".format(len(stacked_flow_scores), stacked_flow_scores_path))


mean_preds = {}
all_scores = [c3d_scores, stacked_flow_scores]

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


prediction_file = 'predictions/c3d_stack_{split}.csv'.format(split=split)

save_preds(mean_preds, prediction_file)
