import numpy as np
import torch

from .evaluation import batch_topk_accuracy


def test_top1_1_accuracy():
    gt = torch.eye(4, 10)
    gt_var = gt
    assert batch_topk_accuracy(gt_var, gt, k=1) == 1


def test_top1_0_4_accuracy():
    n_row = 10
    n_col = 4
    gt = torch.eye(n_row, n_col)
    gt_var = gt
    topk_score = batch_topk_accuracy(gt_var, gt, k=1)
    assert topk_score == n_col / n_row


def test_top1_custom_accuracy():
    np_scores = np.array([[0.2, 0.6, 0.5, -0.1, 0],
                          [1.2, 0.4, 1.5, -0.1, 0]])
    np_gt = np.array([[0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0]])
    var_scores = torch.from_numpy(np_scores)
    t_gt = torch.from_numpy(np_gt)

    top1_score = batch_topk_accuracy(var_scores, t_gt, k=1)
    assert top1_score == 1


def test_top2_custom_accuracy():
    np_scores = np.array([[0.2, 0.6, 0.5, -0.1, 0],
                          [1.2, 0.4, 1.5, -0.1, 0]])
    np_gt = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])
    var_scores = torch.from_numpy(np_scores)
    t_gt = torch.from_numpy(np_gt)

    top2_score = batch_topk_accuracy(var_scores, t_gt, k=2)
    assert top2_score == 1


def test_top3_custom_accuracy():
    np_scores = np.array([[0.2, 0.6, 0.5, -0.1, 0],
                          [1.2, 0.4, 1.5, -0.1, 0]])
    np_gt = np.array([[0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0]])
    var_scores = torch.from_numpy(np_scores)
    t_gt = torch.from_numpy(np_gt)

    top3_score = batch_topk_accuracy(var_scores, t_gt, k=3)
    assert top3_score == 0


def test_top5_custom_accuracy():
    np_scores = np.array([[0.2, 0.6, 0.5, -0.1, 0, 0.2, 0.1],
                          [1.2, 0.4, -1.5, -0.1, 0, 10, 0.3],
                          [1.2, 0.8, 1.5, 0, 0, -5, -0.23],
                          [1.2, 0.1, 1.5, 4, 0.7, 0.9, 0.05]])
    np_gt = np.array([[0, 0, 0, 0, 0, 1, 0],  # Right
                      [0, 0, 0, 1, 0, 0, 0],  # Wrong
                      [0, 0, 0, 1, 0, 0, 0],  # Right
                      [0, 0, 0, 0, 0, 0, 1]])  # Wrong
    var_scores = torch.from_numpy(np_scores)
    t_gt = torch.from_numpy(np_gt)

    top5_score = batch_topk_accuracy(var_scores, t_gt, k=5)
    assert top5_score == 0.5
