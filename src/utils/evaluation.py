import numpy as np
import torch


def batch_topk_accuracy(pred, ground_truth, k=1):
    """
    Computes mean top-k accuracy for the batch
    for samples for which ground truth is provided

    :param ground_truth: tensor of same dimension as pred
    with a one hot encoding
    :param pred: tensor with scores for all classes
    :param k: number of predictions to keep
    """
    pred_val, topk_classes = pred.topk(k)
    gt_vals, gt_classes = ground_truth.max(1)
    gt_classes_rep = gt_classes.repeat(1, k)
    matches = torch.eq(topk_classes, gt_classes_rep)
    acc = matches.float().sum(1).mean()
    return acc


class Metric(object):
    def __init__(self, name, func=None, compute=True, win=None):
        self.name = name
        self.func = func
        self.compute = compute

        self.evolution = []  # Evolution of scores over epoch
        self.epoch_scores = []  # Scores for the current epoch

        self.win = None  # visdom window for display

    def update_epoch(self):
        """
        Update evolution scores by taking mean of
        current epoch_scores
        """
        epoch_score = np.mean(self.epoch_scores)
        self.evolution.append(epoch_score)
        self.epoch_scores = []


def leave_one_out(full_list, idx=0):
    """
    Takes one list and returns the train list and the valid
    list containing just the item at idx

    :param full_list: original list
    :param idx: idx of validation item
    :return train_list: full_list from which the validation
    item has been removed
    :return valid_list: list containing the validation item
    """
    assert idx < len(full_list), "idx of validation item should\
    be smaller then len of original list"
    valid_list = [full_list.pop(idx)]
    train_list = full_list
    return train_list, valid_list
