import numpy as np
import torch


def batch_topk_accuracy(pred, ground_truth, k=1):
    """
    Computes mean top-k accuracy for the batch
    for samples for which ground truth is provided

    Args:
        param ground_truth: tensor of same dimension as pred
            with a one hot encoding
        param pred: tensor with scores for all classes
        param k(int): number of predictions to keep
    """
    _, topk_classes = pred.topk(k)
    _, gt_classes = ground_truth.max(1)
    if k > 1:
        gt_classes_rep = gt_classes.unsqueeze(1).repeat(1, k)
    else:
        gt_classes_rep = gt_classes.unsqueeze(1)
    matches = torch.eq(topk_classes, gt_classes_rep)
    matches = matches.float()
    match_count = matches.sum(1)
    return match_count


class Metric(object):
    """
    Metric object that stores the name, func and scores associated
    to some metric

    Accumulates epoch_scores during epoch as list of
    (batch_score, batch_size) tuples
    """
    def __init__(self, name, func=None, compute=True, win=None):
        self.name = name
        self.func = func
        self.compute = compute

        self.evolution = []  # Evolution of scores over epoch
        # in format (sum_batch_scores, batch_size)
        self.epoch_scores = []  # Scores for the current epoch

        self.win = None  # visdom window for display

    def update_epoch(self):
        """
        Update evolution scores by taking mean of
        current epoch_scores
        """
        sample_score_sum = np.sum([batch_scores[0]
                                   for batch_scores in self.epoch_scores])
        sample_count = np.sum([batch_scores[1]
                               for batch_scores in self.epoch_scores])
        epoch_score = sample_score_sum / sample_count
        self.evolution.append(epoch_score)
        self.epoch_scores = []


def leave_one_out(full_list, idx=0):
    """
    Takes one list and returns the train list and the valid
    list containing just the item at idx

    Args:
        param full_list(list): original list
        param idx(int): idx of validation item
        return train_list: full_list from which the validation
            item has been removed
        return valid_list: list containing the validation item
    """
    assert idx < len(full_list), "idx of validation item should\
    be smaller then len of original list"
    valid_list = [full_list.pop(idx)]
    train_list = full_list
    return train_list, valid_list
