import torch


def batch_topk_accuracy(pred, ground_truth, k=1):
    """
    Computes mean top-k accuracy for the batch

    :param k: number of predictions to keep
    """
    pred_val, topk_classes = pred.topk(k)
    gt_vals, gt_classes = ground_truth.max(1)
    gt_classes_rep = gt_classes.repeat(1, k)
    matches = torch.eq(topk_classes.data, gt_classes_rep)
    acc = matches.float().sum(1).mean()
    return acc
