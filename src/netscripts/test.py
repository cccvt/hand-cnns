import csv
import numpy as np
import os
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


def save_preds(predictions, prediction_file):
    """
    Args:
        predictions(dict): in format {12234: "Hold something", ...}
    """
    with open(prediction_file, 'w', newline='') as save_file:
        writer = csv.writer(save_file, delimiter=';')
        writer.writerows(predictions.items())


def test(dataset, model, viz=None, frame_nb=4, opt=None, save_predictions=False):
    """Performs average pooling on each action clip sample
    """
    sample_scores = []
    if save_predictions:
        predictions = {}
    for idx in tqdm(range(len(dataset)), desc='sample'):
        imgs, class_idx = dataset.get_class_items(idx, frame_nb=frame_nb)
        batches = [imgs[beg:beg + opt.batch_size]
                   for beg in range(0, len(imgs), opt.batch_size)]
        outputs = []
        for batch in batches:
            batch = default_collate(imgs)
            outputs = []
            # Prepare vars

            batch_var = model.prepare_var(batch)

            # Forward pass
            output = model.net(batch_var)
            outputs.append(output.data)
        outputs = torch.cat(outputs)
        mean_scores = outputs.mean(0)
        _, best_idx = mean_scores.max(0)
        if best_idx[0] == class_idx:
            sample_scores.append(1)
        else:
            sample_scores.append(0)
        if save_predictions:
            sample_idx, _, _ = dataset.sample_list[idx]
            predictions[sample_idx] = dataset.classes[best_idx[0]]

    if save_predictions:
        save_dir = os.path.join(
            opt.checkpoint_dir, opt.exp_id, 'predictions.csv')
        save_preds(predictions, save_dir)

    assert len(sample_scores) == len(dataset)
    mean_scores = np.mean(sample_scores)

    return mean_scores
