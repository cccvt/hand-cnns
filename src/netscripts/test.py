import csv
import os
import pickle

import numpy as np
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


def test(dataset,
         model,
         viz=None,
         frame_nb=4,
         opt=None,
         save_predictions=False,
         smthg=True):
    """Performs average pooling on each action clip sample
    """
    sample_scores = []
    if save_predictions:
        # Save final class
        predictions = {}
        # Save predicted scores (for future averaging)
        prediction_scores = {}
    for idx in tqdm(range(len(dataset)), desc='sample'):
        if opt.frame_nb > 0:
            # Get uniformly spaced clips
            imgs, class_idx = dataset.get_class_items(idx, frame_nb=frame_nb)
            # Separate them to batches
            batches = [
                imgs[beg:beg + opt.batch_size]
                for beg in range(0, len(imgs), opt.batch_size)
            ]
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
        else:
            # Take full size clip (with varying size)
            imgs, class_idx = dataset.get_full_sample(idx)
            imgs = imgs.unsqueeze(0)
            # Prepare vars
            imgs_var = model.prepare_var(imgs)
            # Forward pass
            outputs = model.net(imgs_var)
            outputs = outputs.data

        mean_scores = outputs.mean(0)
        _, best_idx = mean_scores.max(0)
        if best_idx[0] == class_idx:
            sample_scores.append(1)
        else:
            sample_scores.append(0)
        if save_predictions:
            if opt.dataset == "smthgsmthg":
                sample_idx, _, _ = dataset.sample_list[idx]
                predictions[sample_idx] = dataset.classes[best_idx[0]]
                prediction_scores[sample_idx] = mean_scores
            elif opt.dataset == "gteagazeplus":
                prediction_scores[idx] = mean_scores
            else:
                raise ValueError(
                    'dataset {} not recognized'.format(opt.dataset))

    assert len(sample_scores) == len(dataset)
    mean_scores = np.mean(sample_scores)
    print('mean score : {}'.format(mean_scores))

    if save_predictions:
        save_dir = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                'predictions.csv')
        save_preds(predictions, save_dir)
        save_scores_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                        'prediction_scores.pickle')
        with open(save_scores_path, 'wb') as score_file:
            pickle.dump(prediction_scores, score_file)

    return mean_scores
