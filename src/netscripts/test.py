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


def test(dataloader,
         model,
         viz=None,
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
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        class_idx = class_idx[0]
        # imgs, class_idx = dataset.get_full_clip(idx)
        # Prepare vars
        imgs_var = model.prepare_var(sample)
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
            if opt.dataset == "smthg":
                sample_idx, _, _, _ = dataset.dataset.all_samples[idx]
                predictions[sample_idx] = dataset.classes[best_idx[0]]
                prediction_scores[sample_idx] = mean_scores
            elif opt.dataset == "gteagazeplus":
                prediction_scores[idx] = mean_scores
            else:
                raise ValueError(
                    'dataset {} not recognized'.format(opt.dataset))

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
        result_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                   'result.txt')
        message = 'mean accuracy :{}'.format(mean_scores)
        with open(result_path, "a") as file:
            file.write(message + '\n')

    return mean_scores
