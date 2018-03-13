import csv
import math
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
    model.net.eval()
    sample_scores = []
    if save_predictions:
        # Save final class
        predictions = {}
        # Save predicted scores (for future averaging)
        prediction_scores = {}
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        # Retrieve class index no mater train formatting
        time_size = sample.shape[2]
        assert sample.shape[
            0] == 1, 'Batch size should be 1 in testing got {}'.format(
                sample.shape[0])

        if time_size < opt.clip_size:
            # Loop clip
            repetitions = math.ceil(opt.clip_size / time_size)
            sample = torch.cat([sample] * repetitions, dim=2)
            sample = sample[:, :, 0:opt.clip_size]

        if opt.mode == 'full':
            sample = sample

        # Extract clips with a fixed stride between them
        elif opt.mode == 'stride':
            max_samples = 44
            sample = sample[0]  # Remove batch dimension
            time_stride = opt.mode_param
            if time_size < opt.clip_size + time_stride:
                subsamples = [sample]
            else:
                subsample_idxs = list(
                    range(0, time_size - opt.clip_size, time_stride))
                subsamples = [
                    sample[:, start_idx:start_idx + opt.clip_size]
                    for start_idx in subsample_idxs
                ]
                subsamples = subsamples[:max_samples]
        elif opt.mode == 'subsample':
            sample = sample[0]  # Remove batch dimension
            if time_size < opt.clip_size:
                subsamples = [sample]
            else:
                subsample_idxs = np.linspace(0, time_size - opt.clip_size,
                                             opt.mode_param)
                subsample_idxs = [
                    int(frame_idx) for frame_idx in subsample_idxs
                ]
                subsamples = [
                    sample[:, start_idx:start_idx + opt.clip_size]
                    for start_idx in subsample_idxs
                ]

        # Extract a fixed number of clips uniformly sampled in video
        if opt.mode == 'subsample' or opt.mode == 'stride':
            print('Got {} subsamples in mode {}'.format(
                len(subsamples), opt.mode))
            sample = torch.stack(subsamples)

        class_idx = dataloader.dataset.dataset.get_class_idx(idx)
        # Prepare vars
        imgs_var = model.prepare_var(sample, volatile=True)
        # Forward pass
        outputs = model.net(imgs_var)
        # Remove batch dimension
        # outputs = torch.nn.functional.softmax(outputs, dim=0)

        # When multiplt classifier use the first for evaluation
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        outputs = outputs.mean(0)
        if outputs.dim() == 3:
            outputs = outputs.mean(2)
        if outputs.dim() == 2:
            outputs = outputs.mean(1)
        outputs = outputs.data

        _, best_idx = outputs.max(0)
        if best_idx[0] == class_idx:
            sample_scores.append(1)
        else:
            sample_scores.append(0)
        if save_predictions:
            if opt.dataset == "smthg":
                sample_idx, _, _, _ = dataloader.dataset.dataset.all_samples[
                    idx]
                predictions[sample_idx] = dataloader.dataset.dataset.classes[
                    best_idx[0]]
                prediction_scores[sample_idx] = outputs
            elif opt.dataset == "gteagazeplus" or opt.dataset == 'gteagazeplus_tres':
                prediction_scores[idx] = outputs
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
