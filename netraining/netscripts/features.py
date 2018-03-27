import csv
import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


def get_features(dataloader, model, viz=None, opt=None, smthg=True):
    """Performs average pooling on each action clip sample
    """
    model.net.eval()
    all_features = []
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        # Retrieve class index no mater train formatting
        time_size = sample.shape[2]
        sample_info = dataloader.dataset.dataset.all_samples[idx]
        assert sample.shape[
            0] == 1, 'Batch size should be 1 in testing got {}'.format(
                sample.shape[0])

        if time_size < opt.clip_size:
            # Loop clip
            repetitions = math.ceil(opt.clip_size / time_size)
            sample = torch.cat([sample] * repetitions, dim=2)
            sample = sample[:, :, 0:opt.clip_size]

        sample = sample

        # Prepare vars
        imgs_var = model.prepare_var(sample, volatile=True)
        # Forward pass
        outputs = model.net(imgs_var, early_stop='5c')
        # Remove spatial dimension
        outputs = outputs.mean(3).mean(3)
        all_features.append((sample_info, outputs.data.cpu().numpy()))

    save_path = os.path.join(opt.checkpoint_dir, opt.exp_id, 'features.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_features, f)
    print('Saved features to {}!'.format(save_path))
