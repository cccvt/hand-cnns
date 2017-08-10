import numpy as np
import torch
from tqdm import tqdm


def test(dataset, model, viz=None, frame_nb=10):
    """Performs average pooling on each action clip sample
    """
    sample_scores = []
    for idx in tqdm(range(len(dataset)), desc='sample'):
        imgs, class_idx = dataset.get_class_items(idx, frame_nb=frame_nb)
        outputs = []
        for img in tqdm(imgs, desc='img'):
            # Prepare vars
            img = model.prepare_var(img.unsqueeze(0))

            # Forward pass
            output = model.net(img)
            outputs.append(output.data)
        outputs = torch.cat(outputs, 0)
        mean_scores = outputs.mean(0)
        _, best_idx = mean_scores.max(0)
        if best_idx[0] == class_idx:
            sample_scores.append(1)
        else:
            sample_scores.append(0)
    assert len(sample_scores) == len(dataset)
    mean_scores = np.mean(sample_scores)
    return mean_scores
