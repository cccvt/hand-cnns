import numpy as np
import torch


class Compose(object):
    """Composes several transforms

    Args:
        transforms (list of ``Transform`` objects): list of transforms
        to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class ToTensor(object):
    """Convert a (H x W x C) numpy.ndarray in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0, 1.0]
    """

    def __call__(self, clip):
        """
        Args: clip (list of numpy.ndarray): clip (list of images)
        to be converted to tensor.
        """
        h, w, c = clip[0].shape
        np_clip = np.zeros([3, len(clip), int(h), int(w)])
        for img_idx, img in enumerate(clip):
            img = self.convert_img(img)
            np_clip[:, img_idx, :, :] = img
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip.float().div(255)

    def convert_img(self, img):
        if isinstance(img, np.ndarray):
            img = img.transpose(2, 0, 1)
            return img
        else:
            raise TypeError('Expected numpy.ndarray, got {0}'.format(
                type(img)))
