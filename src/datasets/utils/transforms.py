import random
import numpy as np
import PIL
import torch
from skimage.transform import resize


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
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, c = clip[0].shape
            assert c == 3, 'should receive 3 channels, got {0}'.format(c)
        if isinstance(clip[0], PIL.Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([3, len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            if isinstance(img, PIL.Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = self.convert_img(img)
            np_clip[:, img_idx, :, :] = img[:,:,::-1]
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip.float().div(255)

    def convert_img(self, img):
        """Converts (H, W, C) numpy.ndarray to (C, W, H) format
        """
        img = img.transpose(2, 0, 1)
        return img


class Scale(object):
    """Scales a list of (H x W x C) numpy.ndarray to the final size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            scaled = [resize(img, self.size) for img in clip]
        if isinstance(clip[0], PIL.Image.Image):
            scaled = [img.resize(self.size) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        return scaled


class RandomCrop(object):
    """Extract random crop at the same location for a list of images

    Args:
        size (sequence or int): Desired output size for the
            crop in format (h, w)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        """
        Args:
            img (PIL.Image or numpy.ndarray): List of images to be cropped
                in format (h, w, c) in numpy.ndarray

        Returns:
            PIL.Image or numpy.ndarray: Cropped list of images
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        if w < im_w or h < im_h:
            raise(ValueError('Initial image size should be larger then cropped size\
                but got cropped sizes : ({w}, {h})\
                while initial image is ({im_w}, {im_h})'.format(im_w=im_w,
                                                                im_h=im_h,
                                                                w=w, h=h)))

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        if isinstance(clip[0], np.ndarray):
            cropped = [img[y1:y1 + h, x1: x1 + w, :] for img in clip]
        if isinstance(clip[0], PIL.Image.Image):
            cropped = [img.crop((x1, y1, x1 + w, y1 + h)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))
        return cropped
