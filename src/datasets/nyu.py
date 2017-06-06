from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import torch.utils.data as data

from src.datasets.utils import visualize, loader

"""
NYU hands is the standard dataset for evaluation
of 3D hand pose estimation from depth data
! RGB data is registered (No rgb values for absent depths)

Each scene (person position) was filmed using 3 camerasi (3 views)
Therefore there are 3*scene_nb images in the train and test folders
"""


class NYU(data.Dataset):
    def __init__(self, transform=None, root_folder="../data/NYU",
                 train=True, depth=True):
        """
        :param train: True to load training samples, false for testing
        :type train: Boolean
        :param depth: True to load depth data
        False for registered rgb
        :type depth: Boolean
        """
        self.transform = transform
        self.train = train
        self.depth = depth

        self.view_nb = 3
        self.links = None

        # set path to data folder
        if (self.train):
            self.path = root_folder + '/train'
            self.scene_nb = 72757
        else:
            self.path = root_folder + '/test'
            self.scene_nb = 8252

        annots = scipy.io.loadmat(self.path + '/joint_data.mat')
        self.annot_uvd = annots['joint_uvd']
        self.annot_xyz = annots['joint_xyz']

    def __getitem__(self, index):
        """
        items are ordered by scene index
        with views kept consecutive and ordered

        :param index: index of the image to load in full data indexing
        should be in range 0, sequence*view_nb
        """
        sequence, view = divmod(index, self.view_nb)
        if self.depth:
            prefix = 'depth'
        else:
            prefix = 'rgb'

        # Get image
        filename = '{pre}_{view}_{seq:07d}.png'.format(pre=prefix,
                                                       seq=sequence + 1,
                                                       view=view + 1)
        image_path = self.path + '/' + filename
        image = loader.load_depth_image(image_path)

        # Get matching annotations
        print(view)
        print(sequence)
        annot = self.annot_uvd[view, sequence]

        return image, annot

    def draw2d(self, idx):
        img, annot = self[idx]
        visualize.draw2d_annotated_img(img, annot, self.links)
