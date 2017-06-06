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
        self.links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                      (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
                      (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
                      (18, 19), (19, 20), (20, 21), (21, 22), (22, 23),
                      (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
                      (35, 5), (35, 11), (35, 17), (35, 23), (35, 29)]

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
        annot = self.annot_uvd[view, sequence]

        return image, annot

    def draw2d(self, idx):
        img, annot = self[idx]
        visualize.draw2d_annotated_img(img, annot, self.links)

    def draw3d(self, idx, xyz=True, angle=320):
        """
        draw 2D rgb image with displayed annotations
        :param idx: idx of the item in the dataset
        :param xyz: True for xyz, False for uvd coordinates
        :param angle: angle in [0:360] for the rotation of the 3d plot
        """
        img, annot = self[idx]
        sequence, view = divmod(idx, self.view_nb)
        if(xyz):
            annot = self.annot_xyz[view, sequence]
        else:
            annot = self.annot_uvd[view, sequence]

        visualize.draw3d_annotated_img(annot, self.links, angle=320)

