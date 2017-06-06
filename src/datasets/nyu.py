from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import torch.utils.data as data

import src.datasets.utils.visualize as visualize

"""
NYU hands is the standard dataset for evaluation
of 3D hand pose estimation from depth data
! RGB data is registered (No rgb values for absent depths)
"""

class NYU(data.Dataset):
    def __init__(self, transform=None, root_folder="../data/NYU",
                 train=True):
        """
        :param train: True to load training samples, false for testing
        :type train: Boolean
        """
        self.transform = transform
        self.train = train

        # set path to data folder
        if (self.train):
            self.path = root_folder + '/train'
        else:
            self.path = root_folder + '/test'
            
        annots = scipy.io.loadmat(self.path + '/joint_data.mat')
        self.annot_uvd = annots['joint_uvd']
        self.annot_xyz = annots['joint_xyz']
