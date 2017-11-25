import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data

from src.datasets.utils import loader, visualize, filesys
"""
Grasp UNderstanding Dataset
Note that rgb and depth images are not aligned in this dataset
(they cannot be superposed perfectly)

The taxonomy can be found here : http://www.cs.cmu.edu/~jialiu1/database.html
"""


class GUN(data.Dataset):
    def __init__(self,
                 transform=None,
                 untransform=None,
                 root_folder="data/gun",
                 rgb=True,
                 seqs=[
                     'Subject1', 'Subject2', 'Subject3', 'Subject4',
                     'Subject5', 'Subject6', 'Subject7', 'Subject8'
                 ]):
        """
        Args:
            rgb (bool) : whether rgb channels should be used
        """
        self.transform = transform
        self.untransform = untransform
        self.path = root_folder
        self.rgb = rgb
        self.class_max_idx = 73
        self.missing_classes = [50, 61]
        self.class_nb = self.class_max_idx - len(self.missing_classes)
        self.classes = ['grasp' + str(i) for i in range(self.class_nb)]

        if rgb:
            filenames = filesys.recursive_files_dataset(self.path, '.jpg', 3)
        else:
            filenames = filesys.recursive_files_dataset(self.path, '.png', 3)

        # Filter sequences out
        filenames = [
            filename for filename in filenames
            if any(seq in filename for seq in seqs)
        ]
        self.image_paths = filenames
        self.item_nb = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        # Load image
        if (self.rgb):
            rgb_img = loader.load_rgb_image(image_path)
            img = rgb_img
        else:
            depth_img = loader.load_depth_image(image_path)
            img = depth_img
        if self.transform is not None:
            img = self.transform(img)

        # Extract grasp index from filename
        prefix = image_path.split("/")[-1]
        grasp_id = int(prefix[1:3])

        # One hot encoding
        annot = np.zeros(self.class_max_idx)
        annot[grasp_id - 1] = 1
        annot = np.delete(annot, self.missing_classes)

        return img, annot

    def __len__(self):
        return self.item_nb

    def draw2d(self, idx):
        """
        draw 2D rgb image with displayed annotations
        :param idx: idx of the item in the dataset
        """
        img, annot = self[idx]
        plt.imshow(img)
        plt.axis('off')
