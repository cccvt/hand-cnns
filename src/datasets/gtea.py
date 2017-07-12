import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch.utils.data as data

from src.datasets.utils import loader, visualize, filesys

"""
Grasp UNderstanding Dataset
Note that rgb and depth images are not aligned in this dataset
"""


class GTEA(data.Dataset):
    def __init__(self, transform=None, root_folder="../data/"):
        """
        :param transform: transformation to apply to the images
        """
        self.transform = transform
        self.path = root_folder
        self.class_nb = 71  # action classes

        filenames = filesys.recursive_files_dataset(self.path, ".png", depth=3)
        self.file_paths = filenames
        self.item_nb = len(self.file_paths)

    def __getitem__(self, index):
        img_path = self.file_paths[index]

        # Load image
        img = loader.load_rgb_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # One hot encoding
        annot = np.zeros(self.class_nb)

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


def process_annots(annot_path):
    """
    Returns a dictionnary with frame as key and
    value (action, [object1, object2, ...]) from
    the gtea annotation text file
    """
    with open(annot_path) as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        matches = re.search('<(.*)><(.*)> \((.*)-(.*)\)', line)
        if matches:
            action_label, object_label = matches.group(1), matches.group(2)
            begin, end = int(matches.group(3)), int(matches.group(4))
            object_labels = object_label.split(',')
            processed_lines.append((action_label, object_labels, begin, end))

    # create annotation_dict
    annot_dict = {}
    for action, object_label, begin, end in processed_lines:
        for frame in range(begin, end + 1):
            annot_dict[frame] = (action, object_label)
    return annot_dict
