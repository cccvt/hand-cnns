import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data

from src.datasets.utils import visualize, loader

"""
UCI ego contains almost 400 annotated frames
in the .txt files a -1 suffix indicates the
right hand while -2 indicates the left hand
"""

def _load_annotation(path):
    """
    loads keypoint annotations from text file at path
    :param path: absolute or relative path to .txt file containing annotations
    :param exclude_joints: list of indexes of the rows to delete
    :return numpy.ndarray:
    """
    annots = np.loadtxt(path, usecols=[2, 3, 4])
    return annots

class UCIEGO(data.Dataset):
    def __init__(self, transform=None, root_folder="data/UCI-EGO",
                 sequences=[1, 2, 3, 4], rgb=True, depth=False):
        """
        :param sequences: indexes of the sequences to load in dataset
        :param rgb: whether rgb channels should be used
        :type rgb: Boolean
        :param depth: whether depth should be used
        :type depth: Boolean
        :type sequences: list of integers among 1 to 4
        """
        self.transform = transform
        self.path = root_folder
        self.rgb = rgb
        self.depth = depth
        self.links = [(5, 6), (6, 7), (7, 8),
                      (9, 10), (10, 11), (11, 12),
                      (13, 14), (14, 15), (15, 16),
                      (17, 18), (18, 19), (19, 20),
                      (21, 22), (22, 23), (23, 24),
                      (25, 21), (25, 17), (25, 13), (25, 9), (25, 5),
                      (4, 25)]
        self.joint_nb = 26

        # self.all_images contains tuples (sequence_idx, image_name)
        # where image_name is the common prefix of the files
        # ('fr187' for instance)
        self.all_images = []
        for seq in sequences:
            seq_path = root_folder + "/Seq" + str(seq)
            files = os.listdir(seq_path)
            # Remove depth files
            files = [filename for filename in files if "_z" not in filename]

            # separate txt and image files
            jpgs = [
                filename for filename in files if filename.endswith(".jpg")]
            annotations = [
                filename for filename in files if filename.endswith(".txt")]

            # Get radical of the image
            file_names = [jpg_file.split(".")[0] for jpg_file in jpgs]

            # Keep only files with annotations
            # TODO for now we only consider frames
            # where the right hand is present
            # we also ignore the left hand
            # this could be improved in the future
            annotated = [
                filename for filename in file_names if filename + "-1.txt" in annotations]
            seq_images = [(seq, file_name) for file_name in annotated]
            self.all_images = self.all_images + seq_images
        self.item_nb = len(self.all_images)

    def __getitem__(self, index):
        if(self.rgb):
            seq, image_name = self.all_images[index]
            seq_path = self.path + "/Seq" + str(seq) + "/"
            image_path = seq_path + image_name + '.jpg'
            # TODO add handling for left hand
            img = loader.load_rgb_image(image_path)
            annot_path = seq_path + image_name + '-1.txt'
            annot = _load_annotation(annot_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, annot

    def __len__(self):
        return self.item_nb

    def draw2d(self, idx):
        """
        draw 2D rgb image with displayed annotations
        :param idx: idx of the item in the dataset
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img, annot = self[idx]
        ax.imshow(img)
        plt.scatter(annot[:, 0], annot[:, 1])
        visualize.draw2djoints(ax, annot, self.links)

    def draw3d(self, idx, angle=320):
        """
        draw 2D rgb image with displayed annotations
        :param idx: idx of the item in the dataset
        :param angle: angle in [0:360] for the rotation of the 3d plot
        """
        img, annot = self[idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, angle)
        visualize.draw3djoints(ax, annot, self.links)
