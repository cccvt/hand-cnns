from PIL import Image
import numpy as np
import os
import torch.utils.data as data

"""
UCI ego contains almost 400 annotated frames
in the .txt files a -1 suffix indicates the right hand while -2 indicates the left hand
"""

def load_image(path):
    """
    loads image from path
    :param path: absolute or relative path to file
    :rtype: PIL Image
    :return: RGB Image
    """
    image = Image.open(path)
    return image.convert("RGB")


def load_annotation(path):
    """
    loads keypoint annotations from text file at path
    :param path: absolute or relative path to .txt file containing annotations
    :return numpy.ndarray:
    """
    return np.loadtxt(path, usecols=[2, 3, 4])



def get_input_target(sequence_nb, image_nb):
    pass


def get_tensors():
    pass


class UCIEGO(data.Dataset):
    def __init__(self, ego_path="../data/UCI-EGO", sequences = [1, 2, 3, 4],
                 rgb=True, depth=False):
        """
        :param sequences: indexes of the sequences to load in dataset
        :param rgb: whether rgb channels should be used
        :type rgb: Boolean
        :param depth: whether depth should be used
        :type depth: Boolean
        :type sequences: list of integers among 1 to 4
        """
        self.path = ego_path
        self.rgb = rgb
        self.depth = depth
        # self.all_images contains tuples (sequence_idx, image_name)
        # where image_name is the common prefix of the files
        # ('fr187' for instance)
        self.all_images = []
        for seq in sequences:
            seq_path = ego_path + "/Seq" + str(seq)
            files = os.listdir(seq_path)
            # Remove depth files
            files = [filename for filename in files if "_z" not in filename]

            # separate txt and image files
            jpgs = [filename for filename in files if filename.endswith(".jpg")]
            annotations = [filename for filename in files if filename.endswith(".txt")]

            # Get radical of the image
            file_names = [jpg_file.split(".")[0] for jpg_file in jpgs]

            # Keep only files with annotations
            # TODO for now we only consider frames where the right hand is present
            # we also ignore the left hand, this could be improved in the future
            annotated = [filename for filename in file_names if filename + "-1.txt" in annotations]
            seq_images = [(seq, file_name) for file_name in file_names]
            self.all_images = self.all_images + seq_images
        self.item_nb = len(self.all_images)

    def __getitem__(self, index):
        if(self.rgb):
            seq, image_name = self.all_images[index]
            seq_path =  self.path + "/Seq"+ str(seq) + "/"
            image_path = seq_path + image_name + '.jpg'
            # TODO add handling for left hand
            rgb_img = load_image(image_path)
            annot_path = seq_path + image_name + '-1.txt'
            annot = load_annotation(annot_path)
            return rgb_img, annot

    def __len__(self):
        return self.item_nb



