from PIL import Image
import sys
import torch.utils.data as data

def load_image(path):
    """
    loads image from path
    :param path: absolute or relative path to file
    :rtype: PIL Image
    :return: RGB Image
    """
    image = Image.open(path)
    return image.convert("RGB")


def get_input_target(sequence_nb, image_nb):
    pass

def get_tensors():
    pass

class UCIEGO(data.Dataset):
    def __init__(self, ego_path="../data/UCI-EGO", sequences = [1, 2, 3, 4]):
        """
        :param sequences: indexes of the sequences to load in dataset
        :type sequences: list of integers among 1 to 4
        """
        self.path = ego_path
        for i in sequences:
            seq_path = ego_path + "/Seq" + str(i)
        


    def __getitem__(self, index):
        raw_img = load_image()
