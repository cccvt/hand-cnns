import csv
import numpy as np
import os
import torch.utils.data as data


class SomethingSomething(data.Dataset):
    def __init__(self, root_folder="data/smthg-smthg",
                 video_transform=None, split='train'):
        """
        Args:
            split(str): train/valid/test
            video_transform : transforms to successively apply
        """
        self.video_transform = video_transform
        self.split = split
        self.path = root_folder
        self.class_nb = None
        self.video_path = os.path.join(self.path,
                                       '20bn-something-something-v1')
        self.label_path = os.path.join(self.path,
                                       'something-something-v1-labels.csv')
        self.train_path = os.path.join(self.path,
                                       'something-something-v1-train.csv')
        self.valid_path = os.path.join(self.path,
                                       'something-something-v1-validation.csv')
        self.test_path = os.path.join(self.path,
                                      'something-something-v1-test.csv')
        self.all_samples = get_samples(self.video_path)
        if split == 'test':
            self.split_path = self.test_path
        elif split == 'valid':
            self.split_path = self.valid_path
        elif split == 'train':
            self.split_path = self.train_path
        else:
            raise ValueError('split should be one of train/test/valid\
                but received {0}'.format(split))
        split_ids = get_split_ids(self.split_path)


def get_samples(video_path):
    video_path, dirnames, filenames = next(os.walk(video_path))
    dirnames = [int(dirname) for dirname in dirnames]
    return dirnames


def get_split_ids(split_path):
    labels = np.loadtxt(split_path, usecols=0, delimiter=';')
    return list(labels)


def get_split_labels(split_path):
    label_dict = {}
    with open(split_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            label_dict[int(row[0])] = row[1]
    return label_dict
