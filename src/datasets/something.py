import csv
import numpy as np
import os
import torch.utils.data as data

from src.datasets.utils import visualize


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
        self.split_ids = get_split_ids(self.split_path)
        self.label_dict = get_split_labels(self.split_path)
        self.sample_list = self.get_dense_samples()

    def get_dense_samples(self, frame_nb=16, clip_stride=1):
        """Gets list of all movie clips by extracting all clips with first
        frames separated by clip_stride
        This returns the samples as (film_id, frame_idx, label) tuples
        where frame_idx is the idx of the first frame, and label is the
        class label
        """
        samples = []
        for film_id in self.split_ids:
            film_path = os.path.join(self.video_path, str(int(film_id)))
            frame_nbs = [int(jpeg.split('.')[0])
                         for jpeg in os.listdir(film_path)]
            max_frames = max(frame_nbs)
            for frame_idx in range(0, max_frames - frame_nb + 1, clip_stride):
                samples.append((film_id, frame_idx,
                                self.label_dict[film_id]))
        return samples

    def plot_hist(self):
        """Plots histogram of classes as sampled in self.sample_list
        """
        labels = [label for (film_id, frame_idx, label) in self.sample_list]
        visualize.plot_hist(labels)


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
