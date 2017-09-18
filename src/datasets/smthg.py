import csv
import numpy as np
import os
import pickle
import torch.utils.data as data
from tqdm import tqdm

from src.datasets.utils import loader
from src.datasets.utils import visualize


class Smthg(data.Dataset):
    def __init__(self, root_folder="data/smthg-smthg",
                 video_transform=None, split='train',
                 clip_size=16, use_flow=False):
        """
        Args:
            split(str): train/valid/test
            video_transform : transforms to successively apply
        """
        self.video_transform = video_transform
        self.use_flow = use_flow
        self.split = split
        self.clip_size = clip_size
        self.class_nb = None
        self.untransform = None  # Needed for visualizer

        # Set paths
        self.path = root_folder
        self.video_path = os.path.join(self.path,
                                       '20bn-something-something-v1')
        self.split_video_path = os.path.join(self.path, 'split-dataset')
        self.split_flow_path = os.path.join(self.path, 'flow-farneback')
        self.label_path = os.path.join(self.path,
                                       'something-something-v1-labels.csv')
        self.train_path = os.path.join(self.path,
                                       'something-something-v1-train.csv')
        self.valid_path = os.path.join(self.path,
                                       'something-something-v1-validation.csv')
        self.test_path = os.path.join(self.path,
                                      'something-something-v1-test.csv')
        if split == 'test':
            self.split_path = self.test_path
        elif split == 'valid':
            self.split_path = self.valid_path
        elif split == 'train':
            self.split_path = self.train_path
        else:
            raise ValueError('split should be one of train/test/valid\
                but received {0}'.format(split))

        # Get split info
        self.split_ids = get_split_ids(self.split_path)
        self.label_dict = get_split_labels(self.split, self.split_path)

        # Get class info
        # sorted list of classes as template strings
        valid_label_dict = get_split_labels('valid', self.valid_path)
        self.classes = sorted(list(set(valid_label_dict.values())))
        self.class_nb = len(self.classes)
        assert self.class_nb == 174

        self.frame_template = '{frame:05d}.jpg'

        # Collect samples
        self.cache_path = os.path.join(self.path, 'cache')
        self.sample_list = self.get_samples()

    def __len__(self):
        return len(self.sample_list)

    def get_samples(self):
        """Gets list of all movie clips in current split
        This returns the samples as (film_id, label, frame_nb) tuples
        """
        pickle_name = 'all_samples_{split}.pickle'.format(split=self.split)
        all_samples_path = os.path.join(self.cache_path, pickle_name)
        if os.path.isfile(all_samples_path):
            with open(all_samples_path, 'rb') as cache_file:
                all_samples = pickle.load(cache_file)
        else:
            all_samples = []

            for film_id in tqdm(sorted(self.split_ids)):
                film_path = self.path_from_id(film_id)
                frame_nbs = [int(jpeg.split('.')[0])
                             for jpeg in os.listdir(film_path)]
                max_frames = max(frame_nbs)
                label = self.label_dict[film_id]
                all_samples.append((film_id, label, max_frames))
        with open(all_samples_path, 'wb') as cache_file:
            pickle.dump(all_samples, cache_file)
        print('Retreived {} samples for {} split in smthg dataset'.format(len(all_samples),
                                                                          self.split))
        return all_samples

    def path_from_id(self, video_id):
        """ Retrieves path to video from idx when dataset is
        split into first-digit folders
        Depending on self.use_flow returns path to rgb or flow folder
        """
        str_video_id = str(video_id)
        # Reconstruct path in format video_folder/8/890 for instance
        if self.use_flow:
            video_path = os.path.join(self.split_flow_path,
                                      str_video_id[0], str_video_id)
        else:
            video_path = os.path.join(
                self.split_video_path, str_video_id[0], str_video_id)
        return video_path

    def plot_hist(self):
        """Plots histogram of classes as sampled in self.sample_list
        """
        labels = [label for (film_id, frame_idx, label) in self.sample_list]
        visualize.plot_hist(labels)


def get_split_labels(split, split_path):
    label_dict = {}
    with open(split_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            if split != 'test':
                label_dict[int(row[0])] = row[1]
            else:
                label_dict[int(row[0])] = None
    return label_dict


def get_split_ids(split_path):
    labels = np.loadtxt(split_path, usecols=0, delimiter=';')
    labels = list(labels)
    labels = [int(label) for label in labels]
    return labels
