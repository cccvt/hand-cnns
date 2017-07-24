from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data

from src.datasets.utils import loader, filesys
from src.datasets.utils import gteaannots

"""
Action labels are composed of a vert and a set of actions such as
('pour', ['honey, bread']), they are then processed to string labels
such as 'pour honey bread'

two action labels have only 1 and 2 occurences in dataset:
('stir', ['cup']) and ('put', ['tea'])
by removing them we get to 71 action classes

Each action must occur at least twice for each subject
==> 44 action classes

Not all frames are annotated !
"""


class GTEAGazePlus(data.Dataset):
    def __init__(self, transform=None, untransform=None,
                 root_folder="data/GTEAGazePlus", no_action_label=True,
                 seqs=['Ahmad', 'Alireza', 'Carlos',
                       'Rahul', 'Shaghayegh', 'Yin']):
        """
        :param transform: transformation to apply to the images
        :param no_action_label: encode absence of action class as class
        if True, 'no action' is encoded as last column in vector
        """
        self.no_action_label = no_action_label
        # Tranform to apply to RGB image
        self.transform = transform
        # Reverse of transform for visualiztion during training
        self.untransform = untransform
        self.path = root_folder
        self.rgb_path = os.path.join(self.path, 'png')
        self.label_path = os.path.join(self.path, 'labels')
        self.all_seqs = ['Ahmad', 'Alireza', 'Carlos',
                         'Rahul', 'Yin', 'Shaghayegh']
        self.seqs = seqs

        filenames = filesys.recursive_files_dataset(self.rgb_path,
                                                    ".png", depth=2)
        filenames = [filename for filename in filenames if
                     any(seq in filename for seq in seqs)]

        self.file_paths = filenames
        self.item_nb = len(self.file_paths)

        self.in_channels = 3

        self.classes = gteaannots.get_all_classes(self.label_path,
                                                  self.inclusion_condition,
                                                  no_action_label=True)
        # Sanity check on computed class nb
        if self.no_action_label:
            self.class_nb = 963
        else:
            self.class_nb = 962
        assert len(self.classes) == self.class_nb,\
            "{0} classes found, should be {1}".format(
                len(self.classes), self.class_nb)

    def __getitem__(self, index):
        img_path = self.file_paths[index]

        # Load image
        img = loader.load_rgb_image(img_path)
        self.img = img
        if self.transform is not None:
            img = self.transform(img)

        # One hot encoding
        annot = np.zeros(self.class_nb)
        root, img_folder, seq, img_file = img_path.rsplit('/', 3)
        annot_path = os.path.join(self.label_path, seq + '.txt')
        sequence_lines = gteaannots.process_lines(annot_path,
                                                  self.inclusion_condition)
        sequence_annots = gteaannots.process_annots(sequence_lines)
        frame_idx = int(img_file.split('.')[0])
        if frame_idx in sequence_annots:
            action_class = sequence_annots[frame_idx]
            class_idx = self.classes.index(action_class)
            annot[class_idx] = 1
        else:
            if self.no_action_label:
                annot[self.class_nb - 1] = 1

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

    def inclusion_condition(self, action, objects):
        if len(objects) != 1:
            return True
        if action == 'stir' and objects[0] == 'cup':
            return False
        elif action == 'put' and objects[0] == 'tea':
            return False
        else:
            return True

    def get_classes(self, repetition_per_subj=2):
        """
        Gets the classes that are repeated at least
        repetition_per_subject times for each subject
        """
        annot_paths = [os.path.join(self.label_path, annot_file)
                       for annot_file in os.listdir(self.label_path)]
        subjects_classes = []

        # Get classes for each subject
        for subject in self.all_seqs:
            subject_annot_files = [filepath for filepath in annot_paths
                                   if subject in filepath]

            # Process files to extract action_lists
            subject_lines = [gteaannots.process_lines(subject_file)
                             for subject_file in subject_annot_files]

            # Flatten actions for each subject
            subject_labels = [label for sequence_labels in subject_lines
                              for label in sequence_labels]

            # Get classes present at least twice for the subject
            duplicate_classes = self.get_repeated_annots(subject_labels,
                                                         repetition_per_subj)

            subjects_classes.append(duplicate_classes)
        shared_classes = []
        first_subject_classes = subjects_classes.pop()
        for subject_class in first_subject_classes:
            shared = all(subject_class in subject_classes
                         for subject_classes in subjects_classes)
            if shared:
                shared_classes.append(subject_class)
        return shared_classes

    def get_repeated_annots(self, annot_lines, repetitions):
        action_labels = [(act, obj) for (act, obj, b, e) in annot_lines]
        counted_labels = defaultdict(int)
        for label in action_labels:
            counted_labels[label] += 1
        repeated_labels = [label for label, count in counted_labels.items()
                           if count >= repetitions]
        return repeated_labels
