from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import os
import torch.utils.data as data

from src.datasets.utils import loader
from src.datasets.utils import gteaannots
from src.utils.debug import timeit

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
                 root_folder="data/GTEAGazePlus", video_transform=None,
                 no_action_label=False,
                 seqs=['Ahmad', 'Alireza', 'Carlos',
                       'Rahul', 'Shaghayegh', 'Yin'],
                 clip_size=16, use_video=False):
        """
        :param transform: transformation to apply to the images
        :param use_video: whether to use video inputs or png inputs
        """
        self.no_action_label = no_action_label
        # Tranform to apply to RGB image
        self.video_transform = video_transform
        # Reverse of transform for visualiztion during training
        self.untransform = untransform
        self.path = root_folder
        self.use_video = use_video
        self.rgb_path = os.path.join(self.path, 'png')
        self.video_path = os.path.join(self.path, 'avi_files')
        self.label_path = os.path.join(self.path, 'labels')
        self.all_seqs = ['Ahmad', 'Alireza', 'Carlos',
                         'Rahul', 'Yin', 'Shaghayegh']

        self.seqs = seqs
        self.clip_size = clip_size

        # Compute classes
        self.classes = self.get_classes(2)
        self.action_clips = self.get_dense_actions(clip_size, self.classes)

        # Sanity check on computed class nb
        if self.no_action_label:
            self.class_nb = 33
        else:
            self.class_nb = 32
        assert len(self.classes) == self.class_nb,\
            "{0} classes found, should be {1}".format(
                len(self.classes), self.class_nb)

    def __getitem__(self, index):
        # Load clip
        subject, recipe, action, objects, frame_idx = self.action_clips[index]
        sequence_name = subject + '_' + recipe

        clip = self.get_clip(sequence_name, frame_idx, self.clip_size)

        # Apply video transform
        if self.video_transform is not None:
            clip = self.video_transform(clip)

        # One hot encoding
        annot = np.zeros(self.class_nb)
        class_idx = self.classes.index((action, objects))
        annot[class_idx] = 1
        return clip, annot

    def __len__(self):
        return self.item_nb

    def get_clip(self, sequence_name, frame_begin, frame_nb):
        if self.use_video:
            video_path = os.path.join(self.video_path,
                                      sequence_name + '.avi')
            video_capture = loader.get_video_capture(video_path)
            clip = loader.get_clip(video_capture, frame_begin, frame_nb)
        else:
            png_path = os.path.join(self.rgb_path,
                                    sequence_name)
            clip = loader.get_stacked_frames(png_path, frame_begin, frame_nb)

        return clip

    def inclusion_condition(self, action, objects):
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
        """
        Given list of annotations in format [action, objects, begin, end]
        returns list of (action, objects) tuples for the (action, objects)
        that appear at least repetitions time
        """
        action_labels = [(act, obj) for (act, obj, b, e) in annot_lines]
        counted_labels = defaultdict(int)
        for label in action_labels:
            counted_labels[label] += 1
        repeated_labels = [label for label, count in counted_labels.items()
                           if count >= repetitions]
        return repeated_labels

    def get_dense_actions(self, frame_nb, action_object_classes):
        """
        Gets dense list of all action movie clips by extracting
        all possible tuples (subject, recipe, action, objects, begin_frame)
        with all begin_frame so that at least frame_nb frames belong
        to the given action

        for frame_nb: 2 and begin: 10, end:17, the candidate begin_frames are:
        10, 11, 12, 13, 14, 15
        This guarantees that we can extract frame_nb of frames starting at
        begin_frame and still be inside the action

        This gives all possible action blocks for the subjects in self.seqs
        """
        annot_paths = [os.path.join(self.label_path, annot_file)
                       for annot_file in os.listdir(self.label_path)]
        actions = []

        # Get classes for each subject
        for subject in self.seqs:
            subject_annot_files = [filepath for filepath in annot_paths
                                   if subject in filepath]
            for annot_file in subject_annot_files:
                recipe = re.search('.*_(.*).txt', annot_file).group(1)
                action_lines = gteaannots.process_lines(annot_file)
                for action, objects, begin, end in action_lines:
                    if (action, objects) in action_object_classes:
                        for frame_idx in range(begin, end - frame_nb + 1):
                            actions.append((subject, recipe, action,
                                            objects, frame_idx))
        return actions
