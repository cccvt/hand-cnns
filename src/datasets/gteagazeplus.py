from collections import defaultdict
import numpy as np
import re
import os
import torch.utils.data as data

from src.datasets.utils import loader
from src.datasets.utils import gteaannots
from src.datasets.utils import visualize

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
    def __init__(self, root_folder="data/GTEAGazePlus", video_transform=None,
                 no_action_label=False, original_labels=True,
                 seqs=['Ahmad', 'Alireza', 'Carlos',
                       'Rahul', 'Shaghayegh', 'Yin'],
                 clip_size=16, use_video=False):
        """
        Args:
            video_transform: transformation to apply to the clips
            use_video (bool): whether to use video inputs or png inputs
        """

        self.cvpr_labels = ['open_fridge', 'close_fridge',
                            'put_cupPlateBowl',
                            'put_spoonForkKnife_cupPlateBowl',
                            'take_spoonForkKnife_cupPlateBowl',
                            'take_cupPlateBowl', 'take_spoonForkKnife',
                            'put_lettuce_cupPlateBowl',
                            'read_recipe', 'take_plastic_spatula',
                            'open_freezer', 'close_freezer',
                            'put_plastic_spatula',
                            'cut_tomato_spoonForkKnife',
                            'put_spoonForkKnife',
                            'take_tomato_cupPlateBowl',
                            'turnon_tap', 'turnoff_tap',
                            'take_cupPlateBowl_plate_container',
                            'turnoff_burner', 'turnon_burner',
                            'cut_pepper_spoonForkKnife',
                            'put_tomato_cupPlateBowl',
                            'put_milk_container', 'put_oil_container',
                            'take_oil_container', 'close_oil_container',
                            'open_oil_container', 'take_lettuce_container',
                            'take_milk_container', 'open_fridge_drawer',
                            'put_lettuce_container', 'close_fridge_drawer',
                            'compress_sandwich',
                            'pour_oil_oil_container_skillet',
                            'take_bread_bread_container',
                            'cut_mushroom_spoonForkKnife',
                            'put_bread_cupPlateBowl', 'put_honey_container',
                            'take_honey_container', 'open_microwave',
                            'crack_egg_cupPlateBowl',
                            'open_bread_container', 'open_honey_container']
        # Label tags
        self.no_action_label = no_action_label
        self.original_labels = original_labels

        # Tranform to apply to video
        self.video_transform = video_transform

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
        if self.original_labels:
            self.classes = self.get_cvpr_classes()
        else:
            self.classes = self.get_repeat_classes(2)
        self.action_clips = self.get_dense_actions(clip_size, self.classes)

        # Sanity check on computed class nb
        if self.original_labels:
            self.class_nb = len(self.cvpr_labels)
        else:
            self.class_nb = 32

        if self.no_action_label:
            self.class_nb += 1

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
        return len(self.action_clips)

    def get_clip(self, sequence_name, frame_begin, frame_nb):
        if self.use_video:
            video_path = os.path.join(self.video_path,
                                      sequence_name + '.avi')
            video_capture = loader.get_video_capture(video_path)
            clip = loader.get_clip(video_capture, frame_begin, frame_nb)
        else:
            png_path = os.path.join(self.rgb_path,
                                    sequence_name)
            clip = loader.get_stacked_frames(png_path, frame_begin, frame_nb,
                                             use_open_cv=False)

        return clip

    def get_repeat_classes(self, repetition_per_subj=2, seqs=None):
        """
        Gets the classes that are repeated at least
        repetition_per_subject times for each subject
        """
        subjects_classes = self._get_subj_classes(seqs=seqs)
        repeated_subjects_classes = []
        for subj_labels in subjects_classes:
            subj_classes = self.get_repeated_annots(subj_labels,
                                                    repetition_per_subj)

            repeated_subjects_classes.append(subj_classes)
        shared_classes = []
        # Get classes present at least twice for the subject
        first_subject_classes = repeated_subjects_classes.pop()
        for subject_class in first_subject_classes:
            shared = all(subject_class in subject_classes
                         for subject_classes in repeated_subjects_classes)
            if shared:
                shared_classes.append(subject_class)
        return shared_classes

    def get_class_str(self, action, objects):
        """Transforms action and objects inputs
        """
        if self.original_labels:
            objects = _original_label_transform(objects)
        action_str = '_'.join((action.replace(' ', ''),
                               '_'.join(objects)))
        return action_str

    def get_cvpr_classes(self, seqs=None):
        """Gets original cvpr classes as list of classes as
        list of strings
        """
        subjects_classes = self._get_subj_classes(seqs=seqs)
        all_classes = []
        for subj_labels in subjects_classes:
            # Remove internal spaces
            for (action, obj, b, e) in subj_labels:
                action_str = self.get_class_str(action, obj)
                if action_str in self.cvpr_labels:
                    all_classes.append((action,
                                        _original_label_transform(obj)))
        return list(set(all_classes))

    def _get_subj_classes(self, seqs=None):
        """Returns a list of label lists where the label lists are
        grouped by subject
        [[(action, obj, b, e), ... ] for subject 1, [],...]
        """
        annot_paths = [os.path.join(self.label_path, annot_file)
                       for annot_file in os.listdir(self.label_path)]
        subjects_classes = []

        # Get classes for each subject
        if seqs is None:
            subjects = self.all_seqs
        else:
            subjects = seqs

        for subject in subjects:
            subject_annot_files = [filepath for filepath in annot_paths
                                   if subject in filepath]

            # Process files to extract action_lists
            subject_lines = [gteaannots.process_lines(subject_file)
                             for subject_file in subject_annot_files]

            # Flatten actions for each subject
            subject_labels = [label for sequence_labels in subject_lines
                              for label in sequence_labels]

            subjects_classes.append(subject_labels)
        return subjects_classes

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
                    if self.original_labels:
                        objects = _original_label_transform(objects)
                    if (action, objects) in action_object_classes:
                        for frame_idx in range(begin, end - frame_nb + 1):
                            actions.append((subject, recipe, action,
                                            objects, frame_idx))
        return actions

    def plot_hist(self):
        """Plots histogram of action classes as sampled in self.action_clips
        """
        labels = [self.get_class_str(action, obj)
                  for (subj, rec, action, obj, frm) in self.action_clips]
        visualize.plot_hist(labels)


def _original_label_transform(objects):
    mutual_1 = ['fork', 'knife', 'spoon']
    mutual_2 = ['cup', 'plate', 'bowl']
    processed_obj = []
    for obj in objects:
        if obj in mutual_1:
            processed_obj.append('spoonForkKnife')
        elif obj in mutual_2:
            processed_obj.append('cupPlateBowl')
        else:
            processed_obj.append(obj)
    return tuple(processed_obj)
