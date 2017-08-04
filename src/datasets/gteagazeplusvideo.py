import numpy as np
import re
import os

from src.datasets.utils import loader, gteaannots
from src.datasets.gteagazeplus import GTEAGazePlus


class GTEAGazePlusVideo(GTEAGazePlus):
    def __init__(self, root_folder="data/GTEAGazePlus",
                 original_labels=True, seqs=['Ahmad', 'Alireza', 'Carlos',
                                             'Rahul', 'Shaghayegh', 'Yin'],
                 video_transform=None, clip_size=16, use_video=False):
        """
        Args:
            video_transform: transformation to apply to the clips
            use_video (bool): whether to use video inputs or png inputs
        """
        super().__init__(root_folder=root_folder,
                         original_labels=original_labels,
                         seqs=seqs)

        # Set video params
        self.video_transform = video_transform
        self.use_video = use_video
        self.rgb_path = os.path.join(self.path, 'png')
        self.video_path = os.path.join(self.path, 'avi_files')
        self.clip_size = clip_size

        self.action_clips = self.get_dense_actions(clip_size, self.classes)

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
                        objects = self.original_label_transform(objects)
                    if (action, objects) in action_object_classes:
                        for frame_idx in range(begin, end - frame_nb + 1):
                            actions.append((subject, recipe, action,
                                            objects, frame_idx))
        return actions
