import random
import os

import numpy as np

from src.datasets.utils import loader, visualize
from src.datasets.gteagazeplus import GTEAGazePlus


class GTEAGazePlusVideo(GTEAGazePlus):
    def __init__(self, root_folder="data/GTEAGazePlus",
                 original_labels=True, seqs=['Ahmad', 'Alireza', 'Carlos',
                                             'Rahul', 'Shaghayegh', 'Yin'],
                 video_transform=None, base_transform=None,
                 clip_size=16, use_video=False):
        """
        Args:
            video_transform: transformation to apply to the clips during
                training
            base_transform: transformation to applay to the clips during
                testing
            use_video (bool): whether to use video inputs or png inputs
        """
        super().__init__(root_folder=root_folder,
                         original_labels=original_labels,
                         seqs=seqs)

        # Set video params
        self.video_transform = video_transform
        self.base_transform = base_transform

        self.use_video = use_video
        self.rgb_path = os.path.join(self.path, 'png')
        self.video_path = os.path.join(self.path, 'avi_files')
        self.clip_size = clip_size

        action_clips = self.get_all_actions(self.classes)
        # Remove actions that are too short
        self.action_clips = [(action, obj, subj, rec, beg, end)
                             for (action, obj, subj, rec, beg, end)
                             in action_clips
                             if end - beg >= self.clip_size]
        action_labels = [(action, obj) for (action, obj, subj, rec,
                                            beg, end) in self.action_clips]
        print(len(action_labels))
        assert len(action_labels) > 100
        self.class_counts = self.get_action_counts(action_labels)
        assert sum(self.class_counts) == len(action_labels)

    def __getitem__(self, index):
        # Load clip
        action, objects, subject, recipe, beg, end = self.action_clips[index]
        sequence_name = subject + '_' + recipe
        frame_idx = random.randint(beg, end - self.clip_size)
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

    def get_class_items(self, index, frame_nb=None):
        # Load clip info
        action, objects, subject, recipe, beg, end = self.action_clips[index]
        sequence_name = subject + '_' + recipe
        frame_idx = random.randint(beg, end - self.clip_size)

        # Get class index
        class_idx = self.classes.index((action, objects))

        # Return list of action tensors
        clips = []

        if frame_nb is None:
            frame_idxs = range(beg, end)
        else:
            frame_idxs = np.linspace(beg, end - self.clip_size, frame_nb)
            frame_idxs = [int(frame_idx) for frame_idx in frame_idxs]

        for frame_idx in frame_idxs:
            clip = self.get_clip(sequence_name, frame_idx, self.clip_size)
            if self.base_transform is not None:
                clip = self.base_transform(clip)
            clips.append(clip)
        return clips, class_idx

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
        all possible tuples (action, objects, subject, recipe, begin_frame)
        with all begin_frame so that at least frame_nb frames belong
        to the given action

        for frame_nb: 2 and begin: 10, end:17, the candidate begin_frames are:
        10, 11, 12, 13, 14, 15
        This guarantees that we can extract frame_nb of frames starting at
        begin_frame and still be inside the action

        This gives all possible action blocks for the subjects in self.seqs
        """
        dense_actions = []
        actions = self.get_all_actions(action_object_classes)
        for action, objects, subject, recipe, begin, end in actions:
            for frame_idx in range(begin, end - frame_nb + 1):
                dense_actions.append((action, objects, subject,
                                      recipe, frame_idx))
        return dense_actions

    def plot_hist(self):
        """Plots histogram of action classes as sampled in self.action_clips
        """
        labels = [self.get_class_str(action, obj)
                  for (action, obj, subj, rec, beg, end) in self.action_clips]
        visualize.plot_hist(labels, proportion=True)
