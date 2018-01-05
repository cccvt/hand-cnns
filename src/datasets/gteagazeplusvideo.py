import random
import os

import numpy as np

from src.datasets.utils import loader, visualize
from src.datasets.gteagazeplus import GTEAGazePlus


class GTEAGazePlusVideo(GTEAGazePlus):
    def __init__(
            self,
            root_folder="data/GTEAGazePlus",
            original_labels=True,
            seqs=['Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Shaghayegh', 'Yin'],
            video_transform=None,
            base_transform=None,
            clip_size=16,
            use_video=False,
            use_flow=False,
            flow_type=None,
            rescale_flows=True):
        """
        Args:
            video_transform: transformation to apply to the clips during
                training
            base_transform: transformation to applay to the clips during
                testing
            use_video (bool): whether to use video inputs or png inputs
        """
        super().__init__(
            root_folder=root_folder,
            original_labels=original_labels,
            seqs=seqs,
            use_flow=use_flow,
            flow_type=flow_type,
            rescale_flows=rescale_flows)

        # Set video params
        self.video_transform = video_transform
        self.base_transform = base_transform

        self.use_video = use_video
        self.clip_size = clip_size

        action_clips = self.get_all_actions(self.classes)
        # Remove actions that are too short
        self.action_clips = [(action, obj, subj, rec, beg, end)
                             for (action, obj, subj, rec, beg,
                                  end) in action_clips
                             if end - beg >= self.clip_size]
        action_labels = [(action, obj)
                         for (action, obj, subj, rec, beg,
                              end) in self.action_clips]
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

    def get_full_sample(self, index, max_size=100):
        """
        Get full clip in time dimension for final fully-convolutional testing
        """
        action, objects, subject, recipe, beg, end = self.action_clips[index]

        sequence_name = subject + '_' + recipe
        if self.use_flow:
            # Last frame not valid for flow
            end = end - 1
        clip_size = end - beg
        if clip_size > max_size:
            center = (beg + end) // 2
            beg = center - max_size // 2
            clip_size = max_size
        clip = self.get_clip(sequence_name, beg, clip_size)

        # Apply video transform
        if self.base_transform is not None:
            clip = self.base_transform(clip)

        # Get class index
        class_idx = self.classes.index((action, objects))
        return clip, class_idx

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
        if self.use_flow:
            # Retrieve stacked flow frames
            flow_path = os.path.join(self.flow_path, sequence_name)
            clip = loader.get_stacked_flow_arrays(
                flow_path,
                frame_begin,
                frame_nb,
                flow_x_template=self.flow_x_template,
                flow_y_template=self.flow_y_template,
                minmax_filename=self.minmax_filename)
        else:
            # Retrieve stacked rgb frames
            if self.use_video:
                video_path = os.path.join(self.video_path,
                                          sequence_name + '.avi')
                video_capture = loader.get_video_capture(video_path)
                clip = loader.get_clip(video_capture, frame_begin, frame_nb)
            else:
                png_path = os.path.join(self.rgb_path, sequence_name)
                clip = loader.get_stacked_frames(
                    png_path, frame_begin, frame_nb, use_open_cv=False)

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
                dense_actions.append((action, objects, subject, recipe,
                                      frame_idx))
        return dense_actions

    def plot_hist(self):
        """Plots histogram of action classes as sampled in self.action_clips
        """
        labels = [
            self.get_class_str(action, obj)
            for (action, obj, subj, rec, beg, end) in self.action_clips
        ]
        visualize.plot_hist(labels, proportion=True)
