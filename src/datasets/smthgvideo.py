import random
import numpy as np

from src.datasets.utils import loader
from src.datasets.smthg import Smthg


class SmthgVideo(Smthg):
    def __init__(self, root_folder="data/smthg-smthg", split='train',
                 video_transform=None, base_transform=None,
                 clip_size=16, use_flows=False, frame_spacing=1):
        """
        Args:
            video_transform: transformation to apply to the clips during
                training
            base_transform: transformation to applay to the clips during
                testing
           frame_spacing(int): time spacing between consecutive frames in clip
        """
        super().__init__(root_folder=root_folder,
                         split=split, use_flows=use_flows)

        # Set video params
        self.video_transform = video_transform
        self.base_transform = base_transform
        self.frame_spacing = 1

        self.clip_size = clip_size

        # Remove actions that are too short
        self.sample_list = [(clip_id, label, max_frame)
                            for (clip_id, label, max_frame)
                            in self.sample_list
                            if max_frame >= self.clip_size + 1]

    def __getitem__(self, index):
        # Load clip
        clip_id, label, max_frame = self.sample_list[index]
        frame_idx = random.randint(1, max_frame - self.clip_size
                                   - self.frame_spacing + 1)
        clip = self.get_clip(clip_id, frame_idx)

        # Apply video transform
        if self.video_transform is not None:
            clip = self.video_transform(clip)

        # One hot encoding
        annot = np.zeros(self.class_nb)
        class_idx = self.classes.index(label)
        annot[class_idx] = 1
        return clip, annot

    def __len__(self):
        return len(self.sample_list)

    def get_class_items(self, index, frame_nb=None):
        # Load clip info
        clip_id, label, max_frame = self.sample_list[index]
        frame_idx = random.randint(1, max_frame - self.clip_size)

        # Get class index
        if self.split == 'test':
            class_idx = 0
        else:
            class_idx = self.classes.index(label)

        # Return list of action tensors
        clips = []

        if frame_nb is None:
            frame_idxs = range(1, max_frame)
        else:
            frame_idxs = np.linspace(1, max_frame - self.clip_size, frame_nb)
            frame_idxs = [int(frame_idx) for frame_idx in frame_idxs]

        for frame_idx in frame_idxs:
            clip = self.get_clip(clip_id, frame_idx)
            if self.base_transform is not None:
                clip = self.base_transform(clip)
            clips.append(clip)
        return clips, class_idx

    def get_clip(self, clip_idx, frame_begin):
        clip_path = self.path_from_id(clip_idx)
        frame_nb = self.clip_size
        if self.use_flows:
            clip = loader.get_stacked_flow_arrays(clip_path, frame_begin,
                                                  frame_nb, 
                                                  flow_x_template="{frame:05d}x.jpg",
                                                  flow_y_template="{frame:05d}y.jpg",
                                                  minmax_filename="minmax.pickle")

        else:
            clip = loader.get_stacked_frames(clip_path, frame_begin,
                                             frame_nb, frame_spacing=self.frame_spacing,
                                             use_open_cv=False,
                                             frame_template=self.frame_template)

        return clip
