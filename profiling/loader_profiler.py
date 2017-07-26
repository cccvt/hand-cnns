import os
import sys

sys.path.insert(0, ".")
from src.datasets.utils import loader

base_folder = '/home/local2/yhasson/datasets/GTEAGazePlus/GTEA'
sequence_name = 'Ahmad_American'
begin_frame = 1
frame_nb = 40

# Load video
video_filename = os.path.join(base_folder, 'avi_files',
                              sequence_name + '.avi')
vid = loader.get_video_capture(video_filename)
loader.get_clip(vid, begin_frame, frame_nb)

# Load stacked frames
image_folder = os.path.join(base_folder, 'png', sequence_name)
loader.get_stacked_frames(image_folder, begin_frame, frame_nb,
                          use_open_cv=False)

