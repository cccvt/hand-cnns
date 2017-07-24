import sys

sys.path.insert(0, ".")
from src.datasets.utils import loader


file_name = '/home/local2/yhasson/datasets/GTEAGazePlus/GTEA/avi_files/Ahmad_American.avi'
vid = loader.get_video_capture(file_name)
loader.get_clip(vid, 10000, 4)
