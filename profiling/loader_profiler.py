import cv2
import os
import sys

sys.path.insert(0, ".")
from src.datasets.utils import loader

file_name = '/home/local2/yhasson/datasets/GTEAGazePlus/GTEA/avi_files/Ahmad_American.avi'
print('file found', os.path.isfile(file_name))
vid = cv2.VideoCapture(file_name)
loader.get_clip(vid, 10000, 4)

