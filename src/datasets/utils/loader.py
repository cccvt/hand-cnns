import cv2
import os
import numpy as np
from PIL import Image
from src.utils.debug import timeit


def load_rgb_image(path):
    """
    loads image from path
    :param path: absolute or relative path to file
    :rtype: PIL Image
    :return: RGB Image
    """
    image = Image.open(path)
    return image.convert("RGB")


def load_depth_image(path):
    """
    loads depth image from path
    :param path: absolute or relative path to depth image
    :rtype: PIL Image
    :return: Depth image
    """
    image = Image.open(path)
    return image


class OpenCvError(Exception):
    pass


def get_video_capture(file_name):
    """
    Returns opencv video_capture name based on file_name
    """
    file_found = os.path.isfile(file_name)

    # Check video exists as file
    if not file_found:
        raise OpenCvError('Video file {0} doesn\'t exist'.format(
            file_name))
    video_capture = cv2.VideoCapture(file_name)

    # Check video could be read
    if not video_capture.isOpened():
        raise OpenCvError('Video is not opened')

    return video_capture


def get_clip(video_capture, frame_begin, frame_nb):
    """
    Returns clip of video as torch Tensor
    of dimensions [channels, frames, height, width]

    :param video_capture: opencv videoCapture object loading a video
    :param frame_begin: first frame from clip
    :param frame_nb: number of frames to retrieve
    """

    # Get video dimensions
    if video_capture.isOpened():
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    else:
        raise OpenCvError('Video is not opened')

    # Retrieve successive frames starting from frame_begin frame
    video_capture.set(1, frame_begin)

    clip = np.zeros([3, frame_nb, int(height), int(width)])
    # Fill clip array with consecutive frames
    for frame_idx in range(frame_nb):
        flag, frame = video_capture.read()
        if not flag:
            raise OpenCvError('Could not read frame {0}'.format(
                frame_idx + frame_begin))
        frame_rgb = frame[:, :, ::-1]
        arranged_frame = np.rollaxis(frame_rgb, 2, 0)
        clip[:, frame_idx, :, :] = arranged_frame
    return clip
