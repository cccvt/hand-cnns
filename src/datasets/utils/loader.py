import cv2
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
        frame_rgb = frame[:, :, ::-1]
        arranged_frame = np.rollaxis(frame_rgb, 2, 0)
        clip[:, frame_idx, :, :] = arranged_frame
    return clip
