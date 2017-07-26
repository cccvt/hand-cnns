import cv2
import os
import numpy as np
from PIL import Image


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


@profile
def get_clip(video_capture, frame_begin, frame_nb):
    """
    Returns clip of video as list of numpy.ndarrays
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

    clip = []
    # Fill clip array with consecutive frames
    for frame_idx in range(frame_nb):
        flag, frame = video_capture.read()
        if not flag:
            raise OpenCvError('Could not read frame {0}'.format(
                frame_idx + frame_begin))

        clip.append(frame)
    return clip


@profile
def get_stacked_frames(image_folder, frame_begin, frame_nb,
                       use_open_cv=True):
    """ Returns numpy array/PIL image of stacked images with dimensions
    [channels, frames, height, width]

    Args:
        image_folder (str): folder containing the images in format
            00{frame}.png
        frame_nb (int): number of consecutive frames to stack
        use_open_cv (bool): wheter to use opencv or PIl image reader
            if PIL is used, PIL image are returned
    """

    frame_template = "{frame:010d}.png"
    clip = []
    for idx in range(frame_nb):
        frame_idx = frame_begin + idx
        image_path = os.path.join(image_folder,
                                  frame_template.format(frame=frame_idx))
        if use_open_cv:
            img = cv2.imread(image_path)
            if img is None:
                raise OpenCvError('Could not open image {0}'.format(image_path))
        else:
            img = Image.open(image_path)
            img = np.asarray(img)
        clip.append(img)
    return clip


def format_img_from_opencv(img_array):
    """
    Transforms an opencv numpy array [width, height, BRG]
    to [RGB, width, height]
    """
    frame_rgb = img_array[:, :, ::-1]
    arranged_frame = np.rollaxis(frame_rgb, 2, 0)
    return arranged_frame
