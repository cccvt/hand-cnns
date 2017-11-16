import argparse
import math
import os
import re

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import torch

from src.utils.filesys import mkdir

from src.datasets.gteagazeplusvideo import GTEAGazePlusVideo


def update_flow_image(frame_idx, axes, folder, x_template,
                      y_template, offset):
    """
    Args:
        offset(int): frame at which to start gif
        x_template(str): template in which to insert frame name
            to retrieve horizontal flow file name
    """
    frame_idx = frame_idx + offset
    x_flow_name = x_template.format(frame_idx)
    x_flow_path = os.path.join(folder, x_flow_name)
    x_flow = plt.imread(x_flow_path)
    x_flow_name = x_template.format(frame_idx)
    x_flow_path = os.path.join(folder, x_flow_name)
    x_flow = plt.imread(x_flow_path)
    y_flow_name = y_template.format(frame_idx)
    y_flow_path = os.path.join(folder, y_flow_name)
    y_flow = plt.imread(y_flow_path)
    y_flow_name = y_template.format(frame_idx)
    y_flow_path = os.path.join(folder, y_flow_name)
    y_flow = plt.imread(y_flow_path)
    img = np.stack((x_flow / 255, y_flow / 255, 0.5 * np.ones_like(x_flow)),
                   axis=2)

    axes.imshow(img)
    axes.axis('off')
    return axes


if __name__ == "__main__":
    print('Starting visualization')

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_folder', type=str,
                        default='data/GTEAGazePlus/tvl1-flow/Ahmad_American')
    parser.add_argument('--out_folder', type=str,
                        default='results/flow_gifs')
    parser.add_argument('--x_template', type=str,
                        default='{:010d}x.jpg')
    parser.add_argument('--y_template', type=str,
                        default='{:010d}y.jpg')
    parser.add_argument('--gif_length', type=int,
                        default='10')
    parser.add_argument('--gif_offset', type=int,
                        default='1')
    args = parser.parse_args()

    # Print options
    print('===== Options ====')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))

    mkdir(args.out_folder)

    fig, axes = plt.subplots(1, 1)
    anim = animation.FuncAnimation(fig, update_flow_image,
                                   args.gif_length,
                                   fargs=(axes,
                                          args.inp_folder,
                                          args.x_template,
                                          args.y_template,
                                          args.gif_offset),
                                   interval=500, repeat=False,
                                   blit=True)
    anim_path = os.path.join(args.out_folder,
                             os.path.basename(args.inp_folder) + '.gif')
    print("Saving gif to {}".format(anim_path))
    # anim.save(anim_path, dpi=80, writer='imagemagick')

    # Action clips to gif
    dataset = GTEAGazePlusVideo(use_flow=1, flow_type='farn')
    for action, objects, subject, recipe, beg, end in dataset.action_clips:
        sequence_name = subject + '_' + recipe
        sequence_path = os.path.join(dataset.flow_path, sequence_name)
        anim = animation.FuncAnimation(fig, update_flow_image,
                                       end - beg,
                                       fargs=(axes,
                                              sequence_path,
                                              args.x_template,
                                              args.y_template,
                                              beg),
                                       interval=500, repeat=False)
        anim_path = os.path.join(args.out_folder,
                                 sequence_name + str(beg) +
                                 dataset.get_class_str(action, objects) + '.gif')
        print("Saving gif to {}".format(anim_path))
        anim.save(anim_path, dpi=80, writer='imagemagick')
