import argparse
import math
import os
import re

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import torch

from src.utils.filesys import mkdir


def update_image_filters(idx, axes, weights_list):
    weights = weights_list[idx]
    filter_nb = weights.shape[0]
    row_nb = int(math.ceil(math.sqrt(filter_nb)))
    for filter_idx in range(filter_nb):
        row = int(filter_idx / row_nb)
        col = filter_idx - row * row_nb
        conv_filter = weights[filter_idx]
        conv_filter = conv_filter.transpose(1, 2, 0)
        axes[row, col].imshow(conv_filter)
        axes[row, col].axis('off')
    return axes


def update_flow_filters(frame_idx, filter_idx, axes,
                        weights_list, cmap='gray'):
    """Print flow filter evolution over epochs
    """
    for epoch_idx, epoch_weights in enumerate(weights_list):
        weights = epoch_weights[filter_idx]
        x_idx = frame_idx * 2
        y_idx = frame_idx * 2 + 1
        x_weights = weights[x_idx]
        y_weights = weights[y_idx]
        axes[0, epoch_idx].imshow(x_weights, cmap=cmap)
        axes[1, epoch_idx].imshow(y_weights, cmap=cmap)

        # Hide axis
        if epoch_idx > 0:
            axes[0, epoch_idx].axis('off')
            axes[1, epoch_idx].axis('off')

        else:
            # Add y axis title
            axes[0, 0].set_ylabel('x flow')
            axes[1, 0].set_ylabel('y flow')
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
    return axes


def update_3d_filters(time_step, axes, weights_list,
                      filter_begin=0, filter_nb=8):
    """Print 3d filter evolution over time

    vertically are the epochs, horizontally the different filters,
    time represents the time dimension of the 3d filter
    """
    for epoch_idx, epoch_weights in enumerate(weights_list):
        for filter_id in range(0, filter_nb):
            filter_idx = filter_id + filter_begin
            filter_weights = weights_list[epoch_idx][filter_idx]
            time_step_weights = filter_weights[:, time_step, :, :]
            time_step_weights.transpose(1, 2, 0)
            axes[filter_id, epoch_idx].imshow(time_step_weights)
            axes[filter_id, epoch_idx].axis('off')

    return axes


if __name__ == "__main__":
    print('Starting visualization')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/run_res_smthg/lr_0.0001')
    args = parser.parse_args()

    # Print options
    print('===== Options ====')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))

    checkpoint_files = os.listdir(args.checkpoint)
    saved_pth_paths = [filename for filename in checkpoint_files
                       if filename.endswith('.pth')]
    # Keep only numbered epochs
    saved_pth_paths = [path for path in saved_pth_paths
                       if 'latest' not in path]
    # Sort by increasing epoch nb
    saved_pth_paths = sorted(
        saved_pth_paths, key=lambda l: int(re.findall(r'\d+', l)[0]))

    # Create filter result folder
    filter_path = os.path.join(args.checkpoint, 'filter-gifs')
    mkdir(filter_path)

    conv1_weights_list = []

    # Aggregate first conv weights into list
    for i, pth_path in enumerate(saved_pth_paths):
        print(pth_path)
        load_path = os.path.join(args.checkpoint, pth_path)
        checkpoint = torch.load(load_path)
        net_weights = checkpoint['net']
        conv1_weights_name = list(net_weights.keys())[0]
        conv1_weights = net_weights[conv1_weights_name].numpy()
        conv1_weights_list.append(conv1_weights)

    filter_nb = conv1_weights_list[0].shape[0]
    epoch_nb = len(conv1_weights_list)

    # Check if 3d convolution
    if len(conv1_weights_list[0].shape) == 5:
        filter_split = 8
        fig, axes = plt.subplots(filter_split, epoch_nb)
        time_steps = 3
        for filter_begin in range(0, filter_nb, filter_split):
            anim = animation.FuncAnimation(fig, update_3d_filters,
                                           time_steps,
                                           fargs=(axes,
                                                  conv1_weights_list,
                                                  filter_begin, filter_split),
                                           interval=500, repeat=False)
            anim_name = '3d_filters_{}.gif'.format(filter_begin)
            anim_path = os.path.join(filter_path, anim_name)
            anim.save(anim_path, dpi=80, writer='imagemagick')
            print('saved filters {} to {} \
                    to {}'.format(filter_begin,
                                  filter_begin + filter_split - 1, anim_path))

    elif conv1_weights_list[0].shape[1] == 3:
        print('creating image filter animation')
        row_nb = 8
        fig, axes = plt.subplots(row_nb, row_nb)
        anim = animation.FuncAnimation(fig, update_image_filters, epoch_nb,
                                       fargs=(axes, conv1_weights_list))
        anim.save('test_img.avi')
    else:
        print('creating flow filter animations')
        conv1_weights_list = conv1_weights_list[::5]
        flow_nb = len(conv1_weights_list)
        fig, axes = plt.subplots(2, flow_nb)
        suptitle = fig.suptitle("filter {}".format(0), fontsize=12)
        print(len(conv1_weights_list))
        print('{} filters in first conv'.format(filter_nb))
        for filter_idx in range(filter_nb):
            suptitle.set_text("filter {}".format(filter_idx))
            anim = animation.FuncAnimation(fig, update_flow_filters, flow_nb,
                                           fargs=(filter_idx, axes,
                                                  conv1_weights_list),
                                           interval=500, repeat=False)
            anim_name = 'filter_{}.gif'.format(filter_idx)
            anim_path = os.path.join(filter_path, anim_name)
            anim.save(anim_path, dpi=80, writer='imagemagick')
            print('saved filter {} to {}'.format(filter_idx, anim_path))
            # plt.show()
