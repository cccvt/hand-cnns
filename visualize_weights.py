import argparse
import math
import os

from matplotlib import pyplot as plt
import torch

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
    saved_pth_paths = sorted([filename for filename in checkpoint_files
                              if filename.endswith('.pth')])

    conv1_weights_list = []
    for i, pth_path in enumerate(saved_pth_paths):
        print(pth_path)
        load_path = os.path.join(args.checkpoint, pth_path)
        checkpoint = torch.load(load_path)
        net_weights = checkpoint['net']
        conv1_weights_name = list(net_weights.keys())[0]
        conv1_weights = net_weights[conv1_weights_name].numpy()
        conv1_weights_list.append(conv1_weights)

    fig = None
    for i, conv1_weights in enumerate(conv1_weights_list):
        print('First filter_name: {}'.format(conv1_weights_name))
        print('Filter shape: {}'.format(tuple(conv1_weights.shape)))
        filter_nb = conv1_weights.shape[0]
        row_nb = int(math.ceil(math.sqrt(filter_nb)))

        if fig is None:
            fig, axes = plt.subplots(row_nb, row_nb)
        for filter_idx in range(filter_nb):
            row = int(filter_idx / row_nb)
            col = filter_idx - row*row_nb
            conv_filter = conv1_weights[filter_idx]
            if conv_filter.shape[0] == 3:
                conv_filter = conv_filter.transpose(1, 2, 0)
            axes[row, col].imshow(conv_filter)
            axes[row, col].axis('off')
        plt.show(block=False)
    plt.show()
    fig, axes = plt.subplots(row_nb, row_nb)

    def update_filters(idx, axes, weights_list):
        weights = weights_list[idx]
        filter_nb = weights.shape[0]
        row_nb = int(math.ceil(math.sqrt(filter_nb)))
        for filter_idx in range(filter_nb):
            row = int(filter_idx / row_nb)
            col = filter_idx - row*row_nb
            conv_filter = weights[filter_idx]
            conv_filter = conv_filter.transpose(1, 2, 0)
            axes[row, col].imshow(conv_filter)
            axes[row, col].axis('off')
        return axes


    animation = FuncAnimation(fig, update_filters, 10, fargs(axes, conv1_weights_list))
