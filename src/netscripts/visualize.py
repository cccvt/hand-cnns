import math
import os

import cv2
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from actiondatasets.utils import display as display_utils

from src.netscripts.visualize_guided import get_guided_by_sample


def visualize(dataloader, model, opt=None):
    model.net.eval()
    get_guided_by_sample(
        dataloader, model, level=opt.level, sample_nb=10, opt=opt)
    get_contributing_pixels(
        dataloader,
        model,
        activation_idx=opt.activation_idx,
        opt=opt,
        level=opt.level)
    get_activations_by_sample(dataloader, model, opt=opt, level=opt.level)


def get_activations_by_sample(dataloader,
                              model,
                              level='4a',
                              filter_chunk_size=8,
                              sample_nb=8,
                              opt=None):
    """
    Produces gifs of activations of filters for a number of samples
    """
    if opt.use_flow:
        print('\033[93m  Make sure rescale_flow option is '
              'same in checkpoint and test\033[0m')
    model.net.eval()
    sample_filters = []
    samples = []
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        if idx < sample_nb:
            # Prepare vars
            imgs_var = model.prepare_var(sample, requires_grad=True)
            # Forward pass
            outputs = model.net(imgs_var, early_stop=level)
            outputs = outputs.data.cpu().numpy()

            # Save inputs
            activations = outputs[0]
            output_time, output_height, output_width = activations.shape[
                1], activations.shape[2], activations.shape[3]

            # Resize samples
            sample = sample.numpy()[0]
            samples.append(sample)
            if idx == 0:
                filter_nb = outputs.shape[1]
                filter_chunk_nb = filter_nb // filter_chunk_size
                print('At level {} output has {} filters'.format(
                    level, filter_nb))
            chunked_filter_activations = []
            for filter_chunk in range(0, filter_chunk_nb):
                idx_start = filter_chunk * filter_chunk_size
                idx_end = (filter_chunk + 1) * filter_chunk_size
                filter_activations = activations[idx_start:idx_end]
                chunked_filter_activations.append(filter_activations)
            sample_filters.append(chunked_filter_activations)
        else:
            break

    # Compute minimal number of time steps for outputs
    time_steps = min(sample_filter[0].shape[1]
                     for sample_filter in sample_filters)

    # Resize input samples in time and space so that they match outputs
    print('Processing inputs')
    resized_samples = []
    for sample_idx, sample in enumerate(samples):
        input_time_size = sample.shape[1]
        output_time_size = sample_filters[sample_idx][0].shape[1]

        time_idxs = np.linspace(0, input_time_size - 1, output_time_size)
        time_samples = []
        for time_idx in time_idxs:
            time_idx = math.floor(time_idx)
            time_sample = sample[:, time_idx].transpose(1, 2, 0)
            resized_sample = cv2.resize(time_sample, (output_width,
                                                      output_height))
            time_samples.append(resized_sample)
        resized_samples.append(time_samples)

    for chunk_idx in range(0, filter_chunk_nb):
        fig, axes = plt.subplots(sample_nb, filter_chunk_size + 1)
        idx_start = chunk_idx * filter_chunk_size
        idx_end = idx_start + filter_chunk_size
        fig.suptitle('{} level {} - dim h: {} w :{} t: {}'.format(
            model.name, level, output_height, output_width, output_time) +
                     '\n filters {} to {}'.format(idx_start, idx_end))

        chunk_activations = [chunks[chunk_idx] for chunks in sample_filters]
        anim = animation.FuncAnimation(
            fig,
            update_activations_by_sample,
            time_steps,
            fargs=(axes, chunk_activations, resized_samples, sample_nb,
                   filter_chunk_size, opt.use_flow),
            interval=500,
            repeat=False)
        anim_name = os.path.join(
            opt.checkpoint_dir, opt.exp_id,
            'activations_samples_{}_epoch_{}_level_{}_chunk_{}.gif'.format(
                idx, opt.epoch, level, chunk_idx))
        anim_path = os.path.join(anim_name)
        anim.save(anim_path, dpi=160, writer='imagemagick')

        print('saved filters {} to {} to {}'.format(idx_start, idx_end,
                                                    anim_path))


def update_activations_by_sample(idx,
                                 axes,
                                 activations,
                                 inputs,
                                 row_nb=None,
                                 col_nb=None,
                                 use_flow=False):
    if row_nb is None:
        row_nb = len(activations)
    if col_nb is None:
        col_nb = activations[0].shape

    for sample_idx in range(row_nb):
        input_img = inputs[sample_idx][idx]
        if use_flow:
            input_img = display_utils.get_draw_flow(input_img * 255)
        axes[sample_idx, 0].imshow(input_img)
        axes[sample_idx, 0].axis('off')
        for filter_idx in range(col_nb):
            sample_activation = activations[sample_idx]
            activation = sample_activation[filter_idx, idx]
            axes[sample_idx, filter_idx + 1].imshow(activation, cmap='gray')
            axes[sample_idx, filter_idx + 1].axis('off')
    return axes


def get_contributing_pixels(dataloader,
                            model,
                            activation_idx=None,
                            level='4a',
                            opt=None):
    model.net.eval()
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        class_idx = class_idx[0]
        # Prepare vars
        imgs_var = model.prepare_var(sample, requires_grad=True)
        # Forward pass
        outputs = model.net(imgs_var, early_stop=level)
        print('At level {} output has shape {}'.format(level, outputs.shape))
        print(imgs_var.shape)
        if not len(activation_idx):
            sample_activation_idx = [dim // 2 for dim in outputs.shape]
        else:
            sample_activation_idx = activation_idx
        print(sample_activation_idx)
        sample_activation_idx[2] = 0
        print(sample_activation_idx)
        activation = outputs[tuple(sample_activation_idx)]
        print(activation.data[0])

        # Back propagate a nan so that all neurons in receptive field become nans
        if activation.data[0] != 0:
            nan_der = torch.Tensor(1)
            nan_der[0] = np.nan
            activation.backward(gradient=nan_der.cuda(), retain_graph=True)
            img_grad_mask = (imgs_var.grad != imgs_var.grad).float()
            img_grad_mask = img_grad_mask.data.cpu().numpy()
            img_grad_mask = img_grad_mask[0]
            channel_nb, time_nb, height_nb, width_nb = img_grad_mask.shape

            fig, axes = plt.subplots(channel_nb, time_nb)
            fig.suptitle('Receptive field for sample of len {} at level {}'.
                         format(time_nb, level))
            for channel in range(channel_nb):
                for time in range(time_nb):
                    axes[channel, time].imshow(
                        img_grad_mask[channel, time] * 0.5,
                        cmap='gray',
                        vmin=0,
                        vmax=1)
                    axes[channel, time].axis('off')
                    axes[channel, time].set_title('{}'.format(time))
        plt.show()
