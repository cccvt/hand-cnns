import math
import os

import cv2
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from actiondatasets.utils import display as display_utils


def visualize(dataloader, model, opt=None):
    model.net.eval()
    get_activations_by_sample(dataloader, model, opt=opt)
    get_contributing_pixels(dataloader, model)
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        class_idx = class_idx[0]
        # Prepare vars
        imgs_var = model.prepare_var(sample, requires_grad=True)
        # Forward pass
        outputs = model.net(imgs_var, early_stop='5a')

        outputs_data = outputs.data

        mean_scores = outputs.mean(0)
        _, best_idx = mean_scores.max(0)


def get_activations_by_filter(dataloader,
                              model,
                              level='2a',
                              filter_chunk_size=64,
                              opt=None):
    model.net.eval()
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        class_idx = class_idx[0]
        # Prepare vars
        imgs_var = model.prepare_var(sample, requires_grad=True)
        # Forward pass
        outputs = model.net(imgs_var, early_stop=level)

        filter_nb = outputs.shape[1]
        print('At level {} output has {} filters'.format(level, filter_nb))
        outputs = outputs.data.cpu().numpy()
        activations = outputs[0]

        ax_nb = math.ceil(math.sqrt(filter_chunk_size))
        fig, axes = plt.subplots(ax_nb, ax_nb)
        time_steps = activations.shape[1]

        filter_chunk_nb = filter_nb // filter_chunk_size
        for filter_chunk in range(0, filter_chunk_nb):
            idx_start = filter_chunk * filter_chunk_size
            idx_end = (filter_chunk + 1) * filter_chunk_size
            filter_activations = activations[idx_start:idx_end]
            anim = animation.FuncAnimation(
                fig,
                update_activations_by_sample,
                time_steps,
                fargs=(axes, filter_activations, ax_nb),
                interval=500,
                repeat=False)
            anim_name = os.path.join(
                opt.checkpoint_dir, opt.exp_id,
                'activations_sample_{}_epoch_{}_level_{}_chunk_{}.gif'.format(
                    idx, opt.epoch, level, filter_chunk))
            anim_path = os.path.join(anim_name)
            anim.save(anim_path, dpi=80, writer='imagemagick')
            print('saved filters {} to {} \
                    to {}'.format(idx_start, idx_end, anim_path))


def update_activations_by_filter(idx, axes, activations, ax_nb=None):
    activations = activations[:, idx]  # Get relevant time step
    filter_nb = activations.shape[0]
    if ax_nb is None:
        row_nb = int(math.ceil(math.sqrt(filter_nb)))
    else:
        row_nb = ax_nb

    for filter_idx in range(filter_nb):
        row = int(filter_idx / row_nb)
        col = filter_idx - row * row_nb
        activation = activations[filter_idx]
        axes[row, col].imshow(activation)
        axes[row, col].axis('off')
    return axes


def get_activations_by_sample(dataloader,
                              model,
                              level='2a',
                              filter_chunk_size=8,
                              sample_nb=8,
                              opt=None):
    model.net.eval()
    sample_filters = []
    samples = []
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        if idx < sample_nb:
            print(idx)
            # Prepare vars
            imgs_var = model.prepare_var(sample, requires_grad=True)
            # Forward pass
            outputs = model.net(imgs_var, early_stop=level)
            outputs = outputs.data.cpu().numpy()

            # Save inputs
            activations = outputs[0]
            output_height, output_width = activations.shape[
                2], activations.shape[3]

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

    fig, axes = plt.subplots(sample_nb, filter_chunk_size + 1)

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
        anim.save(anim_path, dpi=80, writer='imagemagick')

        filter_start = chunk_idx * filter_chunk_size
        print(
            'saved filters {} to {} \
                to {}'
            .format(filter_start, filter_start + filter_chunk_size, anim_path))


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


def get_contributing_pixels(dataloader, model, activation_idx=None,
                            level='4a'):
    model.net.eval()
    for idx, (sample, class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        class_idx = class_idx[0]
        # Prepare vars
        imgs_var = model.prepare_var(sample, requires_grad=True)
        # Forward pass
        outputs = model.net(imgs_var, early_stop=level)
        print('At level {} output has shape {}'.format(level, outputs.shape))
        if activation_idx is None:
            activation_idx = [dim // 2 for dim in outputs.shape]
        activation = outputs[tuple(activation_idx)]
        activation.backward()
        img_grad_mask = imgs_var.grad.ne(0)
        img_idxs = img_grad_mask.shape
        img_idxs = [idx // 2 for idx in img_idxs]
        img_grad_mask = img_grad_mask.data.cpu().numpy()
        img_grad_img = img_grad_mask[img_idxs[0], img_idxs[1], :, :, img_idxs[
            4]]
        plt.imshow(img_grad_img)
        plt.show()
