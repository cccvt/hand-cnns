import os

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from tqdm import tqdm

from actiondatasets.utils import display as display_utils


def relu_forward_gbackprop_hook(module, input, output):
    module.input_kept = input[0]


def relu_backward_gbackprop_hook(module, grad_input, grad_output):
    hook_out = torch.nn.functional.relu(
        grad_output[0]) * torch.nn.functional.relu(module.input_kept).sign(),
    return hook_out


def equip_model_gbackprop(model):
    for m in model.net.modules():
        if isinstance(m, torch.nn.ReLU):
            m.register_forward_hook(relu_forward_gbackprop_hook)
            m.register_backward_hook(relu_backward_gbackprop_hook)


def get_guided_by_sample(dataloader,
                         model,
                         level='4a',
                         sample_nb=2,
                         opt=None,
                         normalize_by_frame=True):
    """
    Produces gifs of guided backprop for a number of samples
    """
    if opt.use_flow:
        print('\033[93m  Make sure rescale_flow option is '
              'same in checkpoint and test\033[0m')
        if isinstance(model.net, torch.nn.DataParallel):
            print('Extracting network from DataParallel')
        model.net = model.net.module

    model.net.eval()
    equip_model_gbackprop(model)

    for sample_idx, (sample,
                     class_idx) in enumerate(tqdm(dataloader, desc='sample')):
        if sample_idx == 0:
            continue
        if sample_idx < sample_nb:
            # Prepare vars
            imgs_var = model.prepare_var(sample, requires_grad=True)
            # Forward pass
            outputs = model.net(imgs_var, early_stop=level)
            results, = torch.autograd.grad(
                outputs.max(), imgs_var, retain_graph=True)
            results = results.data.cpu().numpy()

            # Save inputs
            results = results[0]  # Remove batch dimension

            results = results.transpose(1, 2, 3,
                                        0)  # time, height, width, channels
            # Rescale results between 0 and 1 for display
            print('Range of guided backprop : {} - {}'.format(
                results.min(), results.max()))
            if normalize_by_frame:
                # Local time normalization
                time_mins = results.min(1).min(1).min(1)
                time_maxs = results.max(1).max(1).max(1)
                time_mins = time_mins[:, np.newaxis, np.newaxis, np.newaxis]
                time_maxs = time_maxs[:, np.newaxis, np.newaxis, np.newaxis]
                results = (results - time_mins) / (time_maxs - time_mins)
            else:
                # Global time normalization
                results = (results - results.min()) / (
                    results.max() - results.min())

            # Save samples
            sample = sample.numpy()[0].transpose(1, 2, 3, 0)

            # Save results
            save_guided_backprop(sample, results, sample_idx, level, opt)
        else:
            break


def save_guided_backprop(sample, results, sample_idx, level, opt):
    fig, axes = plt.subplots(1, 2)
    time_steps = results.shape[0]
    print('{} time steps for sample {}'.format(time_steps, sample_idx))
    anim = animation.FuncAnimation(
        fig,
        update_guided_backprop,
        time_steps,
        fargs=(axes, sample, results, opt.use_flow),
        interval=500,
        repeat=False)
    anim_name = os.path.join(
        opt.checkpoint_dir, opt.exp_id,
        'guided_backprop_sample_{}_epoch_{}_level_{}.gif'.format(
            sample_idx, opt.epoch, level))
    anim_path = os.path.join(anim_name)
    anim.save(anim_path, dpi=160, writer='imagemagick')

    print('saved sample {} for level {} to {}'.format(sample_idx, level,
                                                      anim_path))


def update_guided_backprop(idx, axes, samples, results, use_flow=False):
    img_sample = samples[idx]
    img_result = results[idx]
    if use_flow:
        img_sample = display_utils.get_draw_flow(img_sample * 255)
    axes[0].imshow(img_sample)
    axes[0].axis('off')
    axes[1].imshow(img_result)
    axes[1].axis('off')
    return axes
