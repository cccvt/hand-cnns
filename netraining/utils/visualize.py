import os
import subprocess
import time

from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy.misc import imresize
import visdom

from actiondatasets.utils import annots as annot_utils
from actiondatasets.utils import display as displayutils

import logutils


def get_rand_win_id():
    postfix = str(hex(int(time.time() * 10000000))[2:])
    return 'win_{}'.format(postfix)


class Visualize():
    def __init__(self, opt):
        self.vis = visdom.Visdom(port=opt.display_port)

        self.opt = opt
        self.win = None
        checkpoint_folder = os.path.join(opt.checkpoint_dir, opt.exp_id)
        os.makedirs(checkpoint_folder, exist_ok=True)
        self.train_log_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                           'train_log.txt')
        self.valid_log_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                           'valid_log.txt')
        self.valid_aggreg_log_path = os.path.join(
            opt.checkpoint_dir, opt.exp_id, 'valid_aggreg.txt')
        self.lr_history_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                            'lr_history.txt')
        logutils.create_log_file(self.train_log_path, 'Train')
        logutils.create_log_file(self.valid_log_path, 'Valid')
        logutils.create_log_file(self.valid_aggreg_log_path, 'Valid aggreg')
        logutils.create_log_file(self.lr_history_path, 'LR schedule')

        # Launch visdom server
        exp_dir = os.path.join(self.opt.checkpoint_dir, self.opt.exp_id)
        visdom_command = [
            'python', '-m', 'visdom.server', '-port',
            str(opt.display_port), '-env_path', exp_dir
        ]
        sp = subprocess.Popen(
            visdom_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if sp:
            print('Launched visdom server on port {}'.format(opt.display_port))
            self.visdom_subprocess = sp
        else:
            print('Failed to launch visdom on port {} !!'.format(
                opt.display_port))
            self.save()
        self.error_windows = {}
        self.sample_windows = {}
        self.matrix_windows = {}

    def log_errors(self, epoch, errors, valid=False, log_path=None):
        """log_path overrides the destination path of the log
        Args:
            valid(bool): Whether to use the default valid or train log file
                (overriden by log_paht)
            errors(dict): in format {'error_name':error_score, ...}
        """
        if valid and log_path is not None:
            raise ValueError('when log_path is specified, valid is not taken\
                    into account')

        # Write log message to correct file
        if log_path is None:
            if valid:
                log_path = self.valid_log_path
            else:
                log_path = self.train_log_path
        message = logutils.log_errors(epoch, errors, log_path=log_path)
        return message

    def plot_errors(self,
                    window_name,
                    epochs,
                    errors,
                    title='score',
                    legend=None):
        if window_name in self.error_windows:
            win = self.error_windows[window_name]
        else:
            win = get_rand_win_id()
            self.error_windows[window_name] = win

        opts = {'title': title, 'xlabel': 'epoch', 'ylabel': 'score'}
        if legend is not None:
            opts['legend'] = legend
        self.vis.line(
            X=epochs, Y=errors, env=self.opt.exp_id, opts=opts, win=win)
        return win

    def get_sample_strings(self, predictions, gts, classes, display_idx=0,
                           k=1):
        pred_val, topk_classes = predictions.data[display_idx].topk(k)

        if self.opt.use_gpu:
            pred_val = pred_val.cpu()
            gts = gts.data.cpu()
        pred_classes = classes[int(topk_classes[0])]
        pred_string = str(pred_classes)

        real_score, real_class = gts[display_idx].max(0)
        real_string = classes[int(real_class[0])]

        return real_string, pred_string

    def plot_sample(self,
                    window_name,
                    input_imgs,
                    gts,
                    predictions,
                    classes,
                    display_idx=0,
                    k=1,
                    max_frames=8):
        """ Plots in visdom one image with predicted and ground truth labels
        from the given batch

        Args:
            max_frames (int): max number of frames to display

        """
        # Retrieve window id if exists
        if window_name in self.sample_windows:
            win = self.sample_windows[window_name]
        else:
            win = get_rand_win_id()
            self.sample_windows[window_name] = win

        time_input_imgs = input_imgs[display_idx]  # take first batch sample
        if self.opt.use_gpu:
            time_input_imgs = time_input_imgs.cpu()
        if isinstance(gts, dict):
            real_strings = [[] for idx in range(len(gts))]
            predicted_strings = [[] for idx in range(len(gts))]
            for target_idx, (label_name, gts_val) in enumerate(gts.items()):
                predictions_val = predictions[target_idx]
                classes_val = classes[target_idx]
                target_real_string, target_pred_string = self.get_sample_strings(
                    predictions_val,
                    gts_val,
                    classes_val,
                    display_idx=display_idx,
                    k=k)
                real_strings[target_idx].append(target_real_string)
                predicted_strings[target_idx].append(target_pred_string)
            caption = 'true: {} \n predicted: {}'.format(
                annot_utils.stringify(real_strings),
                annot_utils.stringify(predicted_strings))
        else:
            real_string, pred_string = self.get_sample_strings(
                predictions, gts, classes, display_idx=display_idx, k=k)
            caption = 'true : {} \n predicted: {}'.format(
                real_string, pred_string)

        if time_input_imgs.dim() == 4:
            stack_imgs = []
            input_img_nb = time_input_imgs.shape[1]
            time_step = min(1, input_img_nb // max_frames)
            for time_idx in range(0, input_img_nb, time_step):
                input_img = time_input_imgs[:, time_idx, :, :]
                input_img = prepare_img(input_img)
                stack_imgs.append(input_img)
            input_img = np.concatenate(stack_imgs, axis=2)
        else:
            input_img = time_input_imgs
        win = self.vis.image(
            input_img,
            env=self.opt.exp_id,
            opts={'title': 'sample',
                  'caption': caption,
                  'win_size': 256},
            win=win)
        return win

    def plot_mat(self,
                 window_name,
                 mat,
                 title='',
                 normalize_row=True,
                 labels=None):
        if window_name in self.matrix_windows:
            win = self.matrix_windows[window_name]
        else:
            win = get_rand_win_id()
            self.matrix_windows[window_name] = win
        mat = np.copy(mat)
        if normalize_row:
            for i in range(mat.shape[0]):
                norm_row = mat[i].sum() or 1
                mat[i] = mat[i] / norm_row
        opts = {'title': title}
        if labels is not None:
            opts['columnnames'] = labels
            opts['rownames'] = labels
        win = self.vis.heatmap(mat, env=self.opt.exp_id, win=win, opts=opts)

    def save(self):
        self.vis.save([self.opt.exp_id])


def prepare_img(input_img):
    """Identify if image is rgb or flow by determining number of
    channels and process image accordingly
    """
    channel_size = input_img.shape[0]
    # if non canonical number of channels (not 3), extract first channel
    if channel_size > 3:
        input_img = input_img.sum(0)  # sum channels to one dim

    # If two channels treat like flow
    elif channel_size == 2:
        angle, mag = displayutils.radial_flow(
            np.stack((input_img[0], input_img[1]), 2))
        angle, mag = displayutils.normalize_flow(angle, mag, clamp_mag=None)
        input_img = np.stack((angle, mag, np.ones_like(angle)), 2)
        input_img = hsv_to_rgb(input_img)  # In range [0, 1]
        input_img = input_img.transpose(2, 0, 1)
    # Scale color images from [0, 1] to [0, 255]
    if input_img.shape[0] == 3:
        if input_img.max() > 1:
            print('!! Warning, rescaling by *255 image with max {}'.format(
                input_img.max()))
        input_img = input_img * 255
    return input_img
