import os
import subprocess
import time

import numpy as np
import visdom


class Visualize():
    def __init__(self, opt):
        self.vis = visdom.Visdom(port=opt.display_port)

        self.opt = opt
        self.win = None
        self.train_log_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                           'train_log.txt')
        self.valid_log_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                           'valid_log.txt')
        self.valid_aggreg_log_path = os.path.join(
            opt.checkpoint_dir, opt.exp_id, 'valid_aggreg.txt')
        self.lr_history_path = os.path.join(opt.checkpoint_dir, opt.exp_id,
                                            'lr_history.txt')

        # Initialize log files
        with open(self.train_log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Training log at {0} ====\n'.format(now))

        with open(self.valid_log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Valid log at {0} ====\n'.format(now))

        with open(self.valid_aggreg_log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Valid aggreg log at {0} ====\n'.format(now))

        with open(self.lr_history_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== LR schedule log at {0} ====\n'.format(now))

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

        now = time.strftime("%c")
        message = '(epoch: {epoch}, time: {t})'.format(epoch=epoch, t=now)
        for k, v in errors.items():
            message = message + ',{name}:{err}'.format(name=k, err=v)

        # Write log message to correct file
        if log_path is None:
            if valid:
                log_path = self.valid_log_path
            else:
                log_path = self.train_log_path
        with open(log_path, "a") as log_file:
            log_file.write(message + '\n')
        return message

    def plot_errors(self, epochs, errors, title='score', win=None):
        if win is None:
            win = self.vis.line(
                X=epochs,
                Y=errors,
                opts={'title': title,
                      'xlabel': 'epoch',
                      'ylabel': 'score'})
        else:
            self.vis.line(
                X=epochs,
                Y=errors,
                opts={'title': title,
                      'xlabel': 'epoch',
                      'ylabel': 'score'},
                win=win)
        return win

    def plot_sample(self,
                    input_imgs,
                    gts,
                    predictions,
                    classes,
                    win,
                    display_idx=0,
                    k=1,
                    unnormalize=None):
        """ Plots in visdom one image with predicted and ground truth labels
        from the given batch
        """
        input_img = input_imgs[display_idx]
        pred_val, topk_classes = predictions[display_idx].topk(k)

        if self.opt.use_gpu:
            pred_val = pred_val.cpu()
            gts = gts.cpu()
            input_img = input_img.cpu()
        if unnormalize is not None:
            input_img = unnormalize(input_img)
        pred_classes = classes[int(topk_classes[0])]
        pred_string = 'predicted : ' + str(pred_classes)

        real_score, real_class = gts[display_idx].max(0)
        real_class_str = classes[int(real_class[0])]

        real_string = 'true : ' + str(real_class_str)

        caption = pred_string + ';\n' + real_string

        # Extract one image from stacked images
        if input_img.dim() == 4:
            input_img = input_img[:, 0, :, :] * 255

        # if non canonical number of channels (not 3), extract first channel
        if input_img.shape[0] != 3:
            input_img = input_img[0, :, :]
        if win is None:
            win = self.vis.image(
                input_img,
                opts={'title': 'sample',
                      'caption': caption,
                      'win_size': 256})
        else:
            win = self.vis.image(
                input_img,
                opts={'title': 'sample',
                      'caption': caption,
                      'win_size': 256},
                win=win)
        return win

    def plot_mat(self, mat, win=None, title='', normalize_row=True):
        mat = np.copy(mat)
        if normalize_row:
            for i in range(mat.shape[0]):
                norm_row = mat[i].sum() or 1
                mat[i] = mat[i] / norm_row

        if win is None:
            win = self.vis.heatmap(mat, win=win, opts={'title': title})
        else:
            win = self.vis.heatmap(mat, win=win, opts={'title': title})
        return win

    def save(self):
        self.vis.save(['main'])
