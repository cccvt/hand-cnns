import os
import time
import visdom


class Visualize():
    def __init__(self, opt):
        self.vis = visdom.Visdom()
        self.opt = opt
        self.win = None
        self.train_log_path = os.path.join(opt.checkpoint_dir,
                                           opt.exp_id, 'train_log.txt')
        self.valid_log_path = os.path.join(opt.checkpoint_dir,
                                           opt.exp_id,
                                           'valid_log.txt')

        # Initialize log files
        with open(self.train_log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Training log at {0} ====\n'.format(now))

        with open(self.valid_log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Valid log at {0} ====\n'.format(now))

    def log_errors(self, epoch, errors, valid=False):
        now = time.strftime("%c")
        message = '(epoch: {epoch}, time: {t})'.format(epoch=epoch,
                                                       t=now)
        for k, v in errors.items():
            message = message + ',{name}:{err}'.format(name=k,
                                                       err=v)

        # Write log message to correct file
        if valid:
            with open(self.valid_log_path, "a") as log_file:
                log_file.write(message + '\n')
        else:
            with open(self.train_log_path, "a") as log_file:
                log_file.write(message + '\n')
        return message

    def plot_errors(self, epochs, errors, title='score', win=None):
        if win is None:
            win = self.vis.line(
                X=epochs,
                Y=errors,
                opts={
                    'title': title,
                    'xlabel': 'epoch',
                    'ylabel': 'score'
                }
            )
        else:
            self.vis.line(
                X=epochs,
                Y=errors,
                opts={
                    'title': title,
                    'xlabel': 'epoch',
                    'ylabel': 'score'
                },
                win=win
            )
        return win

    def plot_sample(self, input_imgs, gts, predictions, classes, win,
                    display_idx=0, k=1, unnormalize=None):
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
            input_img = input_img[:, 0, :, :]*255
        if win is None:
            win = self.vis.image(input_img,
                                 opts={'title': 'sample',
                                       'caption': caption,
                                       'win_size': 256})
        else:
            win = self.vis.image(input_img,
                                 opts={'title': 'sample',
                                       'caption': caption,
                                       'win_size': 256},
                                 win=win)
        return win
