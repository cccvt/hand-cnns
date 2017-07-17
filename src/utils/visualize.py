import os
import time
import visdom


class Visualize():
    def __init__(self, opt):
        self.vis = visdom.Visdom()
        self.opt = opt
        self.win = None
        self.log_name = os.path.join(opt.checkpoint_dir,
                                     opt.exp_id, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('==== Training loss log at {0} ====\n'.format(now))

    def log_errors(self, epoch, errors):
        now = time.strftime("%c")
        message = '(epoch: {epoch}, time: {t})'.format(epoch=epoch,
                                                       t=now)
        for k, v in errors.items():
            message = message + ',{name}:{err}'.format(name=k,
                                                       err=v)

        with open(self.log_name, "a") as log_file:
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
                    'ylabel': 'loss'
                }
            )
        else:
            self.vis.line(
                X=epochs,
                Y=errors,
                opts={
                    'title': title,
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                },
                win=win
            )
        return win

    def plot_sample(self, input_imgs, gts, predictions, classes, win,
                    display_idx=1, k=1, unnormalize=None):
        """
        Plots in visdom one image with predicted and ground truth labels
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
        pred_classes = classes[int(pred_val[0])]
        pred_string = 'pred class : ' + pred_classes

        real_score, real_class = gts[display_idx].max(0)
        real_class_str = classes[int(real_class[0])]

        real_string = 'True class : ' + real_class_str

        caption = pred_string + '\n' + real_string

        if win is None:
            win = self.vis.image(input_img, opts={'title': 'sample',
                                                  'caption': caption})
        else:
            win = self.vis.image(input_img, opts={'title': 'sample',
                                                  'caption': caption},
                                 win=win)
        return win
