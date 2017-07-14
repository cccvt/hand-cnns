import os
import time
import visdom


class Visualize():
    def __init__(self, opt):
        self.vis = visdom.Visdom()
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
