from collections import OrderedDict
import os
import traceback

import torch

from netraining.options import error


class BaseNet():
    def __init__(self, opt):
        super().__init__()
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.exp_id)
        self.name = None
        self.net = None
        self.opt = opt
        self.lr_scheduler = None
        self.optimzer = None

    def save(self, epoch, opt, latest=False):
        """
        Utility function to save network weights
        If necessary, network should be stored as self.net property
        otherwise, uses the layers at the first level
        (self.fc for instance)
        """
        if latest:
            # Saves latest epoch with "latest" in path
            save_path = self._netfile_path(self.name, 'latest')
        else:
            save_path = self._netfile_path(self.name, epoch)

        self.net.eval()
        if self.optimizer is not None:
            optimizer_state = self.optimizer.state_dict()
        else:
            optimizer_state = None

        checkpoint = {
            'net': self.net.cpu().state_dict(),
            'epoch': epoch,
            'optimizer': optimizer_state
        }
        torch.save(checkpoint, save_path)

        if self.opt.use_gpu:
            self.net.cuda()

    def load(self, epoch=0, load_path=None, latest=False, features=False):
        """
        Utility function to load network weights

        Args:
        load_path: path of checkpoint to load, is set, epoch and
        latest are ignored
        epoch: epoch to load
        latest: whether to use file with 'latest' suffix, if true
        epoch is ignored
        """
        if load_path is None:
            # If load_path not specified load either latest or by epoch
            if latest:
                checkpoint_path = self._netfile_path(self.name, 'latest')
            else:
                checkpoint_path = self._netfile_path(self.name, epoch)
        else:
            checkpoint_path = load_path

        self.net.eval()

        # Load checkpoint state
        checkpoint = torch.load(checkpoint_path)
        if load_path is None:
            if epoch > 0:
                assert checkpoint['epoch'] == epoch, '{} should be {}'.format(
                    checkpoint['epoch'], epoch)
            else:
                epoch = checkpoint['epoch']
        if features:
            strict = False
        else:
            strict = True
        self.net.load_state_dict(checkpoint['net'], strict=strict)
        if not features:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded net from epoch {0}'.format(epoch))
        return epoch

    def update_optimizer(self, lr, momentum, verbose=True):
        """Set new lr and momentum to optimizer
        """
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] != lr:
                lr_message = 'Changing lr from {} to {}'.format(
                    param_group['lr'], lr)
                param_group['lr'] = lr
            else:
                lr_message = 'Learning rate is unchanged :{}'.format(lr)
            if param_group['momentum'] != momentum:
                param_group['momentum'] = momentum
                mom_message = 'Changing momentum from {} to {}'.format(
                    param_group['momentum'], momentum)
            else:
                mom_message = 'Momentum is unchanged :{}'.format(momentum)
            if verbose:
                print(lr_message)
                print(mom_message)

    def set_optimizer(self, optim):
        self.optimizer = optim

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def set_criterion(self, crit):
        self.criterion = crit

    def scheduler_step(self, loss):
        """Function to call at the end of training epoch
        Args:
        loss: New value for loss to monitor
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(loss)

        # Get new learning rate and return it
        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(param_group['lr'])
        if not lrs.count(lrs[0]) == len(lrs):
            raise ValueError('All group learning rates should be the same for '
                             'network groups but got {}'.format(lrs))
        return lrs[0]

    def prepare_var(self, tensor, volatile=False, requires_grad=False):
        """Converts tensors or list of tensors for network processing
        """
        # tensor should be of type float otherwise cuda error
        if isinstance(tensor, (tuple, list)):
            variables = []
            for tens in tensor:
                tens = tens.cuda()
                var = torch.autograd.Variable(
                    tens, volatile=volatile, requires_grad=requires_grad)
                variables.append(var)
            return variables
        elif isinstance(tensor, dict):
            variables = {}
            for key, tens in tensor.items():
                tens = tens.cuda()
                var = torch.autograd.Variable(
                    tens, volatile=volatile, requires_grad=requires_grad)
                variables[key] = var
            return variables

        else:
            if self.opt.use_gpu:
                tensor = tensor.cuda()
            var = torch.autograd.Variable(
                tensor, volatile=volatile, requires_grad=requires_grad)
            return var

    def compute_loss(self, output, target):
        # Compute scores
        if self.opt.criterion == 'MSE':
            loss = self.criterion(output, target)
        elif self.opt.criterion == 'CE':
            # CE expects index of class as ground truth input
            if isinstance(target, (list, tuple)):
                losses = []
                for idx, out in enumerate(output):
                    targ = target[idx]
                    target_vals, target_idxs = targ.max(1)
                    loss = self.criterion(out, target_idxs.view(-1))
                    losses.append(loss)
                return losses
            elif isinstance(target, dict):
                losses = OrderedDict()
                for idx, (target_name,
                          target_value) in enumerate(target.items()):
                    out = output[idx]
                    target_vals, target_idxs = target_value.max(1)
                    if isinstance(self.criterion, dict):
                        # If cross entropy loss, expects indexes
                        if isinstance(self.criterion[target_name],
                                      torch.nn.CrossEntropyLoss):
                            losses[target_name] = self.criterion[target_name](
                                out, target_idxs.view(-1))
                        elif isinstance(self.criterion[target_name],
                                        (torch.nn.MultiLabelMarginLoss,
                                         torch.nn.MultiLabelSoftMarginLoss,
                                         torch.nn.BCEWithLogitsLoss)):
                            losses[target_name] = self.criterion[target_name](
                                out, target_value.float())
                        else:
                            err_msg = ('{} not supported'.format(
                                self.criterion[target_name]))
                            raise ValueError(err_msg)

                    else:
                        losses[target_name] = self.criterion(
                            out, target_idxs.view(-1))
                return losses

            else:
                target_vals, target_idxs = target.max(1)
                loss = self.criterion(output, target_idxs.view(-1))
        else:
            raise error.ArgumentError('{0} is not among known error\
                    functions'.format(self.opt.criterion))
        return loss

    def step_backward(self, loss, multi_weights=None):
        # Compute gradient and do gradient step
        self.optimizer.zero_grad()
        if isinstance(loss, (list, tuple)):
            loss = sum(
                [weight * los for weight, los in zip(multi_weights, loss)])
        if isinstance(loss, dict):
            assert multi_weights is not None, (
                'When loss is dict,'
                'multi_weights should be defined')
            loss = sum([
                weight * los
                for weight, los in zip(multi_weights, list(loss.values()))
            ])
        loss.backward()
        self.optimizer.step()

    def _netfile_path(self, network_name, epoch):
        """
        Constructs path to file where to save/load the network's
        weights
        """
        if epoch is int:
            net_filename = '{net}_epoch_{ep:04d}.pth'.format(
                net=network_name, ep=int(epoch))
        else:
            net_filename = '{net}_epoch_{ep}.pth'.format(
                net=network_name, ep=epoch)
            file_path = os.path.join(self.save_dir, net_filename)
        return file_path
