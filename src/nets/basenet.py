import os
import torch

from src.options import error


class BaseNet(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.exp_id)
        self.opt = opt

    def save_net(self, network, network_name, epoch, opt):
        """
        Utility function to save network weights
        """
        save_path = self._netfile_path(network_name, epoch)
        torch.save(network.cpu().state_dict(), save_path)
        if self.opt.use_gpu:
            network.cuda()

    def load_net(self, network, network_name, epoch):
        """
        Utility function to load network weights
        """
        load_path = self._netfile_path(network_name, epoch)
        network.load_state_dict(torch.load(load_path))

    def set_optimizer(self, optim):
        self.optimizer = optim

    def set_criterion(self, crit):
        self.criterion = crit

    def prepare_var(self, tensor):
        tensor = tensor.float()
        if self.opt.use_gpu:
            tensor = tensor.cuda()
        var = torch.autograd.Variable(tensor)
        return var

    def compute_loss(self, output, target):
        # Compute scores
        if self.opt.criterion == 'MSE':
            loss = self.criterion(output, target)
        elif self.opt.criterion == 'CE':
            # CE expects index of class as ground truth input
            target_vals, target_idxs = target.max(1)
            loss = self.criterion(output, target_idxs.view(-1))
        else:
            raise error.ArgumentError(
                '{0} is not among known error\
                functions'.format(self.opt.criterion))
        return loss

    def step_backward(self, loss):
        # Compute gradient and do gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _netfile_path(self, network_name, epoch):
        """
        Constructs path to file where to save/load the network's
        weights
        """
        net_filename = '{net}_epoch{ep}.pth'.format(net=network_name,
                                                    ep=epoch)
        file_path = os.path.join(self.save_dir, net_filename)
        return file_path
