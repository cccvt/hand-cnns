import os
import torch


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

    def _netfile_path(self, network_name, epoch):
        """
        Constructs path to file where to save/load the network's
        weights
        """
        net_filename = '{net}_epoch{ep}.pth'.format(net=network_name,
                                                    ep=epoch)
        file_path = os.path.join(self.save_dir, net_filename)
        return file_path
