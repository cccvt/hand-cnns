import copy

from torch import nn
from src.nets.basenet import BaseNet


class I3DenseAdapt(BaseNet):
    def __init__(self, opt, i3densenet, class_nb, channel_nb=3):
        super().__init__(opt)
        self.name = 'i3dense_adapt'
        self.class_nb = class_nb
        self.net = i3densenet
        if channel_nb != 3:
            self.net.first_conv = nn.Conv3d(
                channel_nb,
                64,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False)
        self.net.classifier = nn.Conv3d(
            in_channels=1024,
            out_channels=class_nb,
            kernel_size=(1, 1, 1),
            bias=True)
        self.net.conv_class = True
