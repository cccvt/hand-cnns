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
        linear3d = copy.deepcopy(self.net.classifier)
        self.net.classifier = nn.Linear(linear3d.in_features, class_nb)
        import pdb
        pdb.set_trace()

    def forward(self, x):
        return self.net(x)
