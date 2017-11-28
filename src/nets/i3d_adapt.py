from torch import nn
from src.nets.basenet import BaseNet
from src.nets.i3d import Unit3Dpy


class I3DAdapt(BaseNet):
    def __init__(self, opt, i3dnet, class_nb):
        super().__init__(opt)
        self.name = 'i3d_adapt'
        self.class_nb = class_nb
        self.net = i3dnet
        self.net.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.class_nb,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)

    def forward(self, x):
        return self.net(x)
