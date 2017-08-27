from torch import nn
from src.nets.basenet import BaseNet


class C3DAdapt(BaseNet):
    def __init__(self, opt, c3dnet, class_nb, in_channels=3):
        super().__init__(opt)
        self.name = 'c3d_adapt'
        self.net = c3dnet
        self.net.fc8 = nn.Linear(4096, class_nb)
        if in_channels != 3:
            self.net.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, in_channels, in_channels), padding=(1, 1, 1))
    
    def forward(self, x):
        return self.net(x)
