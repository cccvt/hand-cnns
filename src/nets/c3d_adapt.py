from torch import nn
from src.nets.basenet import BaseNet


class C3DAdapt(BaseNet):
    def __init__(self, opt, c3dnet, class_nb):
        super().__init__(opt)
        self.name = 'c3d_adapt'
        self.net = c3dnet
        self.net.fc8 = nn.Linear(4096, class_nb)
    
    def forward(self, x):
        return self.net(x)
