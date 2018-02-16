import copy
import numbers

from torch import nn
from src.nets.basenet import BaseNet


class I3ResAdapt(BaseNet):
    def __init__(self, opt, i3resnet, class_nb, channel_nb=3):
        """
        Args:
            class_nb: if int the unique number of predicted classes
                if tuple, the ordered list of number of classes
        """
        super().__init__(opt)
        self.name = 'i3res_adapt'
        self.class_nb = class_nb
        self.net = i3resnet
        if channel_nb != 3:
            self.net.conv1 = nn.Conv3d(
                channel_nb,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False)
        self.net.conv_class = True

        if isinstance(class_nb, numbers.Number):
            self.net.classifier = nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            classifiers = []
            for nb in class_nb:
                classifier = nn.Conv3d(
                    in_channels=2048,
                    out_channels=class_nb,
                    kernel_size=(1, 1, 1),
                    bias=True)
                classifiers.append(classifier)
            self.net.classifier = classifiers
