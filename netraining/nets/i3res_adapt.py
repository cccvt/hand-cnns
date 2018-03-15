import copy
import numbers

from torch import nn
from netraining.nets.basenet import BaseNet


class I3ResAdapt(BaseNet):
    def __init__(self, opt, i3resnet, class_nb, channel_nb=3, resnext=False):
        """
        Args:
            class_nb: if int the unique number of predicted classes
                if tuple, the ordered list of number of classes
        """
        super().__init__(opt)
        if resnext:
            self.name = 'i3next_adapt'
        else:
            self.name = 'i3res_adapt'
        self.class_nb = class_nb
        self.net = i3resnet
        self.net.conv_class = True
        if channel_nb != 3:
            self.net.conv1 = nn.Conv3d(
                channel_nb,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False)

        if isinstance(class_nb, numbers.Number):
            self.net.classifier = nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)
        else:
            classifier = MultiConvClassifier(class_nb, in_features=2048)
        self.net.classifier = classifier


class MultiConvClassifier(nn.Module):
    def __init__(self, class_nbs, in_features=2048):
        super().__init__()
        self.in_features = in_features
        self.class_nbs = class_nbs
        classifier_names = []
        for classifier_idx, class_nb in enumerate(class_nbs):
            classifier = nn.Conv3d(
                in_channels=in_features,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True)

            classifier_name = 'classifier_{}'.format(classifier_idx)
            classifier_names.append(classifier_name)
            self.add_module(classifier_name, classifier)
        self.classifier_names = classifier_names

    def forward(self, inp):
        outputs = []
        for classifier_name in self.classifier_names:
            classifier = getattr(self, classifier_name)
            out = classifier(inp)
            outputs.append(out)
        return outputs
