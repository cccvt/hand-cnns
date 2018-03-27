import copy

import numbers
import torch
from torch import nn
from src.nets.basenet import BaseNet


class ResNetJoint(BaseNet):
    def __init__(self, opt, resnet, in_channels=None, class_nb, resnext=False):
        """
        Args:
        class_nb: either an int in case of simple classification
        giving the number of output classes or the ordered list
        of class_nbs for each of the predictive tasks
        """
        super().__init__(opt)
        if resnext:
            self.name = 'resnext_adapt'
        else:
            self.name = 'resnet_adapt'
        self.net = resnet
        in_features_nb = self.net.fc.in_features
        if not isinstance(in_channels, numbers.Number):
            resnets = []
            for in_channel in in_channels:
                new_resnet = copy.deepcopy(resnet)
                if in_channel != 3:
                    conv1 = self.new_resnet.conv1
                    self.net.conv1 = nn.Conv2d(
                        in_channels,
                        conv1.out_channels,
                        conv1.kernel_size,
                        stride=conv1.stride,
                        padding=conv1.padding,
                        dilation=conv1.dilation,
                        bias=conv1.bias)
            # TODO add feature concatenation
            self.net.fc = MultiLinearClassifier(in_features_nb, class_nb)
        self.input_size = (224, 224)


class MultiLinearClassifier(nn.Module):
    def __init__(self, in_features, class_nbs):
        super().__init__()
        self.in_features = in_features
        classifier_names = []
        for classifier_idx, class_nb in enumerate(class_nbs):
            classifier = nn.Linear(in_features, class_nb)
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
