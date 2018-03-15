import numbers
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn
from netraining.nets.basenet import BaseNet
from netraining.nets.i3d import Unit3Dpy


class I3DAdapt(BaseNet):
    def __init__(self, opt, i3dnet, class_nb, in_channels=None):
        """
        Args:
        class_nb: either an int in case of simple classification
            giving the number of output classes or the ordered list
            of class_nbs for each of the predictive tasks
        """
        super().__init__(opt)
        self.name = 'i3d_adapt'
        self.class_nb = class_nb
        self.net = i3dnet

        last_feature_channels = 1024
        if isinstance(class_nb, numbers.Number):
            print(class_nb)
            self.net.conv3d_0c_1x1 = Unit3Dpy(
                in_channels=last_feature_channels,
                out_channels=self.class_nb,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)
        else:
            self.net.conv3d_0c_1x1 = MultiConvClassifier(
                last_feature_channels, class_nb)

        if in_channels is not None:
            # Make sure different nb of in_channels is requested
            old_inchannels = self.net.conv3d_1a_7x7.conv3d.in_channels
            if old_inchannels != in_channels:
                conv3d_1a_7x7 = Unit3Dpy(
                    out_channels=64,
                    in_channels=in_channels,
                    kernel_size=(7, 7, 7),
                    stride=(2, 2, 2),
                    padding='SAME')

                # Transfer conv weights
                weight_3d = self.net.conv3d_1a_7x7.conv3d.weight.data
                new_weight_3d = weight_3d.mean(1)
                new_weight_3d = new_weight_3d.unsqueeze(1).repeat(
                    1, in_channels, 1, 1, 1)
                new_weight_3d = new_weight_3d * old_inchannels / in_channels
                conv3d_1a_7x7.conv3d.weight = nn.parameter.Parameter(
                    new_weight_3d)

                # Transfer batch norm params
                conv3d_1a_7x7.batch3d.running_mean = self.net.conv3d_1a_7x7.batch3d.running_mean
                conv3d_1a_7x7.batch3d.running_var = self.net.conv3d_1a_7x7.batch3d.running_var
                conv3d_1a_7x7.batch3d.weight = self.net.conv3d_1a_7x7.batch3d.weight
                conv3d_1a_7x7.batch3d.bias = self.net.conv3d_1a_7x7.batch3d.bias

                self.net.conv3d_1a_7x7 = conv3d_1a_7x7


class MultiConvClassifier(nn.Module):
    def __init__(self, in_features, class_nbs):
        super().__init__()
        self.in_features = in_features
        self.class_nbs = class_nbs
        classifier_names = []
        for classifier_idx, class_nb in enumerate(class_nbs):
            classifier = Unit3Dpy(
                in_channels=in_features,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)

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
