import numbers
from torch import nn
from src.nets.basenet import BaseNet


class ResNetAdapt(BaseNet):
    def __init__(self, opt, resnet, class_nb, in_channels=None):
        """
        Args:
        class_nb: either an int in case of simple classification
        giving the number of output classes or the ordered list
        of class_nbs for each of the predictive tasks
        """
        super().__init__(opt)
        self.name = 'resnet_adapt'
        self.net = resnet
        in_features_nb = self.net.fc.in_features
        if isinstance(class_nb, numbers.Number):
            self.net.fc = nn.Linear(in_features_nb, class_nb)
        else:
            self.net.fc = MultiClassifier(in_features_nb, class_nb)
        if in_channels is not None:
            conv1 = self.net.conv1
            self.net.conv1 = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                dilation=conv1.dilation,
                bias=conv1.bias)
        self.input_size = (224, 224)

    def lr_params(self, lr=0.01, new_layers=['fc']):
        """
        Fixes the learning rate of all layers in new_layers
        Returns a list of params to be passed as first argument to an
        optimizer

        Args:
        new_layers (list of str): layer names for which to modify the lr
        lr (float): learning rate to apply to these layers
        """
        ids = []
        for layer_name in new_layers:
            # add unique parameter ids to list
            layer = getattr(self.net, layer_name)
            full_ids = list(map(id, layer.parameters()))
            ids = ids + full_ids
        base_params = filter(lambda p: id(p) not in ids, self.net.parameters())
        params = []
        params.append({'params': base_params})
        for layer_name in new_layers:
            layer = getattr(self.net, layer_name)
            params.append({'params': layer.parameters(), 'lr': lr})
        return params


class MultiClassifier(nn.Module):
    def __init__(self, in_features, class_nbs):
        super().__init__()
        self.in_features = in_features
        self.class_nbs = class_nbs
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
