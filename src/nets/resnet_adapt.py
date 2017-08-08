from torch import nn
from src.nets.basenet import BaseNet


class ResNetAdapt(BaseNet):
    def __init__(self, opt, resnet, class_nb):
        super().__init__(opt)
        self.name = 'resnet_adapt'
        self.net = resnet
        self.net.fc = nn.Linear(512, class_nb)
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
        base_params = filter(lambda p: id(p) not in ids,
                             self.net.parameters())
        params = []
        params.append({'params': base_params})
        for layer_name in new_layers:
            layer = getattr(self.net, layer_name)
            params.append({'params': layer.parameters(),
                           'lr': lr})
        return params
