from torch import nn
from src.nets.basenet import BaseNet


class ResNetAdapt(BaseNet):
    def __init__(self, opt, resnet, nb_out):
        super().__init__(opt)
        self.name = 'resnet_adapt'
        self.resnet = resnet
        self.resnet.fc = nn.Linear(512, nb_out)
        self.input_size = (224, 224)

    def forward(self, x):
        return self.resnet(x)

    def lr_params(self, lr=0.01, new_layers=['fc']):
        """
        Fixes the learning rate of all layers in new_layers
        Returns a list of params to be passed as first argument to an
        optimizer

        :param new_layers: layer names for which to modify the lr
        :param lr: learning rate to apply to these layers
        """
        ids = []
        for layer_name in new_layers:
            # add unique parameter ids to list
            layer = getattr(self.resnet, layer_name)
            full_ids = list(map(id, layer.parameters()))
            ids = ids + full_ids
        base_params = filter(lambda p: id(p) not in ids,
                             self.resnet.parameters())
        params = []
        params.append({'params': base_params})
        for layer_name in new_layers:
            layer = getattr(self.resnet, layer_name)
            params.append({'params': layer.parameters(),
                           'lr': lr})
        return params

    def save(self, epoch):
        self.save_net(self.resnet, self.name, epoch, self.opt)

    def load(self, epoch):
        self.load_net(self.resnet, self.name, epoch)
