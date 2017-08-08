import copy
import torch
import torchvision.models as models

from src.nets import resnet_adapt
from src.utils.filesys import mkdir


def test_save_load_gpu():
    class Opt():
        def __init__(self):
            self.use_gpu = 0
            self.pretrained = 1
            self.checkpoint_dir = 'test'
            self.exp_id = 'res-test'
    opt = Opt()
    resnet = models.resnet18(pretrained=opt.pretrained)
    class_nb = 10
    model = resnet_adapt.ResNetAdapt(opt, resnet, class_nb)

    # Save weights
    model.name = 'test-resnet'
    mkdir(model.save_dir)
    old_state = copy.deepcopy(model.state_dict())

    # Load weights
    model.save(0, opt)
    model.load(0)
    new_state = model.state_dict()
    equal_layers = []
    for key, items in old_state.items():
        layer_eq = torch.eq(items, new_state[key]).float().mean()
        equal_layers.append(layer_eq)

    assert equal_layers == [1]*len(equal_layers)
