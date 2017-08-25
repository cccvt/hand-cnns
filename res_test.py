import cv2
import torch
from torchvision import transforms
import torchvision.models as models
from src.datasets.smthgimage import SmthgImage

from src.netscripts import test
from src.nets import resnet_adapt
from src.options import test_options
from src.utils.normalize import Unnormalize


def run_testing(opt):
    # Normalize as imageNet
    img_means = [0.485, 0.456, 0.406]
    img_stds = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=img_means,
                                     std=img_stds)

    # Compute reverse of normalize transfor
    unnormalize = Unnormalize(mean=img_means, std=img_stds)

    # Set input tranformations
    transformations = ([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    final_size = 224
    if opt.normalize:
        transformations.append(normalize)
    first_transforms = [transforms.Scale(230),
                        transforms.RandomCrop(final_size)]
    transformations = first_transforms + transformations

    transform = transforms.Compose(transformations)
    base_transform = transforms.Compose([
        transforms.Scale(final_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = SmthgImage(split='test', transform=transform,
                         untransform=unnormalize, base_transform=base_transform)

    # Load model
    resnet = models.resnet50()
    model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb)

    optimizer = torch.optim.SGD(model.net.parameters(), lr=1,
                                momentum=0)
    model.set_optimizer(optimizer)
    # Load existing weights, opt.continue_training is epoch to load
    model.net.eval()
    if opt.use_gpu:
        model.net.cuda()
    model.load(load_path=opt.checkpoint_path)

    test.test(dataset, model, opt=opt, frame_nb=opt.frame_nb,
              save_predictions=True)


if __name__ == "__main__":
    opt = test_options.TestOptions().parse()
    run_testing(opt)
