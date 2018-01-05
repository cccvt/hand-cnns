import cv2
import torch
from torchvision import transforms
import torchvision.models as models
from src.datasets.gteagazeplusimage import GTEAGazePlusImage
from src.datasets.smthgimage import SmthgImage

from src.netscripts import test
from src.nets import resnet_adapt
from src.options import base_options, image_options, test_options
from src.utils.normalize import Unnormalize
from src.utils import evaluation


def run_testing(opt):
    # Normalize as imageNet
    img_means = [0.485, 0.456, 0.406]
    img_stds = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=img_means, std=img_stds)

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
    first_transforms = [
        transforms.Scale(230),
        transforms.RandomCrop(final_size)
    ]
    transformations = first_transforms + transformations

    transform = transforms.Compose(transformations)
    base_transform = transforms.Compose([
        transforms.Scale(final_size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'smthg':
        dataset = SmthgImage(
            split='test',
            transform=transform,
            untransform=unnormalize,
            base_transform=base_transform)
    elif opt.dataset == 'gteagazeplus':
        all_subjects = [
            'Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh'
        ]
        train_seqs, valid_seqs = evaluation.leave_one_out(
            all_subjects, opt.leave_out)
        dataset = GTEAGazePlusImage(
            base_transform=base_transform,
            transform=transform,
            untransform=unnormalize,
            original_labels=True,
            seqs=valid_seqs)

    # Load model
    resnet = models.resnet34()
    model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb)

    optimizer = torch.optim.SGD(model.net.parameters(), lr=1, momentum=0)
    model.set_optimizer(optimizer)
    # Load existing weights, opt.continue_training is epoch to load
    model.net.eval()
    if opt.use_gpu:
        model.net.cuda()
    model.load(load_path=opt.checkpoint_path)

    mean_scores = test.test(
        dataset, model, opt=opt, frame_nb=opt.frame_nb, save_predictions=True)
    print(mean_scores)


if __name__ == "__main__":
    # Initialize base options
    options = base_options.BaseOptions()

    # Add test options and parse
    test_options.add_test_options(options)
    image_options.add_image_options(options)
    opt = options.parse()
    run_testing(opt)
