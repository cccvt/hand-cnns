import cv2
import torch
from torchvision import transforms
import torchvision.models as models

from netraining.options import base_options, image_options, train_options, error
from netraining.nets import resnet_adapt, miniresnet
from netraining.netscripts import train
from netraining.utils.normalize import Unnormalize
from netraining.utils import evaluation

from actiondatasets.gteagazeplus import GTEAGazePlus
from actiondatasets.epicsounds import Epic as EpicSounds


def run_training(opt):
    # Normalize as imageNet
    img_means = [0.485, 0.456, 0.406]
    img_stds = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=img_means, std=img_stds)

    # Set input tranformations
    transformations = ([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    final_size = 224
    scale_size = 230
    transformations.append(normalize)
    first_transforms = [
        transforms.Scale(scale_size),
        transforms.RandomCrop(final_size)
    ]
    transformations = first_transforms + transformations

    transform = transforms.Compose(transformations)

    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    if opt.modality == 'flow':
        use_flow = True
    else:
        use_flow = False

    if opt.dataset == 'gteagazeplus':
        # Create dataset
        all_subjects = [
            'Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh'
        ]
        train_seqs, valid_seqs = evaluation.leave_one_out(
            all_subjects, leave_out_idx)
        dataset = GTEAGazePlus(
            transform=transform,
            seqs=train_seqs,
            flow_type=opt.flow_type,
            heatmaps=opt.use_heatmaps,
            heatmap_size=scale_size,
            original_labels=True,
            rescale_flows=opt.rescale_flows,
            use_flow=use_flow)
        valid_dataset = GTEAGazePlus(
            transform=transform,
            seqs=valid_seqs,
            flow_type=opt.flow_type,
            heatmaps=opt.use_heatmaps,
            heatmap_size=scale_size,
            original_labels=True,
            rescale_flows=opt.rescale_flows,
            use_flow=use_flow)
    if opt.dataset == 'epic':
        if opt.modality == 'melspec':
            class_nbs = (20, 50)
            dataset = EpicSounds('train', limit_classes=class_nbs, time_size=4)
            valid_dataset = EpicSounds(
                'val', limit_classes=class_nbs, time_size=4)

    sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # Initialize dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.threads,
        sampler=sampler)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        num_workers=opt.threads)

    # Load model
    if opt.modality == 'flow':
        in_channels = 2
    elif opt.modality == 'melspec':
        in_channels = 1
    else:
        in_channels = None

    resnet = miniresnet.resnet18(pretrained=opt.pretrained)
    model = resnet_adapt.ResNetAdapt(
        opt, resnet, dataset.class_nb, in_channels=in_channels)

    model_params = model.net.parameters()

    optimizer = torch.optim.SGD(
        model_params,
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay)

    if opt.dataset == 'epic':
        criterion = {
            'verb': torch.nn.CrossEntropyLoss(),
            'objects': torch.nn.BCEWithLogitsLoss()
        }
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.set_criterion(criterion)
    model.set_optimizer(optimizer)

    if opt.plateau_scheduler and opt.continue_training:
        raise ValueError('Plateau scheduler and continue training '
                         'are incompatible for now')
    if opt.plateau_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=opt.plateau_factor,
            patience=opt.plateau_patience,
            threshold=opt.plateau_thresh,
            threshold_mode='rel')
        model.set_lr_scheduler(scheduler)

    # Load existing weights, opt.continue_training is epoch to load
    if opt.continue_training:
        if opt.continue_epoch == 0:
            model.net.eval()
            model.load(latest=True)
        else:
            model.load(epoch=opt.continue_epoch)
        # New learning rate for SGD TODO add momentum update
        model.update_optimizer_lr(lr=opt.lr, momentum=opt.momentum)

    train.train_net(
        dataloader,
        model,
        opt,
        valid_dataloader=valid_dataloader,
        visualize=opt.visualize,
        test_aggreg=opt.test_aggreg)


if __name__ == "__main__":
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    image_options.add_image_options(options)
    opt = options.parse()
    run_training(opt)
