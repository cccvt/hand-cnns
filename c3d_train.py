import copy

import cv2
import torch
import torchvision

from src.datasets.utils import video_transforms, volume_transforms
from src.nets import c3d, c3d_adapt
from src.nets import i3d, i3d_adapt
from src.nets import i3dense, i3dense_adapt
from src.nets import i3res, i3res_adapt
from src.netscripts import train
from src.options import base_options, train_options, video_options
from src.utils import evaluation

from actiondatasets import smthg
from actiondatasets.gteagazeplus import GTEAGazePlus
from actiondatasets.actiondataset import ActionDataset


def run_training(opt):
    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    scale_size = (256, 342)
    crop_size = (224, 224)
    if opt.use_flow:
        channel_nb = 2
    else:
        channel_nb = 3
    base_transform_list = [
        video_transforms.Scale(crop_size),
        volume_transforms.ToTensor(channel_nb=channel_nb)
    ]
    base_transform = video_transforms.Compose(base_transform_list)
    video_transform_list = [
        video_transforms.Scale(scale_size),
        video_transforms.RandomCrop(crop_size),
        volume_transforms.ToTensor(channel_nb=channel_nb)
    ]
    video_transform = video_transforms.Compose(video_transform_list)

    # Initialize datasets
    leave_out_idx = opt.leave_out

    # Initialize dataset
    if opt.dataset == 'gteagazeplus':
        all_subjects = [
            'Ahmad', 'Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh'
        ]
        train_seqs, valid_seqs = evaluation.leave_one_out(
            all_subjects, leave_out_idx)
        dataset = GTEAGazePlus(
            flow_type=opt.flow_type,
            original_labels=True,
            rescale_flows=opt.rescale_flows,
            seqs=train_seqs,
            use_flow=opt.use_flow)
        val_dataset = GTEAGazePlus(
            flow_type=opt.flow_type,
            original_labels=True,
            rescale_flows=opt.rescale_flows,
            seqs=valid_seqs,
            use_flow=opt.use_flow)
    elif opt.dataset == 'smthg':
        dataset = smthg.Smthg(
            flow_type=opt.flow_type,
            rescale_flows=opt.rescale_flows,
            use_flow=opt.use_flow,
            split='train')

        val_dataset = smthg.Smthg(
            flow_type=opt.flow_type,
            rescale_flows=opt.rescale_flows,
            split='valid',
            use_flow=opt.use_flow)
    else:
        raise ValueError('the opt.dataset name provided {0} is not handled\
                by this script'.format(opt.dataset))
    action_dataset = ActionDataset(
        dataset,
        base_transform=base_transform,
        clip_size=opt.clip_size,
        transform=video_transform)
    val_action_dataset = ActionDataset(
        val_dataset,
        base_transform=base_transform,
        clip_size=opt.clip_size,
        transform=video_transform)

    # Initialize sampler
    if opt.weighted_training:
        weights = [1 / k for k in dataset.class_counts]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(action_dataset))
    else:
        sampler = torch.utils.data.sampler.RandomSampler(action_dataset)

    # Initialize dataloaders
    dataloader = torch.utils.data.DataLoader(
        action_dataset,
        sampler=sampler,
        batch_size=opt.batch_size,
        num_workers=opt.threads)

    val_dataloader = torch.utils.data.DataLoader(
        val_action_dataset,
        shuffle=False,
        batch_size=opt.batch_size,
        num_workers=opt.threads)

    # Initialize C3D neural network
    if opt.network == 'c3d':
        c3dnet = c3d.C3D()
        if opt.pretrained:
            c3dnet.load_state_dict(torch.load('data/c3d.pickle'))
        model = c3d_adapt.C3DAdapt(
            opt, c3dnet, action_dataset.class_nb, in_channels=channel_nb)
    elif opt.network == 'i3d':
        if opt.use_flow:
            i3dnet = i3d.I3D(class_nb=400, modality='flow', dropout_rate=0.5)
            if opt.pretrained:
                i3dnet.load_state_dict(torch.load('data/i3d_flow.pth'))
            model = i3d_adapt.I3DAdapt(opt, i3dnet, action_dataset.class_nb)
        else:
            i3dnet = i3d.I3D(class_nb=400, modality='rgb', dropout_rate=0.5)
            if opt.pretrained:
                i3dnet.load_state_dict(torch.load('data/i3d_rgb.pth'))
            model = i3d_adapt.I3DAdapt(opt, i3dnet, action_dataset.class_nb)
    elif opt.network == 'i3dense':
        densenet = torchvision.models.densenet121(pretrained=True)
        i3densenet = i3dense.I3DenseNet(
            copy.deepcopy(densenet), opt.clip_size, inflate_block_convs=True)
        model = i3dense_adapt.I3DenseAdapt(
            opt, i3densenet, action_dataset.class_nb, channel_nb=channel_nb)
    elif opt.network == 'i3res':
        resnet = torchvision.models.resnet50(pretrained=True)
        i3resnet = i3res.I3ResNet(resnet, frame_nb=opt.clip_size)
        model = i3res_adapt.I3ResAdapt(
            opt,
            i3resnet,
            class_nb=action_dataset.class_nb,
            channel_nb=channel_nb)
    else:
        raise ValueError(
            'network should be in [i3res|i3dense|i3d|c3d but got {}]').format(
                opt.network)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.net.parameters(), lr=opt.lr, momentum=opt.momentum)

    model.set_criterion(criterion)
    model.set_optimizer(optimizer)

    # Use multiple GPUS
    if opt.gpu_parallel:
        available_gpus = torch.cuda.device_count()
        device_ids = list(range(opt.gpu_nb))
        print('Using {} out of {} available GPUs'.format(
            len(device_ids), available_gpus))
        model.net = torch.nn.DataParallel(model.net, device_ids=device_ids)

    # Load existing weights, opt.continue_training is epoch to load
    if opt.continue_training:
        if opt.continue_epoch == 0:
            model.net.eval()
            model.load(latest=True)
        else:
            model.load(epoch=opt.continue_epoch)

        # New learning rate for SGD TODO add momentum update
        model.update_optimizer(lr=opt.lr, momentum=opt.momentum)

    train.train_net(
        dataloader,
        model,
        opt,
        valid_dataloader=val_dataloader,
        visualize=opt.visualize,
        test_aggreg=opt.test_aggreg)


if __name__ == '__main__':
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)
    opt = options.parse()
    run_training(opt)
