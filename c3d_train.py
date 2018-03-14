import copy

import cv2
import torch
import torchvision

from actiondatasets import smthg
from actiondatasets import smthgv2
from actiondatasets.gteagazeplus import GTEAGazePlus
from actiondatasets.epic import Epic
from actiondatasets.actiondataset import ActionDataset
from videotransforms import video_transforms, volume_transforms, tensor_transforms

from src.nets import c3d, c3d_adapt
from src.nets import i3d, i3d_adapt
from src.nets import i3dense, i3dense_adapt
from src.nets import i3res, i3res_adapt
from src.nets import resnext as i3next
from src.netscripts import train
from src.options import base_options, train_options, video_options
from src.utils import evaluation


def run_training(opt):
    cv2.setNumThreads(0)
    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    scale_size = (256, 342)
    resize_ratio = (244 / 256,
                    1)  # ratio in (final_size e.g. 256/original_size, 1)
    crop_size = (224, 224)
    if opt.use_heatmaps:
        channel_nb = opt.heatmap_nb
    elif opt.use_flow:
        channel_nb = 2
    elif opt.use_objectness:
        channel_nb = 1
    else:
        channel_nb = 3

    # Initialize transforms
    if not opt.use_heatmaps:
        base_transform_list = [
            video_transforms.Resize(crop_size),
            volume_transforms.ClipToTensor(channel_nb=channel_nb)
        ]
        video_transform_list = [
            video_transforms.RandomResize(ratio=resize_ratio),
            video_transforms.RandomRotation(20),
            video_transforms.RandomCrop(crop_size),
            volume_transforms.ClipToTensor(channel_nb=channel_nb)
        ]
    else:
        base_transform_list = [volume_transforms.ToTensor()]
        video_transform_list = [
            tensor_transforms.SpatialRandomCrop(crop_size),
            volume_transforms.ToTensor()
        ]
    base_transform = video_transforms.Compose(base_transform_list)
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
        if opt.multi_weights is not None and len(opt.multi_weights):
            multi = True
        else:
            multi = False
        dataset = GTEAGazePlus(
            heatmaps=opt.use_heatmaps,
            use_objectness=opt.use_objectness,
            heatmap_size=scale_size,
            label_type='cvpr',
            rescale_flows=opt.rescale_flows,
            seqs=train_seqs,
            use_flow=opt.use_flow,
            multi=multi,
            mini_factor=opt.mini_factor)
        val_dataset = GTEAGazePlus(
            heatmaps=opt.use_heatmaps,
            heatmap_size=scale_size,
            label_type='cvpr',
            rescale_flows=opt.rescale_flows,
            seqs=valid_seqs,
            use_flow=opt.use_flow,
            use_objectness=opt.use_objectness,
            multi=multi,
            mini_factor=opt.mini_factor)
    elif opt.dataset == 'epic':
        dataset = Epic('train', split_seen=False, mini_factor=opt.mini_factor)
        val_dataset = Epic(
            'val', split_seen=False, mini_factor=opt.mini_factor)
    elif opt.dataset == 'gteagazeplus_tres':
        all_subjects = ['Alireza', 'Carlos', 'Rahul', 'Yin', 'Shaghayegh']
        train_seqs, valid_seqs = evaluation.leave_one_out(
            all_subjects, leave_out_idx)
        if opt.multi_weights is not None and len(opt.multi_weights):
            multi = True
        else:
            multi = False
        dataset = GTEAGazePlus(
            heatmaps=opt.use_heatmaps,
            use_objectness=opt.use_objectness,
            heatmap_size=scale_size,
            label_type='rubicon',
            rescale_flows=opt.rescale_flows,
            seqs=train_seqs,
            use_flow=opt.use_flow,
            multi=multi,
            mini_factor=opt.mini_factor)
        val_dataset = GTEAGazePlus(
            heatmaps=opt.use_heatmaps,
            heatmap_size=scale_size,
            label_type='rubicon',
            rescale_flows=opt.rescale_flows,
            seqs=valid_seqs,
            use_flow=opt.use_flow,
            use_objectness=opt.use_objectness,
            multi=multi,
            mini_factor=opt.mini_factor)
    elif opt.dataset == 'smthg':
        dataset = smthg.Smthg(
            flow_type=opt.flow_type,
            rescale_flows=opt.rescale_flows,
            use_flow=opt.use_flow,
            split='train',
            mini_factor=opt.mini_factor)

        val_dataset = smthg.Smthg(
            flow_type=opt.flow_type,
            rescale_flows=opt.rescale_flows,
            split='valid',
            use_flow=opt.use_flow,
            mini_factor=opt.mini_factor)
    elif opt.dataset == 'smthgv2':
        dataset = smthgv2.SmthgV2(
            use_flow=opt.use_flow,
            split='train',
            rescale_flows=opt.rescale_flows,
            mini_factor=opt.mini_factor)

        val_dataset = smthgv2.SmthgV2(
            split='valid',
            use_flow=opt.use_flow,
            rescale_flows=opt.rescale_flows,
            mini_factor=opt.mini_factor)
    else:
        raise ValueError('the opt.dataset name provided {0} is not handled'
                         'by this script'.format(opt.dataset))
    action_dataset = ActionDataset(
        dataset,
        clip_size=opt.clip_size,
        clip_spacing=opt.clip_spacing,
        transform=video_transform)
    val_action_dataset = ActionDataset(
        val_dataset,
        clip_size=opt.clip_size,
        clip_spacing=opt.clip_spacing,
        transform=base_transform)

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
            # Loads RGB weights and then adapts network
            i3dnet = i3d.I3D(class_nb=400, modality='rgb', dropout_rate=0.5)
            if opt.pretrained:
                i3dnet.load_state_dict(torch.load('data/i3d_rgb.pth'))
            model = i3d_adapt.I3DAdapt(
                opt, i3dnet, action_dataset.class_nb, in_channels=channel_nb)
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
    elif opt.network == 'i3next':
        i3resnext = i3next.resnext101(
            sample_duration=opt.clip_size, sample_size=crop_size[0])
        checkpoint = torch.load('data/resnext-101-kinetics.pth')

        # Load state_dict after removing 'module.' prefix in keys
        state_dict = checkpoint['state_dict']
        state_dict = {
            '.'.join(key.split('.')[1:]): value
            for key, value in state_dict.items()
        }
        i3resnext.load_state_dict(state_dict)
        model = i3res_adapt.I3ResAdapt(
            opt,
            i3resnext,
            class_nb=action_dataset.class_nb,
            channel_nb=channel_nb,
            resnext=True)
    else:
        err_msg = 'network should be in [i3res|i3dense|i3d|c3d but got {}]'.format(
            opt.network)
        raise ValueError(err_msg)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.net.parameters(), lr=opt.lr, momentum=opt.momentum)

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
