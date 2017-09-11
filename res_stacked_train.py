import cv2
import torch
import torchvision.models as models

from src.datasets import gun
from src.datasets.gteagazeplusvideo import GTEAGazePlusVideo
from src.datasets.smthgvideo import SmthgVideo
from src.datasets.utils import video_transforms, stack_transforms
from src.options import base_options, video_options, train_options, error, stack_options
from src.nets import resnet_adapt
from src.netscripts import train
from src.utils.normalize import Unnormalize
from src.utils import evaluation


def run_training(opt):
    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    # Set input tranformations
    scale_size = (240, 320)
    final_size = 224
    if opt.use_flow:
        channel_nb = 2
    else:
        channel_nb = 3

    base_transform_list = [video_transforms.Scale(final_size),
                           stack_transforms.ToStackedTensor(channel_nb=channel_nb)]
    base_transform = video_transforms.Compose(base_transform_list)
    video_transform_list = [video_transforms.Scale(scale_size),
                            video_transforms.RandomCrop(
                                (final_size, final_size)),
                            stack_transforms.ToStackedTensor(channel_nb=channel_nb)]
    video_transform = video_transforms.Compose(video_transform_list)

    # Initialize dataset
    if opt.dataset == 'gteagazeplus':
        all_subjects = ['Ahmad', 'Alireza', 'Carlos',
                        'Rahul', 'Yin', 'Shaghayegh']
        train_seqs, valid_seqs = evaluation.leave_one_out(all_subjects,
                                                          leave_out_idx)
        dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                    use_video=False, clip_size=opt.stack_nb,
                                    original_labels=True,
                                    seqs=train_seqs, use_flow=opt.use_flow)
        val_dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                        base_transform=base_transform,
                                        use_video=False,
                                        clip_size=opt.stack_nb,
                                        original_labels=True,
                                        seqs=valid_seqs,
                                        use_flow=opt.use_flow)
    elif opt.dataset == 'smthgsmthg':
        dataset = SmthgVideo(video_transform=video_transform,
                             clip_size=opt.stack_nb, split='train',
                             use_flow=opt.use_flow,
                             frame_spacing=opt.clip_spacing)

        val_dataset = SmthgVideo(video_transform=video_transform,
                                 clip_size=opt.stack_nb, split='valid',
                                 base_transform=base_transform,
                                 use_flow=opt.use_flow)
    else:
        raise ValueError('the opt.dataset name provided {0} is not handled\
                         by this script'.format(opt._dataset))

    print('Dataset size : {0}'.format(len(dataset)))

    # Initialize sampler
    if opt.weighted_training:
        weights = [1 / k for k in dataset.class_counts]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,
                                                                 len(dataset))
    else:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)

    # Initialize dataloaders
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=opt.batch_size,
        num_workers=opt.threads)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, batch_size=opt.batch_size,
        num_workers=opt.threads)

    # Load model
    resnet = models.resnet50(pretrained=opt.pretrained)
    model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb,
                                     in_channels=channel_nb * opt.stack_nb)
    if opt.lr != opt.new_lr:
        model_params = model.lr_params(lr=opt.new_lr)
    else:
        model_params = model.net.parameters()

    optimizer = torch.optim.SGD(model_params, lr=opt.lr,
                                momentum=opt.momentum)

    if opt.criterion == 'MSE':
        criterion = torch.nn.MSELoss()
    elif opt.criterion == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise error.ArgumentError(
            '{0} is not among known error functions'.format(opt.criterion))

    model.set_criterion(criterion)
    model.set_optimizer(optimizer)

    # Load existing weights, opt.continue_training is epoch to load
    if opt.continue_training:
        if opt.continue_epoch == 0:
            model.net.eval()
            model.load('latest')
        else:
            model.load(opt.continue_epoch)

    train.train_net(dataloader, model, opt,
                    valid_dataloader=val_dataloader,
                    visualize=opt.visualize,
                    test_aggreg=opt.test_aggreg)


if __name__ == "__main__":
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)
    stack_options.add_stack_options(options)
    opt = options.parse()
    run_training(opt)
