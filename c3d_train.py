import cv2
import torch

from src.datasets.gteagazeplusvideo import GTEAGazePlusVideo
from src.datasets.smthgvideo import SmthgVideo
from src.datasets.utils import video_transforms, volume_transforms
from src.nets import c3d, c3d_adapt
from src.netscripts import train
from src.options import base_options, train_options, video_options
from src.utils import evaluation


def run_training(opt):
    # Index of sequence item to leave out for validation
    leave_out_idx = opt.leave_out

    scale_size = (128, 171)
    crop_size = (112, 112)
    if opt.use_flow:
        channel_nb = 2
    else:
        channel_nb = 3
    base_transform_list = [video_transforms.Scale(crop_size),
                           volume_transforms.ToTensor(channel_nb=channel_nb)]
    base_transform = video_transforms.Compose(base_transform_list)
    video_transform_list = [video_transforms.Scale(scale_size),
                            video_transforms.RandomCrop(crop_size),
                            volume_transforms.ToTensor(channel_nb=channel_nb)]
    video_transform = video_transforms.Compose(video_transform_list)

    # Initialize datasets
    leave_out_idx = opt.leave_out

    # Initialize dataset
    if opt.dataset == 'gteagazeplus':
        all_subjects = ['Ahmad', 'Alireza', 'Carlos',
                        'Rahul', 'Yin', 'Shaghayegh']
        train_seqs, valid_seqs = evaluation.leave_one_out(all_subjects,
                                                          leave_out_idx)
        dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                    use_video=False, clip_size=16,
                                    original_labels=True,
                                    use_flow=opt.use_flow,
                                    flow_type=opt.flow_type,
                                    rescale_flows=opt.rescale_flows,
                                    seqs=train_seqs)
        val_dataset = GTEAGazePlusVideo(video_transform=video_transform,
                                        base_transform=base_transform,
                                        use_video=False, clip_size=16,
                                        original_labels=True,
                                        seqs=valid_seqs,
                                        use_flow=opt.use_flow,
                                        flow_type=opt.flow_type,
                                        rescale_flows=opt.rescale_flows)
    elif opt.dataset == 'smthgsmthg':
        dataset = SmthgVideo(video_transform=video_transform,
                             clip_size=16, split='train',
                             use_flow=opt.use_flow,
                             flow_type=opt.flow_type,
                             rescale_flows=opt.rescale_flows,
                             frame_spacing=opt.clip_spacing)

        val_dataset = SmthgVideo(video_transform=video_transform,
                                 clip_size=16, split='valid',
                                 base_transform=base_transform,
                                 use_flow=opt.use_flow,
                                 flow_type=opt.flow_type,
                                 rescale_flows=opt.rescale_flows)
    else:
        raise ValueError('the opt.dataset name provided {0} is not handled\
                         by this script'.format(opt._dataset))

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

    # Initialize C3D neural network
    c3dnet = c3d.C3D()
    if opt.pretrained:
        c3dnet.load_state_dict(torch.load('data/c3d.pickle'))
    model = c3d_adapt.C3DAdapt(
        opt, c3dnet, dataset.class_nb, in_channels=channel_nb)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.net.parameters(), lr=0.003)

    model.set_criterion(criterion)
    model.set_optimizer(optimizer)

    # Load existing weights, opt.continue_training is epoch to load
    if opt.continue_training:
        if opt.continue_epoch == 0:
            model.net.eval()
            model.load(latest=True)
        else:
            model.load(epoch=opt.continue_epoch)

    train.train_net(dataloader, model, opt,
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
