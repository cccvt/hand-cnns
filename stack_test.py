import cv2
import torch
import torchvision.models as models

from src.datasets.smthgvideo import SmthgVideo
from src.datasets.gteagazeplusvideo import GTEAGazePlusVideo
from src.datasets.utils import video_transforms, stack_transforms
from src.nets import resnet_adapt
from src.netscripts import test
from src.options import base_options, stack_options, test_options
from src.utils import evaluation
from src.utils.visualize import Visualize


def run_testing(opt):
    final_size = 224
    if opt.use_flow:
        channel_nb = 2
    else:
        channel_nb = 3
    base_transform_list = [video_transforms.Scale(final_size),
                           stack_transforms.ToStackedTensor(channel_nb=channel_nb)]
    base_transform = video_transforms.Compose(base_transform_list)

    if opt.dataset == 'smthgsmthg':
        dataset = SmthgVideo(video_transform=base_transform,
                             base_transform=base_transform,
                             clip_size=opt.stack_nb, split=opt.split,
                             use_flow=opt.use_flow)
    elif opt.dataset == 'gteagazeplus':
        all_subjects = ['Ahmad', 'Alireza', 'Carlos',
                        'Rahul', 'Yin', 'Shaghayegh']
        train_seqs, valid_seqs = evaluation.leave_one_out(all_subjects,
                                                          opt.leave_out)
        dataset = GTEAGazePlusVideo(video_transform=base_transform,
                                    base_transform=base_transform,
                                    use_video=False,
                                    clip_size=opt.stack_nb,
                                    original_labels=True,
                                    seqs=train_seqs, use_flow=opt.use_flow)

    # Initialize neural network
    resnet = models.resnet50()
    model = resnet_adapt.ResNetAdapt(opt, resnet, dataset.class_nb,
                                     in_channels=channel_nb * opt.stack_nb)

    optimizer = torch.optim.SGD(model.net.parameters(), lr=1)

    model.set_optimizer(optimizer)

    # Load existing weights
    model.net.eval()
    if opt.use_gpu:
        model.net.cuda()
    model.load(load_path=opt.checkpoint_path)

    mean_scores = test.test(dataset, model, opt=opt, frame_nb=opt.frame_nb,
                            save_predictions=opt.save_predictions)
    print(mean_scores)

    viz = Visualize(opt)
    # Display and save validations info
    viz.log_errors(epoch=0,
                   errors={'val_aggr_err': mean_scores},
                   log_path=viz.valid_aggreg_log_path)


if __name__ == '__main__':
    # Initialize base options
    options = base_options.BaseOptions()

    # Add test options and parse
    test_options.add_test_options(options)
    stack_options.add_stack_options(options)
    opt = options.parse()
    run_testing(opt)
