import cv2
import torch

from src.datasets.smthgvideo import SmthgVideo
from src.datasets.utils import video_transforms, volume_transforms
from src.nets import c3d, c3d_adapt
from src.netscripts import test
from src.options import base_options, video_options, test_options


def run_testing(opt):
    scale_size = (128, 171)
    crop_size = (112, 112)
    base_transform_list = [video_transforms.Scale(crop_size),
                           volume_transforms.ToTensor()]
    base_transform = video_transforms.Compose(base_transform_list)
    video_transform_list = [video_transforms.Scale(scale_size),
                            video_transforms.RandomCrop(crop_size),
                            volume_transforms.ToTensor()]
    video_transform = video_transforms.Compose(video_transform_list)

    dataset = SmthgVideo(video_transform=video_transform,
                         base_transform=base_transform,
                         clip_size=16, split='test')

    # Initialize C3D neural network
    c3dnet = c3d.C3D()
    model = c3d_adapt.C3DAdapt(opt, c3dnet, dataset.class_nb)

    optimizer = torch.optim.SGD(model.net.parameters(), lr=1)

    model.set_optimizer(optimizer)

    # Load existing weights
    model.net.eval()
    if opt.use_gpu:
        model.net.cuda()
    model.load(load_path=opt.checkpoint_path)

    test.test(dataset, model, opt=opt, frame_nb=opt.frame_nb,
              save_predictions=True)


if __name__ == '__main__':
    # Initialize base options
    options = base_options.BaseOptions()

    # Add test options and parse
    test_options.add_test_options(options)
    video_options.add_video_options(options)
    opt = options.parse()
    run_testing(opt)
