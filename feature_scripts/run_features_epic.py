import os

from c3d_features import get_features
from netraining.options import base_options, test_options, video_options
from netraining.utils.launch import get_test_folder

dataset = 'epic'
experience = 'extract_smthgv2_i3d_features'
modality = 'rgb'  # 'flow'
network = 'i3d'  # 'i3dense'
clip_size = 32
epoch = '25'
split = 'test_unseen'
mode = 'full'
mode_param = 200
multi = True
weights_path = 'checkpoints/train/i3d/rgb/smthgv2/run5_32_lr0.01_sched_fac_0.5_patience_4/i3d_adapt_epoch_25.pth'

test_folder = get_test_folder(
    modality=modality,
    network=network,
    dataset=dataset,
    epoch=epoch,
    experience=experience,
    split=split,
    features=True)
options = base_options.BaseOptions()
test_options.add_test_options(options)
video_options.add_video_options(options)
arguments = [
    '--dataset', dataset, '--split', split, '--batch_size', '1', '--exp_id',
    test_folder + '/{}_{}'.format(mode, mode_param), '--network', network,
    '--gpu_parallel', '--gpu_nb', '1', '--clip_size',
    str(clip_size), '--save_predictions', '--threads', '8',
    '--checkpoint_path', weights_path, '--frame_nb', '10', '--visualize', '0',
    '--mode', mode, '--mode_param',
    str(mode_param)
]
if modality == 'flow':
    arguments = arguments + ['--use_flow', '--flow_type', 'flownet2']
if modality == 'heatmaps':
    arguments = arguments + ['--use_heatmaps']
if multi:
    arguments = arguments + ['--multi']

opt = options.parse(arguments)
get_features(opt)
