import os

from c3d_test import run_testing
from src.options import base_options, test_options, video_options
from src.utils.launch import get_smthg_paths

dataset = 'smthg'
experience = 'run_4_mom_lr0.01'
modality = 'rgb'  # 'flow'
network = 'i3dense'  # 'i3dense'
split = 'test'

train_folder, test_folder = get_smthg_paths(
    network=network,
    experience=experience,
    dataset=dataset,
    modality=modality,
    split=split)

options = base_options.BaseOptions()
test_options.add_test_options(options)
video_options.add_video_options(options)
arguments = [
    '--dataset', 'smthg', '--split', split, '--batch_size', '8', '--exp_id',
    test_folder, '--network', network, '--gpu_parallel', '--gpu_nb', '1',
    '--save_predictions', '--threads', '8', '--checkpoint_path',
    'checkpoints/' + train_folder + '/' + network + '_adapt_epoch_40.pth',
    '--frame_nb', '0', '--visualize', '0'
]
opt = options.parse(arguments)
run_testing(opt)
