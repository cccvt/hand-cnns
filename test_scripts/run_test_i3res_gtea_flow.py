import os

from c3d_test import run_testing
from src.options import base_options, test_options, video_options
from src.utils.launch import get_gtea_paths

dataset = 'gteagazeplus'
experience = 'run_1_leavout_0.01_lr'
modality = 'flow'  # 'flow'
network = 'i3res'  # 'i3dense'

for leave_out in range(0, 6):
    train_folder, test_folder = get_gtea_paths(
        modality=modality,
        network=network,
        leave_out=leave_out,
        dataset=dataset,
        experience=experience)
    options = base_options.BaseOptions()
    test_options.add_test_options(options)
    video_options.add_video_options(options)
    arguments = [
        '--batch_size', '8', '--leave_out',
        str(leave_out), '--use_flow', '--flow_type', 'tvl1', '--exp_id',
        test_folder, '--network', 'i3res', '--gpu_parallel', '--gpu_nb', '1',
        '--save_predictions', '--threads', '8', '--checkpoint_path',
        'checkpoints/' + train_folder + '/i3resnet_adapt_epoch_latest.pth',
        '--frame_nb', '10', '--visualize', '0'
    ]
    opt = options.parse(arguments)

    run_testing(opt)
