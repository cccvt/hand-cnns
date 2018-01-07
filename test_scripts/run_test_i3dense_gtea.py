from c3d_test import run_testing
from src.options import base_options, test_options, video_options
from src.utils.launch import get_gtea_paths

dataset = 'gteagazeplus'
experience = 'run_2_mom_leaveouts_mom_0.08'
modality = 'rgb'  # 'flow'
network = 'i3dense'  # 'i3dense'

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
        str(leave_out), '--exp_id', test_folder, '--network', network,
        '--gpu_parallel', '--gpu_nb', '1', '--save_predictions', '--threads',
        '8', '--checkpoint_path',
        'checkpoints/' + train_folder + '/i3dense_adapt_epoch_latest.pth',
        '--frame_nb', '0', '--visualize', '0'
    ]
    opt = options.parse(arguments)

    run_testing(opt)
