from c3d_train import run_training
from src.options import base_options, train_options, video_options

leave_outs = list(range(0, 6))
lrs = [0.0001]

for lr in lrs:
    for leave_out in leave_outs:
        # Initialize base options
        options = base_options.BaseOptions()

        # Add train options and parse
        train_options.add_train_options(options)
        video_options.add_video_options(options)

        opt = options.parse([
            '--batch_size', '30', '--network', 'i3res', '--gpu_parallel',
            '--gpu_nb', '2', '--leave_out',
            str(leave_out), '--lr',
            str(lr), '--new_lr',
            str(lr), '--threads', '10', '--epochs', '100', '--exp_id',
            'train/i3res/rgb/gteagazeplus/run_2_leavout_0.001_lr_continue_{}/gtea_lo_'.
            format(lr) + str(leave_out), '--visualize', '1', '--test_aggreg',
            '0', '--display_port', '8021', '--momentum', '0.9',
            '--continue_training', '--continue_epoch', '10'
        ])

        run_training(opt)
