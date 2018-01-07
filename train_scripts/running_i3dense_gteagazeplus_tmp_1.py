from c3d_train import run_training
from src.options import base_options, train_options, video_options

leave_outs = list(range(0, 6))
lrs = [0.01]

for lr in lrs:
    for leave_out in leave_outs:
        # Initialize base options
        options = base_options.BaseOptions()

        # Add train options and parse
        train_options.add_train_options(options)
        video_options.add_video_options(options)

        opt = options.parse([
            '--batch_size', '18', '--network', 'i3dense', '--gpu_parallel',
            '--gpu_nb', '2', '--leave_out',
            str(leave_out), '--lr',
            str(lr), '--new_lr',
            str(lr), '--threads', '10', '--epochs', '80', '--exp_id',
            'train/i3dense/rgb/gteagazeplus/run_3_leaveouts_{}/gtea_lo_'.
            format(lr) + str(leave_out), '--visualize', '1', '--test_aggreg',
            '0', '--display_port', '8099', '--momentum', '0.9'
        ])

        run_training(opt)
