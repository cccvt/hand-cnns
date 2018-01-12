from c3d_train import run_training
from src.options import base_options, train_options, video_options

lrs = [0.01]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse([
        '--batch_size', '15', '--dataset', 'smthg', '--network', 'i3res',
        '--gpu_parallel', '--gpu_nb', '3', '--lr',
        str(lr), '--threads', '20', '--epochs', '101', '--exp_id',
        'train/i3res/rgb/smthg/run_3_64_lr' + str(lr), '--clip_size', '64',
        '--visualize', '1', '--display_freq', '100', '--test_aggreg', '0',
        '--clip_spacing', '1', '--display_port', '6401'
    ])

    run_training(opt)
