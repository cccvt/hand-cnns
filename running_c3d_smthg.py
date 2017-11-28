from c3d_train import run_training
from src.options import base_options, train_options, video_options

lrs = [1]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse([
        '--batch_size', '40', '--dataset', 'smthgsmthg', '--network', 'i3d',
        '--gpu_parallel', '--gpu_nb', '2', '--use_flow', '--flow_type', 'tvl1',
        '--lr',
        str(lr), '--new_lr',
        str(lr), '--threads', '8', '--epochs', '101', '--exp_id',
        'train/i3d/rgb/smthg/run_3_lr_' + str(lr), '--visualize', '0',
        '--display_freq', '100', '--test_aggreg', '0', '--clip_spacing', '1'
    ])

    run_training(opt)
