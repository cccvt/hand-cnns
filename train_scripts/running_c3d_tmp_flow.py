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
        '--batch_size', '18', '--dataset', 'smthgsmthg', '--network',
        'i3dense', '--gpu_parallel', '--gpu_nb', '4', '--use_flow',
        '--flow_type', 'tvl1', '--lr',
        str(lr), '--new_lr',
        str(lr), '--threads', '10', '--epochs', '101', '--exp_id',
        'train/i3dense/flow/smthg/run_2_fc_lr_' + str(lr), '--clip_size', '16',
        '--visualize', '0', '--display_freq', '100', '--test_aggreg', '0',
        '--clip_spacing', '1'
        ])

    run_training(opt)
