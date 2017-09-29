from c3d_train import run_training
from src.options import base_options, train_options, video_options

lrs = [0.0001]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse([
        '--batch_size', '6', '--dataset', 'smthgsmthg', '--lr',
        str(lr), '--new_lr',
        str(lr), '--threads', '8', '--epochs', '101', '--exp_id',
        'train/c3d/flow-tvl1/smthg/run_1_lr_' + str(lr), '--visualize', '1',
        '--display_freq', '100', '--use_flow', '1', '--flow_type', 'tvl1',
        '--test_aggreg', '1', '--continue_training', '--continue_epoch', '60',
        '--clip_spacing', '1'
    ])

    run_training(opt)
