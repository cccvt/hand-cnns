from res_stacked_train import run_training
from src.options import base_options, train_options, video_options, stack_options


lrs = [0.0001]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)
    stack_options.add_stack_options(options)

    opt = options.parse(['--batch_size', '8',
                         '--dataset', 'smthgsmthg',
                         '--lr', str(lr),
                         '--new_lr', str(lr),
                         '--threads', '8',
                         '--epochs', '101',
                         '--use_flow', '1',
                         '--flow_type', 'tvl1',
                         '--exp_id',
                         'run_res_stack_smthg_flow_farneback/lr_' +
                         'train/stack/flow-tvl1/smthg/debug_lr_' +
                         str(lr),
                         '--visualize', '1',
                         '--display_freq', '100',
                         '--stack_nb', '10',
                         '--continue_training',
                         '--continue_epoch', '0',
                         '--test_aggreg', '0'])

    run_training(opt)
