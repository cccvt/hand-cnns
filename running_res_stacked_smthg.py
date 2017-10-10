from res_stacked_train import run_training
from src.options import base_options, train_options, video_options, stack_options

# Initialize base options
options = base_options.BaseOptions()

# Add train options and parse
train_options.add_train_options(options)
video_options.add_video_options(options)
stack_options.add_stack_options(options)

lr = '0.0001'
opt = options.parse([
    '--batch_size', '8', '--dataset', 'smthgsmthg', '--lr', lr, '--new_lr', lr,
    '--threads', '8', '--epochs', '200', '--use_flow', '1', '--flow_type',
    'tvl1', '--exp_id', 'train/stack/flow-tvl1/smthg/run_stack_5' + lr,
    '--visualize', '0', '--display_freq', '100', '--stack_nb', '5',
    '--test_aggreg', '1', '--continue_training', '--continue_epoch', '100'
])

run_training(opt)
