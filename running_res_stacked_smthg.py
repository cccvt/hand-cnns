from res_stacked_train import run_training
from src.options import base_options, train_options, video_options, stack_options

# Initialize base options
options = base_options.BaseOptions()

# Add train options and parse
train_options.add_train_options(options)
video_options.add_video_options(options)
stack_options.add_stack_options(options)

lr = '0.00001'
opt = options.parse([
    '--batch_size', '8', '--dataset', 'smthgsmthg', '--lr', lr, '--new_lr', lr,
    '--threads', '8', '--epochs', '200', '--use_flow', '1', '--flow_type',
    'tvl1', '--exp_id', 'train/stack/flow-tvl1/smthg/continue_45_lr_' + lr,
    '--visualize', '1', '--display_freq', '100', '--stack_nb', '10',
    '--continue_training', '--continue_epoch', '45', '--test_aggreg', '0'
])

run_training(opt)
