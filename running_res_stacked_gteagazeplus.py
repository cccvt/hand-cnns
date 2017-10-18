from res_stacked_train import run_training
from src.options import base_options, train_options, video_options, stack_options

leave_outs = list(range(2, 6))

for leave_out in leave_outs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)
    stack_options.add_stack_options(options)

    opt = options.parse([
        '--batch_size', '8', '--leave_out',
        str(leave_out), '--lr', '0.0001', '--new_lr', '0.0001', '--threads',
        '10', '--epochs', '50', '--use_flow', '1', '--flow_type', 'farn',
        '--rescale_flow', '0', '--exp_id',
        'train/stack/flow-farn/ordered_resnet34/gtea_lo_' + str(leave_out),
        '--visualize', '0', '--display_freq', '4', '--stack_nb', '10',
        '--continue_training', '--continue_epoch', '100', '--test_aggreg', '1'
    ])

    run_training(opt)
