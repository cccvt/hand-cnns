from res_stacked_train import run_training
from src.options import base_options, train_options, video_options, stack_options


leave_outs = list(range(0, 6))


for leave_out in leave_outs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)
    stack_options.add_stack_options(options)

    opt = options.parse(['--batch_size', '8',
                         '--leave_out', str(leave_out),
                         '--lr', '0.0001',
                         '--new_lr', '0.0001',
                         '--threads', '10',
                         '--epochs', '101',
                         '--use_flow', '1',
                         '--exp_id',
                         'run_res_stack_gtea_flow_tvl1_leave_outs/gtea_lo_' +
                         str(leave_out),
                         '--visualize', '1',
                         '--display_freq', '4',
                         '--stack_nb', '10',
                         '--test_aggreg', '0'])

    run_training(opt)
