from c3d_train import run_training
from src.options import base_options, train_options, video_options

leave_outs = list(range(0, 6))

for leave_out in leave_outs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse([
        '--batch_size', '18', '--network', 'i3dense', '--gpu_parallel',
        '--gpu_nb', '2', '--leave_out',
        str(leave_out), '--lr', '0.01', '--new_lr', '0.01', '--threads', '10',
        '--epochs', '50', '--exp_id',
        'train/i3dense/flow/gteagazeplus/run_1_leaveouts/gtea_lo_' +
        str(leave_out), '--use_flow', '--flow_type', 'tvl1', '--visualize',
        '1', '--test_aggreg', '0', '--display_port', '8096'
    ])

    run_training(opt)
