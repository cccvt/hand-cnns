from c3d_train import run_training
from src.options import base_options, train_options, video_options


leave_outs = list(range(0, 6))


for leave_out in leave_outs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse(['--batch_size', '6',
                         '--leave_out', str(leave_out),
                         '--lr', '0.0001',
                         '--new_lr', '0.0001',
                         '--threads', '10',
                         '--epochs', '101',
                         '--use_flow', '0',
                         '--exp_id',
                         'run_c3d_gtea_rgb_frame_leave_outs/gtea_lo_' +
                         str(leave_out),
                         '--visualize', '0',
                         '--test_aggreg', '1'])

    run_training(opt)
