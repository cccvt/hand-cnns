from c3d_test import run_testing
from src.options import base_options, test_options, video_options

for i in range(0, 6):
    options = base_options.BaseOptions()
    test_options.add_test_options(options)
    video_options.add_video_options(options)
    arguments = [
        '--batch_size', '8', '--leave_out',
        str(i), '--exp_id', 'test/c3d/rgb/gtea/gtea_lo_' + str(i),
        '--save_predictions', '--threads', '8', '--checkpoint_path',
        'checkpoints/train/c3d/rgb/gteagazeplus/ordered_leave_outs/gtea_lo_' +
        str(i) + '/c3d_adapt_epoch_latest.pth', '--frame_nb', '10',
        '--visualize', '0'
    ]
    opt = options.parse(arguments)
    run_testing(opt)
