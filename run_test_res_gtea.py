from res_test import run_testing
from src.options import base_options, test_options, image_options

for i in range(0, 6):
    # Initialize base options
    options = base_options.BaseOptions()

    # Add test options and parse
    test_options.add_test_options(options)
    image_options.add_image_options(options)

    arguments = [
        '--batch_size', '6', '--leave_out',
        str(i), '--exp_id', 'test/res/rgb/gteagazeplus/gtea_lo_' + str(i),
        '--save_predictions', '--threads', '8', '--checkpoint_path',
        'checkpoints/train/res/rgb/gteagazeplus/resnet34/gtea_lo_' + str(i) +
        '/resnet_adapt_epoch_latest.pth', '--frame_nb', '10', '--visualize',
        '0'
    ]
    opt = options.parse(arguments)
    run_testing(opt)
