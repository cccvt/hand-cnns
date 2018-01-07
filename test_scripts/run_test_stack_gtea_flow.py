from stack_test import run_testing
from src.options import base_options, test_options, stack_options

for i in range(0, 6):
    options = base_options.BaseOptions()
    test_options.add_test_options(options)
    stack_options.add_stack_options(options)
    arguments = [
        '--batch_size', '6', '--leave_out',
        str(i), '--exp_id', 'test/stack/flow-farn/gtea/gtea_lo_' + str(i),
        '--use_flow', '1', '--flow_type', 'farn', '--save_predictions',
        '--threads', '8', '--stack_nb', '10', '--checkpoint_path',
        'checkpoints/train/stack/flow-farn/ordered_resnet34/gtea_lo_' + str(i)
        + '/resnet_adapt_epoch_latest.pth', '--frame_nb', '10', '--visualize',
        '0'
    ]
    opt = options.parse(arguments)
    run_testing(opt)
