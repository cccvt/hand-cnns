from stack_test import run_testing
from src.options import base_options, test_options, stack_options

for i in range(3, 6):
    options = base_options.BaseOptions()
    test_options.add_test_options(options)
    stack_options.add_stack_options(options)
    arguments = ['--batch_size', '6',
                 '--leave_out', str(i),
                 '--exp_id', 'test/gtea_res_flow_leave_out_' + str(i),
                 '--use_flow', '1',
                 '--threads', '8',
                 '--checkpoint_path', 'checkpoints/run_res_stack_gtea_flow_farn_50_first_epochs/gtea_lo_' +
                 str(i) + '/resnet_adapt_epoch50.pth',
                 '--frame_nb', '1',
                 '--visualize', '0']
    opt = options.parse(arguments)
    run_testing(opt)
