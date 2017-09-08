from train import run_training
from src.options import base_options, train_options, image_options


leave_outs = list(range(6))


for leave_out in leave_outs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    image_options.add_image_options(options)

    opt = TrainOptions().parse(['--batch_size', '8',
                                '--leave_out', str(leave_out),
                                '--lr', '0.0001',
                                '--new_lr', '0.0001',
                                '--threads', '1',
                                '--epochs', '50',
                                '--exp_id',
                                'run_res_leave_outs_resnet50/gtea_lo_' +
                                str(leave_out)])

    run_training(opt)
