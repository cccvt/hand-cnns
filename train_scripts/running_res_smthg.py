from train import run_training
from src.options import base_options, train_options, image_options


lrs = [0.00001]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    image_options.add_image_options(options)

    opt = TrainOptions().parse(['--batch_size', '64',
                                '--dataset', 'smthgsmthg',
                                '--lr', str(lr),
                                '--new_lr', str(lr),
                                '--threads', '10',
                                '--epochs', '51',
                                '--exp_id',
                                'run_res_smthg_resnet50/lr_' +
                                str(lr)])

    run_training(opt)
