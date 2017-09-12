from c3d_train import run_training
from src.options import base_options, train_options, video_options


lrs = [0.0001]
for lr in lrs:
    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse(['--batch_size', '6',
                         '--dataset', 'smthgsmthg',
                         '--lr', str(lr),
                         '--new_lr', str(lr),
                         '--threads', '3',
                         '--epochs', '101',
                         '--exp_id',
                         'run_c3d_smthg_flow/lr_' +
                         str(lr),
                         '--visualize', '0',
                         '--use_flow', '1',
                         '--test_aggreg', '0',
                         '--clip_spacing', '1'])

    run_training(opt)
