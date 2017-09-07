from c3d_train import run_training
from src.options.train_options import TrainOptions


lrs = [0.0001]
for lr in lrs:
    opt = TrainOptions().parse(['--batch_size', '6',
                                '--dataset', 'smthgsmthg',
                                '--lr', str(lr),
                                '--new_lr', str(lr),
                                '--threads', '3',
                                '--epochs', '101',
                                '--exp_id',
                                'run_c3d_smthg_spacing_3/lr_' +
                                str(lr),
                                '--visualize', '0',
                                '--clip_spacing', '3'])

    run_training(opt)
