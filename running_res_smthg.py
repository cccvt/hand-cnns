from train import run_training
from src.options.train_options import TrainOptions


lrs = [0.00001]
for lr in lrs:
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
