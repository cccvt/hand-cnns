from train import run_training
from src.options.train_options import TrainOptions


lrs = [0.0001, 0.001, 0.00001]
for lr in lrs:
    opt = TrainOptions().parse(['--batch_size', '128',
                                '--dataset', 'smthgsmthg',
                                '--lr', str(lr),
                                '--new_lr', str(lr),
                                '--threads', '10',
                                '--epochs', '51',
                                '--exp_id',
                                'run_res_smthg/lr_' +
                                str(lr)])

    run_training(opt)
