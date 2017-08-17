from train import run_training
from src.options.train_options import TrainOptions


leave_outs = list(range(3, 6))


for leave_out in leave_outs:
    opt = TrainOptions().parse(['--batch_size', '8',
                                '--leave_out', str(leave_out),
                                '--lr', '0.00001',
                                '--new_lr', '0.00001',
                                '--threads', '3',
                                '--epochs', '101',
                                '--exp_id',
                                'run_c3d_leave_outs/gtea_lo_' +
                                str(leave_out),
                                '--continue_training'])

    run_training(opt)
