from c3d_train import run_training
from src.options.train_options import TrainOptions


leave_outs = list(range(1, 6))


for leave_out in leave_outs:
    opt = TrainOptions().parse(['--batch_size', '8',
                                '--leave_out', str(leave_out),
                                '--lr', '0.00001',
                                '--new_lr', '0.00001',
                                '--threads', '8',
                                '--epochs', '51',
                                '--use_flow', '1',
                                '--exp_id',
                                'running-c3d-gtea-flow-farneback-leave-outs/gtea_lo_' +
                                str(leave_out),
                                '--visualize', '0'])

    run_training(opt)
