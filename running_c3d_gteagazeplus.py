from c3d_train import run_training
from src.options.train_options import TrainOptions


leave_outs = list(range(1, 6))


for leave_out in leave_outs:
    opt = TrainOptions().parse(['--batch_size', '6',
                                '--leave_out', str(leave_out),
                                '--lr', '0.0001',
                                '--new_lr', '0.0001',
                                '--threads', '4',
                                '--epochs', '51',
                                '--use_flow', '1',
                                '--exp_id',
                                'run_c3d_gtea_flow_farneback_leave_outs/gtea_lo_' +
                                str(leave_out),
                                '--visualize', '0',
                                '--test_aggreg', '0'])

    run_training(opt)
