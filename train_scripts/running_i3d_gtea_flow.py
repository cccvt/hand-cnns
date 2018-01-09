from c3d_train import run_training
from src.options import base_options, train_options, video_options
from src.utils.launch import get_gtea_paths

lr = 0.01
network = 'i3d'
modality = 'flow'
dataset = 'gteagazeplus'
experience = 'run_2_real_flow_vs_run_1_lr_{}'.format(lr)

for leave_out in range(3, 6):
    train_path, test_path = get_gtea_paths(network, modality, dataset,
            experience, leave_out)

    # Initialize base options
    options = base_options.BaseOptions()

    # Add train options and parse
    train_options.add_train_options(options)
    video_options.add_video_options(options)

    opt = options.parse([
        '--batch_size', '40', '--dataset', dataset, '--network', network,
        '--use_flow', '--flow_type', 'tvl1',
        '--gpu_parallel', '--gpu_nb', '2', '--lr',
        str(lr), '--threads', '20', '--epochs', '101', '--exp_id', train_path,
        '--clip_size', '16', '--visualize', '1', '--display_freq', '100',
        '--test_aggreg', '0', '--clip_spacing', '1', '--display_port', '8012'
        ])

    run_training(opt)
