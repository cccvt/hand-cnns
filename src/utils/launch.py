import os


def get_smthg_paths(network, modality, dataset, experience, split):
    train_folder = os.path.join('train', network, modality, dataset,
                                experience)
    test_folder = os.path.join('test', network, modality, dataset, split,
                               experience)
    return train_folder, test_folder


def get_gtea_paths(network, modality, dataset, experience, leave_out):
    leave_out_str = 'gtea_lo_{}'.format(leave_out)
    train_folder = os.path.join('train', network, modality, dataset,
                                experience, leave_out_str)
    test_folder = os.path.join('test', network, modality, dataset, experience,
                               leave_out_str)
    return train_folder, test_folder
