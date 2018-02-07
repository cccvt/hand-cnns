import os


def get_smthg_train(network, modality, dataset, experience):
    train_folder = os.path.join('train', network, modality, dataset,
                                experience)
    return train_folder


def get_smthg_test(network, modality, dataset, experience, split, epoch):
    test_folder = os.path.join('test', network, modality, dataset, experience,
                               'epoch_{}'.format(epoch), split)
    return test_folder


def get_smthg_paths(network, modality, dataset, experience, split, epoch):
    train_folder = get_smthg_train(network, modality, dataset, experience)
    test_folder = get_smthg_test(network, modality, dataset, experience, split,
                                 epoch)
    return train_folder, test_folder


def get_gtea_train(network, modality, dataset, experience, leave_out):
    leave_out_str = 'gtea_lo_{}'.format(leave_out)
    train_folder = os.path.join('train', network, modality, dataset,
                                experience, leave_out_str)
    return train_folder


def get_gtea_test(network, modality, dataset, experience, leave_out, epoch):
    leave_out_str = 'gtea_lo_{}'.format(leave_out)
    test_folder = os.path.join('test', network, modality, dataset, experience,
                               'epoch_{}'.format(epoch), leave_out_str)
    return test_folder


def get_gtea_paths(network, modality, dataset, experience, leave_out, epoch):
    train_folder = get_gtea_train(network, modality, dataset, experience,
                                  leave_out)
    test_folder = get_gtea_test(
        network, modality, dataset, experience, leave_out, epoch=epoch)
    return train_folder, test_folder
