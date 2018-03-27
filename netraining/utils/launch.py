import os


def get_train_folder(network, modality, dataset, experience):
    train_folder = os.path.join('train', network, modality, dataset,
                                experience)
    return train_folder


def get_test_folder(network,
                    modality,
                    dataset,
                    experience,
                    split,
                    epoch,
                    features=False):
    if features:
        test_folder = os.path.join('features', network, modality, dataset,
                                   experience, 'epoch_{}'.format(epoch), split)
    else:
        test_folder = os.path.join('test', network, modality, dataset,
                                   experience, 'epoch_{}'.format(epoch), split)
    return test_folder


def get_smthg_viz(network, modality, dataset, experience, split, epoch):
    viz_folder = os.path.join('viz', network, modality, dataset, experience,
                              'epoch_{}'.format(epoch), split)
    return viz_folder


def get_smthg_paths(network, modality, dataset, experience, split, epoch):
    train_folder = get_train_folder(network, modality, dataset, experience)
    test_folder = get_test_folder(network, modality, dataset, experience,
                                  split, epoch)
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
