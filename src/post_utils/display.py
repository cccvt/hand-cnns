from matplotlib import pyplot as plt
import numpy as np


def stringify(nested):
    if isinstance(nested, str):
        return nested
    if (isinstance(nested, tuple)
            or isinstance(nested, list)) and len(nested) == 1:
        return stringify(nested[0])
    else:
        return stringify(nested[0]) + '_' + stringify(nested[1:])


def normalize_rows(mat):
    mat = np.copy(mat)
    for i, row in enumerate(mat):
        norm_row = row.sum() or 1
        mat[i] = row / norm_row
    return mat


def plot_epoch_conf_mat(confmat,
                        title=None,
                        labels=None,
                        epoch=None,
                        both_labels=False,
                        normalize=False,
                        display=True):
    """
    Args:
        score_type (str): label for current curve, [valid|train|aggreg]
    """
    if epoch is None:
        mat = confmat[-1]
    else:
        if epoch > confmat.shape[0]:
            raise ValueError(
                'Epoch {} should be below {}'.format(epoch, confmat.shape[0]))
    mat = confmat[epoch]
    fig, ax = plot_confmat(
        mat,
        labels=labels,
        title=title,
        normalize=normalize,
        both_labels=both_labels,
        display=display)
    return fig, ax


def plot_confmat(confmat,
                 labels=None,
                 title=None,
                 normalize=False,
                 cmap='viridis',
                 both_labels=False,
                 display=True,
                 annotate=False):
    confmat = np.transpose(confmat)
    if normalize:
        confmat = normalize_rows(confmat)  # Percentage of accuracy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(confmat, cmap=cmap)
    fig.colorbar(cax)
    if title is not None:
        ax.set_title(title)
    if labels is not None:
        str_labels = [stringify(label) for label in labels]
        if both_labels:
            ax.set_xticklabels(str_labels, rotation=90)
            ax.set_xticks(range(len(str_labels)))
        ax.set_yticklabels(str_labels)
        ax.set_yticks(range(len(str_labels)))
    if annotate:
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                val = confmat[i, j]
                min_val = 1
                if val > min_val:
                    ax.annotate('{}'.format(int(val)), xy=(i - 0.3, j + 0.3))
    try:
        plt.tight_layout()
    except ValueError:
        pass
    if display:
        plt.show()
    return fig, ax
