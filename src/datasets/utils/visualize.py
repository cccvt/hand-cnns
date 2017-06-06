import matplotlib.pyplot as plt


def draw2d_annotated_img(img, annot, links):
    """
    Draws image img with annotations annot

    :param annot: First axes represent joint indexes
    second the u, v  (and useless d) joint coordinates
    :type annot: numpy ndarray
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.scatter(annot[:, 0], annot[:, 1], s=4, c="r")
    if(links):
        draw2djoints(ax, annot, links)


def draw2dseg(ax, annot, idx1, idx2, color="r", marker="o"):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot plot/subplot
    :param idx1: row of the start point in annot
    :param idx2: row of the end point in annot
    """
    ax.plot([annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            c=color, marker=marker)


def draw2djoints(ax, annots, links):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    """
    for link in links:
        draw2dseg(ax, annots, link[0], link[1])


def draw3dseg(ax, annot, idx1, idx2, color="r"):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot 3d plot/subplot
    :param idx1: row of the start point in annot
    :param idx2: row of the end point in annot
    """
    ax.plot([annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            [annot[idx1, 2], annot[idx2, 2]])


def draw3djoints(ax, annots, links):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot 3d plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    """
    for link in links:
        draw3dseg(ax, annots, link[0], link[1])

