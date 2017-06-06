import matplotlib.pyplot as plt


def draw2d_annotated_img(img, annot, links, keep_joints=None):
    """
    Draws 2d image img with joint annotations

    :param annot: First axes represent joint indexes
    second the u, v  (and useless d) joint coordinates
    :type annot: numpy ndarray
    :param keep_joints: only draws links between joints in keep_joints
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(img)
    ax.scatter(annot[:, 0], annot[:, 1], s=4, c="r")
    if(links):
        draw2djoints(ax, annot, links, keep_joints)


def draw3d_annotated_img(annot, links, keep_joints=None, angle=320):
    """
    Draws 3d image img with joint annotations

    :param annot: First axes represent joint indexes
    second the x, y, z joint coordinates
    :type annot: numpy ndarray
    :param keep_joints: only draws links between joints in keep_joints
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, angle)
    draw3djoints(ax, annot, links, keep_joints)


def draw2djoints(ax, annots, links, keep_joints=None):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    :param keep_joints: only draws links between joints in keep_joints
    """
    for link in links:
        if keep_joints is None or (link[0] in keep_joints
                                   and link[1] in keep_joints):
            draw2dseg(ax, annots, link[0], link[1])


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


def draw3djoints(ax, annots, links, keep_joints=None):
    """
    :param annot: 2d numpy ndarray [[x1, y1, z1], ...]
    :param ax: matplot 3d plot/subplot
    :param links: tuples of annot rows to link [(idx1, idx2), ...]
    :param keep_joints: only draws links between joints in keep_joints
    """
    for link in links:
        if keep_joints is None or (link[0] in keep_joints
                                   and link[1] in keep_joints):
            draw3dseg(ax, annots, link[0], link[1])


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

