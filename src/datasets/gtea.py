import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch.utils.data as data

from src.datasets.utils import loader, filesys

"""
Action labels are composed of a vert and a set of actions such as
('pour', ['honey, bread']), they are then processed to string labels
such as 'pour honey bread'

two action labels have only 1 and 2 occurences in dataset:
('stir', ['cup']) and ('put', ['tea'])
by removing them we get to 71 action classes

Not all frames are annotated !
"""


class GTEA(data.Dataset):
    def __init__(self, transform=None, untransform=None,
                 root_folder="data/GTEA", no_action_label=True,
                 seqs=['S1', 'S2', 'S3', 'S4']):
        """
        :param transform: transformation to apply to the images
        :param no_action_label: encode absence of action class as class
        if True, 'no action' is encoded as last column in vector
        """
        self.no_action_label = no_action_label
        self.transform = transform
        self.untransform = untransform
        self.path = root_folder
        self.label_path = os.path.join(self.path, 'labels')
        self.actions = ['close', 'fold', 'open', 'pour', 'put',
                        'scoop', 'shake', 'spread', 'stir', 'take']

        filenames = filesys.recursive_files_dataset(self.path, ".png", depth=3)
        filenames = [filename for filename in filenames if
                     any(seq in filename for seq in seqs)]
        self.file_paths = filenames
        self.item_nb = len(self.file_paths)

        self.in_channels = 3
        self.in_size = (720, 405)

        self.classes = self._get_all_classes()
        # Sanity check on computed class nb
        if self.no_action_label:
            self.class_nb = 72
        else:
            self.class_nb = 71
        assert len(self.classes) == self.class_nb,\
            "{0} classes found, should be {1}".format(
                len(self.classes), self.class_nb)

    def __getitem__(self, index):
        img_path = self.file_paths[index]

        # Load image
        img = loader.load_rgb_image(img_path)
        self.img = img
        if self.transform is not None:
            img = self.transform(img)

        # One hot encoding
        annot = np.zeros(self.class_nb)
        root, img_folder, seq, img_file = img_path.rsplit('/', 3)
        annot_path = os.path.join(self.label_path, seq + '.txt')
        sequence_lines = process_lines(annot_path)
        sequence_annots = process_annots(sequence_lines)
        frame_idx = int(img_file.split('.')[0])
        if frame_idx in sequence_annots:
            action_class = sequence_annots[frame_idx]
            class_idx = self.classes.index(action_class)
            annot[class_idx] = 1
        else:
            if self.no_action_label:
                annot[self.class_nb - 1] = 1

        return img, annot

    def __len__(self):
        return self.item_nb

    def draw2d(self, idx):
        """
        draw 2D rgb image with displayed annotations
        :param idx: idx of the item in the dataset
        """
        img, annot = self[idx]
        plt.imshow(img)
        plt.axis('off')

    def _get_all_classes(self):
        sequences = os.listdir(self.label_path)
        seqs = [os.path.join(self.label_path, seq) for seq in sequences]
        object_actions = []
        for seq in seqs:
            annots = process_lines(seq)
            for annot in annots:
                object_actions.append((annot[0:2]))
        unique_object_actions = sorted(list(set(object_actions)))
        unique_classes = [_class_string(action, objects)
                          for action, objects in unique_object_actions]
        if self.no_action_label:
            unique_classes.append('None')
        return unique_classes


def process_lines(annot_path):
    """
    Returns list of action_objects ['action object1 object2', ...]
    """
    with open(annot_path) as f:
        lines = f.readlines()
    processed_lines = []
    for line in lines:
        matches = re.search('<(.*)><(.*)> \((.*)-(.*)\)', line)
        if matches:
            action_label, object_label = matches.group(1), matches.group(2)
            begin, end = int(matches.group(3)), int(matches.group(4))
            object_labels = tuple(object_label.split(','))
            if inclusion_condition(action_label, object_labels):
                processed_lines.append((action_label, object_labels,
                                        begin, end))
    return processed_lines


def _class_string(action, objects):
    return action + ' ' + ' '.join(objects)


def process_annots(processed_lines):
    """
    Returns a dictionnary with frame as key and
    value 'action object1 object2 object3' from
    the gtea annotation text file
    """
    # create annotation_dict
    annot_dict = {}
    for action, object_label, begin, end in processed_lines:
        for frame in range(begin, end + 1):
            annot_dict[frame] = _class_string(action, object_label)
    return annot_dict


def inclusion_condition(action, objects):
    if len(objects) != 1:
        return True
    if action == 'stir' and objects[0] == 'cup':
        return False
    elif action == 'put' and objects[0] == 'tea':
        return False
    else:
        return True
