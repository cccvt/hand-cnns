import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data

from src.datasets.utils import loader, filesys
from src.datasets.utils import gteaannots

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

        self.classes = gteaannots.get_all_classes(self.label_path,
                                                  self.inclusion_condition,
                                                  no_action_label=False)
        # Sanity check on computed class nb
        if self.no_action_label:
            self.class_nb = 72
        else:
            self.class_nb = 71

        # Remove frames with no class attached
        if not self.no_action_label:
            self.remove_no_class()

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
        sequence_annots = self.get_seq_annotations(seq)
        frame_idx = int(img_file.split('.')[0])
        if frame_idx in sequence_annots:
            action_class = sequence_annots[frame_idx]
            class_idx = self.classes.index(action_class)
            annot[class_idx] = 1
        else:
            if self.no_action_label:
                annot[self.class_nb - 1] = 1

        return img, annot

    def get_seq_annotations(self, sequence_name):
        annot_path = os.path.join(self.label_path, sequence_name + '.txt')
        sequence_lines = gteaannots.process_lines(annot_path,
                                                  self.inclusion_condition)
        sequence_annots = gteaannots.process_annots(sequence_lines)
        return sequence_annots

    def remove_no_class(self):
        """
        Removes from list of files all frames that do not have
        an action label
        """
        for image_path in self.file_paths:
            root, img_folder, seq, img_file = image_path.rsplit('/', 3)
            sequence_annots = self.get_seq_annotations(seq)
            frame_idx = int(img_file.split('.')[0])
            if frame_idx not in sequence_annots:
                self.file_paths.remove(image_path)

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

    def inclusion_condition(self, action, objects):
        if len(objects) != 1:
            return True
        if action == 'stir' and objects[0] == 'cup':
            return False
        elif action == 'put' and objects[0] == 'tea':
            return False
        else:
            return True
