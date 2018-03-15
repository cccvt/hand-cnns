import torch.nn as nn
import torch.nn.functional as functional


class DeepPrior(nn.Module):
    def __init__(self, joint_nb, joint_dim=3):
        """
        :param joint_dim: number of dimensions for each joints (2 or 3)
        :param joint_nb: number of joints in dataset annotation
        """
        super(DeepPrior, self).__init__()
        self.joint_nb = joint_nb
        self.joint_dim = joint_dim
        self.convs = nn.Sequential(
            nn.Conv2d(3, 8, padding=2, kernel_size=5),
            nn.MaxPool2d(4),
            nn.ReLU(True),
            nn.Conv2d(8, 8, padding=2, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(8, 8, padding=2, kernel_size=3),
        )
        self.fcs = nn.Sequential(
            nn.Linear(40672, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 30),
            nn.ReLU(True),  # note it seems that this relu doesn t exist
            nn.Linear(30, self.joint_dim * self.joint_nb)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
