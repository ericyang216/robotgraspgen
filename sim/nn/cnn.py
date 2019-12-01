import torch
import torch.nn as nn
import numpy as np

import nn.model_utils as models

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn = models.CNN()

        self.pos_net = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 3))

        self.ang_net = nn.Sequential(nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2),
                                     models.Normalize())

    def forward(self, x):
        features = self.cnn(x)
        position = self.pos_net(features)
        angle = self.ang_net(features)
        return torch.cat([position, angle], dim=1)

    def loss(self, a, b):
        return models.DistanceLoss(a, b), models.AngleLoss(a, b)