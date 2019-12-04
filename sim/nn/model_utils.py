import torch
import torch.nn as nn
import numpy as np

def AngleLoss(a, b, norm=False):
    """ a is prediction """
    A = a[:, 3:]
    B = b[:, 3:]
    if norm:
        A = normalize_F(A)
    cos_loss = 1. - ((A[:, 1] * B[:, 1]) + (A[:, 0] * B[:, 0]))
    cos_loss = torch.mean(cos_loss, dim=0)
    return cos_loss

def DistanceLoss(a, b):
    mse_loss = torch.mean(torch.sum(torch.abs(a[:, 0:3] - b[:, 0:3]), dim=1), dim=0)
    return mse_loss

def normalize_F(x):
    sq = x * x
    norm = torch.sqrt(torch.sum(sq, dim=1)).unsqueeze(-1)
    return x / norm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Normalize(nn.Module):
    def forward(self, input):
        sq = input * input
        norm = torch.sqrt(torch.sum(sq, dim=1)).unsqueeze(-1)
        return input / norm

class MiniCNN(nn.Module):
    """ CNN feature extractor for depth image 128x128 """
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 5, stride=2, padding=2),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 8, 5, stride=2, padding=2),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 8, 5, stride=2, padding=2),
                                nn.BatchNorm2d(8),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                Flatten())
                                # nn.Linear(1024, 256))
    
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    """ CNN feature extractor for depth image 128x128 """
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(16, 16, 3, padding=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(16, 16, 3, padding=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(16, 16, 3, padding=1),
                                nn.BatchNorm2d(16),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                Flatten(),
                                nn.Linear(1024, 256))
    
    def forward(self, x):
        return self.net(x)