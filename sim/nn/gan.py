import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import nn.model_utils as models

def gan_loss(g, d, x_real, y):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.z_dim, device=x_real.device, requires_grad=False)
    x_fake = g(z, y)

    real_prob = torch.ones(batch_size, 1, device=x_real.device, requires_grad=False)
    fake_prob = torch.zeros(batch_size, 1, device=x_real.device, requires_grad=False)
    d_loss_fake = F.binary_cross_entropy_with_logits(d(x_fake, y), fake_prob)
    d_loss_real = F.binary_cross_entropy_with_logits(d(x_real, y), real_prob)
    d_loss = d_loss_fake + d_loss_real
    g_loss = -torch.mean(F.logsigmoid(d(x_fake, y)).squeeze(), dim=0)

    return d_loss, g_loss

class GANModel(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim=5, y_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.D = Discriminator(h_dim, x_dim=x_dim, y_dim=y_dim)
        self.G = Generator(z_dim, h_dim, x_dim=x_dim, y_dim=y_dim)
        self.name = 'gan_z{}_h{}'.format(z_dim, h_dim)

    def forward(self, y, z_input=None):
        if z_input is None:
            z = torch.randn(y.shape[0], self.z_dim, device=y.device)
        else:
            # duplicate scalar z_input to fill tensor
            z = torch.ones(y.shape[0], self.z_dim, device=y.device) * z_input

        return self.G(z, y) 

    def loss(self, x, g):
        return gan_loss(self.G, self.D, g, x)    


class Discriminator(nn.Module):
    def __init__(self, h_dim, x_dim=5, y_dim=32):
        super().__init__()
        self.x_fc = nn.Linear(x_dim, h_dim)
        self.y_fc = nn.Linear(y_dim, h_dim)
        self.cnn = models.MiniCNN()

        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x, y):
        features = self.cnn(y)
        logp = self.net(self.x_fc(x) + self.y_fc(features))
        # x_fc = self.x_fc(x)
        # y_fc = self.y_fc(features)
        # logp = self.net(torch.cat([x_fc, y_fc], dim=1))
        return logp

class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim=5, y_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.cnn = models.MiniCNN()

        self.z_fc = nn.Linear(z_dim, h_dim)
        self.y_fc = nn.Linear(y_dim, h_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, x_dim),
        )

        self.norm_layer = models.Normalize()
        
    def forward(self, z, y):
        features = self.cnn(y)
        x = self.net(self.z_fc(z) + self.y_fc(features))
        position = x[:, 0:3]
        angle = self.norm_layer(x[:, 3:])
        return torch.cat([position, angle], dim=1)
