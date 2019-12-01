import torch
import torch.nn as nn
import numpy as np

import nn.model_utils as models
import nn.utils as ut

class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        self.x_dim = 5      # x, y, z, sin, cos
        self.y_dim = 256    # cnn features         
        self.z_dim = 32     # latent variable

        self.dec = Decoder(self.x_dim, self.z_dim, self.y_dim)
        self.enc = Encoder(self.x_dim, self.z_dim, self.y_dim)
        self.cnn = models.CNN()

        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, x):
        """ Inference for a depth image -> grasp """
        # get features from image
        features = self.cnn(x)

        # sample z from prior, batch number of samples
        z = self.sample_z(x.shape[0])

        # decode features and z
        grasp = self.dec.decode(z, features)

        return grasp

    def loss(self, x, g):
        """ Negative evidence lower bound (nelbo)
            nelbo = kl + rec
        """
        # Get features from depth image
        features = self.cnn(x)

        # Encoder evaluates q_phi(z|x), returning m and v 
        qm, qv = self.enc.encode(g, features)

        # Sample z ~ N(m,v)
        z = ut.sample_gaussian(qm, qv)

        # Decoder takes in sampled latent variable and reconstructs output
        g_hat = self.dec.decode(z, features)

        # log p(x | z)
        rec = models.AngleLoss(g_hat, g, norm=True) + models.DistanceLoss(g_hat, g)
        kl = torch.mean(ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v), dim=0)

        nelbo = kl + rec

        return nelbo, kl, rec

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, h_dim=300):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, h_dim=300):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, x_dim)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)