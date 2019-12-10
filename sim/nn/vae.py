import torch
import torch.nn as nn
import numpy as np

import nn.model_utils as models
import nn.utils as ut

class VAEModel(nn.Module):
    def __init__(self, z_dim=32, h_dim=256):
        super(VAEModel, self).__init__()
        self.x_dim = 5      # x, y, z, sin, cos
        self.y_dim = 32    # cnn features         
        self.z_dim = z_dim     # latent variable
        self.name = 'vae_z{}_h{}'.format(z_dim, h_dim)

        self.dec = Decoder(self.x_dim, self.z_dim, self.y_dim, h_dim)
        self.enc = Encoder(self.x_dim, self.z_dim, self.y_dim, h_dim)
        # self.cnn = models.CNN()
        self.cnn = models.MiniCNN()
        self.norm_layer = models.Normalize()
    
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, x, z_input=None):
        """ Inference for a depth image -> grasp """
        # get features from image
        features = self.cnn(x)

        # sample z from prior, batch number of samples
        if z_input is None:
            z = self.sample_z(x.shape[0])
        # z should be size of [batch, self.z_dim]
        else:
            # duplicate scalar z_input to fill tensor
            z = torch.ones(x.shape[0], self.z_dim, device=x.device) * z_input

        # decode features and z
        grasp = self.dec.decode(z, features)
        # return grasp

        position = grasp[:, 0:3]
        angle = self.norm_layer(grasp[:, 3:])
        return torch.cat([position, angle], dim=1)


    def loss(self, x, g, k_angle=1.0):
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
        g_samp = self.dec.decode(z, features)
        position = g_samp[:, 0:3]
        angle = self.norm_layer(g_samp[:, 3:])
        g_hat = torch.cat([position, angle], dim=1)

        # log p(x | z)
        rec = k_angle * models.AngleLoss(g_hat, g, norm=True) + models.DistanceLoss(g_hat, g)
        kl = torch.mean(ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v), dim=0)

        nelbo = kl + rec

        return nelbo, kl, rec

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, h_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_fc = nn.Linear(x_dim, h_dim)
        self.y_fc = nn.Linear(y_dim, h_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 2 * z_dim),
        )

    def encode(self, x, y):
        # xy = x if y is None else torch.cat((x, y), dim=1)
        # h = self.net(xy)
        h = self.net(self.x_fc(x) + self.y_fc(y))
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, h_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.z_fc = nn.Linear(z_dim, h_dim)
        self.y_fc = nn.Linear(y_dim, h_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, x_dim)
        )

    def decode(self, z, y):
        # zy = z if y is None else torch.cat((z, y), dim=1)
        # return self.net(zy)
        x = self.net(self.z_fc(z) + self.y_fc(y))
        return x