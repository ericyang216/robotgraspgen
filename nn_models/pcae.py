# Copyright (c) 2018 Rui Shu
import torch
from torch import nn
from torch.nn import functional as F
from nn_models import model_utils as ut
from chamfer.chamfer_distance import ChamferDistance 
chamfer_dist = ChamferDistance()

class PCAE(nn.Module):
    def __init__(self, name='pcae', z_dim=1024, x_dim=512):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.x_dim = x_dim # N x 3 set of points

        self.enc = Encoder(self.x_dim, self.z_dim)
        self.dec = Decoder(self.x_dim, self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

    def loss_chamfer(self, x):
        x_hat = self.forward(x)
        d1, d2 = chamfer_dist(x, x_hat)
        return torch.mean(d1) + torch.mean(d2)

    def loss_emd(self, x):
        pass

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        # Encoder evaluates q_phi(z|x), returning m and v 
        qm, qv = self.enc.encode(x)

        # Sample z ~ N(m,v)
        z = ut.sample_gaussian(qm, qv)

        # Decoder takes in sampled latent variable and reconstructs output
        x_hat = self.dec.decode(z)

        # log p(x | z)
        # rec = torch.mean(-ut.log_bernoulli_with_logits(x, x_hat), dim=0)
        # rec = torch.mean(chamfer_dist(x, x_hat), dim=0)
        rec = self.loss_chamfer(x, x_hat)
        kl = torch.mean(ut.kl_normal(qm, qv, self.z_prior_m, self.z_prior_v), dim=0)

        nelbo = kl + rec

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        
        # Encoder evaluates q_phi(z|x), returning m and v 
        qm, qv = self.enc.encode(x)
        qm = ut.duplicate(qm.unsqueeze(0), iw)
        qv = ut.duplicate(qv.unsqueeze(0), iw)

        # Sample z ~ N(m,v), iw times per x: [batchsize * iw, z_dim]
        z = ut.sample_gaussian(qm, qv)

        # Decoder takes in sampled latent variable and reconstructs output
        x_hat = self.dec.decode(z)

        # log p(x | z) and kl(q || p)
        # rec = ut.log_bernoulli_with_logits(ut.duplicate(x.unsqueeze(0), iw), x_hat)
        rec = self.loss_chamfer(x, x_hat)

        # Duplicate z_prior to match dimensions [batchsize * iw, z_dim]
        z_prior_m = ut.duplicate(ut.duplicate(self.z_prior_m, self.z_dim).unsqueeze(0), x.shape[0])
        z_prior_v = ut.duplicate(ut.duplicate(self.z_prior_v, self.z_dim).unsqueeze(0), x.shape[0])
        z_prior_m = ut.duplicate(z_prior_m.unsqueeze(0), iw)
        z_prior_v = ut.duplicate(z_prior_v.unsqueeze(0), iw)

        # log(p(z)) - log(q(z|x)) [iw, batchsize]
        log_p_z = ut.log_normal(z, z_prior_m, z_prior_v)
        log_q_zx = ut.log_normal(z, qm, qv)
        kl = log_p_z - log_q_zx 

        # log_exp_mean over iw, mean over batch
        # negate iwae to niwae after log_mean_exp
        niwae = torch.mean(-ut.log_mean_exp(rec + kl, dim=0))

        rec = torch.mean(rec)
        kl = torch.mean(kl)
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    # def sample_sigmoid(self, batch):
    #     z = self.sample_z(batch)
    #     return self.compute_sigmoid_given(z)

    # def compute_sigmoid_given(self, z):
    #     logits = self.dec.decode(z)
    #     return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        # return torch.bernoulli(self.compute_sigmoid_given(z))
        return self.dec(z)
        
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim=0):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=z_dim, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=x_dim)
        )

    def forward(self, x):
        return self.net(x).transpose(2, 1)

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim=0):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=z_dim),
            nn.ReLU(),
            nn.Linear(in_features=z_dim, out_features=z_dim),
            nn.ReLU(),
            nn.Linear(in_features=z_dim, out_features=z_dim),
            nn.ReLU(),
            nn.Linear(in_features=z_dim, out_features=x_dim * 3)
        )

    def forward(self, z):
        x = self.net(z)
        return x.view((-1, 3, self.x_dim))

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)