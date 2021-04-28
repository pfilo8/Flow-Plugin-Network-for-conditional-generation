from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseVAE


class Decoder(nn.Module):
    def __init__(self, z_dim, image_channels=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_block(z_dim, hidden_dim * 4),
            self.make_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_block(hidden_dim * 2, hidden_dim),
            self.make_block(hidden_dim, image_channels, kernel_size=4, final_layer=True),
        )

    def make_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                                   stride=stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                                   stride=stride),
                nn.Sigmoid()
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


class Encoder(nn.Module):

    def __init__(self, image_channels=1, hidden_dim=16):
        super(Encoder, self).__init__()
        self.disc = nn.Sequential(
            self.make_block(image_channels, hidden_dim),
            self.make_block(hidden_dim, hidden_dim * 2),
            self.make_block(hidden_dim * 2, hidden_dim * 4, final_layer=True),
        )

    def make_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class ConvVAE(BaseVAE):
    def __init__(self, latent_dim, hidden_dim=16):
        super(ConvVAE, self).__init__()
        self.enc = Encoder(hidden_dim=hidden_dim)
        self.dec = Decoder(latent_dim)
        self.fc_mu = nn.Linear(hidden_dim * 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4, latent_dim)
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)  # mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decode(self, z):
        h = self.dec(z)
        return h

    def forward(self, x, **kwargs):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def loss_function(self, *args: Any, **kwargs) -> torch.Tensor:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = F.mse_loss(recons, input, reduction='mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device, **kwargs) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples
