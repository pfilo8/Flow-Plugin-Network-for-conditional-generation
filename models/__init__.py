from .base import BaseVAE
from .vanilla_vae import VanillaVAE
from .dfcvae import DFCVAE

# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

VAE_MODELS = {
    'DFCVAE': DFCVAE,
    'VAE': VAE
}
