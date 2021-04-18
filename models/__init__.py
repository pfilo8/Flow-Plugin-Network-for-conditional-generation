from .flows.autoregressive import MaskedAutoregressiveFlow
from .flows.realnvp import SimpleRealNVP

from .vae.vanilla_vae import VanillaVAE
from .vae.dfcvae import DFCVAE

VAE_MODELS = {
    'DFCVAE': DFCVAE,
    'VAE': VanillaVAE
}

FLOWS = {
    'MAF': MaskedAutoregressiveFlow,
    'RNVP': SimpleRealNVP
}


def _get_model_base(model_type, model_name):
    if model_type == 'vae':
        return VAE_MODELS[model_name]
    elif model_type == 'flow':
        return FLOWS[model_name]
    else:
        raise ValueError(f"Model type {model_type} not found.")


def get_model(config):
    model_base = _get_model_base(
        model_type=config['model_params']['type'],
        model_name=config['model_params']['name']
    )
    model_params = config['model_params']['params']
    model = model_base(**model_params)
    return model
