from functools import partial

from .autoregressive import MaskedAutoregressiveFlow
from .base import Flow
from .realnvp import SimpleRealNVP

FLOWS = {
    'MAF': MaskedAutoregressiveFlow,
    'NICE': partial(SimpleRealNVP, use_volumne_perserving=True),
    'RNVP': partial(SimpleRealNVP, use_volumne_perserving=False)
}
