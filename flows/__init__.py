from .autoregressive import MaskedAutoregressiveFlow
from .realnvp import SimpleRealNVP

FLOWS = {
    'MAF': MaskedAutoregressiveFlow,
    'RNVP': SimpleRealNVP
}
