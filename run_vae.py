import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer

from experiments.vae import VAEXperiment
from models import vae_models
from utils import get_config, get_parser_experiment, get_logger

args = get_parser_experiment().parse_args()
config = get_config(args)
tt_logger = get_logger(config)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

runner = Trainer(
    default_root_dir=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
