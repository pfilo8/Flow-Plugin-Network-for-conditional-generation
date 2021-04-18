import pytorch_lightning as pl

from datasets import CelebaDataModule
from experiments.vae import VAEXperiment
from models import VAE_MODELS
from utils import get_config, get_parser_experiment, get_logger

args = get_parser_experiment().parse_args()
config = get_config(args)
tt_logger = get_logger(config)

pl.seed_everything(config['logging_params']['manual_seed'])

dataset = CelebaDataModule(
    data_dir=config['exp_params']['data_path'],
    batch_size=config['exp_params']['batch_size'],
    num_workers=config['exp_params']['num_workers']
)
model = VAE_MODELS[config['model_params']['name']](**config['model_params']['params'])
experiment = VAEXperiment(model, config['exp_params'])
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=config['exp_params'].get('patience', 3),
)

runner = pl.Trainer(
    default_root_dir=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    deterministic=True,
    callbacks=[early_stop_callback],
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, dataset)
