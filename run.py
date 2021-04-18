import pytorch_lightning as pl

from datasets import get_dataset
from experiments.vae import VAEXperiment
from models import get_model
from utils import get_config, get_parser_experiment, get_logger

args = get_parser_experiment().parse_args()
config = get_config(args)
tt_logger = get_logger(config)

pl.seed_everything(config['logging_params']['manual_seed'])

dataset = get_dataset(config)
model = get_model(config)
experiment = VAEXperiment(model, config['exp_params'])

if 'patience' in config.get('exp_params', {}):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['exp_params'].get('patience', 3),
    )
    callbacks = [early_stop_callback]
else:
    callbacks = None

runner = pl.Trainer(
    default_root_dir=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    deterministic=True,
    callbacks=callbacks,
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, dataset)
