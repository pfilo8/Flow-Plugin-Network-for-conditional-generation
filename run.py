import torch
import pytorch_lightning as pl

from datasets import get_dataset
from experiments import get_experiment
from models import get_model
from utils import get_config, get_parser_experiment, get_logger, save_config

args = get_parser_experiment().parse_args()
config = get_config(args.filename)
tt_logger = get_logger(config)

pl.seed_everything(config['logging_params']['manual_seed'])

dataset = get_dataset(config)
model = get_model(config)
experiment = get_experiment(config, model)

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
try:
    runner.fit(experiment, dataset)
finally:
    torch.save(
        runner.model.model,
        f"{runner.logger.save_dir}/{runner.logger.name}/version_{runner.logger.version}/checkpoints/model.pkt"
    )
    save_config(
        config,
        f"{runner.logger.save_dir}/{runner.logger.name}/version_{runner.logger.version}/config.yaml"
    )
