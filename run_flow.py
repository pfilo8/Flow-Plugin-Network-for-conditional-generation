import pytorch_lightning as pl

from experiments.flow import FlowDataModule, FlowExperiment
from flows import FLOWS
from utils import get_config, get_parser_experiment, get_logger

args = get_parser_experiment().parse_args()
config = get_config(args)
tt_logger = get_logger(config)

pl.seed_everything(config['logging_params']['manual_seed'])

dataset = FlowDataModule(
    data_path_train_x=config['exp_params']['data_path_train_x'],
    data_path_train_y=config['exp_params']['data_path_train_y'],
    batch_size=config['exp_params']['batch_size'],
    num_workers=config['exp_params']['num_workers']
)
flow = FLOWS[config['model_params']['name']](
    **config['model_params']['params']
)
experiment = FlowExperiment(flow)

runner = pl.Trainer(
    default_root_dir=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    deterministic=True,
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, dataset)
