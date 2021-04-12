from models import *
from experiments.flow import FlowDataModule, FlowExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

from flows import FLOWS
from utils import get_config, get_parser

args = get_parser().parse_args()
config = get_config(args)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

dataset = FlowDataModule(
    data_path_train_x=config['exp_params']['data_path_train_x'],
    data_path_train_y=config['exp_params']['data_path_train_y'],
    batch_size=config['exp_params']['batch_size'],
)
flow = FLOWS[config['model_params']['name']](
    features=config['model_params']['features'],
    hidden_features=config['model_params']['hidden_features'],
    context_features=config['model_params'].get('context_features')
)
experiment = FlowExperiment(flow)

runner = Trainer(
    default_save_path=f"{tt_logger.save_dir}",
    min_nb_epochs=1,
    logger=tt_logger,
    log_save_interval=100,
    train_percent_check=1.,
    val_percent_check=1.,
    num_sanity_val_steps=5,
    early_stop_callback=False,
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, dataset)
torch.save(runner.model.model, "model.ckpt")
