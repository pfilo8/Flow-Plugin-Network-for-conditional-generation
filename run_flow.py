from models import *
from experiments.flow import FlowDataModule, FlowExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer

from flows import FLOWS
from utils import get_config, get_parser, get_logger

args = get_parser().parse_args()
config = get_config(args)
tt_logger = get_logger(config)

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
    **config['model_params']['params']
)
experiment = FlowExperiment(flow)

runner = Trainer(
    default_root_dir=f"{tt_logger.save_dir}",
    min_epochs=1,
    logger=tt_logger,
    **config['trainer_params']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, dataset)
