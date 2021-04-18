import json
from pathlib import Path

import argparse
import pandas as pd
import torch
import yaml

from pytorch_lightning.loggers import TestTubeLogger

from models import FLOWS
from models import VAE_MODELS


def get_parser_experiment():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--config', '-c',
        dest="filename",
        metavar='FILE',
        help='path to the config file',
        default='configs/vae.yaml'
    )
    return parser


def get_parser_latent_space():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--path', '-p',
        dest="path",
        metavar='FILE',
        help='path to the model'
    )
    return parser


def get_config(args):
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def get_logger(config):
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )
    tt_logger.experiment.tag(config)
    return tt_logger


def load_model(model_dir, model_factory='flow'):
    meta_path = Path(model_dir) / Path('meta_tags.csv')
    state_dict_path = list((Path(model_dir) / Path('checkpoints')).iterdir())[0]

    state_dict = torch.load(state_dict_path)
    state_dict = {str(k).replace('model.', ''): v for k, v in state_dict['state_dict'].items()}

    s = pd.read_csv(meta_path).iloc[0, 1]
    config = json.loads(s.replace("'", '"').replace("True", 'true'))

    if model_factory == 'flow':
        model = FLOWS[config['name']](
            **config['params']
        )
    elif model_factory == 'vae':
        name = config.pop('name')
        model = VAE_MODELS[name](**config)
    else:
        raise ValueError(f'{model_factory} model type not supported.')
    model.load_state_dict(state_dict)
    return model
