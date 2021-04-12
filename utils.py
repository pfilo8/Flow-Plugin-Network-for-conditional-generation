import argparse
import pytorch_lightning as pl
import yaml

from pytorch_lightning.loggers import TestTubeLogger


def get_parser():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--config', '-c',
        dest="filename",
        metavar='FILE',
        help='path to the config file',
        default='configs/vae.yaml'
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


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper
