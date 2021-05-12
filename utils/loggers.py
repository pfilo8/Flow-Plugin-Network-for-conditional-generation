from pytorch_lightning.loggers import TestTubeLogger


def get_logger(config):
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name']
    )
    tt_logger.experiment.tag(config)
    return tt_logger
