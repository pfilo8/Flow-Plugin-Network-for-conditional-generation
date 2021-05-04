from .flow import FlowExperiment
from .vae import VAEXperiment


def get_experiment(config, model):
    experiment_type = config['model_params']['type']
    if experiment_type == 'flow':
        return FlowExperiment(model, config.get('exp_params', {}))
    elif experiment_type == 'vae':
        return VAEXperiment(model, config.get('exp_params', {}))
    else:
        raise ValueError(f"Experiment type: {experiment_type} not found.")
