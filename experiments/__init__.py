from .flow import FlowExperiment
from .vae import VAEXperiment


def get_experiment(config, model):
    experiment_type = config['model_params']['type']
    experiment_config = config.get('exp_params', {})
    experiment_config['num_classes'] = config['model_params']['params']['context_features']
    if experiment_type == 'flow':
        return FlowExperiment(model, experiment_config)
    elif experiment_type == 'vae':
        return VAEXperiment(model, experiment_config)
    else:
        raise ValueError(f"Experiment type: {experiment_type} not found.")
