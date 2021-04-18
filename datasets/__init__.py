from .celeba import CelebaDataModule
from .flow import FlowDataModule
from .mnist import MNISTDataModule


def _dataset_mapper(config):
    dataset = config['dataset']['name']
    if dataset == 'mnist':
        return MNISTDataModule
    elif dataset == 'celeba':
        return CelebaDataModule
    elif dataset == 'flow':
        return FlowDataModule
    else:
        raise ValueError(f"Dataset {dataset} not found.")


def get_dataset(config):
    dataset_base = _dataset_mapper(config)
    dataset = dataset_base(
        **config['dataset']['dataset_params']
    )
    return dataset
