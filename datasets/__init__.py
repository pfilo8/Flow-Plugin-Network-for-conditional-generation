from .celeba import CelebaDataModule
from .flow import FlowDataModule
from .mnist import MNISTDataModule


def _dataset_mapper(config):
    dataset = config['exp_params']['dataset']
    if dataset == 'mnist':
        return MNISTDataModule
    elif dataset == 'celeba':
        return CelebaDataModule
    else:
        raise ValueError(f"Dataset {dataset} not found.")


def get_dataset(config):
    dataset_base = _dataset_mapper(config)
    dataset = dataset_base(
        **config['exp_params']
    )
    return dataset
