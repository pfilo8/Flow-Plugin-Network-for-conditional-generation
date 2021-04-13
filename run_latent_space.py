from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

from utils import get_parser_latent_space, load_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_latent_space(model, dataloader):
    ys = []
    zs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            mu, log_var = model.encode(x)
            z = model.reparameterize(mu, log_var)
            zs.append(z.detach().cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(zs), np.vstack(ys)


args = get_parser_latent_space().parse_args()
path = args.path

data_dir = path / Path('latent_space')
data_dir.mkdir(exist_ok=True)

model = load_model(path, 'vae').to(DEVICE)

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize(64),
    transforms.ToTensor(),
    SetRange
])

bs = 256
dataset_train = CelebA(root='data', split="train", transform=transform, download=False)
dataset_valid = CelebA(root='data', split="valid", transform=transform, download=False)
dataset_test = CelebA(root='data', split="test", transform=transform, download=False)

dataloader_train = DataLoader(dataset_train, batch_size=bs)
dataloader_valid = DataLoader(dataset_valid, batch_size=bs)
dataloader_test = DataLoader(dataset_test, batch_size=bs)

zs_train, ys_train = get_latent_space(model, dataloader_train)
zs_valid, ys_valid = get_latent_space(model, dataloader_valid)
zs_test, ys_test = get_latent_space(model, dataloader_test)

pd.DataFrame(zs_train).to_csv(data_dir / Path('z_train.csv'), index=False)
pd.DataFrame(zs_valid).to_csv(data_dir / Path('z_valid.csv'), index=False)
pd.DataFrame(zs_test).to_csv(data_dir / Path('z_test.csv'), index=False)

pd.DataFrame(ys_train).to_csv(data_dir / Path('y_train.csv'), index=False)
pd.DataFrame(ys_valid).to_csv(data_dir / Path('y_valid.csv'), index=False)
pd.DataFrame(ys_test).to_csv(data_dir / Path('y_test.csv'), index=False)
