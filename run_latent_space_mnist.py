from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder

from datasets import get_dataset
from utils import get_parser_latent_space, get_config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_latent_space(model, dataloader):
    ys = []
    zs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            z = model.encode(x)
            #mu, log_var = model.encode(x)
            #z = model.reparametrize(mu, log_var)
            zs.append(z.detach().cpu().numpy())
            ys.append(y.numpy().reshape(-1, 1))
    return np.vstack(zs), np.vstack(ys)


def get_ohe_labels(encoder, df):
    return pd.DataFrame(encoder.transform(df), columns=encoder.categories_[0])


args = get_parser_latent_space().parse_args()
path = args.path

data_dir = path / Path('latent_space')
data_dir.mkdir(exist_ok=True)

config = get_config(path / Path('config.yaml'))
model = torch.load(path / Path('checkpoints') / Path('model.pkt')).cuda()

dataset = get_dataset(config)
dataset.prepare_data()
dataset.setup()

dataloader_train = dataset.train_dataloader()
dataloader_test = dataset.test_dataloader()

zs_train, ys_train = get_latent_space(model, dataloader_train)
zs_test, ys_test = get_latent_space(model, dataloader_test)

ohe = OneHotEncoder(sparse=False).fit(ys_train)

pd.DataFrame(zs_train).to_csv(data_dir / Path('z_train.csv'), index=False)
pd.DataFrame(zs_test).to_csv(data_dir / Path('z_test.csv'), index=False)

pd.DataFrame(ys_train).to_csv(data_dir / Path('y_train.csv'), index=False)
pd.DataFrame(ys_test).to_csv(data_dir / Path('y_test.csv'), index=False)
get_ohe_labels(ohe, ys_train).to_csv(data_dir / Path('y_train_ohe.csv'), index=False)
get_ohe_labels(ohe, ys_test).to_csv(data_dir / Path('y_test_ohe.csv'), index=False)
