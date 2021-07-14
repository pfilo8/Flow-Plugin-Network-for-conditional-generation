from pathlib import Path

import numpy as np
import pandas as pd
import torch

Y_COLUMN = 'y'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataframes(path):
    def load_df(path_z, path_y):
        df_z = pd.read_csv(path_z)
        df_y = pd.read_csv(path_y)
        df_y = df_y.rename(columns={'0': Y_COLUMN, 'cate': Y_COLUMN})
        return pd.concat([df_z, df_y], axis=1)

    path_train_z = path / Path("latent_space/z_train.csv")
    path_train_y = path / Path("latent_space/y_train_numeric.csv")
    path_test_z = path / Path("latent_space/z_test.csv")
    path_test_y = path / Path("latent_space/y_test_numeric.csv")

    df_train = load_df(path_train_z, path_train_y)
    df_test = load_df(path_test_z, path_test_y)
    return df_train, df_test


def predict(flow, x, num_classes, log_weights=None):
    results = []

    with torch.no_grad():
        for i in range(num_classes):
            context = torch.zeros(len(x), num_classes).to(DEVICE)
            context[:, i] = 1.0
            results.append(flow.log_prob(x, context).detach().cpu().numpy())

    y_prob = np.stack(results, axis=1)
    if log_weights is not None:
        y_prob = y_prob + log_weights
    y_hat = y_prob.argmax(axis=1)
    return y_hat
