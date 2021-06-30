from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
import torch

from utils import get_parser_model_flow

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_parser_model_flow().parse_args()
model_path = args.model_path
flow_path = args.flow_path

df_valid_z = pd.read_csv(model_path / Path("latent_space/z_train.csv"))
df_valid_y = pd.read_csv(model_path / Path("latent_space/y_train.csv"))

df_samples_z = pd.read_csv(flow_path / Path("samples/xs.csv"))
df_samples_z = df_samples_z.dropna()  # Flow may sometimes generate NaNs
df_samples_y = pd.read_csv(flow_path / Path("samples/ys.csv"))
df_samples_y = df_samples_y.loc[df_samples_z.dropna().index]

reducer = umap.UMAP()
print("Creating embeddings.")
embedding = reducer.fit_transform(df_valid_z)
embedding_samples = reducer.transform(df_samples_z)

palette = sns.color_palette("tab10") if 'mnist' in model_path else None

for column in df_valid_y:
    print(column)

    fig, ax = plt.subplots(figsize=(20, 16))

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=df_valid_y[column].values.squeeze(),
        ax=ax,
        legend='full',
        palette=palette
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(flow_path / Path(f"samples/model_latent_space_samples_{column}.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(20, 16))

    sns.scatterplot(
        x=embedding_samples[:, 0],
        y=embedding_samples[:, 1],
        hue=df_samples_y[column].values.squeeze(),
        ax=ax,
        legend='full',
        palette=palette
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(flow_path / Path(f"samples/flow_latent_space_samples_{column}.png"), dpi=150)
