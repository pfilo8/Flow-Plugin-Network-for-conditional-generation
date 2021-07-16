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
df_samples_y = df_samples_y.loc[df_samples_z.index]

if 'shapenet' in model_path:
    classes = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench',
               'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet',
               'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock',
               'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet',
               'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox',
               'microphone', 'microwave', 'monitor', 'motorcycle', 'mug', 'piano',
               'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle',
               'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table',
               'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']
    mapping = {idx: cls for idx, cls in enumerate(classes)}
    df_samples_y['cate'] = df_samples_y['cate'].map(mapping)

    # Class selecting
    # 10 most common classes
    """
    selected_classes = ['table', 'chair', 'airplane', 'car', 'sofa', 'rifle', 'lamp', 'vessel', 'bench', 'speaker']

    df_index = df_valid_y['cate'].isin(selected_classes)
    df_samples_index = df_samples_y['cate'].isin(selected_classes)

    df_valid_y = df_valid_y.loc[df_index]
    df_samples_y = df_samples_y.loc[df_samples_index]
    """

    df_valid_y = df_valid_y.sort_values('cate')  # Persist correct order of y
    df_valid_z = df_valid_z.loc[df_valid_y.index]

    df_samples_y = df_samples_y.sort_values('cate')  # Persist correct order of y
    df_samples_z = df_samples_z.loc[df_samples_y.index]

reducer = umap.UMAP(random_state=42)
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
        legend=None,
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
        legend=None,
        palette=palette
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(flow_path / Path(f"samples/flow_latent_space_samples_{column}.png"), dpi=150)
