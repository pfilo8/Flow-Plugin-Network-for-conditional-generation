from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

from utils import get_parser_latent_space

args = get_parser_latent_space().parse_args()
path = args.path

df_valid_z = pd.read_csv(path / Path("latent_space/z_train.csv"))
df_valid_y = pd.read_csv(path / Path("latent_space/y_train.csv"))

reducer = umap.UMAP()
print("Creating embeddings.")
embedding = reducer.fit_transform(df_valid_z)

for column in df_valid_y:
    print(column)

    fig, ax = plt.subplots(figsize=(20, 16))

    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=df_valid_y[column].values.squeeze(),
        ax=ax,
        legend='full'
    )
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path / Path(f"latent_space/column_{column}.png"), dpi=150)
