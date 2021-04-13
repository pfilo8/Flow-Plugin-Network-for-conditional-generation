from typing import List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim

from nflows.flows.base import Flow
from torch.utils.data import Dataset, DataLoader


class FlowExperiment(pl.LightningModule):

    def __init__(self, model: Flow, params: dict = None) -> None:
        super(FlowExperiment, self).__init__()
        self.model = model
        self.params = params if params is not None else {}

    def forward(self, x: torch.Tensor, y: Union[None, torch.Tensor]) -> torch.Tensor:
        return self.model.log_prob(inputs=x, context=y)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        loss = -self.model.log_prob(inputs=x, context=y).mean()
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        loss = -self.model.log_prob(inputs=x, context=y).mean()
        self.log('val_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.get('LR', 1e-3),
            weight_decay=self.params.get('weight_decay', 0)
        )
        return optimizer


class CSVDataset(Dataset):

    def __init__(self, x_path, y_path=None):
        self.df_x = pd.read_csv(x_path)
        if y_path is not None:
            self.df_y = pd.read_csv(y_path)
            assert len(self.df_x) == len(self.df_y)

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, idx):
        x = self.df_x.iloc[idx, :].values
        x = torch.tensor(x, dtype=torch.float)

        if hasattr(self, 'df_y'):
            y = self.df_y.iloc[idx, :].values
            y = torch.tensor(y, dtype=torch.float)
            return x, y
        return x


class FlowDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_path_train_x: str,
            data_path_train_y: Union[None, str],
            data_path_valid_x: str,
            data_path_valid_y: [Union[None, str]],
            batch_size: int = 64,
            num_workers: int = 0
    ):
        super().__init__()
        self.data_path_train_x = data_path_train_x
        self.data_path_train_y = data_path_train_y
        self.data_path_valid_x = data_path_valid_x
        self.data_path_valid_y = data_path_valid_y
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        self.train_data = CSVDataset(self.data_path_train_x, self.data_path_train_y)
        self.valid_data = CSVDataset(self.data_path_valid_x, self.data_path_valid_y)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers)
