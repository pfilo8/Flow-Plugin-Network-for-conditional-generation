from typing import Union

import pytorch_lightning as pl
import torch
import torch.optim as optim

from nflows.flows.base import Flow


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
