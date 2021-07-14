from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim

from sklearn.metrics import accuracy_score

from nflows.flows.base import Flow
from utils.classification import predict


class FlowExperiment(pl.LightningModule):

    def __init__(self, model: Flow, params: dict = None) -> None:
        super(FlowExperiment, self).__init__()
        self.model = model
        self.params = params if params is not None else {}

    def forward(self, x: torch.Tensor, y: Union[None, torch.Tensor]) -> torch.Tensor:
        return self.model.log_prob(inputs=x, context=y)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        if self.params.get('noise') is not None:
            x = x + self.params['noise'] * torch.rand_like(x)
        loss = -self.model.log_prob(inputs=x, context=y).mean()
        self.log('train_loss', loss.item())
        y_hat = predict(self.model, x, num_classes=self.params.get('num_classes'))
        return {'loss': loss, 'y': y.detach().cpu().numpy().argmax(axis=1), 'y_hat': y_hat}

    def training_epoch_end(self, outputs):
        y = np.hstack([el['y'] for el in outputs])
        y_hat = np.hstack([el['y_hat'] for el in outputs])
        self.log('train_acc', accuracy_score(y, y_hat))

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        x, y = batch
        loss = -self.model.log_prob(inputs=x, context=y).mean()
        self.log('val_loss', loss.item())
        y_hat = predict(self.model, x, num_classes=self.params.get('num_classes'))
        return {'loss': loss, 'y': y.detach().cpu().numpy().argmax(axis=1), 'y_hat': y_hat}

    def validation_epoch_end(self, outputs):
        y = np.hstack([el['y'] for el in outputs])
        y_hat = np.hstack([el['y_hat'] for el in outputs])
        self.log('val_acc', accuracy_score(y, y_hat))

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.get('LR', 1e-3),
            weight_decay=self.params.get('weight_decay', 0)
        )
        return optimizer
