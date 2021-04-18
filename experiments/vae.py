import torch
from torch import optim
from models import BaseVAE
import pytorch_lightning as pl
import torchvision.utils as vutils


class VAEXperiment(pl.LightningModule):

    def __init__(
            self,
            vae_model: BaseVAE,
            params: dict
    ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=0.01,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            # M_N=self.params['batch_size'] / self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )
        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        results = {'val_loss': avg_loss, 'log': tensorboard_logs}
        self.logger.experiment.log(results)
        return results

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/media/{self.logger.name}_{self.current_epoch}.png",
                normalize=True,
                nrow=12
            )
        except:
            pass

        del test_input, recons  # , samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
