import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import CelebA
from torchvision import transforms


class CelebaDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = './',
            batch_size: int = 64,
            num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1.)
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.celeba_train = CelebA(self.data_dir, split='train', transform=self.transform)
        self.celeba_val = CelebA(self.data_dir, split='valid', transform=self.transform)
        self.celeba_test = CelebA(self.data_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        self.num_train_images = len(self.celeba_train)
        return DataLoader(self.celeba_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        self.num_val_images = len(self.celeba_val)
        return DataLoader(self.celeba_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        self.num_test_images = len(self.celeba_test)
        return DataLoader(self.celeba_test, batch_size=self.batch_size, num_workers=self.num_workers)
