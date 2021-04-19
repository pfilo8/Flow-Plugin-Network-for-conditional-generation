import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = './',
            batch_size: int = 64,
            num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1.)
        ])
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
