import torch
import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader
from jigsaw.dataset.dataset import build_geometry_dataloader
from jigsaw.dataset.dataset import build_test_dataloader


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_data, self.val_data = build_geometry_dataloader(cfg)
        self.test_data = build_test_dataloader(cfg)

    def train_dataloader(self):
        return self.train_data

    def val_dataloader(self):
        return self.train_data
    
    # Make the val data to the test data    
    def test_dataloader(self):
        return self.test_data

