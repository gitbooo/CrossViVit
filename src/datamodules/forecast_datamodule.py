from typing import Any, Dict, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datasets.forecast_dataset import TSDataset


class TSDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dict[str, Any],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if stage == "fit" or stage is None:
            self.train_dataset = TSDataset(**self.hparams.dataset, mode="train")
            self.val_dataset = TSDataset(**self.hparams.dataset, mode="val")
        if stage == "test" or stage is None:
            self.test_dataset = TSDataset(**self.hparams.dataset, mode="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    @property
    def num_features(self):
        return self.train_dataset.num_features

    @property
    def feature_names(self):
        return self.train_dataset.feature_names
