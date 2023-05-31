import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
import wandb



class TSForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: dict,
        criterion: torch.nn.Module,
        seq_len: int,
        label_len: int,
        pred_len: int,
        padding: int = 0,
        inverse_scaling: bool = False,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])

        self.model = model
        self.criterion = criterion

        self.train_loss = MeanMetric()

        self.val_loss = MeanMetric()


        self._set_metrics()

    def _set_metrics(self):
        for k in self.hparams.metrics.train:
            setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        for k in self.hparams.metrics.val:
            setattr(self, f"val_{k}", self.hparams.metrics.val[k])

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        if self.hparams.padding == 0:
            decoder_input = torch.zeros((batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))).type_as(batch_y)
        else:  # self.hparams.padding == 1
            decoder_input = torch.ones((batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))).type_as(batch_y)
        decoder_input = torch.cat([batch_y[:, : self.hparams.label_len, :], decoder_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if self.hparams.output_attention:
            outputs = outputs[0]
        return outputs


    def configure_optimizers(self):

        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def prepare_batch(self, batch):
        x_ts = batch["x"].float()
        y_ts = batch["y"].float()
        x_timestamp = batch["x_timestamp"].float()
        y_timestamp = batch["y_timestamp"].float()
        ts_coords= batch["station_coords"].float()
        ts_elevation = batch["station_elevation"].float()
        ghi_id = batch["ghi"].long()[0].item()
        
        T = x_ts.shape[1]
        ts_elevation_x = ts_elevation[..., 0, 0]
        ts_elevation_x = ts_elevation_x[(...,) + (None,) * 2]
        ts_elevation_x = ts_elevation_x.repeat(1, T, 1)

        ts_coords_x = ts_coords.squeeze()
        ts_coords_x = ts_coords_x.unsqueeze(1)
        ts_coords_x = ts_coords_x.repeat(1, T, 1)

        x_ts = torch.cat([x_ts, ts_coords_x, ts_elevation_x], axis=-1)

        
        # do the same with y_ts
        T = y_ts.shape[1]
        ts_elevation_y = ts_elevation[..., 0, 0]
        ts_elevation_y = ts_elevation_y[(...,) + (None,) * 2]
        ts_elevation_y = ts_elevation_y.repeat(1, T, 1)

        ts_coords_y = ts_coords.squeeze()
        ts_coords_y = ts_coords_y.unsqueeze(1)
        ts_coords_y = ts_coords_y.repeat(1, T, 1)
        
        y_ts = torch.cat([y_ts, ts_coords_y, ts_elevation_y], axis=-1)

        return x_ts, y_ts, x_timestamp, y_timestamp, ghi_id

        
    def training_step(self, train_batch, batch_idx):

        batch_x, batch_y, batch_x_mark, batch_y_mark , ghi_id= self.prepare_batch(train_batch)
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark).squeeze() 
        outputs = outputs[:, -self.model.pred_len :, ghi_id]   # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
        batch_y = batch_y[:, -self.model.pred_len :, ghi_id]

        loss = self.criterion(outputs, batch_y)

        self.train_loss(loss)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )


        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            metric(outputs, batch_y)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark, ghi_id = self.prepare_batch(val_batch)
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        print(outputs.shape)
        print(batch_y.shape)
        outputs = outputs[:, -self.model.pred_len :, ghi_id]   # 0 is the index of the target variable GHI , 8 is the index of the target variable DHI
        batch_y = batch_y[:, -self.model.pred_len :, ghi_id]  

        loss = self.criterion(outputs, batch_y)

        self.val_loss(loss)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        


        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            metric(outputs, batch_y)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return {'grountruth': batch_y, 'prediction': outputs}
