from abc import ABC, abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


class ContextMixerModule(ABC, pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])

        self.model = model
        self.criterion = criterion

        self._set_metrics()

    def _set_metrics(self):
        if self.hparams.metrics.train is not None:
            for k in self.hparams.metrics.train:
                setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        if self.hparams.metrics.val is not None:
            for k in self.hparams.metrics.val:
                setattr(self, f"val_{k}", self.hparams.metrics.val[k])

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

    def calc_dot_product(
        self,
        optflow_data: torch.Tensor,
        coords: torch.Tensor,
        station_coords: torch.Tensor,
    ):
        """
        Args:
            optflow_data: Optical flow data. Tensor of shape [B, T, 2*C, H, W]
            coords: Coordinates of each pixel. Tensor of shape [B, 2, H, W]
            station_coords: Coordinates of the station. Tensor of shape [B, 2, 1, 1]
        Returns:
            dp: Dot product between the optical flow and station vectors of shape [B, T, C, H, W]
        """
        optflow_data = rearrange(optflow_data, "b t (c n) h w -> b t c n h w", n=2)
        dist = station_coords - coords
        dist = repeat(
            dist,
            "b n h w -> b t c n h w",
            t=optflow_data.shape[1],
            c=optflow_data.shape[2],
        )
        optflow_data = F.normalize(
            rearrange(optflow_data, "b t c n h w -> b t c h w n"), p=2, dim=-1
        )
        dist = F.normalize(rearrange(dist, "b t c n h w -> b t c h w n"), p=2, dim=-1)
        dp = optflow_data[..., 0] * dist[..., 0] + optflow_data[..., 1] * dist[..., 1]
        return dp

    def prepare_batch(self, batch, use_target=True):
        x_ctx = batch["context"].float()
        x_opt = batch["optical_flow"].float()
        x_ts = batch["timeseries"].float()
        if use_target:
            y_ts = batch["target"].float()
            y_previous_ts = batch["target_previous"].float()
        spatial_coords = batch["spatial_coordinates"].float()
        time_coords = batch["time_coordinates"].float()
        ts_coords = batch["station_coords"].float()
        ts_elevation = batch["station_elevation"].float()

        aux_data = []
        for k in batch["auxiliary_data"]:
            ad = batch["auxiliary_data"][k].float()  # B, H, W
            ad = ad.unsqueeze(1).unsqueeze(1)
            ad = ad.repeat(1, x_ctx.shape[1], 1, 1, 1)
            aux_data.append(ad)
        x_ctx = torch.cat(
            [
                x_ctx,
            ]
            + aux_data,
            axis=2,
        )

        if self.hparams.use_dp:
            x_dp = self.calc_dot_product(x_opt, spatial_coords, ts_coords)
            x_ctx = torch.cat([x_ctx, x_dp], axis=2)
        else:
            x_ctx = torch.cat([x_ctx, x_opt], axis=2)
        ts_elevation = ts_elevation[..., 0, 0]
        ts_elevation = ts_elevation[(...,) + (None,) * 2]
        ts_elevation = ts_elevation.repeat(1, x_ts.shape[1], 1)
        x_ts = torch.cat([x_ts, ts_elevation], axis=-1)

        H, W = x_ctx.shape[-2:]
        ctx_coords = F.interpolate(
            spatial_coords,
            size=(H // self.model.patch_size[0], W // self.model.patch_size[1]),
            mode="bilinear",
        )
        if use_target:
            return x_ts, x_ctx, y_ts, y_previous_ts, ctx_coords, ts_coords, time_coords
        return x_ts, x_ctx, ctx_coords, ts_coords, time_coords

    @abstractmethod
    def forward(
        self,
        x_ctx: Tensor,
        ctx_coords: Tensor,
        x_ts: Tensor,
        ts_coords: Tensor,
        time_coords: Tensor,
    ) -> Tensor:
        pass

    @abstractmethod
    def training_step(self, train_batch, batch_idx) -> Any:
        pass

    @abstractmethod
    def validation_step(self, val_batch, batch_idx) -> Any:
        pass
