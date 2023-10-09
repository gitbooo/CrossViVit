import torch
import wandb
from torchmetrics import MeanMetric

from . import ContextMixerModule


class CrossViViTQuantileMask(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.cutoff_epoch = self.hparams.cutoff_epoch

        self.train_quantile_loss = MeanMetric()
        self.train_ce_loss = MeanMetric()
        self.train_loss = MeanMetric()

        self.val_quantile_loss = MeanMetric()
        self.val_ce_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask):
        out, quantile_mask, self_attention_scores, cross_attention_scores = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask
        )

        return out, quantile_mask

    def get_bce_loss(self, y_hat, y_ts, mask):
        n_heads = y_hat.shape[2]
        y_ts_ = y_ts.unsqueeze(2).repeat(1, 1, n_heads, 1)
        error = (y_hat - y_ts_).abs().mean(dim=-1)  # B, T, Q
        indices = torch.argmin(error, dim=-1)  # B, T
        indices = indices.reshape(-1)
        mask = mask.reshape(-1, n_heads)
        return self.ce_loss(mask, indices)

    def get_mae_loss(self, y_hat, y_ts, mask):
        n_heads = y_hat.shape[2]
        y_ts_ = y_ts.unsqueeze(2).repeat(1, 1, n_heads, 1)
        error = (y_hat - y_ts_).abs().mean(dim=-1)  # B, T, Q
        mask = torch.nn.functional.softmax(mask, dim=-1)
        return (error * mask).sum() / (mask.sum())

    def training_step(self, train_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(train_batch)

        y_hat, quantile_mask = self(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True
        )

        quantile_loss = self.criterion(y_hat, y_ts)

        loss = quantile_loss
        if self.current_epoch >= self.cutoff_epoch:
            mae_loss = self.get_mae_loss(y_hat, y_ts, quantile_mask)
            loss += mae_loss
            self.train_ce_loss(mae_loss)
            self.log("train/mae_loss", self.train_ce_loss, on_step=True, prog_bar=True)

        self.train_loss(loss)

        self.train_quantile_loss(quantile_loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        self.log(
            "train/quantile_loss", self.train_quantile_loss, on_step=True, prog_bar=True
        )
        indices = torch.argmax(
            quantile_mask.unsqueeze(-1).repeat(1, 1, 1, y_hat.shape[-1]), dim=2
        )
        y_hat = y_hat.gather(dim=2, index=indices.unsqueeze(2)).squeeze(2)
        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)
        y_hat, quantile_mask = self(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False
        )
        quantile_loss = self.criterion(y_hat, y_ts)

        loss = quantile_loss
        if self.current_epoch >= self.cutoff_epoch:
            mae_loss = self.get_mae_loss(y_hat, y_ts, quantile_mask)
            loss += mae_loss
            self.val_ce_loss(mae_loss)
            self.log("val/mae_loss", self.val_ce_loss, on_step=True, prog_bar=True)

        self.val_loss(loss)

        self.val_quantile_loss(quantile_loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        self.log(
            "val/quantile_loss", self.val_quantile_loss, on_step=True, prog_bar=True
        )
        indices = torch.argmax(
            quantile_mask.unsqueeze(-1).repeat(1, 1, 1, y_hat.shape[-1]), dim=2
        )
        y_hat = y_hat.gather(dim=2, index=indices.unsqueeze(2)).squeeze(2)
        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}
