from typing import Any, Tuple

import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError(
            f"Predictions and targets are expected to have the same shape, but got {preds.shape} and {target.shape}."
        )


def _mean_aufec_compute(sum_abs_error: Tensor, n_obs: int) -> Tensor:
    """Computes Mean Absolute Error.
    Args:
        sum_abs_error: Sum of absolute value of errors over all observations
        n_obs: Number of predictions or observations
    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)
        >>> _mean_absolute_error_compute(sum_abs_error, n_obs)
        tensor(0.2500)
    """

    return sum_abs_error / n_obs


def _mean_aufec_update(
    preds: Tensor,
    target: Tensor,
    dim: int,
    mode="low",
    cutoff_mode=2,
    **kwargs,
) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean AUFEC.
    Checks for same shape of input tensors.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim: Temporal dimension
        mode: Whether to return AUFEC on low frequencies or high frequencies.
        cutoff_mode: Cut-off mode in frequency domain.
    """
    _check_same_shape(preds, target)
    assert mode in [
        "low",
        "high",
    ], f"mode can only be 'low' or 'high' but you chose {mode}"
    preds = preds if preds.is_floating_point else preds.float()
    target = target if target.is_floating_point else target.float()

    preds_fft = torch.fft.rfft(preds, dim=dim)
    target_fft = torch.fft.rfft(target, dim=dim)
    abs_diff = torch.abs(preds_fft - target_fft)
    if mode == "low":
        error = (
            torch.index_select(
                abs_diff,
                dim=dim,
                index=torch.Tensor(range(cutoff_mode)).long().to(target.device),
            )
            .mean(dim=dim)
            .sum()
        )
    elif mode == "high":
        error = (
            torch.index_select(
                abs_diff,
                dim=dim,
                index=torch.Tensor(range(cutoff_mode, abs_diff.shape[dim]))
                .long()
                .to(target.device),
            )
            .mean(dim=dim)
            .sum()
        )
    else:
        NotImplementedError("mode can only be 'low' or 'high' but you chose {mode}")

    n_obs = target.numel() // target.shape[dim]

    return error, n_obs


class AUFEC(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        dim=1,
        cutoff_mode=2,
        mode="low",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dim = dim
        self.cutoff_mode = cutoff_mode
        self.mode = mode
        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        sum_abs_error, n_obs = _mean_aufec_update(
            preds,
            target,
            self.dim,
            mode=self.mode,
            cutoff_mode=self.cutoff_mode,
        )

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return _mean_aufec_compute(self.sum_abs_error, self.total)
