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


def _mean_absolute_scaled_error_compute(sum_abs_error: Tensor, n_obs: int) -> Tensor:
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


def _mean_absolute_scaled_error_update(
    preds: Tensor,
    target: Tensor,
    dim: int,
    baseline: str = "persistence",
    epsilon: float = 1e-2,
    **kwargs,
) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Absolute Scaled Error.
    Checks for same shape of input tensors.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        dim: Temporal dimension
        baseline: Which baseline to compare against. Can choose 'persistence' or 'fourier' in which case
            you must provide the cut-off mode in `kwargs`
        epsilon: Small value to prevent division by zero
    """
    _check_same_shape(preds, target)
    assert baseline in [
        "persistence",
        "fourier",
    ], f"baseline can only be 'persistence' or 'fourier' but you chose {baseline}"
    preds = preds if preds.is_floating_point else preds.float()
    target = target if target.is_floating_point else target.float()
    abs_diff = torch.abs(preds - target).mean(dim=dim)
    if baseline == "persistence":
        previous_day = kwargs["x"]
        scaling = torch.clamp((previous_day - target).abs().mean(dim=dim), min=epsilon)
    elif baseline == "fourier":
        mode = kwargs["mode"]
        baseline_pred = torch.fft.irfft(
            torch.index_select(
                torch.fft.rfft(target, dim=dim),
                dim=dim,
                index=torch.arange(mode).to(target.device),
            ),
            n=target.shape[dim],
            dim=dim,
        )
        scaling = torch.clamp(
            torch.abs(baseline_pred - target).mean(dim=dim), min=epsilon
        )
    else:
        NotImplementedError(
            "baseline can only be 'persistence' or 'fourier' but you chose {baseline}"
        )
    error = (abs_diff / scaling).sum()

    n_obs = target.numel() // target.shape[dim]

    return error, n_obs


class MeanAbsoluteScaledError(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        dim=1,
        baseline="persistence",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.dim = dim
        self.baseline = baseline
        self.cut_off_mode = kwargs.get("mode", None)
        self.needs_previous = True
        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, target_previous: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        sum_abs_error, n_obs = _mean_absolute_scaled_error_update(
            preds,
            target,
            self.dim,
            self.baseline,
            mode=self.cut_off_mode,
            x=target_previous,
        )

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_scaled_error_compute(self.sum_abs_error, self.total)
