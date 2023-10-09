from typing import Any, Optional, Sequence, Union

import torch
import torchmetrics
from torch import Tensor, tensor
from torchmetrics.functional.regression.mape import (
    _mean_absolute_percentage_error_compute,
    _mean_absolute_percentage_error_update,
)
from torchmetrics.metric import Metric


class MeanAbsolutePercentageError(Metric):
    r"""Compute `Mean Absolute Percentage Error`_ (MAPE).

    .. math:: \text{MAPE} = \frac{1}{n}\sum_{i=1}^n\frac{|   y_i - \hat{y_i} |}{\max(\epsilon, | y_i |)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mean_abs_percentage_error`` (:class:`~torch.Tensor`): A tensor with the mean absolute percentage error over
      state

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Note:
        MAPE output is a non-negative floating point. Best result is ``0.0`` . But it is important to note that,
        bad predictions, can lead to arbitarily large values. Especially when some ``target`` values are close to 0.
        This `MAPE implementation returns`_ a very large number instead of ``inf``.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.regression import MeanAbsolutePercentageError
        >>> target = tensor([1, 10, 1e6])
        >>> preds = tensor([0.9, 15, 1.2e6])
        >>> mean_abs_percentage_error = MeanAbsolutePercentageError()
        >>> mean_abs_percentage_error(preds, target)
        tensor(0.2667)
    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_per_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        epsilon=1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        
        self.epsilon = epsilon
        self.add_state("sum_abs_per_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _mean_absolute_percentage_error_update(preds, target, epsilon=self.epsilon)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> torch.Tensor:
        """Compute mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)