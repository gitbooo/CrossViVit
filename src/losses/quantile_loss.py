import warnings

import torch
import torch.nn.functional as F
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, input, target):
        if not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()
                ),
                stacklevel=2,
            )
        error = (target - input).float()
        return torch.max((self.quantile - 1) * error, self.quantile * error).mean()


class QuantileLossMulti(nn.Module):
    def __init__(
        self,
        quantiles=[round(0.1 * i, 1) for i in range(1, 10)],
        quantile_weights=[1.0 / 9 for i in range(1, 10)],
        use_qweights=False,
    ):
        super(QuantileLossMulti, self).__init__()
        self.quantiles = quantiles
        self.quantile_losses = nn.ModuleList(
            [QuantileLoss(quantile) for quantile in quantiles]
        )
        self.quantile_weights = quantile_weights
        self.use_qweights = use_qweights

    def forward(self, outputs, input):
        if not (outputs.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    outputs.size(), input.size()
                ),
                stacklevel=2,
            )

        losses = []
        for i in range(len(self.quantiles)):
            quantile_loss = self.quantile_losses[i](outputs[:, :, i], input)
            losses.append(quantile_loss)

        # Return combined quantile loss and predictions
        # total_loss = torch.mean(torch.stack(losses))

        # --------- the total loss is a weighted sum of the quantile losses ------------
        # use weights proportional to the quantiles frequencies
        if self.use_qweights:
            freq = torch.tensor(
                [
                    round(
                        float(
                            sum(
                                1
                                * (
                                    input
                                    > torch.quantile(
                                        input, i * 0.1, dim=0, keepdim=True
                                    )
                                )
                            )
                            / len(input)
                        ),
                        1,
                    )
                    for i in range(1, input.shape[1] + 1)
                ]
            )
            weights = torch.exp(freq) / torch.sum(torch.exp(freq))

            total_loss = 0
            for i in range(len(self.quantiles)):
                total_loss += weights[i] * losses[i]

        # use predefined weights
        else:
            total_loss = 0
            for i in range(len(self.quantiles)):
                total_loss += self.quantile_weights[i] * losses[i]

        return total_loss
