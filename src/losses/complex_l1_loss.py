from torch import nn
import torch
import torch.nn.functional as F
import warnings


class ComplexL1Loss(nn.Module):
    def __init__(self):
        super(ComplexL1Loss, self).__init__()

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
        error_amplitude = (input - target).abs().float()
        zeros = torch.zeros_like(error_amplitude).float()
        l1 = F.l1_loss(error_amplitude, zeros)
        return l1
