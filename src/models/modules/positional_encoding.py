import torch
from torch import nn


class NeRF_embedding(nn.Module):
    def __init__(self, n_layers: int = 5):
        super().__init__()
        self.n_layers = n_layers
        self.dim = self.n_layers * 4

    def forward(self, spatial_coords: torch.Tensor):
        """
        Args:
            spatial_coords (torch.Tensor): Spatial coordinates of shape [B, 2, H, W]
        """
        embeddings = []
        for i in range(self.n_layers):
            embeddings += [
                torch.sin((2**i * torch.pi) * spatial_coords),
                torch.cos((2**i * torch.pi) * spatial_coords),
            ]
        embeddings = torch.cat(embeddings, axis=1)
        return embeddings


class Cyclical_embedding(nn.Module):
    def __init__(self, frequencies: list):
        super().__init__()
        self.frequencies = frequencies
        self.dim = len(self.frequencies) * 2

    def forward(self, time_coords: torch.Tensor):
        """
        Args:
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
        """
        embeddings = []
        for i, frequency in enumerate(self.frequencies):
            embeddings += [
                torch.sin(2 * torch.pi * time_coords[:, :, i] / frequency),
                torch.cos(2 * torch.pi * time_coords[:, :, i] / frequency),
            ]
        embeddings = torch.stack(embeddings, axis=2)
        return embeddings
