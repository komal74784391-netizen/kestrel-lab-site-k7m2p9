"""Contrastive spatial encoder."""

import torch
import torch.nn as nn


class SpatialEncoder(nn.Module):
    def __init__(self, n_genes, embed_dim=64, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        z = self.backbone(x)
        return z, self.projection(z)
