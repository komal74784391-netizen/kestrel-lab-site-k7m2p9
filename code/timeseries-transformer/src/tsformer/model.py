"""Minimal transformer forecaster."""

import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_channels, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        encoder = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.head = nn.Linear(d_model, n_channels)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.head(h[:, -1])
