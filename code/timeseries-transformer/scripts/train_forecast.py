"""Train a forecasting baseline."""

import argparse

import numpy as np
import torch

from tsformer.model import TimeSeriesTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    series = np.load(args.data)["series"]
    model = TimeSeriesTransformer(series.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    x = torch.as_tensor(series[:, :-1], dtype=torch.float32)
    y = torch.as_tensor(series[:, -1], dtype=torch.float32)
    for epoch in range(args.epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("final forecast loss", float(loss))


if __name__ == "__main__":
    main()
