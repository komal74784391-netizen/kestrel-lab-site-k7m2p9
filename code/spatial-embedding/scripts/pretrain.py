"""Pretrain the contrastive spatial encoder."""

import argparse

import numpy as np
import torch
import yaml

from spatial_embed.encoder import SpatialEncoder
from spatial_embed.losses import neighbourhood_contrastive_loss


def build_pairs(coords, k=6):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    return idx[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/contrastive.yaml")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    bundle = np.load(args.data)
    x = torch.as_tensor(bundle["expression"], dtype=torch.float32)
    coords = bundle["coords"]
    pairs = build_pairs(coords, cfg["neighbourhood"])
    model = SpatialEncoder(x.shape[1], cfg["embed_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(cfg["epochs"]):
        z, proj = model(x)
        pos = proj[torch.as_tensor(pairs[:, 0])]
        neg_idx = np.random.default_rng(epoch).integers(0, len(x), (len(x), cfg["negatives"]))
        neg = proj[torch.as_tensor(neg_idx)]
        loss = neighbourhood_contrastive_loss(proj, pos, neg, cfg["temperature"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), "spatial_encoder.pt")
    print("final loss", float(loss))


if __name__ == "__main__":
    main()
