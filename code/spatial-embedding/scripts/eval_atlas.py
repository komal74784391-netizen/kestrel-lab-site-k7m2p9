"""Evaluate embeddings on clustering and trajectory tasks."""

import argparse

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from spatial_embed.encoder import SpatialEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    bundle = np.load(args.data)
    x = torch.as_tensor(bundle["expression"], dtype=torch.float32)
    model = SpatialEncoder(x.shape[1])
    model.load_state_dict(torch.load(args.checkpoint))
    with torch.no_grad():
        z, _ = model(x)
    pred = KMeans(n_clusters=8, n_init=10).fit_predict(z.numpy())
    truth = np.load(args.labels)["labels"]
    print("ARI", adjusted_rand_score(truth, pred))


if __name__ == "__main__":
    main()
