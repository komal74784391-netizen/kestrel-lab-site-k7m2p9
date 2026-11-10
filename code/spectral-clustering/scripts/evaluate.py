"""Cluster quality evaluation against held-out marker annotations."""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    pred = pd.read_csv(args.clusters)["spectral_cluster"].to_numpy()
    truth = pd.read_csv(args.labels)["label"].to_numpy()
    print("ARI", float(adjusted_rand_score(truth, pred)))
    print("NMI", float(normalized_mutual_info_score(truth, pred)))


if __name__ == "__main__":
    main()
