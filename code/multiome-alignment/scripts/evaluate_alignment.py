"""Evaluate alignment quality via label transfer accuracy."""

import argparse

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    data = np.load(args.aligned)
    labels = np.load(args.labels)
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(data["rna"], labels["rna"])
    acc = knn.score(data["atac"], labels["atac"])
    print("label transfer accuracy", acc)


if __name__ == "__main__":
    main()
