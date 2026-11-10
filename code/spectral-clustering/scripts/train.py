"""End-to-end spectral clustering run."""

import argparse

import scanpy as sc
import yaml

from spectral_sc.cluster import select_clusters
from spectral_sc.embedding import spectral_embedding
from spectral_sc.graph import build_affinity_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="clusters.csv")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    adata = sc.read_h5ad(args.data)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    graph = build_affinity_graph(adata.X.toarray(),
                                 n_neighbors=cfg["n_neighbors"])
    eigvals, eigvecs = spectral_embedding(graph, cfg["n_eigenvectors"])
    labels, k = select_clusters(eigvals, eigvecs,
                                random_state=cfg.get("random_state", 17))

    adata.obs["spectral_cluster"] = labels
    adata.obs[["spectral_cluster"]].to_csv(args.out)
    print(f"selected {k} clusters -> {args.out}")


if __name__ == "__main__":
    main()
