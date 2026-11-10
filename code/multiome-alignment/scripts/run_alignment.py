"""Run joint multiome alignment and save latent coordinates."""

import argparse

import anndata as ad
import numpy as np
import yaml

from afa.align import align_modalities
from afa.factor_graph import AnchoredFactorGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--atac", required=True)
    parser.add_argument("--config", default="configs/align.yaml")
    parser.add_argument("--out", default="aligned.npz")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    rna = ad.read_h5ad(args.rna)
    atac = ad.read_h5ad(args.atac)
    rna_latent = np.asarray(rna.obsm["X_pca"])[:, : cfg["latent_dim"]]
    atac_latent = np.asarray(atac.obsm["X_lsi"])[:, : cfg["latent_dim"]]

    fg = AnchoredFactorGraph(cfg["n_anchors"], cfg["latent_dim"], cfg["ridge"])
    anchors = fg.select_anchors(rna_latent, atac_latent,
                                np.random.default_rng(cfg["random_state"]))
    rna_aligned, atac_aligned = align_modalities(
        rna_latent, atac_latent, anchors, cfg["max_iter"], cfg["ridge"])
    np.savez(args.out, rna=rna_aligned, atac=atac_aligned, anchors=anchors)
    print("saved", args.out)


if __name__ == "__main__":
    main()
