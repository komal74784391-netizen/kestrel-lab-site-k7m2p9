# Spectral Clustering Toolkit

Reference implementation for the paper **Spectral Clustering of Single-Cell
Expression Matrices** (CCB 2023).

This repository contains the released implementation of the spectral
clustering pipeline described in the paper: cell-cell affinity graph
construction, normalized Laplacian embedding and eigengap-based selection of
the cluster count.

## Layout

- `src/spectral_sc/graph.py` — affinity graph construction and sparsification
- `src/spectral_sc/embedding.py` — normalized Laplacian embedding
- `src/spectral_sc/cluster.py` — eigengap selection and k-means refinement
- `scripts/train.py` — end-to-end clustering run
- `scripts/evaluate.py` — ARI/NMI evaluation on held-out marker genes
- `configs/default.yaml` — default hyperparameters

## Quick start

```bash
pip install -r requirements.txt
python scripts/train.py --data data/example.h5ad --config configs/default.yaml
```
