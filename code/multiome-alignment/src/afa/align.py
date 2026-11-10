"""Block-coordinate solver for anchored factor alignment."""

import numpy as np


def _procrustes(x, y, ridge=0.01):
    u, _, vt = np.linalg.svd(x.T @ y)
    return u @ vt


def align_modalities(rna_latent, atac_latent, anchors, max_iter=60, ridge=0.01):
    """Iteratively align modality latents through shared anchor cells."""
    rna = np.asarray(rna_latent, dtype=np.float64)
    atac = np.asarray(atac_latent, dtype=np.float64)
    rot = np.eye(rna.shape[1])
    for _ in range(max_iter):
        moved = atac @ rot
        rot = _procrustes(rna[anchors], moved[anchors], ridge=ridge)
    return rna, atac @ rot
