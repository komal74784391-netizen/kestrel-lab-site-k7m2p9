"""Normalized Laplacian embedding."""

import numpy as np
from scipy.sparse.linalg import eigsh

from .graph import normalize_affinity


def spectral_embedding(graph, n_eigenvectors=10):
    """Return the leading eigenvectors of the normalized adjacency matrix."""
    norm = normalize_affinity(graph)
    k = min(n_eigenvectors, norm.shape[0] - 1)
    values, vectors = eigsh(norm.astype(np.float64), k=k, which="LA")
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]
