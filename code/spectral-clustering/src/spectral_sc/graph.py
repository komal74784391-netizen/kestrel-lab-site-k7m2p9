"""Affinity graph construction for single-cell expression matrices."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph


def _cosine_similarity_matrix(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms) @ (x / norms).T


def build_affinity_graph(expression, n_neighbors=15, sparsify=True):
    """Build a symmetric kNN affinity graph from a cells x genes matrix."""
    sim = _cosine_similarity_matrix(np.asarray(expression, dtype=np.float64))
    nn = kneighbors_graph(sim, n_neighbors=n_neighbors, mode="distance",
                          include_self=False)
    if sparsify:
        nn = nn.maximum(nn.T)
    return csr_matrix(nn)


def normalize_affinity(graph):
    """Symmetric normalization A_norm = D^-1/2 A D^-1/2."""
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    d_mat = csr_matrix((d_inv_sqrt, (np.arange(len(d_inv_sqrt)),
                                    np.arange(len(d_inv_sqrt)))),
                       shape=graph.shape)
    return d_mat @ graph @ d_mat
