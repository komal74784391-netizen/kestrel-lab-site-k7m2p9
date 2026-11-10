"""Anchored factor graph for joint multiome alignment."""

import numpy as np


class AnchoredFactorGraph:
    def __init__(self, n_anchors, latent_dim=32, ridge=0.01):
        self.n_anchors = n_anchors
        self.latent_dim = latent_dim
        self.ridge = ridge
        self.anchors = None

    def select_anchors(self, rna_embedding, atac_embedding, rng=None):
        rng = rng or np.random.default_rng(0)
        n = min(self.n_anchors, len(rna_embedding), len(atac_embedding))
        idx = rng.choice(len(rna_embedding), size=n, replace=False)
        self.anchors = idx
        return idx

    def adjacency(self, embeddings, k=10):
        """Mutual kNN graph shared by both modalities."""
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(embeddings, k, mode="connectivity", include_self=False)
        return g.maximum(g.T)
