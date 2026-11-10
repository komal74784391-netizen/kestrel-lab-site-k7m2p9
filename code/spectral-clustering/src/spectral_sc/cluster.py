"""Cluster-count selection and refinement."""

import numpy as np
from sklearn.cluster import KMeans


def eigengap_scores(eigenvalues, max_clusters=8):
    eig = np.asarray(eigenvalues, dtype=np.float64)
    gaps = np.abs(np.diff(eig))
    return gaps[: max(1, min(max_clusters, len(gaps)))]


def select_clusters(eigenvalues, eigenvectors, max_clusters=8, random_state=17):
    gaps = eigengap_scores(eigenvalues, max_clusters=max_clusters)
    k = int(np.argmax(gaps) + 2)
    k = max(2, min(k, max_clusters))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(eigenvectors[:, :k])
    return labels, k
