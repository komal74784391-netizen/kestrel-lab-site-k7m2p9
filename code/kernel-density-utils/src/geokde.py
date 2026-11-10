"""Kernel density helpers for geospatial point layers."""

import numpy as np


def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))


def kde_grid(points, bandwidth=0.01, grid_size=256):
    xs = np.linspace(points[:, 0].min(), points[:, 0].max(), grid_size)
    ys = np.linspace(points[:, 1].min(), points[:, 1].max(), grid_size)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.zeros_like(gx)
    for px, py in points:
        grid += gaussian_kernel(np.hypot(gx - px, gy - py), bandwidth)
    return xs, ys, grid
