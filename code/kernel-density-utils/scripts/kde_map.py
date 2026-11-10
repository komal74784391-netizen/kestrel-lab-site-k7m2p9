"""Render a kernel density surface for a point layer."""

import argparse

import numpy as np

from geokde import kde_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True)
    parser.add_argument("--out", default="kde.npy")
    args = parser.parse_args()
    points = np.loadtxt(args.points)[:, :2]
    xs, ys, grid = kde_grid(points)
    np.savez(args.out, xs=xs, ys=ys, grid=grid)
    print("grid", grid.shape)


if __name__ == "__main__":
    main()
