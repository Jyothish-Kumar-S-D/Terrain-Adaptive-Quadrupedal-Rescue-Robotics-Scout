#!/usr/bin/env python3
"""Load and view saved point cloud from NPZ file."""

import sys
from pathlib import Path

import numpy as np


def view_pointcloud(npz_path):
    """Load and display point cloud from NPZ file."""
    data = np.load(npz_path)
    points = data['points']
    colors = data['colors']

    print(f"Loaded: {npz_path}")
    print(f"Points: {len(points)}")
    print(f"Bounds: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}] "
          f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}] "
          f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    import matplotlib.pyplot as plt

    fig = plt.figure("Point Cloud Viewer", figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Point Cloud ({len(points)} points)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=2, depthshade=False)
        margin = 0.5
        ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
        ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
        ax.set_zlim(points[:, 2].min() - margin, points[:, 2].max() + margin)

    plt.show()


def list_saved_clouds(cloud_dir="point_clouds"):
    """List all saved point cloud files."""
    path = Path(cloud_dir)
    if not path.exists():
        print(f"Directory {cloud_dir} does not exist")
        return []
    files = sorted(path.glob("*.npz"), key=lambda f: f.stat().st_mtime, reverse=True)
    for i, f in enumerate(files):
        size = f.stat().st_size / 1024
        mtime = f.stat().st_mtime
        import time
        print(f"  {i+1}. {f.name} ({size:.1f} KB, {time.ctime(mtime)})")
    return files


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View saved point clouds")
    parser.add_argument("file", nargs="?", help="NPZ file to view")
    parser.add_argument("--list", "-l", action="store_true", help="List saved clouds")
    parser.add_argument("--dir", "-d", default="point_clouds", help="Cloud directory")
    args = parser.parse_args()

    if args.list:
        list_saved_clouds(args.dir)
    elif args.file:
        view_pointcloud(args.file)
    else:
        # Show latest by default
        files = list_saved_clouds(args.dir)
        if files:
            view_pointcloud(files[0])
        else:
            print("No point clouds found")
            parser.print_help()