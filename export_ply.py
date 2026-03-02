#!/usr/bin/env python3
"""
Export PhysTwin GT and predicted trajectories as colored PLY files per frame.

GT = blue, Predicted = pink, Controller = red

Usage:
    python export_ply.py --case pgnd_cloth_ep0162
    python export_ply.py --case single_clift_cloth_3
    python export_ply.py --case pgnd_cloth_ep0162 --frames 0 30 60 90 119
"""

import argparse
import os
import pickle
import json
import numpy as np


def write_ply(path, points, colors):
    """Write a colored point cloud to PLY format.
    
    Args:
        path: output file path
        points: (N, 3) float array
        colors: (N, 3) uint8 array (RGB)
    """
    N = len(points)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="Specific frames to export (default: every 10th + first/last)")
    parser.add_argument("--base", type=str, default="./data/different_types")
    parser.add_argument("--exp", type=str, default="./experiments")
    args = parser.parse_args()

    case = args.case
    data_path = f"{args.base}/{case}/final_data.pkl"
    inf_path = f"{args.exp}/{case}/inference.pkl"

    # Load GT
    with open(data_path, 'rb') as f:
        gt_data = pickle.load(f)
    gt_obj = gt_data['object_points']
    gt_ctrl = gt_data['controller_points']

    # Load split
    split_path = f"{args.base}/{case}/split.json"
    with open(split_path) as f:
        split = json.load(f)
    test_start = split['test'][0]
    T = gt_obj.shape[0]
    N_obj = gt_obj.shape[1]

    # Load predicted (if exists)
    has_pred = os.path.exists(inf_path)
    if has_pred:
        with open(inf_path, 'rb') as f:
            pred = pickle.load(f)
        print(f"Loaded inference: {pred.shape}")
    else:
        print(f"No inference.pkl found — exporting GT only")

    # Select frames
    if args.frames:
        frames = args.frames
    else:
        frames = list(range(0, T, 10))
        if T - 1 not in frames:
            frames.append(T - 1)
        if test_start not in frames:
            frames.append(test_start)
        frames = sorted(set(frames))

    # Output directory
    out_dir = f"{args.exp}/{case}/ply_frames"
    os.makedirs(out_dir, exist_ok=True)

    # Colors
    blue = np.array([70, 130, 255], dtype=np.uint8)     # GT object
    pink = np.array([255, 100, 180], dtype=np.uint8)     # Predicted
    red = np.array([255, 30, 30], dtype=np.uint8)        # Controller

    for t in frames:
        if t >= T:
            print(f"  Frame {t} > max {T-1}, skipping")
            continue

        phase = "TEST" if t >= test_start else "TRAIN"

        # GT object + controller combined
        gt_pts = gt_obj[t]
        gt_colors = np.tile(blue, (len(gt_pts), 1))

        ctrl_pts = gt_ctrl[t]
        ctrl_colors = np.tile(red, (len(ctrl_pts), 1))

        combined_pts = np.vstack([gt_pts, ctrl_pts])
        combined_colors = np.vstack([gt_colors, ctrl_colors])

        gt_path = f"{out_dir}/frame_{t:04d}_gt_{phase}.ply"
        write_ply(gt_path, combined_pts, combined_colors)

        # Predicted
        if has_pred:
            pred_pts = pred[t, :N_obj]
            pred_colors = np.tile(pink, (len(pred_pts), 1))

            pred_path = f"{out_dir}/frame_{t:04d}_pred_{phase}.ply"
            write_ply(pred_path, pred_pts, pred_colors)

            # Combined: GT blue + Pred pink + Controller red
            all_pts = np.vstack([gt_pts, pred_pts, ctrl_pts])
            all_colors = np.vstack([gt_colors, pred_colors, ctrl_colors])

            both_path = f"{out_dir}/frame_{t:04d}_both_{phase}.ply"
            write_ply(both_path, all_pts, all_colors)

        print(f"  Frame {t:4d} [{phase}] — {gt_path}")

    print(f"\nExported {len(frames)} frames to {out_dir}/")
    print(f"  *_gt_*.ply    = GT (blue) + controller (red)")
    print(f"  *_pred_*.ply  = Predicted (pink)")
    print(f"  *_both_*.ply  = Combined overlay")


if __name__ == "__main__":
    main()
