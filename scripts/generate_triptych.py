#!/usr/bin/env python3
"""
Generate triptych comparison videos for a trained PhysTwin trajectory.

Video 1 (triptych.mp4):     RGB | RGB + Digital Twin overlay | Digital Twin only
Video 2 (triptych_pcd.mp4): RGB | Fused PCD (camera colors)  | Digital Twin only

Usage:
    python scripts/generate_triptych.py --case-name traj_right_02
    python scripts/generate_triptych.py --case-name trajectory_01
    python scripts/generate_triptych.py --case-name traj_right_02 --fps 10 --cam 1
"""

import argparse
import os
import sys

import cv2
import json
import numpy as np
import pickle


def project_to_2d(pts_3d, K):
    """Project 3D points to 2D pixel coords (negates Z for PhysTwin convention)."""
    pts = pts_3d.copy()
    pts[:, 2] *= -1
    px = (K @ pts.T).T
    return (px[:, :2] / px[:, 2:3]).astype(int)


def height_colors_bgr(z):
    """Height-based coloring: red = lifted, blue = on table, gray = invalid."""
    real = z[z < -0.01]
    if len(real) < 10:
        return np.full((len(z), 3), 150, dtype=np.uint8)
    zmin, zmax = real.min(), real.max()
    if zmax - zmin < 0.01:
        zmin, zmax = zmin - 0.05, zmax + 0.05
    z_norm = np.clip(1.0 - (z - zmin) / (zmax - zmin), 0, 1)
    colors = np.zeros((len(z), 3), dtype=np.uint8)
    colors[:, 2] = (z_norm * 255).astype(np.uint8)       # R channel
    colors[:, 0] = ((1 - z_norm) * 255).astype(np.uint8)  # B channel
    colors[:, 1] = 30                                      # G channel
    colors[z > -0.01] = [120, 120, 120]
    return colors


def draw_gripper(img, ctrl_i, K, sx, sy, W, H):
    """Draw gripper marker (red dot with white border) at controller centroid."""
    center = ctrl_i.mean(axis=0)
    cc = center.copy()
    cc[2] *= -1
    px = K @ cc
    px = px[:2] / px[2]
    cx, cy = int(px[0] * sx), int(px[1] * sy)
    if 0 <= cx < W and 0 <= cy < H:
        cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Generate triptych comparison videos')
    parser.add_argument('--case-name', required=True, help='Trajectory case name (e.g. traj_right_02)')
    parser.add_argument('--base-path', default='data/so101_cloth', help='Base data path')
    parser.add_argument('--cam', type=int, default=0, help='Camera index (0 or 1)')
    parser.add_argument('--fps', type=int, default=4, help='Output video FPS')
    parser.add_argument('--panel-width', type=int, default=427, help='Width of each panel')
    parser.add_argument('--panel-height', type=int, default=360, help='Height of each panel')
    args = parser.parse_args()

    case = args.case_name
    base = args.base_path
    cam = args.cam
    W, H = args.panel_width, args.panel_height

    data_dir = f'{base}/{case}'
    exp_dir = f'experiments/{case}'
    os.makedirs(exp_dir, exist_ok=True)

    # Load metadata and intrinsics
    with open(f'{data_dir}/metadata.json') as f:
        meta = json.load(f)
    K = np.array(meta['intrinsics'])[cam]

    # Load final_data (downsampled particles used for training)
    with open(f'{data_dir}/final_data.pkl', 'rb') as f:
        fd = pickle.load(f)
    obj = np.array(fd['object_points'])
    ctrl = np.array(fd['controller_points'])
    n_frames = obj.shape[0]
    print(f'Case: {case}')
    print(f'Frames: {n_frames}, Object pts: {obj.shape[1]}, Ctrl pts: {ctrl.shape[1]}')

    # Load track_process_data (raw tracked points with camera colors)
    tpd_path = f'{data_dir}/track_process_data.pkl'
    has_raw = os.path.exists(tpd_path)
    if has_raw:
        with open(tpd_path, 'rb') as f:
            tpd = pickle.load(f)
        raw_pts = np.array(tpd['object_points'])
        raw_cols = np.array(tpd['object_colors'])
        print(f'Raw points: {raw_pts.shape}, colors range: [{raw_cols.min():.2f}, {raw_cols.max():.2f}]')
        if raw_cols.max() <= 1.0:
            raw_cols_u8 = (raw_cols * 255).astype(np.uint8)
        else:
            raw_cols_u8 = raw_cols.astype(np.uint8)
    else:
        print(f'No track_process_data.pkl found — skipping PCD triptych video')

    # Load inference results if available
    inf_path = f'{exp_dir}/inference.pkl'
    has_inference = os.path.exists(inf_path)
    if has_inference:
        with open(inf_path, 'rb') as f:
            pred_vertices = pickle.load(f)
        if hasattr(pred_vertices, 'shape'):
            print(f'Inference data: {pred_vertices.shape}')
        else:
            print(f'Inference data loaded (type: {type(pred_vertices).__name__})')

    # Scale factors
    sx, sy = W / 1280, H / 720

    # Load split info
    with open(f'{data_dir}/split.json') as f:
        split = json.load(f)
    train_end = split['train'][1]

    # === VIDEO 1: RGB | RGB+overlay | Digital Twin ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v1_path = f'{exp_dir}/triptych_cam{cam}.mp4'
    out1 = cv2.VideoWriter(v1_path, fourcc, args.fps, (W * 3, H))

    print(f'\nRecording triptych → {v1_path}')
    for i in range(n_frames):
        rgb_path = f'{data_dir}/color/{cam}/{i}.png'
        if not os.path.exists(rgb_path):
            continue
        rgb = cv2.imread(rgb_path)
        rgb = cv2.resize(rgb, (W, H))

        # Panel 1: raw RGB
        p1 = rgb.copy()
        phase = 'TRAIN' if i < train_end else 'TEST'
        cv2.putText(p1, f'RGB {i}/{n_frames} [{phase}]', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Panel 2: RGB + overlay
        p2 = rgb.copy()
        obj_px = project_to_2d(obj[i], K)
        z = obj[i][:, 2]
        hcols = height_colors_bgr(z)
        for k in range(len(obj_px)):
            x, y = int(obj_px[k, 0] * sx), int(obj_px[k, 1] * sy)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(p2, (x, y), 1, hcols[k].tolist(), -1)
        draw_gripper(p2, ctrl[i], K, sx, sy, W, H)
        cv2.putText(p2, 'RGB + Twin', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Panel 3: Digital twin only
        p3 = np.full((H, W, 3), 25, dtype=np.uint8)
        for k in range(len(obj_px)):
            x, y = int(obj_px[k, 0] * sx), int(obj_px[k, 1] * sy)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(p3, (x, y), 2, hcols[k].tolist(), -1)
        draw_gripper(p3, ctrl[i], K, sx, sy, W, H)
        cv2.putText(p3, 'Digital Twin', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out1.write(np.hstack([p1, p2, p3]))
        if i % 50 == 0:
            print(f'  V1 Frame {i}/{n_frames}')

    out1.release()
    print(f'Saved: {v1_path}')

    # === VIDEO 2: RGB | Fused PCD (camera colors) | Digital Twin ===
    if has_raw:
        v2_path = f'{exp_dir}/triptych_pcd_cam{cam}.mp4'
        out2 = cv2.VideoWriter(v2_path, fourcc, args.fps, (W * 3, H))

        print(f'\nRecording PCD triptych → {v2_path}')
        for i in range(n_frames):
            rgb_path = f'{data_dir}/color/{cam}/{i}.png'
            if not os.path.exists(rgb_path):
                continue
            rgb = cv2.imread(rgb_path)
            rgb = cv2.resize(rgb, (W, H))

            # Panel 1: raw RGB
            p1 = rgb.copy()
            phase = 'TRAIN' if i < train_end else 'TEST'
            cv2.putText(p1, f'RGB {i}/{n_frames} [{phase}]', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Panel 2: fused PCD with camera colors
            p2 = np.full((H, W, 3), 25, dtype=np.uint8)
            rp = raw_pts[i]
            valid = np.any(rp != 0, axis=1)
            rp_v = rp[valid]
            rc_v = raw_cols_u8[i][valid]

            rp_proj = rp_v.copy()
            if rp_v[:, 2].mean() < 0:
                rp_proj[:, 2] *= -1
            px = (K @ rp_proj.T).T
            px2d = (px[:, :2] / px[:, 2:3]).astype(int)

            for k in range(len(px2d)):
                x, y = int(px2d[k, 0] * sx), int(px2d[k, 1] * sy)
                if 0 <= x < W and 0 <= y < H:
                    b, g, r = int(rc_v[k, 2]), int(rc_v[k, 1]), int(rc_v[k, 0])
                    cv2.circle(p2, (x, y), 2, (b, g, r), -1)
            draw_gripper(p2, ctrl[i], K, sx, sy, W, H)
            cv2.putText(p2, 'Fused PCD', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Panel 3: Digital twin (height colored)
            p3 = np.full((H, W, 3), 25, dtype=np.uint8)
            obj_px = project_to_2d(obj[i], K)
            z = obj[i][:, 2]
            hcols = height_colors_bgr(z)
            for k in range(len(obj_px)):
                x, y = int(obj_px[k, 0] * sx), int(obj_px[k, 1] * sy)
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(p3, (x, y), 2, hcols[k].tolist(), -1)
            draw_gripper(p3, ctrl[i], K, sx, sy, W, H)
            cv2.putText(p3, 'Digital Twin', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            out2.write(np.hstack([p1, p2, p3]))
            if i % 50 == 0:
                print(f'  V2 Frame {i}/{n_frames}')

        out2.release()
        print(f'Saved: {v2_path}')

    print('\nDone! Videos:')
    print(f'  {v1_path}')
    if has_raw:
        print(f'  {v2_path}')


if __name__ == '__main__':
    main()
