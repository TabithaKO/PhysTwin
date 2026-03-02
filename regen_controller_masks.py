#!/usr/bin/env python3
"""
Regenerate controller masks from CoTracker gripper tracks with larger radius.

The original 15px radius was too small — tracked points drifted outside the masks
and data_process_track.py filtered them out, resulting in 0 controller points.

This script:
1. Loads gripper tracks from /tmp/gripper_tracks_{case}_{cam}.npz
2. Computes consensus keypoint per arm per frame (median + outlier rejection)
3. Draws circles with larger radius (default 50px)
4. Saves updated controller masks to PhysTwin mask dirs
5. Rebuilds processed_masks.pkl

Usage:
    python regen_controller_masks.py --radius 50
    python regen_controller_masks.py --radius 50 --case trajectory_01
    python regen_controller_masks.py --radius 50 --all
"""

import argparse
import pickle
import json
from pathlib import Path

import cv2
import numpy as np


CASES = [
    "trajectory_01",
    "traj_left_02", "traj_left_03", "traj_left_04", "traj_left_05",
    "traj_right_01", "traj_right_02", "traj_right_03", "traj_right_04", "traj_right_05",
]

PHYSTWIN_BASE = Path.home() / "PhysTwin/data/so101_cloth"


def compute_consensus(tracks, vis, arm_mask, frame_idx):
    """Compute consensus keypoint from multiple tracked points using median + outlier rejection."""
    pts = tracks[frame_idx, arm_mask]
    v = vis[frame_idx, arm_mask]
    
    visible_pts = pts[v]
    if len(visible_pts) < 2:
        return None
    
    # Median
    median = np.median(visible_pts, axis=0)
    
    # Outlier rejection: remove points > 2 * median absolute deviation
    dists = np.linalg.norm(visible_pts - median, axis=1)
    mad = np.median(dists)
    if mad > 0:
        inliers = visible_pts[dists < 3 * mad]
    else:
        inliers = visible_pts
    
    if len(inliers) < 1:
        return median
    
    return np.mean(inliers, axis=0)


def process_case(case, radius, phystwin_base):
    """Regenerate controller masks for a single trajectory."""
    case_dir = phystwin_base / case
    
    if not case_dir.exists():
        print(f"  SKIP {case}: dir not found")
        return False
    
    # Load metadata for frame count
    meta_path = case_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_frames = meta["frame_num"]
    
    # Load existing processed_masks to get object masks
    masks_path = case_dir / "mask/processed_masks.pkl"
    if not masks_path.exists():
        print(f"  SKIP {case}: no processed_masks.pkl")
        return False
    
    with open(masks_path, "rb") as f:
        processed_masks = pickle.load(f)
    
    # Determine image size from first object mask
    if isinstance(processed_masks, list):
        h, w = processed_masks[0][0]["object"].shape
    elif isinstance(processed_masks, dict):
        first_key = min(processed_masks.keys())
        h, w = processed_masks[first_key][0]["object"].shape
    else:
        h, w = 720, 1280
    
    # Load gripper tracks for both cameras
    cam_tracks = {}
    for cam in [0, 1]:
        track_path = f"/tmp/gripper_tracks_{case}_cam{cam}.npz"
        if not Path(track_path).exists():
            print(f"  WARNING: {track_path} not found, using empty controller masks for cam{cam}")
            cam_tracks[cam] = None
            continue
        
        d = np.load(track_path, allow_pickle=True)
        cam_tracks[cam] = {
            "tracks": d["tracks"],        # (T, N, 2) - xy coords
            "visibility": d["visibility"], # (T, N)
            "arm_labels": d["arm_labels"], # (N,) - 'left' or 'right'
            "n_frames": int(d["n_frames_used"]),
        }
        
        # Handle subsampled tracks (n_frames_used < n_frames)
        track_frames = cam_tracks[cam]["tracks"].shape[0]
        if track_frames != n_frames:
            print(f"  cam{cam}: track has {track_frames} frames, need {n_frames}, will interpolate")
    
    # Regenerate masks
    new_processed_masks = []
    ctrl_point_counts = []
    
    for frame_idx in range(n_frames):
        frame_masks = []
        
        for cam in [0, 1]:
            # Get existing object mask
            if isinstance(processed_masks, list):
                obj_mask = processed_masks[frame_idx][cam]["object"]
            else:
                obj_mask = processed_masks.get(frame_idx, processed_masks[min(processed_masks.keys())])[cam]["object"]
            
            # Generate new controller mask
            ctrl_mask = np.zeros((h, w), dtype=np.uint8)
            
            if cam_tracks[cam] is not None:
                tracks = cam_tracks[cam]["tracks"]
                vis = cam_tracks[cam]["visibility"]
                arm_labels = cam_tracks[cam]["arm_labels"]
                track_frames = tracks.shape[0]
                
                # Map frame index if subsampled
                if track_frames != n_frames:
                    t_idx = int(frame_idx * (track_frames - 1) / (n_frames - 1))
                else:
                    t_idx = frame_idx
                t_idx = min(t_idx, track_frames - 1)
                
                # Compute consensus for each arm
                for arm in np.unique(arm_labels):
                    arm_mask = arm_labels == arm
                    kp = compute_consensus(tracks, vis, arm_mask, t_idx)
                    
                    if kp is not None:
                        x, y = int(kp[0]), int(kp[1])
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(ctrl_mask, (x, y), radius, 255, -1)
            
            ctrl_mask_bool = ctrl_mask > 127
            
            # Remove controller from object mask
            obj_mask_clean = obj_mask & ~ctrl_mask_bool
            
            # Save PNGs
            ctrl_dir = case_dir / f"mask/{cam}/controller"
            ctrl_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ctrl_dir / f"{frame_idx}.png"), ctrl_mask)
            
            obj_dir = case_dir / f"mask/{cam}/object"
            obj_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(obj_dir / f"{frame_idx}.png"), 
                        (obj_mask_clean * 255).astype(np.uint8))
            
            frame_masks.append({
                "object": obj_mask_clean,
                "controller": ctrl_mask_bool,
            })
            
            if frame_idx == 0:
                ctrl_point_counts.append(ctrl_mask_bool.sum())
        
        new_processed_masks.append(frame_masks)
    
    # Handle both list and dict format
    # data_process_mask.py indexes by frame number, data_process_track.py expects dict or list
    # Check original format
    if isinstance(processed_masks, dict):
        new_pm_dict = {}
        for i, fm in enumerate(new_processed_masks):
            new_pm_dict[i] = fm
        save_masks = new_pm_dict
    else:
        save_masks = new_processed_masks
    
    with open(masks_path, "wb") as f:
        pickle.dump(save_masks, f)
    
    print(f"  {case}: {n_frames} frames, radius={radius}px, "
          f"ctrl pixels f0: cam0={ctrl_point_counts[0]}, cam1={ctrl_point_counts[1]}")
    
    return True


def main():
    p = argparse.ArgumentParser(description="Regenerate controller masks with larger radius")
    p.add_argument("--radius", type=int, default=50, help="Circle radius in pixels (default: 50)")
    p.add_argument("--case", type=str, default=None, help="Single case to process")
    p.add_argument("--all", action="store_true", help="Process all cases")
    p.add_argument("--phystwin-base", type=str, default=str(PHYSTWIN_BASE))
    args = p.parse_args()
    
    phystwin_base = Path(args.phystwin_base)
    
    if args.case:
        cases = [args.case]
    elif args.all:
        cases = CASES
    else:
        cases = CASES  # default to all
    
    print(f"Regenerating controller masks: radius={args.radius}px")
    print(f"Cases: {len(cases)}")
    print()
    
    success = 0
    for case in cases:
        if process_case(case, args.radius, phystwin_base):
            success += 1
    
    print(f"\nDone! Processed {success}/{len(cases)} cases")
    print(f"\nNext steps:")
    print(f"  1. Re-run data_process_track.py for each case")
    print(f"  2. Re-run data_process_sample.py")
    print(f"  3. Re-run train_warp.py")


if __name__ == "__main__":
    main()
