#!/usr/bin/env python3
"""
Convert PGND episode data to PhysTwin final_data.pkl format.

PGND format (per episode directory):
  - traj.npz: xyz (T, N, 3), v (T, N, 3) — particle positions and velocities
  - eef_traj.txt: (T, 6) — left/right EEF translations [lx,ly,lz, rx,ry,rz]
  - eef_rot.txt: (T, 18) — left/right EEF rotation matrices (flattened 3x3 each)
  - eef_gripper.txt: (T, 2) — left/right gripper states

PhysTwin format (final_data.pkl):
  - controller_mask: (total_points,) int64 — 0=object, 1=controller
  - controller_points: (T, n_ctrl, 3) float64 — gripper/controller positions per frame
  - object_points: (T, N_obj, 3) float64 — object particle positions per frame
  - object_colors: (T, N_obj, 3) float64 — RGB colors [0,1] per particle per frame
  - object_visibilities: (T, N_obj) bool — which particles are visible
  - object_motions_valid: (T, N_obj) bool — which particles have valid motion
  - surface_points: (0, 3) float64 — empty for cloth
  - interior_points: (0, 3) float64 — empty for cloth
  + split.json: {"frame_len": T, "train": [0, train_frame], "test": [train_frame, T]}

Usage:
  python pgnd_to_phystwin.py \
    --episode-dir ~/pgnd/experiments/log/data/cloth_merged/sub_episodes_v/episode_0162 \
    --output-dir ~/PhysTwin/data/different_types/pgnd_cloth_ep0162 \
    --train-ratio 0.7 \
    --max-particles 8000 \
    --n-controller-points 30

  # Batch convert multiple episodes:
  python pgnd_to_phystwin.py \
    --episodes-dir ~/pgnd/experiments/log/data/cloth_merged/sub_episodes_v \
    --output-base ~/PhysTwin/data/different_types \
    --start-episode 162 --end-episode 242 \
    --train-ratio 0.7 \
    --max-particles 8000
"""

import argparse
import json
import os
import pickle
import numpy as np
from pathlib import Path


def load_pgnd_episode(episode_dir: str) -> dict:
    """Load a single PGND episode from its directory."""
    episode_dir = Path(episode_dir)

    # Load trajectory (particles)
    traj = np.load(episode_dir / "traj.npz")
    xyz = traj["xyz"]  # (T, N, 3)
    vel = traj["v"]    # (T, N, 3)

    # Load EEF data
    eef_traj = np.loadtxt(episode_dir / "eef_traj.txt")      # (T, 6)
    eef_rot = np.loadtxt(episode_dir / "eef_rot.txt")         # (T, 18)
    eef_gripper = np.loadtxt(episode_dir / "eef_gripper.txt")  # (T, 2)

    # Handle single-frame edge case
    if eef_traj.ndim == 1:
        eef_traj = eef_traj.reshape(1, -1)
        eef_rot = eef_rot.reshape(1, -1)
        eef_gripper = eef_gripper.reshape(1, -1)

    return {
        "xyz": xyz,
        "vel": vel,
        "eef_traj": eef_traj,
        "eef_rot": eef_rot,
        "eef_gripper": eef_gripper,
    }


def generate_controller_points(eef_traj: np.ndarray, eef_rot: np.ndarray,
                                n_points_per_gripper: int = 15,
                                gripper_radius: float = 0.04) -> np.ndarray:
    """
    Generate controller point clouds around each gripper position.
    
    PhysTwin expects a small point cloud representing the controller/gripper
    at each timestep. We generate points in a small sphere around each EEF.
    
    Args:
        eef_traj: (T, 6) left/right EEF positions
        eef_rot: (T, 18) left/right EEF rotation matrices
        n_points_per_gripper: number of points per gripper
        gripper_radius: radius of the point cloud around the gripper
    
    Returns:
        controller_points: (T, n_points_total, 3) 
    """
    T = eef_traj.shape[0]
    n_grippers = eef_traj.shape[1] // 3
    n_total = n_points_per_gripper * n_grippers

    # Generate fixed offsets in a small sphere (same offsets each frame for consistency)
    rng = np.random.RandomState(42)
    # Use fibonacci sphere for even distribution
    offsets = _fibonacci_sphere(n_points_per_gripper) * gripper_radius

    controller_points = np.zeros((T, n_total, 3), dtype=np.float64)

    for t in range(T):
        for g in range(n_grippers):
            pos = eef_traj[t, g*3:(g+1)*3]  # (3,)
            rot = eef_rot[t, g*9:(g+1)*9].reshape(3, 3)  # (3, 3)

            # Rotate offsets by gripper orientation and translate
            rotated_offsets = (rot @ offsets.T).T  # (n_points, 3)
            start_idx = g * n_points_per_gripper
            end_idx = start_idx + n_points_per_gripper
            controller_points[t, start_idx:end_idx] = pos + rotated_offsets

    return controller_points


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n evenly-distributed points on a unit sphere."""
    indices = np.arange(n, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def downsample_particles(xyz: np.ndarray, vel: np.ndarray,
                          max_particles: int) -> tuple:
    """
    Downsample particles if there are too many.
    Uses farthest point sampling on the first frame for consistent indices.
    
    Args:
        xyz: (T, N, 3) particle positions
        vel: (T, N, 3) particle velocities
        max_particles: maximum number of particles
    
    Returns:
        xyz_down: (T, M, 3) downsampled positions
        vel_down: (T, M, 3) downsampled velocities
        indices: (M,) selected particle indices
    """
    N = xyz.shape[1]
    if N <= max_particles:
        return xyz, vel, np.arange(N)

    # Farthest point sampling on first frame
    points = xyz[0]  # (N, 3)
    indices = np.zeros(max_particles, dtype=int)
    distances = np.full(N, np.inf)

    # Start from random point
    indices[0] = np.random.randint(N)

    for i in range(1, max_particles):
        last = points[indices[i-1]]
        dist = np.linalg.norm(points - last, axis=-1)
        distances = np.minimum(distances, dist)
        indices[i] = np.argmax(distances)

    indices = np.sort(indices)
    return xyz[:, indices], vel[:, indices], indices


def generate_visibility(xyz: np.ndarray, vel: np.ndarray) -> tuple:
    """
    Generate visibility and motion validity masks.
    
    For real cloth data, we assume:
    - All particles are visible (they came from depth cameras)
    - Motion is valid where velocity is non-zero or position changes
    
    Args:
        xyz: (T, N, 3) particle positions
        vel: (T, N, 3) particle velocities
    
    Returns:
        visibilities: (T, N) bool
        motions_valid: (T, N) bool
    """
    T, N, _ = xyz.shape

    # All particles visible (they were reconstructed from cameras)
    visibilities = np.ones((T, N), dtype=bool)

    # Motion valid where we have non-trivial velocity or position change
    vel_magnitude = np.linalg.norm(vel, axis=-1)  # (T, N)
    motions_valid = vel_magnitude > 1e-8

    # Also mark as valid if position changes between frames
    if T > 1:
        pos_change = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)  # (T-1, N)
        # Pad to match T
        pos_change = np.vstack([pos_change, pos_change[-1:]])
        motions_valid = motions_valid | (pos_change > 1e-8)

    return visibilities, motions_valid


def generate_colors(xyz: np.ndarray, color: tuple = (0.6, 0.2, 0.2)) -> np.ndarray:
    """
    Generate synthetic colors for particles.
    
    PGND doesn't track per-particle colors, so we assign a uniform cloth color
    with slight variation for visual distinction.
    
    Args:
        xyz: (T, N, 3) particle positions
        color: base RGB color tuple [0, 1]
    
    Returns:
        colors: (T, N, 3) float64 in [0, 1]
    """
    T, N, _ = xyz.shape
    colors = np.zeros((T, N, 3), dtype=np.float64)

    # Base color with slight per-particle variation for visual diversity
    rng = np.random.RandomState(123)
    noise = rng.uniform(-0.05, 0.05, (N, 3))

    base = np.array(color).reshape(1, 3)
    particle_colors = np.clip(base + noise, 0.0, 1.0)  # (N, 3)

    # Same colors across all frames
    for t in range(T):
        colors[t] = particle_colors

    return colors


def convert_episode(episode_dir: str, output_dir: str,
                     max_particles: int = 8000,
                     n_controller_points: int = 30,
                     gripper_radius: float = 0.04,
                     train_ratio: float = 0.7,
                     cloth_color: tuple = (0.6, 0.2, 0.2)) -> dict:
    """
    Convert a single PGND episode to PhysTwin format.
    
    Args:
        episode_dir: path to PGND episode (e.g., .../episode_0162)
        output_dir: path to output PhysTwin case directory
        max_particles: max object particles (PhysTwin default ~8000)
        n_controller_points: total controller points (split across grippers)
        gripper_radius: radius of controller point cloud
        train_ratio: fraction of frames for training
        cloth_color: RGB base color for synthetic particle colors
    
    Returns:
        info dict with conversion statistics
    """
    episode_dir = Path(episode_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PGND episode from {episode_dir}...")
    data = load_pgnd_episode(episode_dir)

    xyz = data["xyz"]       # (T, N, 3)
    vel = data["vel"]       # (T, N, 3)
    eef_traj = data["eef_traj"]  # (T, 6)
    eef_rot = data["eef_rot"]    # (T, 18)

    T, N_orig, _ = xyz.shape
    print(f"  Frames: {T}, Original particles: {N_orig}")

    # Downsample particles if needed
    xyz_ds, vel_ds, ds_indices = downsample_particles(xyz, vel, max_particles)
    N = xyz_ds.shape[1]
    print(f"  Particles after downsampling: {N}")

    # Generate controller points around grippers
    n_per_gripper = n_controller_points // 2  # bimanual
    ctrl_pts = generate_controller_points(
        eef_traj, eef_rot,
        n_points_per_gripper=n_per_gripper,
        gripper_radius=gripper_radius
    )
    n_ctrl = ctrl_pts.shape[1]
    print(f"  Controller points: {n_ctrl}")

    # Generate visibility and motion masks
    visibilities, motions_valid = generate_visibility(xyz_ds, vel_ds)

    # Generate synthetic colors
    colors = generate_colors(xyz_ds, color=cloth_color)

    # Build controller mask: 0=object, 1=controller
    # Total points = N_object + N_controller (at frame 0)
    total_pts = N + n_ctrl
    controller_mask = np.zeros(total_pts, dtype=np.int64)
    controller_mask[N:] = 1  # last n_ctrl points are controller

    # Empty surface/interior (cloth is a surface, no interior distinction needed)
    surface_points = np.zeros((0, 3), dtype=np.float64)
    interior_points = np.zeros((0, 3), dtype=np.float64)

    # Build final_data dict
    final_data = {
        "controller_mask": controller_mask,
        "controller_points": ctrl_pts,          # (T, n_ctrl, 3)
        "object_points": xyz_ds,                # (T, N, 3)
        "object_colors": colors,                # (T, N, 3)
        "object_visibilities": visibilities,     # (T, N)
        "object_motions_valid": motions_valid,   # (T, N)
        "surface_points": surface_points,        # (0, 3)
        "interior_points": interior_points,      # (0, 3)
    }

    # Save final_data.pkl
    pkl_path = output_dir / "final_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(final_data, f)
    print(f"  Saved: {pkl_path}")

    # Generate split.json
    train_frame = int(T * train_ratio)
    split = {
        "frame_len": T,
        "train": [0, train_frame],
        "test": [train_frame, T],
    }
    split_path = output_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split, f)
    print(f"  Saved: {split_path} (train: 0-{train_frame}, test: {train_frame}-{T})")

    info = {
        "episode": str(episode_dir),
        "output": str(output_dir),
        "frames": T,
        "particles_original": N_orig,
        "particles_final": N,
        "controller_points": n_ctrl,
        "train_frame": train_frame,
    }
    return info


def batch_convert(episodes_dir: str, output_base: str,
                   start_episode: int, end_episode: int,
                   **kwargs) -> list:
    """
    Batch convert multiple PGND episodes to PhysTwin format.
    
    Args:
        episodes_dir: base directory containing episode_XXXX folders
        output_base: base output directory for PhysTwin cases
        start_episode: first episode index
        end_episode: last episode index (exclusive)
        **kwargs: passed to convert_episode
    
    Returns:
        list of info dicts
    """
    episodes_dir = Path(episodes_dir)
    output_base = Path(output_base)
    results = []

    for ep_idx in range(start_episode, end_episode):
        ep_name = f"episode_{ep_idx:04d}"
        ep_dir = episodes_dir / ep_name

        if not ep_dir.exists():
            print(f"Skipping {ep_name} (not found)")
            continue

        out_name = f"pgnd_cloth_ep{ep_idx:04d}"
        out_dir = output_base / out_name

        try:
            info = convert_episode(str(ep_dir), str(out_dir), **kwargs)
            results.append(info)
            print(f"  ✓ {ep_name} → {out_name}\n")
        except Exception as e:
            print(f"  ✗ {ep_name} FAILED: {e}\n")
            results.append({"episode": str(ep_dir), "error": str(e)})

    # Save summary
    summary_path = output_base / "pgnd_conversion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Converted {len([r for r in results if 'error' not in r])}/{len(results)} episodes")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert PGND episode data to PhysTwin final_data.pkl format"
    )

    # Single episode mode
    parser.add_argument("--episode-dir", type=str, default=None,
                        help="Path to single PGND episode directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for single episode")

    # Batch mode
    parser.add_argument("--episodes-dir", type=str, default=None,
                        help="Base directory containing episode_XXXX folders")
    parser.add_argument("--output-base", type=str, default=None,
                        help="Base output directory for batch conversion")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="First episode index (batch mode)")
    parser.add_argument("--end-episode", type=int, default=10,
                        help="Last episode index, exclusive (batch mode)")

    # Common parameters
    parser.add_argument("--max-particles", type=int, default=8000,
                        help="Max object particles (default: 8000)")
    parser.add_argument("--n-controller-points", type=int, default=30,
                        help="Total controller points across all grippers (default: 30)")
    parser.add_argument("--gripper-radius", type=float, default=0.04,
                        help="Radius of controller point cloud (default: 0.04)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of frames for training (default: 0.7)")
    parser.add_argument("--cloth-color", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                        help="Base RGB cloth color (default: 0.6 0.2 0.2)")

    args = parser.parse_args()

    common_kwargs = {
        "max_particles": args.max_particles,
        "n_controller_points": args.n_controller_points,
        "gripper_radius": args.gripper_radius,
        "train_ratio": args.train_ratio,
        "cloth_color": tuple(args.cloth_color),
    }

    if args.episode_dir:
        # Single episode mode
        output_dir = args.output_dir or args.episode_dir + "_phystwin"
        convert_episode(args.episode_dir, output_dir, **common_kwargs)

    elif args.episodes_dir:
        # Batch mode
        if not args.output_base:
            print("Error: --output-base required for batch mode")
            return
        batch_convert(
            args.episodes_dir, args.output_base,
            args.start_episode, args.end_episode,
            **common_kwargs,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Single episode:")
        print("  python pgnd_to_phystwin.py \\")
        print("    --episode-dir ~/pgnd/experiments/log/data/cloth_merged/sub_episodes_v/episode_0162 \\")
        print("    --output-dir ~/PhysTwin/data/different_types/pgnd_cloth_ep0162")
        print()
        print("  # Batch convert episodes 162-242:")
        print("  python pgnd_to_phystwin.py \\")
        print("    --episodes-dir ~/pgnd/experiments/log/data/cloth_merged/sub_episodes_v \\")
        print("    --output-base ~/PhysTwin/data/different_types \\")
        print("    --start-episode 162 --end-episode 242")


if __name__ == "__main__":
    main()
