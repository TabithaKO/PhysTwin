#!/usr/bin/env python3
"""
Adapter: Convert SO-101 raw captures → PhysTwin data format.

Converts data from the SO-101 dual-arm capture format:
    ~/data_captures/data_capture_{name}/
        cam0/         000000.jpg, 000001.jpg, ...
        cam0_depth/   000000.png, 000001.png, ...
        cam1/         000000.jpg, 000001.jpg, ...
        cam1_depth/   000000.png, 000001.png, ...
        robot/        000000.txt, 000001.txt, ...
        recording.json

Into PhysTwin's expected format:
    {output_base}/{case_name}/
        metadata.json
        calibrate.pkl
        color/0/{frame}.png  color/1/{frame}.png
        depth/0/{frame}.npy  depth/1/{frame}.npy
        mask/{cam}/object/{frame}.png  mask/{cam}/controller/{frame}.png
        mask/processed_masks.pkl
        mask/mask_info_0.json  mask/mask_info_1.json

Then you can run PhysTwin's pipeline:
    python dense_track.py --base_path {output_base} --case_name {case_name}
    python data_process_track.py --base_path {output_base} --case_name {case_name}
    python data_process_sample.py --base_path {output_base} --case_name {case_name}

Usage:
    # Convert a single capture:
    python so101_to_phystwin.py \\
        --capture-dir ~/data_captures/data_capture_black_cloth_fold_demo_00_26_02_04_175714 \\
        --calib-dir ~/calibration \\
        --output-base ~/PhysTwin/data/different_types \\
        --case-name black_cloth_fold_00 \\
        --yolo-weights ~/yolov5/runs/train/black_cloth3/weights/best.pt

    # Convert with frame subsampling (e.g., 1389 frames @ 3Hz is too many):
    python so101_to_phystwin.py ... --frame-step 3 --max-frames 200

    # Skip segmentation (if you want to do it manually later):
    python so101_to_phystwin.py ... --skip-masks

Dependencies:
    - numpy, opencv-python, open3d, pickle
    - For mask generation: torch, yolov5, ultralytics (SAM2)
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Calibration conversion
# ─────────────────────────────────────────────────────────────

def load_so101_calibration(calib_dir: Path, img_size: tuple):
    """
    Load SO-101 stereo calibration and convert to PhysTwin format.

    SO-101 format:
        intrinsic_cam_0.json, intrinsic_cam_1.json
        extrinsic_cam_0_to_cam_1.json

    PhysTwin format:
        intrinsics: (num_cam, 3, 3) array
        c2ws: list of (4, 4) camera-to-world transforms
    """
    img_w, img_h = img_size

    intrinsics = []
    for cam_id in range(2):
        intr_path = calib_dir / f"intrinsic_cam_{cam_id}.json"
        with open(intr_path) as f:
            d = json.load(f)
        cw, ch = d["size"]
        fx = d["fx"] * img_w / cw
        fy = d["fy"] * img_h / ch
        cx = d["cx"] * img_w / cw
        cy = d["cy"] * img_h / ch
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        intrinsics.append(K)

    # Load extrinsics
    ext_path = calib_dir / "extrinsic_cam_0_to_cam_1.json"
    with open(ext_path) as f:
        d = json.load(f)
    R_01 = np.array(d["R"], dtype=np.float64)
    T_01 = np.array(d["T"], dtype=np.float64).reshape(3)

    # PhysTwin uses camera-to-world (c2w) transforms
    # cam0 is the world frame (identity)
    c2w_0 = np.eye(4, dtype=np.float64)

    # cam1 → world: inverse of cam0→cam1
    # cam0_to_cam1 = [R_01 | T_01], so cam1_to_cam0 = [R_01^T | -R_01^T @ T_01]
    R_10 = R_01.T
    T_10 = -R_10 @ T_01
    c2w_1 = np.eye(4, dtype=np.float64)
    c2w_1[:3, :3] = R_10
    c2w_1[:3, 3] = T_10

    return np.array(intrinsics), [c2w_0, c2w_1]


# ─────────────────────────────────────────────────────────────
# Robot data parsing
# ─────────────────────────────────────────────────────────────

def parse_robot_file(robot_path: Path) -> dict:
    """
    Parse a single robot state file.

    Format (5 lines):
        Line 0: left EEF position (x, y, z)
        Lines 1-3: left EEF rotation matrix (3x3, row by row)
        Line 4: right EEF position (x, y, z)

    Returns dict with left/right pos and rot.
    """
    lines = robot_path.read_text().strip().split('\n')
    if len(lines) < 5:
        raise ValueError(f"Robot file {robot_path} has {len(lines)} lines, expected 5")

    left_pos = np.array([float(x) for x in lines[0].split()], dtype=np.float64)
    left_rot = np.array([
        [float(x) for x in lines[1].split()],
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
    ], dtype=np.float64)
    right_pos = np.array([float(x) for x in lines[4].split()], dtype=np.float64)

    return {
        "left_pos": left_pos,
        "left_rot": left_rot,
        "right_pos": right_pos,
    }


def load_robot_trajectory(robot_dir: Path, frame_indices: list) -> dict:
    """Load robot data for selected frames."""
    left_positions = []
    right_positions = []
    left_rotations = []

    for idx in frame_indices:
        robot_file = robot_dir / f"{idx:06d}.txt"
        if not robot_file.exists():
            # Repeat last known state
            if left_positions:
                left_positions.append(left_positions[-1])
                right_positions.append(right_positions[-1])
                left_rotations.append(left_rotations[-1])
            continue

        data = parse_robot_file(robot_file)
        left_positions.append(data["left_pos"])
        right_positions.append(data["right_pos"])
        left_rotations.append(data["left_rot"])

    return {
        "left_pos": np.array(left_positions),
        "right_pos": np.array(right_positions),
        "left_rot": np.array(left_rotations),
    }


# ─────────────────────────────────────────────────────────────
# Frame conversion
# ─────────────────────────────────────────────────────────────

def convert_frames(capture_dir: Path, output_dir: Path,
                   frame_indices: list, recording_data: dict):
    """
    Convert SO-101 frames to PhysTwin layout.

    SO-101:  cam0/000042.jpg, cam0_depth/000042.png
    PhysTwin: color/0/42.png, depth/0/42.npy
    """
    for cam_idx in range(2):
        color_out = output_dir / f"color/{cam_idx}"
        depth_out = output_dir / f"depth/{cam_idx}"
        color_out.mkdir(parents=True, exist_ok=True)
        depth_out.mkdir(parents=True, exist_ok=True)

    for new_idx, orig_idx in enumerate(frame_indices):
        frame = recording_data["frames"][orig_idx]
        imgs = frame["images"]

        for cam_idx, cam_key in enumerate(["cam0", "cam1"]):
            # Color: jpg → png
            rgb_path = capture_dir / imgs[cam_key]
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                print(f"  WARNING: missing {rgb_path}")
                continue
            cv2.imwrite(str(output_dir / f"color/{cam_idx}/{new_idx}.png"), rgb)

            # Depth: 16-bit PNG (mm) → npy (mm as float)
            depth_path = capture_dir / imgs[f"{cam_key}_depth"]
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"  WARNING: missing {depth_path}")
                continue
            # PhysTwin expects depth in mm as float
            np.save(str(output_dir / f"depth/{cam_idx}/{new_idx}.npy"),
                    depth.astype(np.float64))

    # Also create video files for CoTracker (it reads mp4)
    for cam_idx, cam_key in enumerate(["cam0", "cam1"]):
        first_frame = cv2.imread(
            str(capture_dir / recording_data["frames"][frame_indices[0]]["images"][cam_key]))
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_path = output_dir / f"color/{cam_idx}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))

        if not writer.isOpened():
            # Fallback codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))

        for new_idx, orig_idx in enumerate(frame_indices):
            frame_img = cv2.imread(
                str(capture_dir / recording_data["frames"][orig_idx]["images"][cam_key]))
            if frame_img is not None:
                writer.write(frame_img)

        writer.release()
        print(f"  Video: {video_path} ({len(frame_indices)} frames)")


# ─────────────────────────────────────────────────────────────
# Mask generation using YOLO + SAM2
# ─────────────────────────────────────────────────────────────

def generate_masks(output_dir: Path, frame_indices: list,
                   robot_data: dict, intrinsics: np.ndarray, c2ws: list,
                   yolo_weights: str = None, sam_model_name: str = "sam2_b.pt",
                   conf_thresh: float = 0.25):
    """
    Generate object and controller masks for each frame and camera.

    Object mask: cloth segmentation via YOLO + SAM2
    Controller mask: robot gripper projection into image

    Produces:
        mask/processed_masks.pkl — list of [per_frame][per_cam] dicts
        mask/mask_info_{cam}.json — dummy mask info for pipeline compat
        mask/{cam}/object/{frame}.png
        mask/{cam}/controller/{frame}.png
    """
    import torch

    num_frames = len(frame_indices)

    # Create mask directories
    for cam_idx in range(2):
        (output_dir / f"mask/{cam_idx}/object").mkdir(parents=True, exist_ok=True)
        (output_dir / f"mask/{cam_idx}/controller").mkdir(parents=True, exist_ok=True)

    # Load YOLO
    yolo_model = None
    sam_model = None
    if yolo_weights:
        print("  Loading YOLO...")
        yolo_model = torch.hub.load("ultralytics/yolov5", "custom",
                                     path=yolo_weights, force_reload=False)
        yolo_model.conf = conf_thresh
        print("  Loading SAM2...")
        from ultralytics import SAM
        sam_model = SAM(sam_model_name)

    processed_masks = []

    for frame_idx in range(num_frames):
        frame_masks = []

        for cam_idx in range(2):
            rgb = cv2.imread(str(output_dir / f"color/{cam_idx}/{frame_idx}.png"))
            if rgb is None:
                frame_masks.append({
                    "object": np.zeros((480, 640), dtype=bool),
                    "controller": np.zeros((480, 640), dtype=bool),
                })
                continue

            h, w = rgb.shape[:2]
            rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # Object mask via YOLO + SAM2
            object_mask = np.zeros((h, w), dtype=bool)
            if yolo_model is not None:
                results = yolo_model(rgb_rgb[..., ::-1])
                dets = results.xyxy[0].cpu().numpy()
                dets = dets[dets[:, 4] >= conf_thresh] if len(dets) else dets
                if len(dets) > 0:
                    best = dets[np.argmax(dets[:, 4])]
                    bbox = best[:4].tolist()
                    sam_results = sam_model(rgb_rgb, bboxes=[bbox])
                    if sam_results[0].masks is not None and len(sam_results[0].masks.data) > 0:
                        mask = sam_results[0].masks.data[0].cpu().numpy().astype(bool)
                        if mask.shape != (h, w):
                            mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
                        object_mask = mask

            # Controller mask: project robot EEF positions into image
            controller_mask = np.zeros((h, w), dtype=bool)
            K = intrinsics[cam_idx]
            c2w = c2ws[cam_idx]
            w2c = np.linalg.inv(c2w)

            for eef_pos in [robot_data["left_pos"][frame_idx],
                            robot_data["right_pos"][frame_idx]]:
                # Transform EEF world position to camera frame
                pos_h = np.append(eef_pos, 1.0)
                pos_cam = (w2c @ pos_h)[:3]

                if pos_cam[2] > 0:  # In front of camera
                    px = K @ pos_cam
                    u, v = int(px[0] / px[2]), int(px[1] / px[2])

                    # Paint a circle around the projected position
                    radius = 30  # pixels
                    cv2.circle(controller_mask.astype(np.uint8), (u, v), radius, 1, -1)
                    controller_mask = controller_mask.astype(bool) | (
                        cv2.circle(np.zeros((h, w), dtype=np.uint8), (u, v), radius, 1, -1).astype(bool)
                    )

            # Remove controller from object mask
            object_mask = object_mask & ~controller_mask

            # Save masks as images
            cv2.imwrite(str(output_dir / f"mask/{cam_idx}/object/{frame_idx}.png"),
                        (object_mask * 255).astype(np.uint8))
            cv2.imwrite(str(output_dir / f"mask/{cam_idx}/controller/{frame_idx}.png"),
                        (controller_mask * 255).astype(np.uint8))

            frame_masks.append({
                "object": object_mask,
                "controller": controller_mask,
            })

        processed_masks.append(frame_masks)

        if (frame_idx + 1) % 50 == 0:
            print(f"    Masks: {frame_idx + 1}/{num_frames}")

    # Save processed_masks.pkl
    with open(output_dir / "mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)

    # Save dummy mask_info files (required by data_process_track.py to count cameras)
    for cam_idx in range(2):
        with open(output_dir / f"mask/mask_info_{cam_idx}.json", "w") as f:
            json.dump({"cam_idx": cam_idx, "type": "auto_generated"}, f)

    print(f"  Masks saved: {num_frames} frames × 2 cameras")


# ─────────────────────────────────────────────────────────────
# Metadata generation
# ─────────────────────────────────────────────────────────────

def write_metadata(output_dir: Path, intrinsics: np.ndarray,
                   c2ws: list, recording_data: dict,
                   num_frames: int, img_size: tuple):
    """Write PhysTwin metadata.json and calibrate.pkl."""
    img_w, img_h = img_size

    metadata = {
        "intrinsics": intrinsics.tolist(),
        "WH": [img_w, img_h],
        "frame_num": num_frames,
        "serial_numbers": recording_data["metadata"].get("cameras", ["cam0", "cam1"]),
        "source": "so101_adapter",
        "original_name": recording_data["metadata"]["name"],
        "record_hz": recording_data["metadata"].get("record_hz", 3),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "calibrate.pkl", "wb") as f:
        pickle.dump(c2ws, f)


# ─────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────

def convert_capture(capture_dir: str, calib_dir: str, output_base: str,
                    case_name: str, frame_step: int = 1, max_frames: int = None,
                    yolo_weights: str = None, sam_model_name: str = "sam2_b.pt",
                    skip_masks: bool = False, skip_frames: bool = False):
    """
    Full conversion pipeline: SO-101 capture → PhysTwin format.
    """
    capture_dir = Path(capture_dir).expanduser()
    calib_dir = Path(calib_dir).expanduser()
    output_base = Path(output_base).expanduser()
    output_dir = output_base / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting: {capture_dir.name} → {case_name}")

    # Load recording metadata
    with open(capture_dir / "recording.json") as f:
        recording_data = json.load(f)

    total_frames = len(recording_data["frames"])
    print(f"  Total frames: {total_frames}")

    # Select frame indices
    frame_indices = list(range(0, total_frames, frame_step))
    if max_frames and len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]
    num_frames = len(frame_indices)
    print(f"  Selected frames: {num_frames} (step={frame_step})")

    # Get image size from first frame
    first_img_path = capture_dir / recording_data["frames"][0]["images"]["cam0"]
    first_img = cv2.imread(str(first_img_path))
    img_h, img_w = first_img.shape[:2]
    img_size = (img_w, img_h)
    print(f"  Image size: {img_w}×{img_h}")

    # Load calibration
    intrinsics, c2ws = load_so101_calibration(calib_dir, img_size)
    print(f"  Calibration loaded (baseline: {np.linalg.norm(c2ws[1][:3, 3])*1000:.1f}mm)")

    # Write metadata
    write_metadata(output_dir, intrinsics, c2ws, recording_data, num_frames, img_size)
    print(f"  metadata.json + calibrate.pkl written")

    # Convert frames (color + depth + video)
    if not skip_frames:
        print(f"  Converting {num_frames} frames...")
        convert_frames(capture_dir, output_dir, frame_indices, recording_data)
    else:
        print(f"  Skipping frame conversion (--skip-frames)")

    # Load robot data
    robot_dir = capture_dir / "robot"
    if robot_dir.exists():
        print(f"  Loading robot data...")
        robot_data = load_robot_trajectory(robot_dir, frame_indices)
        # Save for later use
        with open(output_dir / "robot_data.pkl", "wb") as f:
            pickle.dump(robot_data, f)
        print(f"  Robot trajectory: {len(robot_data['left_pos'])} frames")
    else:
        print(f"  WARNING: No robot directory found")
        robot_data = None

    # Generate masks
    if not skip_masks and yolo_weights:
        print(f"  Generating masks...")
        generate_masks(output_dir, frame_indices, robot_data, intrinsics, c2ws,
                       yolo_weights=yolo_weights, sam_model_name=sam_model_name)
    elif skip_masks:
        print(f"  Skipping mask generation (--skip-masks)")
    else:
        print(f"  Skipping mask generation (no --yolo-weights provided)")

    # Print next steps
    print(f"\n{'='*60}")
    print(f"Conversion complete: {output_dir}")
    print(f"\nPhysTwin pipeline expects {num_frames} frames, 2 cameras.")
    print(f"\nNOTE: PhysTwin's dense_track.py expects num_cam=3 by default.")
    print(f"You need to edit dense_track.py line ~22: num_cam = 2")
    print(f"\nNext steps:")
    print(f"  cd ~/PhysTwin/data_process")
    print(f"")
    print(f"  # Step 1: Generate point clouds")
    print(f"  python data_process_pcd.py --base_path {output_base} --case_name {case_name}")
    print(f"")
    print(f"  # Step 2: Dense tracking (edit num_cam=2 in dense_track.py first!)")
    print(f"  python dense_track.py --base_path {output_base} --case_name {case_name}")
    print(f"")
    print(f"  # Step 3: Process tracks")
    print(f"  python data_process_track.py --base_path {output_base} --case_name {case_name}")
    print(f"")
    print(f"  # Step 4: Sample final data")
    print(f"  python data_process_sample.py --base_path {output_base} --case_name {case_name}")
    print(f"")
    print(f"  # Step 5: Train PhysTwin")
    print(f"  cd ~/PhysTwin")
    print(f"  python train_warp.py --base_path {output_base} --case_name {case_name} \\")
    print(f"    --train_frame $(python -c \"import json; print(json.load(open('{output_dir}/split.json'))['train'][1])\")")

    # Write split.json
    train_frame = int(num_frames * 0.7)
    split = {"frame_len": num_frames, "train": [0, train_frame], "test": [train_frame, num_frames]}
    with open(output_dir / "split.json", "w") as f:
        json.dump(split, f)
    print(f"\n  split.json: train 0-{train_frame}, test {train_frame}-{num_frames}")

    return {
        "case_name": case_name,
        "output_dir": str(output_dir),
        "num_frames": num_frames,
        "img_size": img_size,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert SO-101 captures to PhysTwin data format"
    )
    parser.add_argument("--capture-dir", type=str, required=True,
                        help="Path to SO-101 capture directory")
    parser.add_argument("--calib-dir", type=str, required=True,
                        help="Path to calibration directory")
    parser.add_argument("--output-base", type=str, required=True,
                        help="Base output directory (PhysTwin data path)")
    parser.add_argument("--case-name", type=str, default=None,
                        help="Output case name (default: derived from capture name)")

    parser.add_argument("--frame-step", type=int, default=1,
                        help="Take every Nth frame (default: 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to convert (default: all)")

    parser.add_argument("--yolo-weights", type=str, default=None,
                        help="Path to YOLO weights for cloth detection")
    parser.add_argument("--sam-model", type=str, default="sam2_b.pt",
                        help="SAM2 model name (default: sam2_b.pt)")

    parser.add_argument("--skip-masks", action="store_true",
                        help="Skip mask generation")
    parser.add_argument("--skip-frames", action="store_true",
                        help="Skip frame conversion (reuse existing)")

    args = parser.parse_args()

    # Derive case name from capture directory if not provided
    if args.case_name is None:
        name = Path(args.capture_dir).name
        # Strip prefix and timestamp
        name = name.replace("data_capture_", "")
        # Remove trailing timestamp (26_02_04_175714)
        parts = name.split("_")
        # Find where the date starts (look for 2-digit year)
        for i, p in enumerate(parts):
            if len(p) == 2 and p.isdigit() and i > 0:
                # Check if next parts look like date
                if i + 2 < len(parts) and all(parts[j].isdigit() for j in range(i, min(i+4, len(parts)))):
                    name = "_".join(parts[:i])
                    break
        args.case_name = name

    convert_capture(
        capture_dir=args.capture_dir,
        calib_dir=args.calib_dir,
        output_base=args.output_base,
        case_name=args.case_name,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        yolo_weights=args.yolo_weights,
        sam_model_name=args.sam_model,
        skip_masks=args.skip_masks,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    main()
