#!/usr/bin/env python3
"""
Export first-frame Gaussian data for SO-101 2-camera setup.

Creates:
    data/gaussian_data/{case_name}/
        0.png, 1.png          — first frame RGB per camera
        mask_0.png, mask_1.png — first frame object mask per camera
        pcd.ply               — first frame point cloud
        cameras.json          — camera intrinsics + extrinsics
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
import open3d as o3d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", required=True)
    p.add_argument("--case_name", required=True)
    p.add_argument("--output_path", default="./data/gaussian_data")
    p.add_argument("--controller_name", default="gripper")
    args = p.parse_args()

    base = f"{args.base_path}/{args.case_name}"
    out = f"{args.output_path}/{args.case_name}"
    os.makedirs(out, exist_ok=True)

    # Load metadata
    with open(f"{base}/metadata.json") as f:
        meta = json.load(f)
    intrinsics = np.array(meta["intrinsics"])
    num_cam = len(intrinsics)
    WH = meta["WH"]

    with open(f"{base}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)

    print(f"Exporting GS data for {args.case_name} ({num_cam} cameras)")

    # Copy first-frame images and masks
    for i in range(num_cam):
        # RGB
        src = f"{base}/color/{i}/0.png"
        dst = f"{out}/{i}.png"
        os.system(f"cp {src} {dst}")
        print(f"  RGB: {dst}")

        # Object mask
        with open(f"{base}/mask/mask_info_{i}.json") as f:
            mask_info = json.load(f)

        obj_idx = None
        for key, value in mask_info.items():
            if value != args.controller_name:
                try:
                    obj_idx = int(key)
                except ValueError:
                    continue

        if obj_idx is not None:
            mask_src = f"{base}/mask/{i}/{obj_idx}/0.png"
            mask_dst = f"{out}/mask_{i}.png"
            os.system(f"cp {mask_src} {mask_dst}")
            print(f"  Mask: {mask_dst}")
        else:
            # Create full white mask as fallback
            mask = np.ones((WH[1], WH[0]), dtype=np.uint8) * 255
            mask_dst = f"{out}/mask_{i}.png"
            cv2.imwrite(mask_dst, mask)
            print(f"  Mask (fallback): {mask_dst}")

    # Generate first-frame point cloud from pcd data
    pcd_path = f"{base}/pcd/0.npz"
    if os.path.exists(pcd_path):
        data = np.load(pcd_path)
        points = data["points"]
        colors = data["colors"]
        masks = data["masks"]

        # Load processed masks for object filtering
        processed_masks_path = f"{base}/mask/processed_masks.pkl"
        if os.path.exists(processed_masks_path):
            with open(processed_masks_path, "rb") as f:
                processed_masks = pickle.load(f)

            all_pts = []
            all_cols = []
            for i in range(num_cam):
                obj_mask = processed_masks[0][i]["object"]
                valid = np.logical_and(masks[i], obj_mask)
                pts = points[i][valid]
                cols = colors[i][valid]
                all_pts.append(pts)
                all_cols.append(cols)

            all_pts = np.concatenate(all_pts)
            all_cols = np.concatenate(all_cols)
        else:
            # No processed masks, use raw depth masks
            all_pts = []
            all_cols = []
            for i in range(num_cam):
                pts = points[i][masks[i]]
                cols = colors[i][masks[i]]
                all_pts.append(pts)
                all_cols.append(cols)
            all_pts = np.concatenate(all_pts)
            all_cols = np.concatenate(all_cols)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_cols)

        # Outlier removal
        cl, ind = pcd.remove_radius_outlier(nb_points=20, radius=0.01)
        pcd = pcd.select_by_index(ind)

        pcd_out = f"{out}/pcd.ply"
        o3d.io.write_point_cloud(pcd_out, pcd)
        print(f"  PCD: {pcd_out} ({len(pcd.points)} points)")

    # Save camera info
    cameras = {
        "num_cam": num_cam,
        "WH": WH,
        "intrinsics": intrinsics.tolist(),
        "c2ws": [c.tolist() for c in c2ws],
    }
    with open(f"{out}/cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"  Cameras: {out}/cameras.json")

    print(f"\nDone! GS data in: {out}")
    print(f"\nNext: train GS with:")
    print(f"  python gs_train.py \\")
    print(f"    -s {out} \\")
    print(f"    -m ./gaussian_output/{args.case_name}/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0 \\")
    print(f"    --iterations 10000 \\")
    print(f"    --lambda_depth 0.001 --lambda_normal 0.0 --lambda_anisotropic 0.0 --lambda_seg 1.0 \\")
    print(f"    --use_masks --isotropic --gs_init_opt hybrid")


if __name__ == "__main__":
    main()
