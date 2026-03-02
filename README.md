# PhysTwin — Custom Cloth Manipulation Data Pipeline

This is my fork of [PhysTwin](https://github.com/Jianghanxiao/PhysTwin), extended with a full data collection and processing pipeline for bimanual cloth manipulation using my custom SO-101 robot platform (the Sew Unit).

The goal: fit PhysTwin's particle-based cloth simulator to real fabric by running it on data I collected myself, rather than the original benchmark data.

---

## What I Added

### SO-101 Cloth Data Pipeline

PhysTwin was originally built around a different robot and data format. I wrote the full pipeline to get it working with my setup:

- **`data_process/so101_to_phystwin.py`** — converts raw RGB-D recordings from the Sew Unit's 4-camera setup into the particle track format PhysTwin trains on
- **`data_process/data_process_sample.py`** / **`data_process_track.py`** — updated to handle SO-101 gripper geometry and the ChAruco-calibrated multi-camera rig
- **`regen_controller_masks.py`** — regenerates controller masks (gripper segmentation) when HSV thresholding fails under changing lab lighting; replaced with CoTracker 3 for robustness
- **`export_gaussian_data_so101.py`** — exports Gaussian splatting training data in the format expected by the downstream PGND pipeline
- **`export_ply.py`** — exports point cloud PLY files for external visualization and validation
- **`pgnd_to_phystwin.py`** — converts PGND episode format back into PhysTwin format for cross-pipeline evaluation
- **`train_phystwin_cloth.sh`** — training script configured for cloth trajectories on the SO-101 setup

### Trajectory Generation Scripts (`scripts/`)

Scripts for generating and visualizing bimanual manipulation trajectories used during data collection and evaluation:

- `generate_dual_v5.py` / `v5b` / `v6` / `v7` — bimanual pull-apart, push-together, fold, and lift trajectories (iterated versions as the robot setup evolved)
- `generate_single_gripper.py` / `generate_single_gripper_right02.py` — single-arm baseline trajectories
- `generate_triptych.py` — side-by-side comparison visualization: real RGB | point cloud | PhysTwin prediction
- `test_lift.py` / `test_lift_4v.py` / `test_lifts.py` / `test_single.py` — live trajectory test scripts used during hardware debugging

---

## Hardware Setup

The data was collected on the **Sew Unit** — a bimanual cloth manipulation platform I built from scratch:

- Two SO-101 arms (LeRobot platform) mounted inverted on a custom aluminum extrusion frame
- 4x Intel RealSense cameras (D435i + D405) with ChAruco calibration (best stereo pair: 0.68 RMS reprojection error)
- Ender 3 printer bed as workspace
- STS/SCS servo motors, unified serial controller

Full hardware writeup: [tabithako.github.io/projects/sew-unit](https://tabithako.github.io/projects/sew-unit)

---

## Dataset

11 annotated bimanual cloth manipulation trajectories collected on the Sew Unit. Each trajectory includes:
- Synchronized RGB-D from 4 cameras at 30 Hz
- Robot joint states and gripper poses
- Controller masks (gripper segmentation) — generated via CoTracker 3
- Particle tracks exported for PhysTwin training

---

## Results

PhysTwin fit to a single training trajectory successfully learns cloth physical parameters (stiffness, damping, mass) and generalises to novel bimanual actions not seen during training.

**Training trajectory — particle simulation fit to a real cloth episode:**

<video src="https://github.com/user-attachments/assets/9487b403-865a-417e-8830-3e5de7cf80ef" autoplay loop muted playsinline></video>

**Robot executing a recorded bimanual trajectory:** [video](https://tabithako.github.io/assets/videos/sew-unit-mirror-bimanual.mp4) · **Leader-follower teleoperation:** [video](https://tabithako.github.io/assets/videos/sew-unit-teleop-leader.mp4)

**Triptych validation** (real RGB | point cloud | PhysTwin prediction):

<video src="https://github.com/user-attachments/assets/549f60a3-1575-40cb-999a-32c8a42e4c1e" autoplay loop muted playsinline></video>

<video src="https://github.com/user-attachments/assets/25435962-691d-44a5-9ca9-f97fedea93ff" autoplay loop muted playsinline></video>

![Point cloud tracking grid](https://tabithako.github.io/assets/images/cloth-dynamics-pointcloud-grid.png)

**Novel actions generated with fitted cloth parameters:**

<video src="https://github.com/user-attachments/assets/79245128-d3a3-4459-9fd2-85e3f5b8d116" autoplay loop muted playsinline></video>

*Fold left over right — bimanual pull-apart [video](https://tabithako.github.io/assets/videos/sew-unit-dual-pull-apart.mp4) · push-together [video](https://tabithako.github.io/assets/videos/sew-unit-dual-push-together.mp4)*

Full writeup: [tabithako.github.io/projects/cloth-dynamics](https://tabithako.github.io/projects/cloth-dynamics)

---

## Original Paper

**PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos**
Jianglong Ye et al. — [project page](https://github.com/Jianghanxiao/PhysTwin)
