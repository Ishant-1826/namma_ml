#!/usr/bin/env python3
"""Runner for SfM experiments following the prompt's protocol and AGENTS.md rules."""

import os
import sys
import subprocess
import shutil
import math
from pathlib import Path
import numpy as np

# Ensure root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)

def run_cmd(cmd_list, desc):
    print(f"\n>>> [{desc}] Running: {' '.join(str(x) for x in cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FAILED with return code {res.returncode}")
        print("STDERR:\n", res.stderr[-2000:])
        print("STDOUT:\n", res.stdout[-2000:])
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd_list)}")
    print("SUCCESS.")
    return res.stdout

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def parse_model(sparse_model_dir, total_v1_frames=222, total_v2_frames=228):
    """Parse TXT or BIN model to extract metrics."""
    # Convert model to TXT if not already
    txt_images = sparse_model_dir / "images.txt"
    txt_cameras = sparse_model_dir / "cameras.txt"
    txt_points = sparse_model_dir / "points3D.txt"
    
    if not (txt_images.exists() and txt_cameras.exists() and txt_points.exists()):
        run_cmd([
            "colmap", "model_converter",
            "--input_path", str(sparse_model_dir),
            "--output_path", str(sparse_model_dir),
            "--output_type", "TXT"
        ], "Convert model to TXT")
        
    # Parse cameras.txt
    cameras = {}
    with open(txt_cameras, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            w = int(parts[2])
            h = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}
            
    # Parse images.txt
    images = {}
    with open(txt_images, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            img_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]])
            tvec = np.array([float(x) for x in parts[5:8]])
            cam_id = int(parts[8])
            name = parts[9]
            
            # Camera center in world frame: C = -R^T * t
            R = qvec2rotmat(qvec)
            C = -R.T @ tvec
            images[name] = {
                "img_id": img_id,
                "qvec": qvec,
                "tvec": tvec,
                "cam_id": cam_id,
                "center": C
            }
            # Skip 2D points line
            i += 2
            
    # Parse points3D.txt for reprojection errors
    reproj_errors = []
    with open(txt_points, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            err = float(parts[7])
            reproj_errors.append(err)
            
    mean_reproj_error = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    
    # Registration count by video
    v1_registered = sum(1 for name in images if int(Path(name).stem.split("_")[1]) <= 222)
    v2_registered = sum(1 for name in images if int(Path(name).stem.split("_")[1]) > 222)
    
    # Intrinsics
    first_cam = next(iter(cameras.values()))
    cam_model = first_cam["model"]
    params = first_cam["params"]
    if cam_model == "SIMPLE_RADIAL":
        fx = params[0]
        fy = params[0]
        k1 = params[3]
    elif cam_model in ("PINHOLE", "RADIAL"):
        fx = params[0]
        fy = params[1] if cam_model == "PINHOLE" else params[0]
        k1 = params[3] if cam_model == "RADIAL" else 0.0
    elif cam_model == "OPENCV":
        fx = params[0]
        fy = params[1]
        k1 = params[4]
    else:
        fx = params[0]
        fy = params[0]
        k1 = 0.0
        
    # Trajectory step lengths & start-to-end gap
    # Sort registered images by frame number
    sorted_names = sorted(images.keys(), key=lambda n: int(Path(n).stem.split("_")[1]))
    step_lengths = []
    for k in range(len(sorted_names) - 1):
        c1 = images[sorted_names[k]]["center"]
        c2 = images[sorted_names[k+1]]["center"]
        step = float(np.linalg.norm(c2 - c1))
        step_lengths.append(step)
        
    if step_lengths:
        median_step = float(np.median(step_lengths))
        largest_step = float(np.max(step_lengths))
        step_ratio = largest_step / median_step if median_step > 0 else 0.0
    else:
        median_step = 0.0
        largest_step = 0.0
        step_ratio = 0.0
        
    # Start to end gap
    if len(sorted_names) >= 2:
        c_start = images[sorted_names[0]]["center"]
        c_end = images[sorted_names[-1]]["center"]
        start_end_gap = float(np.linalg.norm(c_end - c_start))
    else:
        start_end_gap = 0.0
        
    return {
        "registered_total": len(images),
        "v1_registered": v1_registered,
        "v2_registered": v2_registered,
        "mean_reproj_error": mean_reproj_error,
        "fx": fx,
        "fy": fy,
        "k1": k1,
        "median_step": median_step,
        "largest_step": largest_step,
        "step_ratio": step_ratio,
        "start_end_gap": start_end_gap,
        "cameras": cameras,
        "image_count": len(images),
        "points_count": len(reproj_errors)
    }

print("Loaded sfm_experiment_runner helper functions.")
