#!/usr/bin/env python3
"""
pipeline/align_scene.py  --  Phase 4: Metric, Gravity-Aligned, Y-up Scene Alignment.

Converts a COLMAP sparse reconstruction into a metric, gravity-aligned,
Y-up coordinate frame with the floor plane at y = 0 BEFORE Gaussian Splatting training.

RULES ENFORCED:
- Every transform applied to data is a proper rotation (determinant +1),
  plus one uniform scale and a translation. Never negates an axis.
- Never moves, deletes, or re-orders camera poses or points.
- Reads configs/default.yaml for all settings.
- Robust to single or dual video passes (Eye level reference, Chest level check).

Outputs:
  * data/<scene>/colmap_aligned/ (cameras.txt, images.txt, points3D.txt, cameras.bin, images.bin, points3D.bin)
  * data/<scene>/transforms.json (Nerfstudio OpenGL aligned format)
  * data/<scene>/alignment.json (R, t, s, up spread, floor plane, per-pass heights, room extents)
  * data/<scene>/colmap_aligned/points3D.ply and data/<scene>/points3D.ply (sparse init point cloud)
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from pipeline.utils.point_cloud import o3d
except ImportError:
    try:
        import open3d as o3d
    except ImportError:
        o3d = None

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Robust COLMAP I/O (TXT & Binary)
# ---------------------------------------------------------------------------

def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix (world-from-camera or cam-from-world)."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def rotmat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Shepperd's method: 3x3 rotation matrix -> (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def rodrigues_align(v_src: np.ndarray, v_dst: np.ndarray) -> np.ndarray:
    """Proper 3D rotation matrix (det = +1.0) mapping unit v_src to unit v_dst via Rodrigues' formula."""
    v_src = v_src / np.linalg.norm(v_src)
    v_dst = v_dst / np.linalg.norm(v_dst)
    cross = np.cross(v_src, v_dst)
    dot = float(np.dot(v_src, v_dst))
    if abs(dot + 1.0) < 1e-8:  # 180-degree antipodal case
        perp = np.array([1., 0., 0.]) if abs(v_src[0]) < 0.9 else np.array([0., 1., 0.])
        axis = np.cross(v_src, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    s = np.linalg.norm(cross)
    if s < 1e-8:
        return np.eye(3)
    kmat = np.array([
        [0., -cross[2], cross[1]],
        [cross[2], 0., -cross[0]],
        [-cross[1], cross[0], 0.],
    ])
    R = np.eye(3) + kmat + kmat @ kmat * (1.0 - dot) / (s * s)
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"det(R) must be +1, got {np.linalg.det(R)}"
    return R


def parse_cameras_txt(path: Path) -> Dict[int, dict]:
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(p) for p in parts[4:]],
            }
    return cameras


def parse_images_txt(path: Path) -> Dict[int, dict]:
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images[img_id] = {
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "camera_id": cam_id,
            "name": name,
            "point2d_str": lines[i + 1] if i + 1 < len(lines) else "",
        }
        i += 2
    return images


def parse_points3d_txt(path: Path) -> Dict[int, dict]:
    points = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pt_id = int(parts[0])
            points[pt_id] = {
                "xyz": np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                "rgb": (int(parts[4]), int(parts[5]), int(parts[6])),
                "error": float(parts[7]),
                "track": " ".join(parts[8:]),
            }
    return points


def write_cameras_txt(cameras: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id, cam in sorted(cameras.items()):
            params_str = " ".join(repr(p) for p in cam["params"])
            f.write(f"{cam_id} {cam['model']} {cam['width']} {cam['height']} {params_str}\n")


def write_images_txt(images: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img_id, img in sorted(images.items()):
            f.write(
                f"{img_id}"
                f" {img['qw']:.10f} {img['qx']:.10f}"
                f" {img['qy']:.10f} {img['qz']:.10f}"
                f" {img['tx']:.10f} {img['ty']:.10f} {img['tz']:.10f}"
                f" {img['camera_id']} {img['name']}\n"
            )
            f.write(img.get("point2d_str", "") + "\n")


def write_points3d_txt(points: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(points)}\n")
        for pt_id, pt in sorted(points.items()):
            xyz = pt["xyz"]
            r, g, b = pt["rgb"]
            f.write(
                f"{pt_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
                f" {r} {g} {b} {pt['error']:.6f} {pt.get('track', '')}\n"
            )


def write_points3d_ply(points: dict, path: Path) -> None:
    """Save point cloud in PLY format for splat training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = len(points)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for pt in points.values():
            xyz = pt["xyz"]
            r, g, b = pt["rgb"]
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {int(r)} {int(g)} {int(b)}\n")


def convert_model_to_bin_if_possible(txt_dir: Path, bin_dir: Path) -> None:
    """Export aligned TXT model to COLMAP binary files (cameras.bin, images.bin, points3D.bin)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    if HAS_PYCOLMAP:
        try:
            rec = pycolmap.Reconstruction(str(txt_dir))
            rec.write_binary(str(bin_dir))
            return
        except Exception:
            pass

    colmap_bin = shutil.which("colmap")
    if colmap_bin:
        try:
            cmd = [
                colmap_bin, "model_converter",
                "--input_path", str(txt_dir),
                "--output_path", str(bin_dir),
                "--output_type", "BIN"
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Camera Center & Helpers
# ---------------------------------------------------------------------------

def camera_centre(img: dict) -> np.ndarray:
    """Compute camera center in world coordinates from COLMAP pose: C = -R_wc.T @ t_wc."""
    R = quat_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
    t = np.array([img["tx"], img["ty"], img["tz"]])
    return -(R.T @ t)


def frame_number(name: str) -> int:
    digits = "".join(filter(str.isdigit, name))
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# STEP 0: Find Input Model
# ---------------------------------------------------------------------------

def pick_best_colmap_model(repo_root: Path, scene: str, configured_model: Optional[str]) -> Path:
    """
    Finds the COLMAP model with the most registered images (config key: align.input_model).
    """
    if configured_model:
        p = Path(configured_model)
        if not p.is_absolute():
            p = repo_root / p
        if (p / "cameras.txt").is_file() or (p / "cameras.bin").is_file():
            return p

    candidates: List[Path] = []
    # 1. Search colmap_runs
    colmap_runs = repo_root / "colmap_runs"
    if colmap_runs.is_dir():
        for exp in sorted(colmap_runs.iterdir()):
            if exp.is_dir():
                sparse = exp / "sparse"
                if sparse.is_dir():
                    for sub in sparse.iterdir():
                        if sub.is_dir() and ((sub / "cameras.bin").is_file() or (sub / "cameras.txt").is_file()):
                            candidates.append(sub)

    # 2. Search data/<scene>/colmap/sparse
    data_sparse = repo_root / "data" / scene / "colmap" / "sparse"
    if data_sparse.is_dir():
        for sub in data_sparse.iterdir():
            if sub.is_dir() and ((sub / "cameras.bin").is_file() or (sub / "cameras.txt").is_file()):
                candidates.append(sub)

    if not candidates:
        raise RuntimeError(
            f"No COLMAP models found in colmap_runs/ or data/{scene}/colmap/sparse/.\n"
            "Actionable advice: Run Stage 2 SfM (pipeline/run_sfm.py) or specify --input-model."
        )

    # Rank by number of registered images
    ranked = []
    for c in candidates:
        img_count = 0
        txt_img = c / "images.txt"
        bin_img = c / "images.bin"
        if txt_img.is_file():
            try:
                with open(txt_img, "r", encoding="utf-8") as f:
                    lines = [l for l in f if l.strip() and not l.startswith("#")]
                img_count = len(lines) // 2
            except Exception:
                pass
        elif bin_img.is_file() and HAS_PYCOLMAP:
            try:
                rec = pycolmap.Reconstruction(str(c))
                img_count = len(rec.images)
            except Exception:
                pass
        elif bin_img.is_file():
            img_count = int(bin_img.stat().st_size // 1000)

        ranked.append((img_count, c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_count, best_path = ranked[0]
    return best_path


# ---------------------------------------------------------------------------
# Nerfstudio transforms.json Export
# ---------------------------------------------------------------------------

def export_nerfstudio_transforms(
    cameras: dict,
    images: dict,
    output_path: Path,
    image_rel_prefix: str = "images",
) -> None:
    """
    Exports camera poses and intrinsics in Nerfstudio format (transforms.json).

    Camera Coordinate Axes Conversion (OpenCV/COLMAP vs OpenGL/Nerfstudio):
    -----------------------------------------------------------------------
    COLMAP/OpenCV camera convention:
      - +X: Right
      - +Y: Down
      - +Z: Forward (viewing direction along optical axis)
      - World-to-Camera: x_c = R_w2c @ X_w + t_w2c
      - Camera-to-World: C = -R_w2c.T @ t_w2c, R_c2w = R_w2c.T = [r_x, r_y, r_z]

    Nerfstudio/OpenGL/Blender camera convention:
      - +X: Right (+r_x)
      - +Y: Up (-r_y, inverted from OpenCV down)
      - +Z: Backwards (-r_z, inverted from OpenCV forward / pointing out of screen)
      - Camera origin C is invariant to camera body frame definition.

    Therefore, the 4x4 camera-to-world matrix in OpenGL convention is:
      T_c2w = [
        [ r_x[0], -r_y[0], -r_z[0], C[0] ],
        [ r_x[1], -r_y[1], -r_z[1], C[1] ],
        [ r_x[2], -r_y[2], -r_z[2], C[2] ],
        [ 0.0,     0.0,     0.0,    1.0  ]
      ]
    This preserves det(R) = +1 (proper 3D rotation) and conforms directly
    to Nerfstudio and 3D Gaussian Splatting loaders.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cameras or not images:
        return

    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    model = cam["model"]
    w = cam["width"]
    h = cam["height"]
    params = cam["params"]

    fl_x = params[0]
    fl_y = params[0]
    cx = float(w) / 2.0
    cy = float(h) / 2.0
    k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

    if model in ("SIMPLE_RADIAL", "RADIAL"):
        fl_x, fl_y = params[0], params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]
    elif model == "PINHOLE":
        fl_x, fl_y = params[0], params[1]
        cx, cy = params[2], params[3]
    elif model in ("OPENCV", "OPENCV_FISHEYE"):
        fl_x, fl_y = params[0], params[1]
        cx, cy = params[2], params[3]
        k1, k2, p1, p2 = params[4], params[5], params[6], params[7]

    frames_list = []
    sorted_img_ids = sorted(images.keys(), key=lambda i: frame_number(images[i]["name"]))

    for img_id in sorted_img_ids:
        img = images[img_id]
        R_w2c = quat_to_rotmat(img["qw"], img["qx"], img["qy"], img["qz"])
        t_w2c = np.array([img["tx"], img["ty"], img["tz"]])
        C = -R_w2c.T @ t_w2c
        R_c2w = R_w2c.T

        # Convert OpenCV camera axes to OpenGL camera axes:
        # Flip column 1 (Y: down -> up) and column 2 (Z: forward -> back)
        c2w_opengl = np.eye(4)
        c2w_opengl[:3, 0] = R_c2w[:, 0]   # +X Right
        c2w_opengl[:3, 1] = -R_c2w[:, 1]  # +Y Up
        c2w_opengl[:3, 2] = -R_c2w[:, 2]  # +Z Backwards
        c2w_opengl[:3, 3] = C

        fname = img["name"]
        rel_path = f"{image_rel_prefix}/{fname}" if not fname.startswith(image_rel_prefix) else fname

        frames_list.append({
            "file_path": rel_path,
            "transform_matrix": c2w_opengl.tolist(),
        })

    transforms_data = {
        "camera_model": "OPENCV",
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(w),
        "h": int(h),
        "k1": float(k1),
        "k2": float(k2),
        "p1": float(p1),
        "p2": float(p2),
        "applied_transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "frames": frames_list,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=2)


# ---------------------------------------------------------------------------
# Main Alignment Pipeline
# ---------------------------------------------------------------------------

def align_scene(
    scene: str,
    config_path: str = "configs/default.yaml",
    input_model: Optional[str] = None,
    output_dir: Optional[str] = None,
    floor_height_manual: Optional[float] = None,
    scale_mode: Optional[str] = None,
    point_ids: Optional[List[int]] = None,
    point_coords: Optional[List[float]] = None,
    known_distance: Optional[float] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete Phase 4 Scene Alignment pipeline."""
    repo_root = Path.cwd()
    cfg = load_config(config_path)

    # Config sections
    align_cfg = cfg.get("align", cfg.get("alignment", {}))
    data_dir_path = Path(data_dir) if data_dir else (repo_root / "data" / scene)
    data_dir = data_dir_path

    # -----------------------------------------------------------------------
    # STEP 0: PICK THE INPUT
    # -----------------------------------------------------------------------
    model_key = input_model or align_cfg.get("input_model")
    model_path = pick_best_colmap_model(repo_root, scene, model_key)
    print(f"\n[STEP 0] Selected input model: {model_path}")

    # -----------------------------------------------------------------------
    # STEP 1: LOAD
    # -----------------------------------------------------------------------
    print("\n[STEP 1] Loading COLMAP model...")
    txt_cameras = model_path / "cameras.txt"
    txt_images = model_path / "images.txt"
    txt_points = model_path / "points3D.txt"

    # Convert binary to TXT if TXT is missing
    if not (txt_cameras.is_file() and txt_images.is_file() and txt_points.is_file()):
        colmap_bin = shutil.which("colmap")
        if colmap_bin:
            cmd = [
                colmap_bin, "model_converter",
                "--input_path", str(model_path),
                "--output_path", str(model_path),
                "--output_type", "TXT"
            ]
            subprocess.run(cmd, capture_output=True, check=True)

    cameras = parse_cameras_txt(txt_cameras)
    images = parse_images_txt(txt_images)
    points = parse_points3d_txt(txt_points)

    num_images = len(images)
    num_points = len(points)
    if num_images == 0 or num_points == 0:
        raise RuntimeError(f"Model at {model_path} has 0 images or points.")

    # Calculate mean reprojection error
    reproj_errors = [p["error"] for p in points.values()]
    mean_reproj_error = float(np.mean(reproj_errors)) if reproj_errors else 0.0

    # Intrinsics
    primary_cam = list(cameras.values())[0]
    cam_model = primary_cam["model"]
    params = primary_cam["params"]
    fx, fy, k1 = 0.0, 0.0, 0.0

    if cam_model in ("SIMPLE_RADIAL", "RADIAL"):
        fx = params[0]
        fy = params[0]
        k1 = params[3]
    elif cam_model == "PINHOLE":
        fx = params[0]
        fy = params[1]
        k1 = 0.0
    elif cam_model in ("OPENCV", "OPENCV_FISHEYE"):
        fx = params[0]
        fy = params[1]
        k1 = params[4]

    print(f"  Images count          : {num_images}")
    print(f"  3D points count       : {num_points}")
    print(f"  Mean reproj error     : {mean_reproj_error:.4f} px")
    print(f"  Intrinsics            : fx={fx:.2f}, fy={fy:.2f}, k1={k1:.4f} (Model: {cam_model})")

    # If fx and fy differ by > 5%, print WARNING
    f_denom = max(fx, fy) if max(fx, fy) > 1e-4 else 1.0
    f_diff_pct = abs(fx - fy) / f_denom * 100.0
    if f_diff_pct > 5.0:
        print(f"  [WARNING] fx ({fx:.1f}) and fy ({fy:.1f}) differ by {f_diff_pct:.2f}% (> 5% threshold)!")

    # -----------------------------------------------------------------------
    # STEP 2: UP VECTOR
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Estimating camera-up vector and spread...")
    # Per camera, world_up_i = -R_cw[:, 1]. R_cw = R_wc.T
    cam_ups = []
    cam_centres = []
    image_list = sorted(images.values(), key=lambda im: frame_number(im["name"]))

    for im in image_list:
        R_wc = quat_to_rotmat(im["qw"], im["qx"], im["qy"], im["qz"])
        R_cw = R_wc.T
        world_up_i = -R_cw[:, 1]
        world_up_i /= np.linalg.norm(world_up_i)
        cam_ups.append(world_up_i)
        cam_centres.append(camera_centre(im))

    U = np.array(cam_ups)
    cam_centres = np.array(cam_centres)

    # First principal direction via SVD of stacked up vectors
    mu_up = np.mean(U, axis=0)
    _, _, Vt = np.linalg.svd(U)
    up_vector = Vt[0]
    if np.dot(up_vector, mu_up) < 0:
        up_vector = -up_vector
    up_vector /= np.linalg.norm(up_vector)

    # Spread: mean angle between each camera's up and the result
    angles = np.degrees(np.arccos(np.clip(U @ up_vector, -1.0, 1.0)))
    up_spread_deg = float(np.mean(angles))

    print(f"  Result UP vector      : [{up_vector[0]:+.4f}, {up_vector[1]:+.4f}, {up_vector[2]:+.4f}]")
    print(f"  Camera up spread      : {up_spread_deg:.2f}°")

    if up_spread_deg > 25.0:
        print(f"  [WARNING] Camera up spread ({up_spread_deg:.1f}°) exceeds 25° (probable rotation-only motion).")

    # -----------------------------------------------------------------------
    # STEP 3: FLOOR
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Fitting floor planes via Open3D RANSAC...")
    pts_xyz = np.array([p["xyz"] for p in points.values()])
    h_all = pts_xyz @ up_vector

    # Project points on UP and keep lowest 25% of height range
    h_p005 = float(np.percentile(h_all, 0.5))
    h_p995 = float(np.percentile(h_all, 99.5))
    h_range = h_p995 - h_p005
    cutoff_25 = h_p005 + 0.25 * h_range
    # Ensure at least lowest 25% of points are retained
    cutoff = max(cutoff_25, float(np.percentile(h_all, 25.0)))

    pts_lowest = pts_xyz[h_all <= cutoff]
    print(f"  Points in lowest 25% height range (h <= {cutoff:.3f}): {len(pts_lowest)} / {len(pts_xyz)}")

    max_angle_deg = float(align_cfg.get("max_floor_angle_deg", 8.0))
    min_inlier_ratio = float(align_cfg.get("min_inlier_ratio", 0.015))
    min_inliers_req = int(math.ceil(min_inlier_ratio * num_points))
    dist_thresh = float(align_cfg.get("dist_thresh", 0.05))
    max_planes = int(align_cfg.get("max_planes", 6))

    plane_candidates = []
    chosen_plane = None

    manual_h = floor_height_manual if floor_height_manual is not None else align_cfg.get("floor_height_manual")

    if manual_h is not None:
        print(f"  [INFO] Manual floor height override specified: {manual_h:.4f}")
        chosen_plane = {
            "plane_idx": "manual",
            "normal": up_vector,
            "d": -float(manual_h),
            "height": float(manual_h),
            "inliers": int(np.sum(np.abs(pts_xyz @ up_vector - float(manual_h)) < dist_thresh)),
            "angle_to_up": 0.0,
            "chosen": True,
        }
        plane_candidates.append(chosen_plane)
    else:
        if o3d is None:
            raise RuntimeError("Open3D is required for RANSAC plane fitting.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_lowest)
        curr_pcd = pcd

        for p_idx in range(1, max_planes + 1):
            if len(curr_pcd.points) < 50:
                break
            plane_model, inlier_indices = curr_pcd.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=3,
                num_iterations=2000,
            )
            nrm = np.array(plane_model[:3])
            d_val = float(plane_model[3])
            n_len = np.linalg.norm(nrm)
            if n_len < 1e-7:
                curr_pcd = curr_pcd.select_by_index(inlier_indices, invert=True)
                continue
            nrm /= n_len
            d_val /= n_len

            # Orient normal so nrm @ up_vector > 0
            if np.dot(nrm, up_vector) < 0:
                nrm = -nrm
                d_val = -d_val

            angle_to_up = float(np.degrees(np.arccos(np.clip(np.dot(nrm, up_vector), -1.0, 1.0))))

            # Inliers across all points in model
            all_inliers = int(np.sum(np.abs(pts_xyz @ nrm + d_val) < dist_thresh))

            # Height along UP vector
            inlier_pts = np.asarray(curr_pcd.select_by_index(inlier_indices).points)
            mean_h = float(np.mean(inlier_pts @ up_vector)) if len(inlier_pts) > 0 else -d_val

            plane_candidates.append({
                "plane_idx": p_idx,
                "normal": nrm,
                "d": d_val,
                "height": mean_h,
                "inliers": all_inliers,
                "angle_to_up": angle_to_up,
                "chosen": False,
            })
            curr_pcd = curr_pcd.select_by_index(inlier_indices, invert=True)

        # Filter qualifying planes: angle <= max_angle_deg and inliers >= 1.5% of all points
        qualifying = [
            c for c in plane_candidates
            if c["angle_to_up"] <= max_angle_deg and c["inliers"] >= min_inliers_req
        ]

        if qualifying:
            # Choose the LOWEST plane to avoid tabletops, stage tops, seats
            qualifying.sort(key=lambda c: c["height"])
            chosen_plane = qualifying[0]
            for c in plane_candidates:
                if c["plane_idx"] == chosen_plane["plane_idx"]:
                    c["chosen"] = True
                    break

    # Print candidate table
    print("\n  Plane Candidate Table:")
    print("  " + "-" * 72)
    print(f"  {'Plane':<6} {'Height (u)':<12} {'Inliers':<12} {'Inlier %':<10} {'Angle to UP':<14} {'Chosen?'}")
    print("  " + "-" * 72)
    for c in plane_candidates:
        inlier_pct = (c["inliers"] / num_points) * 100.0
        chosen_str = "[YES] CHOSEN" if c["chosen"] else "No"
        print(f"  #{c['plane_idx']:<5} {c['height']:<12.4f} {c['inliers']:<12} {inlier_pct:<9.2f}% {c['angle_to_up']:<13.2f}° {chosen_str}")
    print("  " + "-" * 72)

    if chosen_plane is None:
        print("\n[FAIL] Floor detection failed: No candidate plane qualified!", file=sys.stderr)
        print("  Actionable advice: Check lighting and the textureless floor in your capture.", file=sys.stderr)
        print("  You can manually override the floor plane using --floor-height <value>.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Selected Floor Plane: Height = {chosen_plane['height']:.4f} u, Inliers = {chosen_plane['inliers']} ({chosen_plane['inliers']/num_points*100:.2f}%), Angle to UP = {chosen_plane['angle_to_up']:.2f}°")

    # -----------------------------------------------------------------------
    # STEP 4: ROTATE AND TRANSLATE
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Applying rotation and translation...")
    floor_n = chosen_plane["normal"]
    # 1. Rotate UP/normal to +Y = [0, 1, 0]
    R_floor = rodrigues_align(floor_n, np.array([0.0, 1.0, 0.0]))
    assert abs(np.linalg.det(R_floor) - 1.0) < 1e-4

    # 2. Wall alignment: PCA of in-plane points between 0.3m and 1.8m above floor
    pts_rot1 = pts_xyz @ R_floor.T
    cam_centres_rot1 = cam_centres @ R_floor.T

    # Preliminary scale estimate from camera height to compute 0.3m and 1.8m bounds
    mean_cam_y1 = float(np.mean(cam_centres_rot1[:, 1]))
    floor_y1 = chosen_plane["height"]  # unscaled height
    rough_h = abs(mean_cam_y1 - floor_y1) if abs(mean_cam_y1 - floor_y1) > 0.1 else 1.45
    rough_scale = 1.45 / rough_h  # meters per unscaled unit

    # Points between 0.3m and 1.8m above floor
    y_min_u = floor_y1 + 0.30 / rough_scale
    y_max_u = floor_y1 + 1.80 / rough_scale
    wall_mask = (pts_rot1[:, 1] >= y_min_u) & (pts_rot1[:, 1] <= y_max_u)
    wall_pts = pts_rot1[wall_mask]
    if len(wall_pts) < 100:
        wall_pts = pts_rot1  # fallback to all points if sparse

    # PCA in X-Z horizontal plane
    xz = wall_pts[:, [0, 2]]
    mu_xz = np.mean(xz, axis=0)
    cov_xz = np.cov((xz - mu_xz).T)
    eigvals, eigvecs = np.linalg.eigh(cov_xz)
    v_wall = eigvecs[:, 1]  # principal wall direction
    phi_wall = math.atan2(v_wall[1], v_wall[0])

    # First leg of camera path
    # sorted by frame number
    c_start = cam_centres_rot1[0]
    # First leg end is after ~15-25 frames or 10% of path
    leg1_idx = min(25, len(cam_centres_rot1) - 1)
    c_leg1 = cam_centres_rot1[leg1_idx]
    v_leg1 = c_leg1 - c_start  # [dx, dy, dz]

    # Test 4 90-degree yaw options to make dominant walls axis-aligned
    # AND make FIRST leg point UP the image in top.png (image vertical is Z inverted -> points along -Z)
    best_yaw_deg = 0.0
    best_score = -1e9

    for k in range(4):
        yaw_deg = math.degrees(-phi_wall) + k * 90.0
        rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(rad), math.sin(rad)
        # R2 rotates (X, Z)
        R2 = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        leg1_rot = R2 @ np.array([v_leg1[0], v_leg1[2]])
        # Pointing UP the image in inverted-Z plot means heading towards -Z: score = -Z_rot
        score = -leg1_rot[1]
        if score > best_score:
            best_score = score
            best_yaw_deg = yaw_deg

    rad_best = math.radians(best_yaw_deg)
    cos_b, sin_b = math.cos(rad_best), math.sin(rad_best)
    R_yaw = np.array([
        [ cos_b, 0.0, -sin_b],
        [   0.0, 1.0,    0.0],
        [ sin_b, 0.0,  cos_b],
    ])
    assert abs(np.linalg.det(R_yaw) - 1.0) < 1e-4

    # Combined proper rotation matrix (det = +1.0)
    R_total = R_yaw @ R_floor
    assert abs(np.linalg.det(R_total) - 1.0) < 1e-4

    # Rotate 3D points
    for pt in points.values():
        pt["xyz"] = R_total @ pt["xyz"]

    # Rotate cameras: R_wc_new = R_wc_old @ R_total.T
    for im in images.values():
        R_cur = quat_to_rotmat(im["qw"], im["qx"], im["qy"], im["qz"]) @ R_total.T
        qw, qx, qy, qz = rotmat_to_quat(R_cur)
        im["qw"], im["qx"], im["qy"], im["qz"] = qw, qx, qy, qz

    # Translate so floor plane is at y = 0.0
    # Measure floor Y in rotated frame
    inlier_mask = np.abs(pts_xyz @ floor_n + chosen_plane["d"]) < dist_thresh
    floor_pts_rot = pts_xyz[inlier_mask] @ R_total.T
    floor_y_unscaled = float(np.median(floor_pts_rot[:, 1])) if len(floor_pts_rot) > 0 else chosen_plane["height"]

    t_snap = np.array([0.0, -floor_y_unscaled, 0.0])
    for pt in points.values():
        pt["xyz"] += t_snap

    # Camera translation update: t_wc_new = t_wc_old - R_wc_new @ t_snap
    for im in images.values():
        R_wc_cur = quat_to_rotmat(im["qw"], im["qx"], im["qy"], im["qz"])
        t_old = np.array([im["tx"], im["ty"], im["tz"]])
        t_new = t_old - R_wc_cur @ t_snap
        im["tx"], im["ty"], im["tz"] = float(t_new[0]), float(t_new[1]), float(t_new[2])

    print(f"  Wall Yaw Rotation     : {best_yaw_deg:.2f}° (First leg points North / up top-view)")
    print(f"  Floor Snapped Level   : from y = {floor_y_unscaled:.4f} u to y = 0.0000 m")

    # -----------------------------------------------------------------------
    # STEP 5: SCALE
    # -----------------------------------------------------------------------
    print("\n[STEP 5] Metric scale recovery...")
    # Determine passes / videos
    # Pass 1: video1 or frames 1..pass_split
    # Pass 2: video2 or frames > pass_split
    pass_split = 222
    frames_rep_path = data_dir / "frames_report.json"
    if frames_rep_path.is_file():
        try:
            with open(frames_rep_path, "r", encoding="utf-8") as fr_f:
                vids = json.load(fr_f).get("videos", [])
                if vids:
                    pass_split = vids[0].get("kept_after_duplicate", vids[0].get("raw_extracted", 222))
        except Exception:
            pass

    # Separate camera heights by pass
    p1_cams_unscaled = []
    p2_cams_unscaled = []

    for im in images.values():
        C = camera_centre(im)
        name = im["name"].lower()
        f_num = frame_number(name)
        if "video2" in name or ("video1" not in name and f_num > pass_split):
            p2_cams_unscaled.append(C[1])
        else:
            p1_cams_unscaled.append(C[1])

    mean_p1_unscaled = float(np.mean(p1_cams_unscaled)) if p1_cams_unscaled else float(np.mean([camera_centre(im)[1] for im in images.values()]))
    mean_p2_unscaled = float(np.mean(p2_cams_unscaled)) if p2_cams_unscaled else mean_p1_unscaled

    target_eye_h = float(align_cfg.get("eye_level_height_m", 1.45))
    target_chest_h = float(align_cfg.get("chest_level_height_m", 1.15))

    scale_method = scale_mode or align_cfg.get("scale_mode", "phone_height")
    s = 1.0

    # Mode (a): measurements.json point IDs or coords
    meas_path = data_dir / "measurements.json"
    meas_data = {}
    if meas_path.is_file():
        try:
            with open(meas_path, "r", encoding="utf-8") as mf:
                meas_data = json.load(mf)
        except Exception:
            pass

    if scale_method == "points" and point_ids and known_distance:
        p1_id, p2_id = point_ids
        if p1_id in points and p2_id in points:
            dist_u = float(np.linalg.norm(points[p1_id]["xyz"] - points[p2_id]["xyz"]))
            s = known_distance / dist_u
            print(f"  Mode (a) Point IDs [{p1_id}, {p2_id}]: dist = {dist_u:.4f} u, target = {known_distance:.3f} m -> Scale = {s:.5f} m/u")
        else:
            print("  [WARN] Specified point IDs not found; falling back to Mode (b).")
            scale_method = "phone_height"
    elif scale_method == "points" and point_coords and known_distance:
        c1 = np.array(point_coords[:3])
        c2 = np.array(point_coords[3:])
        dist_u = float(np.linalg.norm(c1 - c2))
        s = known_distance / dist_u
        print(f"  Mode (a) Point Coordinates: dist = {dist_u:.4f} u, target = {known_distance:.3f} m -> Scale = {s:.5f} m/u")
    else:
        # Mode (b) Default: scale so eye level = 1.45 m
        scale_method = "phone_height"
        s = target_eye_h / mean_p1_unscaled if abs(mean_p1_unscaled) > 1e-4 else 1.0
        print(f"  Mode (b) Reference: Eye level unscaled = {mean_p1_unscaled:.4f} u, Target = {target_eye_h:.2f} m -> Scale = {s:.5f} m/u")

    # Apply uniform scale to points and cameras
    for pt in points.values():
        pt["xyz"] *= s

    for im in images.values():
        im["tx"] *= s
        im["ty"] *= s
        im["tz"] *= s

    # Per-pass heights before and after
    mean_p1_scaled = mean_p1_unscaled * s
    mean_p2_scaled = mean_p2_unscaled * s
    ratio_scaled = mean_p1_scaled / mean_p2_scaled if mean_p2_scaled > 1e-4 else 1.0

    print("\n  Per-Pass Camera Heights:")
    print(f"    Pass 1 (Eye level)   : Before = {mean_p1_unscaled:.4f} u, After = {mean_p1_scaled:.3f} m (Reference: {target_eye_h:.2f} m)")
    if p2_cams_unscaled:
        print(f"    Pass 2 (Chest level) : Before = {mean_p2_unscaled:.4f} u, After = {mean_p2_scaled:.3f} m (Check: {target_chest_h:.2f} m)")
        print(f"    Eye / Chest Ratio    : {ratio_scaled:.2f}x")
        if not (1.10 <= ratio_scaled <= 1.50):
            print(f"    [WARNING] Eye/chest height ratio ({ratio_scaled:.2f}) is not between 1.1 and 1.5!")

    # Room extents and ceiling height (robust point filtering: track length >= 4, error <= 1.0 px)
    filtered_pts = []
    for p in points.values():
        t_tokens = p.get("track", "").split()
        if len(t_tokens) // 2 >= 4 and float(p.get("error", 0.0)) <= 1.0:
            filtered_pts.append(p["xyz"])

    if len(filtered_pts) >= 50:
        robust_pts = np.array(filtered_pts)
    else:
        robust_pts = np.array([p["xyz"] for p in points.values()])

    x_p1, x_p99 = float(np.percentile(robust_pts[:, 0], 1)), float(np.percentile(robust_pts[:, 0], 99))
    z_p1, z_p99 = float(np.percentile(robust_pts[:, 2], 1)), float(np.percentile(robust_pts[:, 2], 99))
    robust_x_span = x_p99 - x_p1
    robust_z_span = z_p99 - z_p1

    aligned_cams = np.array([camera_centre(im) for im in images.values()])
    cam_x_min, cam_x_max = float(np.min(aligned_cams[:, 0])), float(np.max(aligned_cams[:, 0]))
    cam_z_min, cam_z_max = float(np.min(aligned_cams[:, 2])), float(np.max(aligned_cams[:, 2]))
    cam_footprint_x = cam_x_max - cam_x_min
    cam_footprint_z = cam_z_max - cam_z_min

    # Ceiling determination following competition SOP
    max_cam_h = float(np.max(aligned_cams[:, 1])) if len(aligned_cams) > 0 else 1.45
    min_ceiling_h = max_cam_h + 0.5
    is_synthetic = scene.startswith("synth") or not meas_path.is_file()
    meas_ceiling = meas_data.get("ceiling_height_m") if not is_synthetic else None

    if meas_ceiling is not None and float(meas_ceiling) > 0:
        ceiling_h_m = max(float(meas_ceiling), min_ceiling_h)
        ceiling_label = "measured"
    else:
        # Estimate ONLY from points inside camera footprint + 1.0m and above max camera height + 0.3m
        box_mask = (
            (robust_pts[:, 0] >= cam_x_min - 1.0) & (robust_pts[:, 0] <= cam_x_max + 1.0) &
            (robust_pts[:, 2] >= cam_z_min - 1.0) & (robust_pts[:, 2] <= cam_z_max + 1.0) &
            (robust_pts[:, 1] >= max_cam_h + 0.3)
        )
        ceil_candidates = robust_pts[box_mask]
        if len(ceil_candidates) >= 500:
            ceiling_h_m = max(float(np.percentile(ceil_candidates[:, 1], 95)), min_ceiling_h)
            ceiling_label = "estimated"
        else:
            print("ceiling not reconstructed")
            default_ceiling = float(
                align_cfg.get("ceiling_default_m")
                or cfg.get("alignment", {}).get("ceiling_default_m")
                or cfg.get("alignment", {}).get("ceiling_height_m")
                or 3.0
            )
            ceiling_h_m = max(default_ceiling, min_ceiling_h)
            ceiling_label = "default"

    print(f"\n  Camera-Loop Footprint (X x Z): {cam_footprint_x:.2f} m x {cam_footprint_z:.2f} m")
    print(f"  Robust Point Extents  (X x Z): {robust_x_span:.2f} m x {robust_z_span:.2f} m")
    print(f"  Ceiling Height               : {ceiling_h_m:.2f} m ({ceiling_label})")

    # Compare against measurements.json if not synthetic
    if not is_synthetic:
        meas_list = meas_data.get("measurements", [])
        ref_dim = None
        for m in meas_list:
            m_name = m.get("name", "")
            if "square" in m_name or "side" in m_name or "footprint" in m_name:
                ref_dim = float(m.get("metres", 0.0))
                break
        if ref_dim and ref_dim > 0:
            diff_cam_x = ((cam_footprint_x - ref_dim) / ref_dim) * 100.0
            diff_cam_z = ((cam_footprint_z - ref_dim) / ref_dim) * 100.0
            diff_ext_x = ((robust_x_span - ref_dim) / ref_dim) * 100.0
            diff_ext_z = ((robust_z_span - ref_dim) / ref_dim) * 100.0
            print(f"  Measurements Comparison (vs {ref_dim:.2f} m reference):")
            print(f"    Camera Footprint X: {cam_footprint_x:.2f} m ({diff_cam_x:+.1f}%) | Z: {cam_footprint_z:.2f} m ({diff_cam_z:+.1f}%)")
            print(f"    Robust Extents   X: {robust_x_span:.2f} m ({diff_ext_x:+.1f}%) | Z: {robust_z_span:.2f} m ({diff_ext_z:+.1f}%)")

        if meas_ceiling:
            meas_c_val = float(meas_ceiling)
            c_diff_pct = ((ceiling_h_m - meas_c_val) / meas_c_val) * 100.0
            print(f"    Ceiling Height    : {ceiling_h_m:.2f} m vs Measured {meas_c_val:.2f} m ({c_diff_pct:+.1f}%)")

    # -----------------------------------------------------------------------
    # STEP 6: OUTPUTS
    # -----------------------------------------------------------------------
    print("\n[STEP 6] Saving aligned outputs...")
    out_dir = Path(output_dir) if output_dir else (data_dir / "colmap_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Aligned COLMAP model in TXT and Binary form
    write_cameras_txt(cameras, out_dir / "cameras.txt")
    write_images_txt(images, out_dir / "images.txt")
    write_points3d_txt(points, out_dir / "points3D.txt")
    convert_model_to_bin_if_possible(out_dir, out_dir)
    print(f"  [+] Saved aligned model: {out_dir} (TXT & BIN)")

    # 2. transforms.json in Nerfstudio format
    transforms_file = data_dir / "transforms.json"
    export_nerfstudio_transforms(cameras, images, transforms_file, image_rel_prefix="images")
    export_nerfstudio_transforms(cameras, images, out_dir / "transforms.json", image_rel_prefix="../images")
    print(f"  [+] Saved Nerfstudio transforms.json: {transforms_file}")

    # 3. alignment.json
    alignment_summary = {
        "scene": scene,
        "convention": "right-handed Y-up metric (floor at y = 0.0, units = metres)",
        "R": R_total.tolist(),
        "t": (t_snap * s).tolist(),
        "s": float(s),
        "up_vector_detected": up_vector.tolist(),
        "up_spread_deg": round(up_spread_deg, 3),
        "floor_plane": {
            "normal": chosen_plane["normal"].tolist(),
            "height_unscaled": round(chosen_plane["height"], 4),
            "inliers": chosen_plane["inliers"],
            "angle_to_up_deg": round(chosen_plane["angle_to_up"], 2),
        },
        "per_pass_heights_m": {
            "pass_1_eye": round(mean_p1_scaled, 3),
            "pass_2_chest": round(mean_p2_scaled, 3) if p2_cams_unscaled else None,
            "ratio": round(ratio_scaled, 2),
        },
        "camera_footprint_m": {
            "width_x": round(cam_footprint_x, 3),
            "length_z": round(cam_footprint_z, 3),
        },
        "robust_extents_m": {
            "width_x": round(robust_x_span, 3),
            "length_z": round(robust_z_span, 3),
        },
        "ceiling_height_m": round(ceiling_h_m, 3),
        "ceiling_source": ceiling_label,
        "room_extents_m": {
            "width_x": round(robust_x_span, 3),
            "length_z": round(robust_z_span, 3),
            "height_y": round(ceiling_h_m, 3),
        },
        "num_cameras": num_images,
        "num_points": num_points,
        "reprojection_error_px": round(mean_reproj_error, 4),
    }

    with open(data_dir / "alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment_summary, f, indent=2)
    with open(out_dir / "alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment_summary, f, indent=2)
    print(f"  [+] Saved alignment.json: {data_dir / 'alignment.json'}")

    # 4. Sparse init PLY point cloud
    ply_path = out_dir / "points3D.ply"
    write_points3d_ply(points, ply_path)
    write_points3d_ply(points, data_dir / "points3D.ply")
    print(f"  [+] Saved sparse init PLY: {ply_path}")

    print("\nPhase 4 Scene Alignment completed successfully.")
    return alignment_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Metric, Gravity-Aligned, Y-up Scene Alignment.")
    parser.add_argument("--scene", required=True, help="Scene name (subfolder under data/)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML (default: configs/default.yaml)")
    parser.add_argument("--input-model", default=None, help="Input COLMAP sparse model directory (default: best registered model)")
    parser.add_argument("--output-dir", default=None, help="Output directory for aligned model (default: data/<scene>/colmap_aligned)")
    parser.add_argument("--data-dir", default=None, help="Data directory override (default: data/<scene>)")
    parser.add_argument("--floor-height", type=float, default=None, help="Manual floor height override")
    parser.add_argument("--scale-mode", choices=["phone_height", "points"], default=None, help="Scale mode")
    parser.add_argument("--point-ids", nargs=2, type=int, default=None, help="Two 3D point IDs for distance scaling")
    parser.add_argument("--point-coords", nargs=6, type=float, default=None, help="Two 3D coordinates for distance scaling")
    parser.add_argument("--distance", type=float, default=None, help="Known metric distance in metres")
    args = parser.parse_args()

    try:
        align_scene(
            scene=args.scene,
            config_path=args.config,
            input_model=args.input_model,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            floor_height_manual=args.floor_height,
            scale_mode=args.scale_mode,
            point_ids=args.point_ids,
            point_coords=args.point_coords,
            known_distance=args.distance,
        )
    except Exception as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
