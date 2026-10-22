#!/usr/bin/env python3
"""Render sparse COLMAP point cloud and camera trajectory to top-view PNG images.

Implements robust top-view generation:
1. Auto-detect up axis / floor normal via camera up poses + RANSAC plane fit.
2. Clean point cloud (outlier removal, reprojection error, track length, height band, in-plane clipping).
3. Render floor plan via 2D density histogram (0.05-unit bins, log scale) with PCA axis alignment.
4. Render multi-pass camera trajectory with Start/End markers and directional arrows.
5. Scale bar matching metric or unaligned units.
6. 1920x1080 output with comprehensive stats header and optional side-view debug plots.

Usage:
  python scripts/view_sparse.py --scene home
  python scripts/view_sparse.py --scene home --debug
  python scripts/view_sparse.py --model-dir data/home/colmap/sparse/0 --output data/home/colmap/top_view.png
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    for p in sys.path:
        o3d_dir = Path(p) / "open3d"
        if o3d_dir.is_dir():
            os.environ["PATH"] = f"{o3d_dir};{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(o3d_dir))
                except Exception:
                    pass
            break

import cv2
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix (world -> cam)."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def rodrigues_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that rotates unit vector v_from into v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    c = float(np.dot(v_from, v_to))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        # 180-deg flip around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v_from[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        ortho = np.cross(v_from, axis)
        ortho = ortho / np.linalg.norm(ortho)
        return 2.0 * np.outer(ortho, ortho) - np.eye(3)

    v_cross = np.cross(v_from, v_to)
    s = np.linalg.norm(v_cross)
    K = np.array([
        [0.0, -v_cross[2], v_cross[1]],
        [v_cross[2], 0.0, -v_cross[0]],
        [-v_cross[1], v_cross[0], 0.0],
    ])
    R = np.eye(3) + K + (K @ K) * ((1.0 - c) / (s ** 2))
    return R


# ---------------------------------------------------------------------------
# Robust COLMAP Model Parsers (Both .bin and .txt)
# ---------------------------------------------------------------------------

def parse_cameras(model_dir: Path) -> Dict[int, Dict]:
    """Parse cameras from cameras.bin or cameras.txt."""
    cams: Dict[int, Dict] = {}
    bin_path = model_dir / "cameras.bin"
    txt_path = model_dir / "cameras.txt"

    if bin_path.is_file():
        try:
            with open(bin_path, "rb") as f:
                num_cams = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_cams):
                    cam_id, model_id, w, h = struct.unpack("<2I2Q", f.read(24))
                    # Number of params varies by model; 4 to 12
                    # OPENCV has 8, PINHOLE has 4, SIMPLE_RADIAL has 3
                    # Read remaining floats based on model or fallback
                    # Model IDs: 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), 2=SIMPLE_RADIAL(4), 3=RADIAL(5), 4=OPENCV(8)
                    param_counts = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12}
                    n_p = param_counts.get(model_id, 4)
                    params = struct.unpack(f"<{n_p}d", f.read(n_p * 8))
                    fx = params[0]
                    fy = params[1] if n_p > 1 else fx
                    cams[cam_id] = {"w": w, "h": h, "fx": fx, "fy": fy}
            if cams:
                return cams
        except Exception:
            pass

    if txt_path.is_file():
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                w, h = float(parts[2]), float(parts[3])
                fx = float(parts[4])
                fy = float(parts[5]) if len(parts) > 5 else fx
                cams[cam_id] = {"w": w, "h": h, "fx": fx, "fy": fy}

    return cams


def parse_images(model_dir: Path) -> List[Dict[str, Any]]:
    """Parse camera poses from images.bin or images.txt.

    Returns sorted list of dicts with keys:
      'id', 'R', 'C', 'up', 'fwd', 'cam_id', 'f_num', 'name'
    """
    poses: List[Dict[str, Any]] = []
    bin_path = model_dir / "images.bin"
    txt_path = model_dir / "images.txt"

    if bin_path.is_file():
        try:
            with open(bin_path, "rb") as f:
                num_reg = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_reg):
                    iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<I4d3dI", f.read(64))
                    name_bytes = []
                    while True:
                        ch = f.read(1)
                        if not ch or ch == b"\x00":
                            break
                        name_bytes.append(ch)
                    name = b"".join(name_bytes).decode("utf-8", errors="replace")
                    num_pts2d = struct.unpack("<Q", f.read(8))[0]
                    f.seek(num_pts2d * 24, 1)

                    qvec = np.array([qw, qx, qy, qz])
                    tvec = np.array([tx, ty, tz])
                    R = qvec2rotmat(qvec)
                    C = -R.T @ tvec

                    # COLMAP convention:
                    # Camera up is negative Y axis in camera coords: [0, -1, 0]
                    # In world coords: u = R^T @ [0, -1, 0] = -R[1, :]
                    cam_up = -R[1, :].copy()
                    # Camera forward is +Z axis in camera coords: [0, 0, 1]
                    # In world coords: fwd = R^T @ [0, 0, 1] = R[2, :]
                    cam_fwd = R[2, :].copy()

                    digits = "".join(filter(str.isdigit, name))
                    f_num = int(digits) if digits else iid

                    poses.append({
                        "id": iid,
                        "R": R,
                        "C": C,
                        "up": cam_up,
                        "fwd": cam_fwd,
                        "cam_id": cid,
                        "f_num": f_num,
                        "name": name,
                    })
            if poses:
                poses.sort(key=lambda p: p["f_num"])
                return poses
        except Exception:
            pass

    if txt_path.is_file():
        with open(txt_path, "r", encoding="utf-8") as f:
            header = True
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if header:
                    parts = line.split(maxsplit=9)
                    if len(parts) == 10 and parts[0].isdigit():
                        iid = int(parts[0])
                        qvec = np.array([float(x) for x in parts[1:5]])
                        tvec = np.array([float(x) for x in parts[5:8]])
                        cid = int(parts[8])
                        name = parts[9]
                        digits = "".join(filter(str.isdigit, name))
                        f_num = int(digits) if digits else iid

                        R = qvec2rotmat(qvec)
                        C = -R.T @ tvec
                        cam_up = -R[1, :].copy()
                        cam_fwd = R[2, :].copy()

                        poses.append({
                            "id": iid,
                            "R": R,
                            "C": C,
                            "up": cam_up,
                            "fwd": cam_fwd,
                            "cam_id": cid,
                            "f_num": f_num,
                            "name": name,
                        })
                    header = False
                else:
                    header = True

    poses.sort(key=lambda p: p["f_num"])
    return poses


def parse_points3d(model_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse 3D points from points3D.bin or points3D.txt.

    Returns:
      pts: (N, 3) float64
      cols: (N, 3) float64 in [0, 1]
      errs: (N,) float64 reprojection errors in px
      tracks: (N,) int64 number of observations
    """
    bin_path = model_dir / "points3D.bin"
    txt_path = model_dir / "points3D.txt"

    if bin_path.is_file():
        try:
            with open(bin_path, "rb") as f:
                num_pts = struct.unpack("<Q", f.read(8))[0]
                pts = np.empty((num_pts, 3), dtype=np.float64)
                cols = np.empty((num_pts, 3), dtype=np.float64)
                errs = np.empty(num_pts, dtype=np.float64)
                tracks = np.empty(num_pts, dtype=np.int64)

                for i in range(num_pts):
                    pid, x, y, z, r, g, b, err, trk_len = struct.unpack("<Q3d3BdQ", f.read(51))
                    f.seek(trk_len * 8, 1)
                    pts[i] = [x, y, z]
                    cols[i] = [r / 255.0, g / 255.0, b / 255.0]
                    errs[i] = err
                    tracks[i] = trk_len

            return pts, cols, errs, tracks
        except Exception:
            pass

    if txt_path.is_file():
        pts_list, cols_list, errs_list, tracks_list = [], [], [], []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                        err = float(parts[7])
                        trk_len = (len(parts) - 8) // 2
                        pts_list.append([x, y, z])
                        cols_list.append([r / 255.0, g / 255.0, b / 255.0])
                        errs_list.append(err)
                        tracks_list.append(trk_len)
                    except ValueError:
                        continue
        if pts_list:
            return (
                np.array(pts_list, dtype=np.float64),
                np.array(cols_list, dtype=np.float64),
                np.array(errs_list, dtype=np.float64),
                np.array(tracks_list, dtype=np.int64),
            )

    # Fallback to PLY
    ply_path = model_dir / "points3D.ply"
    if ply_path.is_file():
        try:
            pcd = o3d.io.read_point_cloud(str(ply_path))
            n = len(pcd.points)
            if n > 0:
                pts = np.asarray(pcd.points)
                cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.full((n, 3), 0.7)
                errs = np.zeros(n, dtype=np.float64)
                tracks = np.full(n, 3, dtype=np.int64)
                return pts, cols, errs, tracks
        except Exception:
            pass

    return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Up-Axis Auto-Detection & Floor Alignment
# ---------------------------------------------------------------------------

def detect_up_vector_and_floor(
    poses: List[Dict[str, Any]],
    pts: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Auto-detect up axis and floor offset using cameras up vector + RANSAC plane fit.

    Returns:
      up_vector: (3,) unit vector pointing up in SfM coordinate frame
      floor_offset: scalar distance along up_vector for floor plane
      confidence_deg: angle between camera up and RANSAC floor normal
    """
    if not poses:
        return np.array([0.0, 1.0, 0.0]), 0.0, 0.0

    # 1. Average cameras up vector
    cam_ups = np.array([p["up"] for p in poses])
    # Dominant direction: compute mean and normalize
    avg_up = np.mean(cam_ups, axis=0)
    norm_up = np.linalg.norm(avg_up)
    if norm_up < 1e-6:
        avg_up = np.array([0.0, 1.0, 0.0])
    else:
        avg_up = avg_up / norm_up

    if len(pts) < 50:
        return avg_up, 0.0, 0.0

    # 2. Cross-check with RANSAC plane fit on lowest ~20% of points along avg_up
    h = pts @ avg_up
    p20 = np.percentile(h, 20.0)
    floor_candidates = pts[h <= p20]

    if len(floor_candidates) < 10:
        return avg_up, float(p20), 0.0

    # RANSAC plane fitting
    best_normal = avg_up.copy()
    best_dist = float(p20)
    best_inliers = 0
    inlier_thresh = 0.05  # 5cm tolerance

    rng = np.random.default_rng(42)
    n_candidates = len(floor_candidates)
    num_iters = min(600, max(100, n_candidates * 2))

    for _ in range(num_iters):
        idx = rng.choice(n_candidates, 3, replace=False)
        tri = floor_candidates[idx]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-6:
            continue
        normal = normal / n_len
        # Normal should point upward (agreement with camera up)
        if np.dot(normal, avg_up) < 0:
            normal = -normal

        d = -float(np.dot(normal, tri[0]))
        dists = np.abs(floor_candidates @ normal + d)
        inliers = int(np.sum(dists < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_dist = -d  # floor level along normal

    cos_ang = float(np.clip(np.dot(best_normal, avg_up), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))

    # If RANSAC normal agrees with camera up within 20 degrees, use it; else fallback
    if angle_deg <= 20.0 and best_inliers >= 20:
        up_vector = best_normal
        floor_offset = best_dist
    else:
        up_vector = avg_up
        floor_offset = float(np.percentile(h, 2.0))

    return up_vector, floor_offset, angle_deg


# ---------------------------------------------------------------------------
# Point Cloud Cleaning
# ---------------------------------------------------------------------------

def clean_point_cloud(
    pts: np.ndarray,
    cols: np.ndarray,
    errs: np.ndarray,
    tracks: np.ndarray,
    poses: List[Dict[str, Any]],
    max_reproj_error: float = 3.0,
    min_observations: int = 3,
    std_ratio: float = 2.0,
    nb_neighbors: int = 20,
    height_margin: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Clean point cloud:

    - Filter points with reprojection error > max_reproj_error or observations < min_observations
    - Statistical outlier removal (k=20, std_ratio=2.0)
    - Clip to 2nd-98th percentile of in-plane coordinates
    - Keep height band between floor+0.2m and ceiling-0.2m (floor is at y=0)

    Returns:
      clean_pts: (M, 3)
      clean_cols: (M, 3)
      floor_y: float (0.0)
      ceiling_y: float
    """
    if len(pts) == 0:
        return pts, cols, 0.0, 3.0

    # 1. Observation count and reprojection error filter
    mask = (tracks >= min_observations)
    if errs is not None and len(errs) == len(pts):
        mask &= (errs <= max_reproj_error)
    if np.sum(mask) >= 20:
        pts = pts[mask]
        cols = cols[mask]

    # 2. Statistical outlier removal via Open3D
    if len(pts) >= nb_neighbors:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcd_clean, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        if len(inlier_indices) >= 20:
            pts = np.asarray(pcd_clean.points)
            cols = np.asarray(pcd_clean.colors)

    # 3. Determine ceiling height along Y axis (floor is snapped at Y=0)
    # Estimate ceiling from 98th percentile of points and camera heights
    y_pts = pts[:, 1]
    p98_y = float(np.percentile(y_pts, 98.0))
    cam_ys = [p["C"][1] for p in poses] if poses else [1.5]
    max_cam_y = float(np.max(cam_ys)) if cam_ys else 1.5

    ceiling_y = max(p98_y, max_cam_y + 0.6)
    floor_y = 0.0

    # 4. Height band filter: floor + 0.2m to ceiling - 0.2m
    # Keeps vertical walls and furniture while removing floor tiles/carpets and ceiling points
    y_lo = floor_y + height_margin
    y_hi = max(ceiling_y - height_margin, y_lo + 0.5)
    h_mask = (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)

    if np.sum(h_mask) >= 20:
        pts = pts[h_mask]
        cols = cols[h_mask]

    # 5. Clip to 2nd-98th percentile of in-plane coordinates (X and Z)
    if len(pts) >= 20:
        x_lo, x_hi = np.percentile(pts[:, 0], [2.0, 98.0])
        z_lo, z_hi = np.percentile(pts[:, 2], [2.0, 98.0])
        # Add 5% padding so we clip only true distant stray points
        dx = max(x_hi - x_lo, 0.1) * 0.05
        dz = max(z_hi - z_lo, 0.1) * 0.05
        box_mask = (
            (pts[:, 0] >= x_lo - dx) & (pts[:, 0] <= x_hi + dx) &
            (pts[:, 2] >= z_lo - dz) & (pts[:, 2] <= z_hi + dz)
        )
        if np.sum(box_mask) >= 20:
            pts = pts[box_mask]
            cols = cols[box_mask]

    return pts, cols, floor_y, ceiling_y


# ---------------------------------------------------------------------------
# In-Plane PCA Axis Alignment
# ---------------------------------------------------------------------------

def compute_pca_yaw(pts_2d: np.ndarray) -> float:
    """Compute 2D PCA angle of in-plane points to align dominant walls with coordinate axes."""
    if len(pts_2d) < 10:
        return 0.0
    centroid = np.mean(pts_2d, axis=0)
    centered = pts_2d - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Dominant direction corresponds to largest eigenvalue
    dom_vec = eigvecs[:, -1]
    yaw = float(np.arctan2(dom_vec[1], dom_vec[0]))
    # Snap to nearest 90-degree quadrant to keep upright / readable
    # We want dominant walls horizontal or vertical
    yaw = yaw % (np.pi / 2.0)
    if yaw > np.pi / 4.0:
        yaw -= np.pi / 2.0
    return yaw


def rotate_2d(xy: np.ndarray, angle_rad: float, center: np.ndarray = np.array([0.0, 0.0])) -> np.ndarray:
    """Rotate 2D points around center by angle_rad."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (xy - center) @ R.T + center


# ---------------------------------------------------------------------------
# Real 2D Density Histogram Floor Plan Renderer
# ---------------------------------------------------------------------------

def render_topview_density(
    pts_clean: np.ndarray,
    poses: List[Dict[str, Any]],
    output_path: Path,
    up_vector: np.ndarray,
    split_indices: Optional[List[int]] = None,
    bin_size: float = 0.05,
    width: int = 1920,
    height: int = 1080,
    is_metric: bool = False,
    scale_label: str = "1 metre (metric)",
    arrow_every: int = 5,
) -> None:
    """Render crisp 2-D orthographic floor-plan density map + multi-pass trajectory."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Modern dark blueprint background
    canvas[:] = (22, 20, 28)

    # In Y-up coordinate frame, horizontal axes are X (axis 0) and Z (axis 2)
    # In-plane coordinates: u = X, v = Z
    pts_uv = pts_clean[:, [0, 2]] if len(pts_clean) > 0 else np.zeros((0, 2))
    cam_uv = np.array([p["C"][[0, 2]] for p in poses]) if poses else np.zeros((0, 2))

    # 1. Orient plan via PCA of in-plane points so dominant walls are axis-aligned
    pca_yaw = compute_pca_yaw(pts_uv) if len(pts_uv) >= 20 else 0.0
    centroid_2d = np.mean(pts_uv, axis=0) if len(pts_uv) > 0 else np.zeros(2)

    if len(pts_uv) > 0:
        pts_uv = rotate_2d(pts_uv, -pca_yaw, center=centroid_2d)
    if len(cam_uv) > 0:
        cam_uv = rotate_2d(cam_uv, -pca_yaw, center=centroid_2d)

    # Also rotate camera forward direction vectors
    for idx, p in enumerate(poses):
        fwd_uv = p["fwd"][[0, 2]]
        fwd_uv_rot = rotate_2d(fwd_uv.reshape(1, 2), -pca_yaw)[0]
        norm_fwd = np.linalg.norm(fwd_uv_rot)
        p["fwd_2d"] = fwd_uv_rot / max(norm_fwd, 1e-6)
        p["C_2d"] = cam_uv[idx]

    # 2. Determine equal-aspect data bounds with robust 1st-99th percentile clipping
    # so stray outlier points or cameras do not shrink the real room layout.
    if len(pts_uv) > 10:
        u_p1, u_p99 = np.percentile(pts_uv[:, 0], [1.0, 99.0])
        v_p1, v_p99 = np.percentile(pts_uv[:, 1], [1.0, 99.0])
        pts_span_u = max(u_p99 - u_p1, 0.5)
        pts_span_v = max(v_p99 - v_p1, 0.5)

        # Include cameras if they are reasonably near the point cloud
        if len(cam_uv) > 0:
            cam_valid_u = cam_uv[(cam_uv[:, 0] >= u_p1 - 0.5 * pts_span_u) & (cam_uv[:, 0] <= u_p99 + 0.5 * pts_span_u), 0]
            cam_valid_v = cam_uv[(cam_uv[:, 1] >= v_p1 - 0.5 * pts_span_v) & (cam_uv[:, 1] <= v_p99 + 0.5 * pts_span_v), 1]
            u_min = min(u_p1, float(np.min(cam_valid_u))) if len(cam_valid_u) > 0 else u_p1
            u_max = max(u_p99, float(np.max(cam_valid_u))) if len(cam_valid_u) > 0 else u_p99
            v_min = min(v_p1, float(np.min(cam_valid_v))) if len(cam_valid_v) > 0 else v_p1
            v_max = max(v_p99, float(np.max(cam_valid_v))) if len(cam_valid_v) > 0 else v_p99
        else:
            u_min, u_max = u_p1, u_p99
            v_min, v_max = v_p1, v_p99
    elif len(cam_uv) > 0:
        u_p1, u_p99 = np.percentile(cam_uv[:, 0], [1.0, 99.0])
        v_p1, v_p99 = np.percentile(cam_uv[:, 1], [1.0, 99.0])
        u_min, u_max = u_p1, u_p99
        v_min, v_max = v_p1, v_p99
    else:
        u_min, u_max = -5.0, 5.0
        v_min, v_max = -5.0, 5.0

    span_u = max(u_max - u_min, 0.2)
    span_v = max(v_max - v_min, 0.2)

    # 10% breathing margin around the room layout
    margin_ratio = 0.10
    u_min -= span_u * margin_ratio
    u_max += span_u * margin_ratio
    v_min -= span_v * margin_ratio
    v_max += span_v * margin_ratio
    span_u = u_max - u_min
    span_v = v_max - v_min

    # Screen layout
    pad_left, pad_right = 90, 90
    pad_top, pad_bottom = 120, 80
    draw_w = width - pad_left - pad_right
    draw_h = height - pad_top - pad_bottom

    # Equal aspect scale
    scale = min(draw_w / span_u, draw_h / span_v)
    drawn_w = span_u * scale
    drawn_h = span_v * scale

    x_off = pad_left + (draw_w - drawn_w) / 2.0
    y_off = pad_top + (draw_h - drawn_h) / 2.0

    def to_px(u: float, v: float) -> Tuple[int, int]:
        px = int(round(x_off + (u - u_min) * scale))
        # Top-down mapping: -Z / North is up, +Z is down
        py = int(round(y_off + (v - v_min) * scale))
        return px, py

    # 3. 2D Point Density Map (0.05-unit bins, log scale)
    if len(pts_uv) > 10:
        bin_w = max(bin_size, 0.02)
        bins_u = max(int(np.ceil(span_u / bin_w)), 30)
        bins_v = max(int(np.ceil(span_v / bin_w)), 30)

        H, u_edges, v_edges = np.histogram2d(
            pts_uv[:, 0], pts_uv[:, 1],
            bins=[bins_u, bins_v],
            range=[[u_min, u_max], [v_min, v_max]]
        )

        # Log-scale density so dense walls pop while sparse points remain subtle
        log_H = np.log1p(H)
        max_log = np.max(log_H)
        if max_log > 0:
            norm_density = (log_H / max_log).T  # Transpose for (v, u) image indexing

            # Render into drawn area
            dw_int = max(int(round(drawn_w)), 10)
            dh_int = max(int(round(drawn_h)), 10)
            density_resized = cv2.resize(norm_density, (dw_int, dh_int), interpolation=cv2.INTER_LINEAR)
            density_resized = np.clip(density_resized, 0.0, 1.0)

            # Modern electric cyan/teal glow colormap for wall density
            # Background: (22, 20, 28)
            # Walls glow brightly: cyan (255, 220, 40) in BGR
            wall_b = (22 + (255 - 22) * (density_resized ** 0.65)).astype(np.uint8)
            wall_g = (20 + (220 - 20) * (density_resized ** 0.75)).astype(np.uint8)
            wall_r = (28 + (40 - 28)  * (density_resized ** 1.10)).astype(np.uint8)
            density_bgr = cv2.merge([wall_b, wall_g, wall_r])

            # Alpha blend onto canvas
            alpha = (density_resized ** 0.8)[..., np.newaxis]
            x0 = int(round(x_off))
            y0 = int(round(y_off))
            x1 = min(x0 + dw_int, width)
            y1 = min(y0 + dh_int, height)
            sub_w = x1 - x0
            sub_h = y1 - y0

            if sub_w > 0 and sub_h > 0:
                fg = density_bgr[:sub_h, :sub_w].astype(float)
                bg = canvas[y0:y1, x0:x1].astype(float)
                a = alpha[:sub_h, :sub_w]
                blended = (fg * a + bg * (1.0 - a)).astype(np.uint8)
                canvas[y0:y1, x0:x1] = blended

    # 4. Multi-Pass Camera Trajectory
    passes = []
    if split_indices:
        last_s = 0
        for s in split_indices:
            sub = [p for p in poses if last_s < p["f_num"] <= s]
            if sub:
                passes.append(sub)
            last_s = s
        sub = [p for p in poses if p["f_num"] > last_s]
        if sub:
            passes.append(sub)
    else:
        passes = [poses]

    pass_styles = [
        {"name": "Pass 1 (Eye level)", "color_start": (255, 210, 0), "color_end": (255, 80, 0), "line_col": (245, 170, 30)},
        {"name": "Pass 2 (Chest level)", "color_start": (0, 240, 255), "color_end": (0, 110, 255), "line_col": (0, 200, 245)},
        {"name": "Pass 3", "color_start": (255, 0, 200), "color_end": (160, 0, 255), "line_col": (220, 0, 210)},
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    total_rendered_cams = 0

    for p_i, p_poses in enumerate(passes):
        style = pass_styles[p_i % len(pass_styles)]
        n_p = len(p_poses)
        cam_pixels: List[Tuple[int, int]] = []
        valid_poses = []

        for idx, p in enumerate(p_poses):
            u, v = p["C_2d"]
            px, py = to_px(u, v)
            if 0 <= px < width and 0 <= py < height:
                cam_pixels.append((px, py))
                valid_poses.append(p)
                t = idx / max(n_p - 1, 1)
                b = int(style["color_start"][0] * (1 - t) + style["color_end"][0] * t)
                g = int(style["color_start"][1] * (1 - t) + style["color_end"][1] * t)
                r = int(style["color_start"][2] * (1 - t) + style["color_end"][2] * t)
                cv2.circle(canvas, (px, py), 4, (b, g, r), -1)

        total_rendered_cams += len(cam_pixels)

        # Draw trajectory lines and small view-direction arrows
        for i in range(len(cam_pixels) - 1):
            pt_a = cam_pixels[i]
            pt_b = cam_pixels[i+1]
            dist_px = float(np.hypot(pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]))
            # Avoid connecting wildly discontinuous camera teleports
            if dist_px < 350:
                cv2.line(canvas, pt_a, pt_b, style["line_col"], 2, cv2.LINE_AA)

            # View-direction arrows every arrow_every frames
            if i % arrow_every == 0:
                p_curr = valid_poses[i]
                fwd_2d = p_curr.get("fwd_2d", np.array([0.0, -1.0]))
                arrow_len = 10.0
                ax = float(pt_a[0] + fwd_2d[0] * arrow_len)
                ay = float(pt_a[1] + fwd_2d[1] * arrow_len)
                cv2.arrowedLine(canvas, pt_a, (int(round(ax)), int(round(ay))),
                               (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)

        # Pass Start marker (Green circle + dot)
        if cam_pixels:
            cv2.circle(canvas, cam_pixels[0], 10, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(canvas, cam_pixels[0], 4, (0, 255, 120), -1)
            cv2.putText(canvas, f"Start (P{p_i+1})", (cam_pixels[0][0] + 14, cam_pixels[0][1] + 4),
                        font, 0.42, (0, 255, 160), 1, cv2.LINE_AA)

            # Pass End marker (Red circle + dot)
            cv2.circle(canvas, cam_pixels[-1], 10, (40, 60, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, cam_pixels[-1], 4, (40, 60, 255), -1)
            cv2.putText(canvas, f"End (P{p_i+1})", (cam_pixels[-1][0] + 14, cam_pixels[-1][1] + 4),
                        font, 0.42, (60, 80, 255), 1, cv2.LINE_AA)

    # 5. Scale Bar
    sb_unit = 1.0  # 1 metre or 1 unit
    sb_px = int(round(sb_unit * scale))
    if sb_px > 10:
        sx0, sy0 = pad_left, height - 35
        cv2.line(canvas, (sx0, sy0), (sx0 + sb_px, sy0), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (sx0, sy0 - 5), (sx0, sy0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (sx0 + sb_px, sy0 - 5), (sx0 + sb_px, sy0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, scale_label, (sx0, sy0 - 10), font, 0.40, (180, 180, 190), 1, cv2.LINE_AA)

    # 6. Compass Rose (North / -Z points Up on map)
    comp_x, comp_y = width - 110, 110
    cv2.circle(canvas, (comp_x, comp_y), 24, (36, 34, 46), -1)
    cv2.circle(canvas, (comp_x, comp_y), 24, (110, 110, 130), 1, cv2.LINE_AA)
    cv2.line(canvas, (comp_x, comp_y + 14), (comp_x, comp_y - 14), (50, 140, 255), 2, cv2.LINE_AA)
    pts_n = np.array([[comp_x, comp_y - 19], [comp_x - 5, comp_y - 10], [comp_x + 5, comp_y - 10]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts_n], (50, 140, 255))
    cv2.putText(canvas, "N", (comp_x - 4, comp_y - 23), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "E", (comp_x + 14, comp_y + 4), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, "W", (comp_x - 22, comp_y + 4), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, "S", (comp_x - 3, comp_y + 20), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)

    # 7. Legend for Passes & Start/End
    leg_y = 120
    for p_i, p_poses in enumerate(passes):
        style = pass_styles[p_i % len(pass_styles)]
        cv2.line(canvas, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), style["line_col"], 3)
        cv2.putText(canvas, f"{style['name']} ({len(p_poses)} cams)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22

    cv2.circle(canvas, (pad_left + 6, leg_y - 4), 6, (0, 255, 120), 2)
    cv2.circle(canvas, (pad_left + 6, leg_y - 4), 3, (0, 255, 120), -1)
    cv2.putText(canvas, "Start Point", (pad_left + 18, leg_y), font, 0.42, (0, 255, 120), 1, cv2.LINE_AA)

    cv2.circle(canvas, (pad_left + 115, leg_y - 4), 6, (40, 60, 255), 2)
    cv2.circle(canvas, (pad_left + 115, leg_y - 4), 3, (40, 60, 255), -1)
    cv2.putText(canvas, "End Point", (pad_left + 127, leg_y), font, 0.42, (60, 80, 255), 1, cv2.LINE_AA)

    # 8. Title and Comprehensive Stats Header
    up_str = f"[{up_vector[0]:.2f}, {up_vector[1]:.2f}, {up_vector[2]:.2f}]"
    cv2.putText(canvas, "NAMMA SPACE -- Floor Plan & Multi-Pass Trajectory",
                (pad_left, 48), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    stats_line = (
        f"Cameras: {total_rendered_cams} / {len(poses)}   |"
        f"   3D Points: {len(pts_clean):,} (density map: {bin_size:.2f}m bins)   |"
        f"   View axes: X-Z (looking along -Y)   |"
        f"   Detected Up: {up_str}"
    )
    cv2.putText(canvas, stats_line, (pad_left, 80), font, 0.44, (170, 175, 190), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"  [+] Saved Floor Plan top_view: {output_path}")


# ---------------------------------------------------------------------------
# Side-View Elevation Plots (--debug)
# ---------------------------------------------------------------------------

def render_side_views(
    pts_clean: np.ndarray,
    poses: List[Dict[str, Any]],
    output_dir: Path,
    floor_y: float,
    ceiling_y: float,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render side-view elevation plots (X-Y and Y-Z) to verify horizontal floor and vertical walls."""
    # Side View 1: X-Y (Elevation looking along Z)
    canvas_xy = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_xy[:] = (22, 20, 28)

    # Side View 2: Y-Z (Elevation looking along X)
    canvas_yz = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_yz[:] = (22, 20, 28)

    pad_h = 100
    pad_v = 100
    dw = width - 2 * pad_h
    dh = height - 2 * pad_v

    font = cv2.FONT_HERSHEY_SIMPLEX

    def render_elevation(canvas: np.ndarray, horiz_idx: int, horiz_name: str, title: str):
        if len(pts_clean) == 0:
            return
        h_vals = pts_clean[:, horiz_idx]
        y_vals = pts_clean[:, 1]
        cam_h = [p["C"][horiz_idx] for p in poses] if poses else [0.0]
        cam_y = [p["C"][1] for p in poses] if poses else [1.5]

        min_h = min(np.min(h_vals), min(cam_h))
        max_h = max(np.max(h_vals), max(cam_h))
        min_y = min(floor_y - 0.3, np.min(y_vals))
        max_y = max(ceiling_y + 0.3, max(cam_y) + 0.3)

        span_h = max(max_h - min_h, 0.5)
        span_y = max(max_y - min_y, 0.5)

        scale = min(dw / span_h, dh / span_y)
        x_off = pad_h + (dw - span_h * scale) / 2.0
        y_off = pad_v + (dh - span_y * scale) / 2.0

        def to_side_px(h_val: float, y_val: float) -> Tuple[int, int]:
            px = int(round(x_off + (h_val - min_h) * scale))
            py = int(round(height - (y_off + (y_val - min_y) * scale)))
            return px, py

        # Points
        sub_pts = pts_clean[::max(1, len(pts_clean) // 30000)]
        for pt in sub_pts:
            px, py = to_side_px(pt[horiz_idx], pt[1])
            if 0 <= px < width and 0 <= py < height:
                canvas[py, px] = (180, 190, 150)

        # Floor line at y = floor_y
        fl_y = to_side_px(0.0, floor_y)[1]
        cv2.line(canvas, (pad_h, fl_y), (width - pad_h, fl_y), (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Floor (y = {floor_y:.2f} m)", (pad_h + 10, fl_y - 10), font, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        # Ceiling line
        cl_y = to_side_px(0.0, ceiling_y)[1]
        cv2.line(canvas, (pad_h, cl_y), (width - pad_h, cl_y), (60, 160, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Ceiling (y = {ceiling_y:.2f} m)", (pad_h + 10, cl_y - 10), font, 0.45, (60, 160, 255), 1, cv2.LINE_AA)

        # Camera trajectory
        cam_pxs = [to_side_px(p["C"][horiz_idx], p["C"][1]) for p in poses]
        for i in range(len(cam_pxs) - 1):
            cv2.line(canvas, cam_pxs[i], cam_pxs[i+1], (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, cam_pxs[i], 3, (255, 200, 0), -1)

        # Header
        cv2.putText(canvas, f"NAMMA SPACE -- {title}", (pad_h, 50), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Horizontal: {horiz_name} axis   |   Vertical: Y axis (Up)   |   Verification: Floor horizontal at y=0, vertical walls",
                    (pad_h, 80), font, 0.45, (170, 175, 190), 1, cv2.LINE_AA)

    render_elevation(canvas_xy, horiz_idx=0, horiz_name="X", title="Side View Elevation (X-Y)")
    out_xy = output_dir / "side_view_xy.png"
    cv2.imwrite(str(out_xy), canvas_xy)
    print(f"  [+] Saved debug elevation plot: {out_xy}")

    render_elevation(canvas_yz, horiz_idx=2, horiz_name="Z", title="Side View Elevation (Y-Z)")
    out_yz = output_dir / "side_view_yz.png"
    cv2.imwrite(str(out_yz), canvas_yz)
    print(f"  [+] Saved debug elevation plot: {out_yz}")


# ---------------------------------------------------------------------------
# Helper: Find Best Model
# ---------------------------------------------------------------------------

def find_best_sparse_model(sparse_root: Path) -> Optional[Path]:
    """Find the sparse model directory with the highest registered image count."""
    if not sparse_root.is_dir():
        return None

    candidates = [d for d in sparse_root.iterdir() if d.is_dir()]
    if not candidates:
        return None

    best_dir = None
    max_cams = -1

    for c in candidates:
        img_bin = c / "images.bin"
        img_txt = c / "images.txt"
        n = 0
        if img_bin.is_file():
            try:
                with open(img_bin, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
            except Exception:
                pass
        elif img_txt.is_file():
            try:
                with open(img_txt, "r", encoding="utf-8") as f:
                    n = len([l for l in f if l.strip() and not l.startswith("#")]) // 2
            except Exception:
                pass
        if n > max_cams:
            max_cams = n
            best_dir = c

    return best_dir


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render COLMAP sparse model as top-view floor-plan PNG."
    )
    parser.add_argument("--scene",        type=str, help="Scene name (data/<scene>/)")
    parser.add_argument("--model-dir",    type=str, help="Direct path to COLMAP sparse model dir")
    parser.add_argument("--output",       type=str, help="Output PNG path (default: data/<scene>/colmap/top_view.png)")
    parser.add_argument("--bin-size",     type=float, default=0.05, help="2D density histogram bin size in units (default: 0.05)")
    parser.add_argument("--max-reproj-error", type=float, default=3.0, help="Reprojection error threshold in px (default: 3.0)")
    parser.add_argument("--min-obs",      type=int, default=3, help="Minimum observations per 3D point (default: 3)")
    parser.add_argument("--height-margin", type=float, default=0.20, help="Margin inside floor and ceiling in metres (default: 0.20)")
    parser.add_argument("--arrow-every",  type=int, default=5, help="Interval for trajectory view arrows (default: 5)")
    parser.add_argument("--no-filter",    action="store_true", help="Skip outlier filtering")
    parser.add_argument("--debug",        action="store_true", help="Save side-view plots (X-Y and Y-Z) for alignment verification")
    parser.add_argument("--scale",        type=float, default=None, help="Explicit scale multiplier")
    args = parser.parse_args()

    # 1. Resolve model directory
    is_aligned = False
    model_dir = None

    if args.model_dir:
        model_dir = Path(args.model_dir)
        is_aligned = "aligned" in model_dir.parts
    elif args.scene:
        aligned_dir = Path("data") / args.scene / "colmap" / "aligned"
        sparse_root = Path("data") / args.scene / "colmap" / "sparse"
        if aligned_dir.is_dir() and (aligned_dir / "images.txt").is_file():
            model_dir = aligned_dir
            is_aligned = True
        elif sparse_root.is_dir():
            model_dir = find_best_sparse_model(sparse_root)
            if not model_dir:
                model_dir = sparse_root / "0"
        else:
            model_dir = Path("data") / args.scene / "colmap" / "sparse" / "0"
    else:
        parser.print_help()
        print("\n[ERROR] Either --scene or --model-dir is required.")
        return 1

    if not model_dir or not model_dir.is_dir():
        print(f"[FAIL] Model directory not found: {model_dir}")
        return 1

    # 2. Output paths
    if args.output:
        out_2d = Path(args.output)
    elif args.scene:
        out_2d = Path("data") / args.scene / "colmap" / "top_view.png"
    else:
        out_2d = model_dir.parent / "top_view.png"

    print("=" * 70)
    print("  NAMMA SPACE -- TOP-VIEW FLOOR PLAN & TRAJECTORY VIEWER")
    print("=" * 70)
    print(f"  Model dir          : {model_dir}")
    print(f"  Output path        : {out_2d}")
    print(f"  Debug side-views   : {args.debug}")
    print(f"  Density bin size   : {args.bin_size} m")
    print("-" * 70)

    # 3. Load cameras, poses, and 3D points
    print("Loading camera poses...")
    poses = parse_images(model_dir)
    print(f"  -> {len(poses)} camera poses loaded.")
    if not poses:
        print("[FAIL] No camera poses found in model directory.")
        return 1

    print("Loading 3D point cloud...")
    pts, cols, errs, tracks = parse_points3d(model_dir)
    print(f"  -> {len(pts):,} points loaded.")

    # 4. Auto-detect up vector & floor plane
    print("\n[Step 1/4] Auto-detecting up vector and floor plane...")
    up_vector, floor_offset, conf_deg = detect_up_vector_and_floor(poses, pts)
    print(f"  -> Detected up vector: [{up_vector[0]:.4f}, {up_vector[1]:.4f}, {up_vector[2]:.4f}]")
    print(f"  -> Camera-up to RANSAC floor agreement: {conf_deg:.2f} deg")

    # 5. Build rotation matrix to align normal with +Y (Metric Y-up convention)
    target_up = np.array([0.0, 1.0, 0.0])
    R_align = rodrigues_align(up_vector, target_up)

    # Rotate 3D points and camera centers
    if len(pts) > 0:
        pts_aligned = (R_align @ pts.T).T
    else:
        pts_aligned = pts.copy()

    # Find floor Y offset after alignment and snap to Y = 0
    if len(pts_aligned) > 0:
        h_aligned = pts_aligned[:, 1]
        # Inliers around detected floor level
        floor_y = float(np.percentile(h_aligned, 2.0))
        pts_aligned[:, 1] -= floor_y
    else:
        floor_y = 0.0

    for p in poses:
        p["C"] = R_align @ p["C"]
        p["C"][1] -= floor_y
        p["up"] = R_align @ p["up"]
        p["fwd"] = R_align @ p["fwd"]

    print(f"  -> Aligned to canonical +Y up axis and snapped floor to y = 0.00 m")

    # 6. Clean Point Cloud
    print("\n[Step 2/4] Cleaning point cloud...")
    if not args.no_filter and len(pts_aligned) > 0:
        pts_clean, cols_clean, floor_y_snap, ceiling_y = clean_point_cloud(
            pts_aligned, cols, errs, tracks, poses,
            max_reproj_error=args.max_reproj_error,
            min_observations=args.min_obs,
            height_margin=args.height_margin,
        )
        print(f"  -> {len(pts_clean):,} clean points remaining ({len(pts_clean)/max(1, len(pts_aligned))*100:.1f}%).")
        print(f"  -> Walls height band: {floor_y_snap + args.height_margin:.2f} m to {ceiling_y - args.height_margin:.2f} m")
    else:
        pts_clean, cols_clean = pts_aligned, cols
        floor_y_snap = 0.0
        ceiling_y = float(np.percentile(pts_aligned[:, 1], 98.0)) if len(pts_aligned) > 0 else 3.0

    # 7. Multi-pass video detection
    split_indices = []
    scene_name = args.scene
    if not scene_name and args.model_dir:
        parts = Path(args.model_dir).parts
        if "data" in parts:
            d_idx = parts.index("data")
            if d_idx + 1 < len(parts):
                scene_name = parts[d_idx + 1]

    if scene_name:
        report_file = Path("data") / scene_name / "frames_report.json"
        if report_file.is_file():
            try:
                with open(report_file, "r", encoding="utf-8") as rf:
                    f_rep = json.load(rf)
                v_list = f_rep.get("videos", [])
                total_kept = sum(v.get("kept_after_duplicate", v.get("raw_extracted", 0)) for v in v_list)
                final_count = f_rep.get("summary", {}).get("final_frame_count", total_kept)
                scale_ratio = final_count / total_kept if total_kept > final_count and total_kept > 0 else 1.0

                acc = 0
                for v in v_list[:-1]:
                    acc += v.get("kept_after_duplicate", v.get("raw_extracted", 0))
                    split_indices.append(int(round(acc * scale_ratio)))
                if split_indices:
                    print(f"  -> Multi-video passes detected with split at frame(s): {split_indices}")
            except Exception:
                pass

    # 8. Scale detection
    scale_label = "1 metre (metric)" if is_aligned else "1 unit (unaligned)"
    if args.scale:
        scale_label = f"1.0 m (scale {args.scale:.2f})"

    # 9. Render 2D Top-View Density Floor Plan
    print("\n[Step 3/4] Rendering 2D density histogram floor plan...")
    render_topview_density(
        pts_clean, poses, out_2d,
        up_vector=up_vector,
        split_indices=split_indices,
        bin_size=args.bin_size,
        is_metric=is_aligned,
        scale_label=scale_label,
        arrow_every=args.arrow_every,
    )

    # 10. Debug Side-Views
    if args.debug:
        print("\n[Step 4/4] Rendering side-view elevation plots (--debug)...")
        render_side_views(pts_clean, poses, out_2d.parent, floor_y=0.0, ceiling_y=ceiling_y)

    print("\n" + "=" * 70)
    print("  [SUCCESS] TOP-VIEW FLOOR PLAN GENERATION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
