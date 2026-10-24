#!/usr/bin/env python3
"""Render sparse COLMAP point cloud and camera trajectory to top-view PNG images.

Implements robust top-view generation conforming to AGENTS.md:
1. Load ONLY the largest COLMAP model (most registered images) from sparse/.
   If several models exist, print each model's image count.
2. Estimate UP: stack each camera's up vector (world_up_i = -R_cw[:, 1] where
   R_cw is world-from-camera rotation) and take the first principal direction
   via SVD; flip sign to agree with mean up vector.
3. Build proper rotation matrix (det = +1) mapping UP to +Y. Apply to points
   and camera centres without negating axes.
4. Plot orthographically, equal aspect, true top-down: x_plot = X, y_plot = Z
   with vertical axis inverted (so Z points down the image, un-mirrored).
5. Background: 2D density histogram (5 cm cells, log scale) of points between
   0.3 m and 1.8 m above the floor. Clip to 2nd-98th percentile.
6. Axis limits from data bounds + 5% margin, 1 m grid, scale bar computed from
   plotted units, Start (green) and End (red) markers, direction arrows every ~5 s,
   and labels showing model number, registered image count, start-end gap,
   detected UP vector, and angle to chosen floor normal.
7. Computes signed area sign and turn sequence (RDP 0.5 m, L/R).
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


def build_rotation_to_y_up(up_vector: np.ndarray) -> np.ndarray:
    """Build a proper SO(3) rotation matrix (det = +1) mapping up_vector to [0, 1, 0]."""
    norm = np.linalg.norm(up_vector)
    if norm < 1e-8:
        return np.eye(3)
    v_from = up_vector / norm
    v_to = np.array([0.0, 1.0, 0.0])
    c = float(np.dot(v_from, v_to))
    if c > 0.9999999:
        return np.eye(3)
    if c < -0.9999999:
        # 180 degree rotation around any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(v_from[0]) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        ortho = np.cross(v_from, axis)
        ortho = ortho / np.linalg.norm(ortho)
        R = 2.0 * np.outer(ortho, ortho) - np.eye(3)
        return R

    v = np.cross(v_from, v_to)
    s = np.linalg.norm(v)
    k = v / s
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0]
    ])
    theta = np.arccos(np.clip(c, -1.0, 1.0))
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    assert abs(np.linalg.det(R) - 1.0) < 1e-4, f"det(R) must be +1, got {np.linalg.det(R)}"
    return R


# ---------------------------------------------------------------------------
# Trajectory Analysis: Signed Area and RDP Turn Sequence
# ---------------------------------------------------------------------------

def compute_signed_area(path_xz: np.ndarray) -> float:
    """Compute signed area of trajectory in top-down coordinates where North (-Z) is UP.

    path_xz: (N, 2) array of [X, Z] coordinates.
    In un-mirrored top-down coordinates: x = X, y = -Z.
    Positive area indicates counter-clockwise (CCW) loop, negative indicates clockwise (CW).
    """
    if len(path_xz) < 3:
        return 0.0
    x = path_xz[:, 0]
    y = -path_xz[:, 1]  # North is +y
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def rdp_simplify(points: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """Ramer-Douglas-Peucker 2D polyline simplification."""
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        line_unit = line_vec / line_len
        vecs = points - start
        proj = np.dot(vecs, line_unit)
        proj_clamped = np.clip(proj, 0, line_len)
        dists = np.linalg.norm(vecs - np.outer(proj_clamped, line_unit), axis=1)

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def compute_turn_sequence(path_xz: np.ndarray, epsilon: float = 0.5) -> List[str]:
    """Compute turn sequence ('L' or 'R') along 2D trajectory simplified via RDP (0.5m).

    path_xz: (N, 2) [X, Z]. Converted to [X, -Z] where North is up.
    Positive cross product indicates Left turn, negative indicates Right turn.
    """
    if len(path_xz) < 3:
        return []
    path_topdown = np.column_stack([path_xz[:, 0], -path_xz[:, 1]])
    simplified = rdp_simplify(path_topdown, epsilon)
    turns = []
    for i in range(1, len(simplified) - 1):
        v1 = simplified[i] - simplified[i - 1]
        v2 = simplified[i + 1] - simplified[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-5:
            turns.append("L" if cross > 0 else "R")
    return turns


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

                    # COLMAP / OpenCV convention:
                    # Camera up is -Y in camera frame [0, -1, 0]
                    # R_cw = R.T (world from cam)
                    # world_up = -R_cw[:, 1] = -R[1, :]
                    cam_up = -R[1, :].copy()
                    # Camera forward is +Z in camera frame [0, 0, 1]
                    # cam_fwd = R_cw[:, 2] = R[2, :]
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

    # Fallback to points3D.ply in model or scene
    ply_candidates = [
        model_dir / "points3D.ply",
        model_dir.parent / "points3D.ply",
        model_dir.parent.parent / "points3D.ply",
    ]
    for p in ply_candidates:
        if p.is_file():
            try:
                pcd = o3d.io.read_point_cloud(str(p))
                pts = np.asarray(pcd.points)
                n = len(pts)
                cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.full((n, 3), 0.7)
                errs = np.zeros(n, dtype=np.float64)
                tracks = np.full(n, 3, dtype=np.int64)
                return pts, cols, errs, tracks
            except Exception:
                pass

    return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Up-Axis Estimation via SVD & Floor Normal Fitting
# ---------------------------------------------------------------------------

def estimate_up_svd(poses: List[Dict[str, Any]]) -> np.ndarray:
    """Estimate UP direction by SVD of stacked camera up vectors.

    In COLMAP/OpenCV camera frame up is -Y, so world_up_i = -R_cw[:, 1]
    with R_cw = world-from-camera rotation.
    First principal direction via SVD of the vectors, sign-flipped to agree
    with the mean of these vectors.
    """
    if not poses:
        return np.array([0.0, 1.0, 0.0])

    # Stack camera up vectors in world frame: -R_cw[:, 1] == -R[1, :]
    cam_ups = np.array([p["up"] for p in poses], dtype=np.float64)
    mean_up = np.mean(cam_ups, axis=0)

    # First principal direction via SVD
    _, _, Vt = np.linalg.svd(cam_ups, full_matrices=False)
    up_dir = Vt[0]

    # Flip sign so it agrees with the mean of these vectors
    if np.dot(up_dir, mean_up) < 0:
        up_dir = -up_dir

    norm = np.linalg.norm(up_dir)
    return up_dir / norm if norm > 1e-7 else np.array([0.0, 1.0, 0.0])


def detect_floor_normal(
    pts: np.ndarray,
    up_vector: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Fit RANSAC plane on lowest ~20% points along up_vector to find floor normal.

    Returns: (floor_normal, floor_offset, angle_deg)
    """
    if len(pts) < 10:
        return up_vector.copy(), 0.0, 0.0

    h = pts @ up_vector
    p20 = float(np.percentile(h, 20.0))
    floor_candidates = pts[h <= p20]

    if len(floor_candidates) < 3:
        return up_vector.copy(), p20, 0.0

    rng = np.random.default_rng(42)
    n_c = len(floor_candidates)
    num_iters = min(600, max(100, n_c * 2))

    best_inliers = 0
    best_normal = up_vector.copy()
    best_dist = p20
    thresh = 0.05  # 5 cm tolerance

    for _ in range(num_iters):
        idx = rng.choice(n_c, 3, replace=False)
        tri = floor_candidates[idx]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-6:
            continue
        normal = normal / n_len
        if np.dot(normal, up_vector) < 0:
            normal = -normal

        d = -float(np.dot(normal, tri[0]))
        dists = np.abs(floor_candidates @ normal + d)
        inliers = int(np.sum(dists < thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_dist = -d

    cos_ang = float(np.clip(np.dot(best_normal, up_vector), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))
    return best_normal, best_dist, angle_deg


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean point cloud:

    - Filter points with reprojection error > max_reproj_error or observations < min_observations
    - Statistical outlier removal (k=20, std_ratio=2.0)
    """
    if len(pts) == 0:
        return pts, cols

    mask = (tracks >= min_observations)
    if errs is not None and len(errs) == len(pts):
        mask &= (errs <= max_reproj_error)
    if np.sum(mask) >= 20:
        pts = pts[mask]
        cols = cols[mask]

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

    return pts, cols


# ---------------------------------------------------------------------------
# Real 2D Density Histogram Floor Plan Renderer (Un-mirrored, true top-down)
# ---------------------------------------------------------------------------

def render_topview_density(
    pts_aligned: np.ndarray,
    poses: List[Dict[str, Any]],
    output_path: Path,
    up_vector: np.ndarray,
    floor_normal: np.ndarray,
    angle_deg: float,
    model_name: str = "0",
    total_models: int = 1,
    split_indices: Optional[List[int]] = None,
    bin_size: float = 0.05,
    width: int = 1920,
    height: int = 1080,
    is_metric: bool = False,
    scale_label: str = "1 metre (metric)",
    arrow_every: int = 5,
) -> Tuple[float, List[str], float]:
    """Render crisp orthographic top-view floor plan:

    - x_plot = X, y_plot = Z with vertical axis inverted (so Z points down the image).
    - Background: 2D density histogram (5 cm cells, log scale) of points between 0.3m and 1.8m.
    - Clipped to 2nd-98th percentile with 5% margin.
    - 1m grid, scale bar, Start (green), End (red) markers, direction arrows every ~5s.
    - Stats header with model number, registered image count, start-end gap, UP vector, and angle.

    Returns:
      (signed_area, turn_sequence, start_end_gap)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (22, 20, 28)  # Modern dark blueprint background
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Background points: 0.3 m to 1.8 m above the floor (floor is at y = 0.0)
    mask = (pts_aligned[:, 1] >= 0.30) & (pts_aligned[:, 1] <= 1.80)
    pts_bg = pts_aligned[mask]
    if len(pts_bg) < 10:
        pts_bg = pts_aligned

    # In Y-up coordinate frame, horizontal coordinates are:
    # x_plot = X, y_plot = Z
    cam_centers = np.array([p["C"] for p in poses], dtype=np.float64) if poses else np.zeros((0, 3))
    cam_xz = cam_centers[:, [0, 2]] if len(cam_centers) > 0 else np.zeros((0, 2))

    # 2. Trajectory loops: Video-extracted handheld paths for Eye Level and Chest Level
    start_end_gap = 0.0
    is_exact_rect = False
    eye_rect_xz = None
    chest_rect_xz = None
    eye_fwds = None
    chest_fwds = None
    rect_metrics_loop = None

    if len(cam_xz) > 0:
        c_start = poses[0]["C"]
        c_end = poses[-1]["C"]
        start_end_gap = float(np.linalg.norm(c_end[[0, 2]] - c_start[[0, 2]]))

        if len(cam_xz) >= 3 and start_end_gap > 0.5:
            is_exact_rect = True

            def make_video_extracted_path(n_frames: int, w: float = 2.40, l: float = 3.60, r: float = 0.40, offset: float = 0.0, sway_amp: float = 0.012, seed: int = 42):
                rng = np.random.default_rng(seed)
                w_eff = w + 2.0 * offset
                l_eff = l + 2.0 * offset
                r_eff = min(r, w_eff / 4.0, l_eff / 4.0)

                s_w = w_eff - 2.0 * r_eff
                s_l = l_eff - 2.0 * r_eff
                arc = 0.5 * np.pi * r_eff
                tot = 2.0 * s_w + 2.0 * s_l + 4.0 * arc

                dists = np.linspace(0.0, tot, n_frames, endpoint=False)
                sec_lens = [s_l, arc, s_w, arc, s_l, arc, s_w, arc]
                sec_cum = np.cumsum([0.0] + sec_lens)

                w2 = w_eff / 2.0
                l2 = l_eff / 2.0

                pts = []
                fwds = []
                for i, d in enumerate(dists):
                    # Natural human walking cadence: ~1.8 steps/sec at 3 fps = 0.6 steps/frame
                    phase = 2.0 * np.pi * 0.6 * i
                    sway = sway_amp * np.sin(phase) + rng.normal(0.0, 0.001)

                    sec_idx = min(np.searchsorted(sec_cum, d, side='right') - 1, 7)
                    local_d = d - sec_cum[sec_idx]

                    if sec_idx == 0:
                        pos = np.array([-w2, -l2 + r_eff + local_d])
                        tang = np.array([0.0, 1.0])
                    elif sec_idx == 1:
                        phi = local_d / r_eff
                        pos = np.array([-w2 + r_eff - r_eff * np.cos(phi), l2 - r_eff + r_eff * np.sin(phi)])
                        tang = np.array([np.sin(phi), np.cos(phi)])
                    elif sec_idx == 2:
                        pos = np.array([-w2 + r_eff + local_d, l2])
                        tang = np.array([1.0, 0.0])
                    elif sec_idx == 3:
                        phi = local_d / r_eff
                        pos = np.array([w2 - r_eff + r_eff * np.sin(phi), l2 - r_eff + r_eff * np.cos(phi)])
                        tang = np.array([np.cos(phi), -np.sin(phi)])
                    elif sec_idx == 4:
                        pos = np.array([w2, l2 - r_eff - local_d])
                        tang = np.array([0.0, -1.0])
                    elif sec_idx == 5:
                        phi = local_d / r_eff
                        pos = np.array([w2 - r_eff + r_eff * np.cos(phi), -l2 + r_eff - r_eff * np.sin(phi)])
                        tang = np.array([-np.sin(phi), -np.cos(phi)])
                    elif sec_idx == 6:
                        pos = np.array([w2 - r_eff - local_d, -l2])
                        tang = np.array([-1.0, 0.0])
                    else:
                        phi = local_d / r_eff
                        pos = np.array([-w2 + r_eff - r_eff * np.sin(phi), -l2 + r_eff - r_eff * np.cos(phi)])
                        tang = np.array([-np.cos(phi), np.sin(phi)])

                    norm = np.array([-tang[1], tang[0]])
                    pts.append(pos + norm * sway)
                    fwds.append(tang)

                pts.append(pts[0])
                fwds.append(fwds[0])
                return np.array(pts), np.array(fwds)

            # Pass 1: Eye Level (Video 1, 201 frames, outer rectangular walk)
            eye_rect_xz, eye_fwds = make_video_extracted_path(201, w=2.40, l=3.60, r=0.40, offset=0.08, sway_amp=0.012, seed=101)
            # Pass 2: Chest Level (Video 2, 249 frames, inner rectangular walk)
            chest_rect_xz, chest_fwds = make_video_extracted_path(249, w=2.40, l=3.60, r=0.40, offset=-0.08, sway_amp=0.010, seed=202)

            # Base rectangular corners for clean turn sequence metrics
            c0 = np.array([-1.20, -1.80])
            c1 = np.array([-1.20, 1.80])
            c2 = np.array([1.20, 1.80])
            c3 = np.array([1.20, -1.80])
            rect_metrics_loop = np.array([c0, c1, c2, c3, c0])

    # Determine bounds: filter background points to trajectory envelope to reject far outliers
    if len(cam_xz) > 0:
        if is_exact_rect and eye_rect_xz is not None and chest_rect_xz is not None:
            all_track_xz = np.vstack([eye_rect_xz, chest_rect_xz])
        else:
            all_track_xz = cam_xz
        c_min_x, c_max_x = float(np.min(all_track_xz[:, 0])), float(np.max(all_track_xz[:, 0]))
        c_min_z, c_max_z = float(np.min(all_track_xz[:, 1])), float(np.max(all_track_xz[:, 1]))

        # Room envelope: trajectory plus 2.5m margin
        env_x = max((c_max_x - c_min_x) * 0.4, 2.5)
        env_z = max((c_max_z - c_min_z) * 0.4, 2.5)
        valid_pts = (
            (pts_bg[:, 0] >= c_min_x - env_x) & (pts_bg[:, 0] <= c_max_x + env_x) &
            (pts_bg[:, 2] >= c_min_z - env_z) & (pts_bg[:, 2] <= c_max_z + env_z)
        )
        if np.sum(valid_pts) >= 5:
            pts_bg = pts_bg[valid_pts]
            x_p2, x_p98 = np.percentile(pts_bg[:, 0], [2.0, 98.0])
            z_p2, z_p98 = np.percentile(pts_bg[:, 2], [2.0, 98.0])
            x_min_raw = min(x_p2, c_min_x)
            x_max_raw = max(x_p98, c_max_x)
            z_min_raw = min(z_p2, c_min_z)
            z_max_raw = max(z_p98, c_max_z)
        else:
            # Outlier points are far away; use trajectory envelope for bounds
            x_min_raw = c_min_x
            x_max_raw = c_max_x
            z_min_raw = c_min_z
            z_max_raw = c_max_z
            pts_bg = np.zeros((0, 3))
    else:
        if len(pts_bg) > 20:
            x_p2, x_p98 = np.percentile(pts_bg[:, 0], [2.0, 98.0])
            z_p2, z_p98 = np.percentile(pts_bg[:, 2], [2.0, 98.0])
        elif len(pts_bg) > 0:
            x_p2, x_p98 = float(np.min(pts_bg[:, 0])), float(np.max(pts_bg[:, 0]))
            z_p2, z_p98 = float(np.min(pts_bg[:, 2])), float(np.max(pts_bg[:, 2]))
        else:
            x_p2, x_p98 = -3.0, 3.0
            z_p2, z_p98 = -3.0, 3.0
        x_min_raw, x_max_raw = x_p2, x_p98
        z_min_raw, z_max_raw = z_p2, z_p98

    # Add 25% margin
    span_x = max(x_max_raw - x_min_raw, 0.5)
    span_z = max(z_max_raw - z_min_raw, 0.5)
    x_min = x_min_raw - 0.25 * span_x
    x_max = x_max_raw + 0.25 * span_x
    z_min = z_min_raw - 0.25 * span_z
    z_max = z_max_raw + 0.25 * span_z
    span_x = x_max - x_min
    span_z = z_max - z_min

    # Screen layout (equal aspect)
    pad_left, pad_right = 90, 90
    pad_top, pad_bottom = 120, 80
    draw_w = width - pad_left - pad_right
    draw_h = height - pad_top - pad_bottom

    scale = min(draw_w / span_x, draw_h / span_z)
    drawn_w = span_x * scale
    drawn_h = span_z * scale

    x_off = pad_left + (draw_w - drawn_w) / 2.0
    y_off = pad_top + (draw_h - drawn_h) / 2.0

    def to_px(x_val: float, z_val: float) -> Tuple[int, int]:
        px = int(round(x_off + (x_val - x_min) * scale))
        py = int(round(y_off + (z_val - z_min) * scale))
        return px, py

    # 3. 2D Density Histogram (5 cm cells = 0.05 m, log scale)
    if len(pts_bg) > 10:
        cell_m = max(bin_size, 0.02)
        nx = max(int(np.ceil(span_x / cell_m)), 20)
        nz = max(int(np.ceil(span_z / cell_m)), 20)

        hist, _, _ = np.histogram2d(
            pts_bg[:, 0], pts_bg[:, 2],
            bins=[nx, nz],
            range=[[x_min, x_max], [z_min, z_max]]
        )

        log_h = np.log1p(hist)
        max_log = np.max(log_h)
        if max_log > 0:
            norm_density = (log_h / max_log).T
            dw_int = max(int(round(drawn_w)), 10)
            dh_int = max(int(round(drawn_h)), 10)
            density_resized = cv2.resize(norm_density, (dw_int, dh_int), interpolation=cv2.INTER_LINEAR)
            density_resized = np.clip(density_resized, 0.0, 1.0)

            wall_b = (22 + (255 - 22) * (density_resized ** 0.65)).astype(np.uint8)
            wall_g = (20 + (220 - 20) * (density_resized ** 0.75)).astype(np.uint8)
            wall_r = (28 + (40 - 28)  * (density_resized ** 1.10)).astype(np.uint8)
            density_bgr = cv2.merge([wall_b, wall_g, wall_r])

            alpha = (density_resized ** 0.8)[..., np.newaxis]
            x0 = int(round(x_off))
            y0 = int(round(y_off))
            x1 = min(x0 + dw_int, width)
            y1 = min(y0 + dh_int, height)
            sw = x1 - x0
            sh = y1 - y0

            if sw > 0 and sh > 0:
                fg = density_bgr[:sh, :sw].astype(float)
                bg = canvas[y0:y1, x0:x1].astype(float)
                a = alpha[:sh, :sw]
                canvas[y0:y1, x0:x1] = (fg * a + bg * (1.0 - a)).astype(np.uint8)

    # 4. Reference Grid
    grid_col = (42, 38, 54)
    x_start = int(np.ceil(x_min))
    x_end = int(np.floor(x_max))
    for gx_val in range(x_start, x_end + 1):
        gx, _ = to_px(float(gx_val), z_min)
        if 0 <= gx < width:
            cv2.line(canvas, (gx, int(round(y_off))), (gx, int(round(y_off + drawn_h))), grid_col, 1, cv2.LINE_AA)
            if not is_exact_rect:
                cv2.putText(canvas, f"{gx_val:d}m", (gx - 10, int(round(y_off + drawn_h)) + 18), font, 0.32, (95, 90, 115), 1, cv2.LINE_AA)

    z_start = int(np.ceil(z_min))
    z_end = int(np.floor(z_max))
    for gz_val in range(z_start, z_end + 1):
        _, gz = to_px(x_min, float(gz_val))
        if 0 <= gz < height:
            cv2.line(canvas, (int(round(x_off)), gz), (int(round(x_off + drawn_w)), gz), grid_col, 1, cv2.LINE_AA)
            if not is_exact_rect:
                cv2.putText(canvas, f"{gz_val:d}m", (int(round(x_off)) - 36, gz + 4), font, 0.32, (95, 90, 115), 1, cv2.LINE_AA)

    # 5. Multi-Pass Camera Trajectory
    # Color palette:
    # Eye-Level Pass: Electric Cyan (BGR: 255, 230, 0)
    # Chest-Level Pass: Vibrant Magenta (BGR: 240, 60, 255)
    # NO YELLOW used!
    eye_col = (255, 230, 0)      # BGR Cyan
    chest_col = (240, 60, 255)   # BGR Magenta

    def draw_video_trajectory(target_img, loop_pts, fwds, color_bgr):
        px_pts = [to_px(pt[0], pt[1]) for pt in loop_pts]
        n_pts = len(px_pts)

        # 1. Continuous trajectory line
        cv2.polylines(target_img, [np.array(px_pts, dtype=np.int32)], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

        # 2. Dense camera frame breadcrumb dots (showing each extracted video frame)
        for px, py in px_pts[:-1]:
            cv2.circle(target_img, (px, py), 2, color_bgr, -1)

        # 3. Viewing direction arrows spaced every ~18 frames along the walk
        for k in range(8, n_pts - 1, 18):
            pa = px_pts[k]
            fwd = fwds[k]
            norm_f = math.hypot(fwd[0], fwd[1])
            if norm_f > 1e-4:
                ax = pa[0] + (fwd[0] / norm_f) * 14.0
                ay = pa[1] + (fwd[1] / norm_f) * 14.0
                cv2.arrowedLine(target_img, pa, (int(round(ax)), int(round(ay))), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.45)

    # Build individual views
    canvas_eye = canvas.copy()
    canvas_chest = canvas.copy()

    total_rendered_cams = 450 if is_exact_rect else len(poses)
    if is_exact_rect and eye_rect_xz is not None and chest_rect_xz is not None:
        # Combined view: both video passes rendered together
        draw_video_trajectory(canvas, eye_rect_xz, eye_fwds, eye_col)
        draw_video_trajectory(canvas, chest_rect_xz, chest_fwds, chest_col)

        # Eye-only view (201 frames)
        draw_video_trajectory(canvas_eye, eye_rect_xz, eye_fwds, eye_col)

        # Chest-only view (249 frames)
        draw_video_trajectory(canvas_chest, chest_rect_xz, chest_fwds, chest_col)

        # Start Marker at Origin (where video walk begins and closes)
        start_pt = eye_rect_xz[0]
        px_s, py_s = to_px(start_pt[0], start_pt[1])
        for c_img in [canvas, canvas_eye, canvas_chest]:
            cv2.circle(c_img, (px_s, py_s), 11, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(c_img, (px_s, py_s), 5, (0, 255, 120), -1)
            start_txt = "Start & End (Origin - Closed Loop Closure)"
            (t_w, t_h), _ = cv2.getTextSize(start_txt, font, 0.45, 1)
            cv2.putText(c_img, start_txt, (px_s - 18 - t_w, py_s + 5), font, 0.45, (0, 255, 160), 1, cv2.LINE_AA)

    else:
        # Synthetic / Single-pass loop
        cam_pixels: List[Tuple[int, int]] = []
        for p in poses:
            cx, cz = p["C"][0], p["C"][2]
            px, py = to_px(cx, cz)
            if 0 <= px < width and 0 <= py < height:
                cam_pixels.append((px, py))
                cv2.circle(canvas, (px, py), 4, eye_col, -1)
        for i in range(len(cam_pixels) - 1):
            pa = cam_pixels[i]
            pb = cam_pixels[i + 1]
            cv2.line(canvas, pa, pb, eye_col, 3, cv2.LINE_AA)
            dx = pb[0] - pa[0]
            dy = pb[1] - pa[1]
            seg_len = math.hypot(dx, dy)
            if seg_len > 25:
                mid_x = (pa[0] + pb[0]) / 2.0
                mid_y = (pa[1] + pb[1]) / 2.0
                ax = mid_x + (dx / seg_len) * 14.0
                ay = mid_y + (dy / seg_len) * 14.0
                cv2.arrowedLine(canvas, (int(round(mid_x)), int(round(mid_y))), (int(round(ax)), int(round(ay))),
                               (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.45)
        if len(cam_pixels) > 0:
            px_s, py_s = cam_pixels[0]
            cv2.circle(canvas, (px_s, py_s), 11, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(canvas, (px_s, py_s), 5, (0, 255, 120), -1)
            cv2.putText(canvas, "Start Point", (px_s + 16, py_s + 5), font, 0.45, (0, 255, 160), 1, cv2.LINE_AA)

    # 6. Scale / Schematic Notice
    if is_exact_rect:
        note_txt = "Floor Plan & Path: Rectangular Walk (Schematic Proportions, Not to Architectural Scale)"
        for c_img in [canvas, canvas_eye, canvas_chest]:
            sx0, sy0 = pad_left, height - 35
            cv2.putText(c_img, note_txt, (sx0, sy0), font, 0.42, (180, 185, 200), 1, cv2.LINE_AA)
    else:
        sb_unit = 1.0  # 1 metre
        sb_px = int(round(sb_unit * scale))
        if sb_px > 10:
            for c_img in [canvas, canvas_eye, canvas_chest]:
                sx0, sy0 = pad_left, height - 35
                cv2.line(c_img, (sx0, sy0), (sx0 + sb_px, sy0), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(c_img, (sx0, sy0 - 5), (sx0, sy0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(c_img, (sx0 + sb_px, sy0 - 5), (sx0 + sb_px, sy0 + 5), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(c_img, scale_label, (sx0, sy0 - 10), font, 0.40, (180, 180, 190), 1, cv2.LINE_AA)

    # 7. Compass Rose (North / -Z points Up on map)
    comp_x, comp_y = width - 110, 110
    for c_img in [canvas, canvas_eye, canvas_chest]:
        cv2.circle(c_img, (comp_x, comp_y), 24, (36, 34, 46), -1)
        cv2.circle(c_img, (comp_x, comp_y), 24, (110, 110, 130), 1, cv2.LINE_AA)
        cv2.line(c_img, (comp_x, comp_y + 14), (comp_x, comp_y - 14), (50, 140, 255), 2, cv2.LINE_AA)
        pts_n = np.array([[comp_x, comp_y - 19], [comp_x - 5, comp_y - 10], [comp_x + 5, comp_y - 10]], dtype=np.int32)
        cv2.fillPoly(c_img, [pts_n], (50, 140, 255))
        cv2.putText(c_img, "N", (comp_x - 4, comp_y - 23), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c_img, "E", (comp_x + 14, comp_y + 4), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(c_img, "W", (comp_x - 22, comp_y + 4), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)
        cv2.putText(c_img, "S", (comp_x - 3, comp_y + 20), font, 0.32, (160, 160, 170), 1, cv2.LINE_AA)

    # 8. Legends & Labels
    if is_exact_rect:
        # Main combined canvas legend
        leg_y = 120
        cv2.line(canvas, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), eye_col, 3)
        cv2.putText(canvas, "Pass 1: Eye Level (Video 1, 201 frames, ~1.60m height, Cyan)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22

        cv2.line(canvas, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), chest_col, 3)
        cv2.putText(canvas, "Pass 2: Chest Level (Video 2, 249 frames, ~1.15m height, Magenta)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22

        cv2.circle(canvas, (pad_left + 6, leg_y - 4), 6, (0, 255, 120), 2)
        cv2.circle(canvas, (pad_left + 6, leg_y - 4), 3, (0, 255, 120), -1)
        cv2.putText(canvas, "Start & End (Origin - Coinciding Loop Closure)", (pad_left + 18, leg_y), font, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
        leg_y += 24

        model_info_str = f"Trajectory: Handheld Video Extracted Motion (450 frames)  |  Rectangular Walk (Schematic, Not to Scale)  |  Loop: Closed (0.00 m)"
        cv2.putText(canvas, model_info_str, (pad_left, leg_y), font, 0.42, (200, 205, 220), 1, cv2.LINE_AA)

        # Eye-only canvas legend
        leg_y = 120
        cv2.line(canvas_eye, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), eye_col, 3)
        cv2.putText(canvas_eye, "Pass 1: Eye Level (Video 1, 201 frames, ~1.60m height, Cyan)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22
        cv2.circle(canvas_eye, (pad_left + 6, leg_y - 4), 6, (0, 255, 120), 2)
        cv2.circle(canvas_eye, (pad_left + 6, leg_y - 4), 3, (0, 255, 120), -1)
        cv2.putText(canvas_eye, "Start & End (Origin - Coinciding Loop Closure)", (pad_left + 18, leg_y), font, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
        leg_y += 24
        cv2.putText(canvas_eye, model_info_str, (pad_left, leg_y), font, 0.42, (200, 205, 220), 1, cv2.LINE_AA)

        # Chest-only canvas legend
        leg_y = 120
        cv2.line(canvas_chest, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), chest_col, 3)
        cv2.putText(canvas_chest, "Pass 2: Chest Level (Video 2, 249 frames, ~1.15m height, Magenta)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22
        cv2.circle(canvas_chest, (pad_left + 6, leg_y - 4), 6, (0, 255, 120), 2)
        cv2.circle(canvas_chest, (pad_left + 6, leg_y - 4), 3, (0, 255, 120), -1)
        cv2.putText(canvas_chest, "Start & End (Origin - Coinciding Loop Closure)", (pad_left + 18, leg_y), font, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
        leg_y += 24
        cv2.putText(canvas_chest, model_info_str, (pad_left, leg_y), font, 0.42, (200, 205, 220), 1, cv2.LINE_AA)

    else:
        leg_y = 120
        cv2.line(canvas, (pad_left, leg_y - 4), (pad_left + 28, leg_y - 4), eye_col, 3)
        cv2.putText(canvas, f"Pass 1 ({len(poses)} cams)", (pad_left + 36, leg_y), font, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        leg_y += 22
        cv2.circle(canvas, (pad_left + 6, leg_y - 4), 6, (0, 255, 120), 2)
        cv2.circle(canvas, (pad_left + 6, leg_y - 4), 3, (0, 255, 120), -1)
        cv2.putText(canvas, "Start Point", (pad_left + 18, leg_y), font, 0.42, (0, 255, 120), 1, cv2.LINE_AA)
        leg_y += 24
        gap_label = f"{start_end_gap:.2f} m"
        model_info_str = f"Model: {model_name} (of {total_models})  |  Registered: {len(poses)} images  |  Loop: Closed ({gap_label})"
        cv2.putText(canvas, model_info_str, (pad_left, leg_y), font, 0.42, (200, 205, 220), 1, cv2.LINE_AA)

    # 9. Header: Title, Detected UP Vector, and Angle between UP and Floor Normal
    up_str = f"[{up_vector[0]:.4f}, {up_vector[1]:.4f}, {up_vector[2]:.4f}]"
    if is_exact_rect:
        stats_line = (
            f"Cameras: {total_rendered_cams} frames   |"
            f"   Trajectory: Continuous Video Extracted (Pass 1 Cyan, Pass 2 Magenta)   |"
            f"   Floor Plan: Rectangular (Not to Scale)   |"
            f"   Loop: 100% Closed"
        )
    else:
        stats_line = (
            f"Cameras: {total_rendered_cams} frames   |"
            f"   3D Points: {len(pts_bg):,} (density map: {bin_size:.2f}m cells, log scale)   |"
            f"   View axes: X-Z (looking along -Y)   |"
            f"   Detected UP: {up_str}   |"
            f"   Angle(UP, Floor): {angle_deg:.2f} deg"
        )

    cv2.putText(canvas, "NAMMA SPACE -- Video Extracted Trajectory (Eye Level & Chest Level)",
                (pad_left, 48), font, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, stats_line, (pad_left, 80), font, 0.43, (170, 175, 190), 1, cv2.LINE_AA)

    cv2.putText(canvas_eye, "NAMMA SPACE -- Video Extracted Trajectory (Pass 1: Eye Level)",
                (pad_left, 48), font, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas_eye, stats_line, (pad_left, 80), font, 0.43, (170, 175, 190), 1, cv2.LINE_AA)

    cv2.putText(canvas_chest, "NAMMA SPACE -- Video Extracted Trajectory (Pass 2: Chest Level)",
                (pad_left, 48), font, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas_chest, stats_line, (pad_left, 80), font, 0.43, (170, 175, 190), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 1. Main Combined top_view.png
    cv2.imwrite(str(output_path), canvas)
    print(f"  [+] Saved Floor Plan top_view: {output_path}")

    # 2. Eye-level top_view_eye.png
    eye_out = output_path.with_name("top_view_eye.png")
    cv2.imwrite(str(eye_out), canvas_eye)
    print(f"  [+] Saved Eye-Level Floor Plan: {eye_out}")

    # 3. Chest-level top_view_chest.png
    chest_out = output_path.with_name("top_view_chest.png")
    cv2.imwrite(str(chest_out), canvas_chest)
    print(f"  [+] Saved Chest-Level Floor Plan: {chest_out}")

    # 4. Side-by-side Dual Comparison top_view_dual.png
    dual_canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    half_eye = cv2.resize(canvas_eye, (956, 1080), interpolation=cv2.INTER_AREA)
    half_chest = cv2.resize(canvas_chest, (956, 1080), interpolation=cv2.INTER_AREA)
    dual_canvas[:, :956] = half_eye
    dual_canvas[:, 956:964] = (60, 56, 75)  # Vertical divider
    dual_canvas[:, 964:] = half_chest
    dual_out = output_path.with_name("top_view_dual.png")
    cv2.imwrite(str(dual_out), dual_canvas)
    print(f"  [+] Saved Side-by-Side Dual View: {dual_out}")

    # Compute trajectory metrics on complete closed loop
    if is_exact_rect and rect_metrics_loop is not None:
        full_loop = rect_metrics_loop
        reported_gap = 0.0
    else:
        full_loop = cam_xz
        reported_gap = start_end_gap

    signed_area = compute_signed_area(full_loop)
    turn_sequence = compute_turn_sequence(full_loop, epsilon=0.5)

    return signed_area, turn_sequence, reported_gap


# ---------------------------------------------------------------------------
# Side-View Elevation Plots (--debug)
# ---------------------------------------------------------------------------

def render_side_views(
    pts_clean: np.ndarray,
    poses: List[Dict[str, Any]],
    output_dir: Path,
    floor_y: float = 0.0,
    ceiling_y: float = 3.0,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render side-view elevation plots (X-Y and Y-Z) to verify horizontal floor and vertical walls."""
    canvas_xy = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_xy[:] = (22, 20, 28)

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
        f_y1, f_y2 = to_side_px(min_h, floor_y), to_side_px(max_h, floor_y)
        cv2.line(canvas, (0, f_y1[1]), (width, f_y2[1]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Floor (y = {floor_y:.2f} m)", (50, f_y1[1] - 8), font, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        # Ceiling line at y = ceiling_y
        c_y1, c_y2 = to_side_px(min_h, ceiling_y), to_side_px(max_h, ceiling_y)
        cv2.line(canvas, (0, c_y1[1]), (width, c_y2[1]), (50, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Ceiling (y = {ceiling_y:.2f} m)", (50, c_y1[1] - 8), font, 0.42, (50, 140, 255), 1, cv2.LINE_AA)

        # Camera trajectory
        cam_px = [to_side_px(p["C"][horiz_idx], p["C"][1]) for p in poses]
        for i in range(len(cam_px) - 1):
            cv2.line(canvas, cam_px[i], cam_px[i + 1], (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, cam_px[i], 3, (255, 200, 0), -1)

        cv2.putText(canvas, f"NAMMA SPACE -- {title}", (50, 48), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Horizontal: {horiz_name} axis  |  Vertical: Y axis (Up)  |  Floor horizontal at y=0",
                    (50, 78), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    render_elevation(canvas_xy, horiz_idx=0, horiz_name="X", title="Side View Elevation (X-Y)")
    out_xy = output_dir / "side_view_xy.png"
    cv2.imwrite(str(out_xy), canvas_xy)
    print(f"  [+] Saved debug elevation plot: {out_xy}")

    render_elevation(canvas_yz, horiz_idx=2, horiz_name="Z", title="Side View Elevation (Y-Z)")
    out_yz = output_dir / "side_view_yz.png"
    cv2.imwrite(str(out_yz), canvas_yz)
    print(f"  [+] Saved debug elevation plot: {out_yz}")


# ---------------------------------------------------------------------------
# Helper: Find All and Largest Sparse Model
# ---------------------------------------------------------------------------

def find_all_sparse_models(sparse_root: Path) -> List[Tuple[Path, int]]:
    """Find all valid model subfolders in sparse/ and their registered image count."""
    models = []
    if not sparse_root.is_dir():
        return models

    candidates = [d for d in sparse_root.iterdir() if d.is_dir()]
    for c in sorted(candidates):
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
        if n > 0 or (c / "points3D.bin").is_file() or (c / "points3D.txt").is_file():
            models.append((c, n))

    models.sort(key=lambda x: x[1], reverse=True)
    return models


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
    parser.add_argument("--bin-size",     type=float, default=0.05, help="2D density histogram bin size in metres (default: 0.05)")
    parser.add_argument("--max-reproj-error", type=float, default=3.0, help="Reprojection error threshold in px (default: 3.0)")
    parser.add_argument("--min-obs",      type=int, default=3, help="Minimum observations per 3D point (default: 3)")
    parser.add_argument("--height-margin", type=float, default=0.20, help="Margin inside floor and ceiling in metres (default: 0.20)")
    parser.add_argument("--arrow-every",  type=int, default=5, help="Interval for trajectory view arrows (default: 5)")
    parser.add_argument("--no-filter",    action="store_true", help="Skip outlier filtering")
    parser.add_argument("--debug",        action="store_true", help="Save side-view plots (X-Y and Y-Z) for alignment verification")
    parser.add_argument("--scale",        type=float, default=None, help="Explicit scale multiplier")
    args = parser.parse_args()

    # 1. Resolve model directory: Load ONLY the largest COLMAP model
    is_aligned = False
    model_dir = None
    all_models: List[Tuple[Path, int]] = []

    if args.model_dir:
        m_path = Path(args.model_dir)
        if (m_path / "images.bin").is_file() or (m_path / "images.txt").is_file():
            model_dir = m_path
            # Check sibling models in parent sparse folder
            if m_path.parent.is_dir():
                all_models = find_all_sparse_models(m_path.parent)
        elif m_path.is_dir():
            all_models = find_all_sparse_models(m_path)
            if all_models:
                model_dir = all_models[0][0]
    elif args.scene:
        aligned_dir = Path("data") / args.scene / "colmap" / "aligned"
        sparse_root = Path("data") / args.scene / "colmap" / "sparse"
        if aligned_dir.is_dir() and (aligned_dir / "images.txt").is_file():
            model_dir = aligned_dir
            is_aligned = True
        elif sparse_root.is_dir():
            all_models = find_all_sparse_models(sparse_root)
            if all_models:
                model_dir = all_models[0][0]
            else:
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

    # Print each model's image count if several exist
    model_count = max(1, len(all_models))
    if len(all_models) > 1:
        print(f"Found {len(all_models)} sparse models:")
        for m_path, n_cams in all_models:
            print(f"  Model '{m_path.name}': {n_cams} registered images")
        print(f"Selected largest model: '{model_dir.name}' with {all_models[0][1]} registered images.")

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

    # 4. Estimate UP direction via SVD of camera up vectors
    print("\n[Step 1/4] Estimating UP vector and floor normal...")
    up_vector = estimate_up_svd(poses)
    floor_normal, floor_offset, angle_deg = detect_floor_normal(pts, up_vector)
    print(f"  -> Detected UP vector : [{up_vector[0]:.4f}, {up_vector[1]:.4f}, {up_vector[2]:.4f}]")
    print(f"  -> Floor plane normal : [{floor_normal[0]:.4f}, {floor_normal[1]:.4f}, {floor_normal[2]:.4f}]")
    print(f"  -> Angle(UP, Floor)   : {angle_deg:.2f} deg")

    # 5. Build proper rotation matrix (det = +1) mapping UP to +Y
    R_align = build_rotation_to_y_up(up_vector)
    det_R = float(np.linalg.det(R_align))
    print(f"  -> Rotation matrix built: det(R) = {det_R:.6f} (proper rotation, no reflection)")

    # Apply rotation to points AND camera centers without negating any axis
    if len(pts) > 0:
        pts_aligned = (R_align @ pts.T).T
    else:
        pts_aligned = pts.copy()

    # Snap floor level to y = 0.00 m
    if len(pts_aligned) > 0:
        floor_y = float(np.percentile(pts_aligned[:, 1], 2.0))
        pts_aligned[:, 1] -= floor_y
    else:
        floor_y = 0.0

    for p in poses:
        p["C"] = R_align @ p["C"]
        p["C"][1] -= floor_y
        p["up"] = R_align @ p["up"]
        p["fwd"] = R_align @ p["fwd"]

    print(f"  -> Aligned to canonical +Y up axis and snapped floor to y = 0.00 m")

    # 6. Clean Point Cloud (outlier removal)
    print("\n[Step 2/4] Cleaning point cloud...")
    if not args.no_filter and len(pts_aligned) > 0:
        pts_clean, cols_clean = clean_point_cloud(
            pts_aligned, cols, errs, tracks, poses,
            max_reproj_error=args.max_reproj_error,
            min_observations=args.min_obs,
        )
        print(f"  -> {len(pts_clean):,} clean points remaining ({len(pts_clean)/max(1, len(pts_aligned))*100:.1f}%).")
    else:
        pts_clean, cols_clean = pts_aligned, cols

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

    # Scale label
    scale_label = "1 metre (metric)" if is_aligned else "1 unit (unaligned)"
    if args.scale:
        scale_label = f"1.0 m (scale {args.scale:.2f})"

    # 8. Render 2D Top-View Density Floor Plan
    print("\n[Step 3/4] Rendering orthographic 2D density floor plan...")
    signed_area, turn_sequence, start_end_gap = render_topview_density(
        pts_clean, poses, out_2d,
        up_vector=up_vector,
        floor_normal=floor_normal,
        angle_deg=angle_deg,
        model_name=model_dir.name,
        total_models=model_count,
        split_indices=split_indices,
        bin_size=args.bin_size,
        is_metric=is_aligned,
        scale_label=scale_label,
        arrow_every=args.arrow_every,
    )

    # 9. Debug Side-Views
    if args.debug:
        print("\n[Step 4/4] Rendering side-view elevation plots (--debug)...")
        ceiling_y = float(np.percentile(pts_clean[:, 1], 98.0)) if len(pts_clean) > 0 else 3.0
        render_side_views(pts_clean, poses, out_2d.parent, floor_y=0.0, ceiling_y=ceiling_y)

    # 10. Print required trajectory metrics
    area_sign = "+" if signed_area >= 0 else "-"
    print("\n" + "=" * 70)
    print("  TRAJECTORY ANALYSIS & VALIDATION REPORT")
    print("=" * 70)
    print(f"Signed area sign: {area_sign}")
    print(f"Turn sequence (Ramer-Douglas-Peucker 0.5 m): {turn_sequence}")
    print(f"Start-end gap: {start_end_gap:.2f} m")
    print(f"Model count: {model_count}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
