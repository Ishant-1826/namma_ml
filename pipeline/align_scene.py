#!/usr/bin/env python3
"""
pipeline/align_scene.py  --  Stage 3: Metric, Gravity-Aligned, Y-up Floor Alignment.

Converts a COLMAP sparse reconstruction into a metric, gravity-aligned,
Y-up coordinate frame with the floor plane at y = 0 BEFORE Gaussian Splatting training.

Key Stages:
-----------
1. Load sparse/0 (pycolmap if available, else robust COLMAP text parser).
2. Floor detection: Iterative RANSAC plane fitting (Open3D segment_plane).
   Selects the lowest large horizontal plane whose normal agrees with the
   mean camera up vector, with cameras verified to be strictly above the floor.
   Fails with an actionable error message if inliers are insufficient.
3. Rigid Rotation & Translation:
   Rotates floor normal to +Y (proper SO(3) rotation with det = +1), aligns
   dominant room walls axis-aligned with Leg 1 heading North (-Z), and
   translates the scene so the floor plane lies at y = 0.0.
4. Metric Scaling:
   - Mode (a) Default: Scales the model so the mean camera height above floor
     equals the assumed smartphone capture height (config, default 1.45 m).
   - Mode (b): Scales according to a user-supplied metric distance between two
     point IDs or coordinates.
   - Compares implied ceiling height and room extents against measurements.json.
5. Outputs:
   - data/<scene>/colmap_aligned/ (cameras.txt, images.txt, points3D.txt)
   - data/<scene>/transforms.json (Nerfstudio format in aligned frame, with
     explicit COLMAP/OpenCV vs OpenGL camera axes conversion)
   - data/<scene>/alignment.json (R, t, s, and detailed geometry statistics)

Usage:
------
    python pipeline/align_scene.py --scene home
    python pipeline/align_scene.py --scene home --phone-height 1.45
    python pipeline/align_scene.py --scene home --scale-mode points --point-ids 10 25 --distance 2.45
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from pipeline.utils.point_cloud import o3d
except ImportError:
    import open3d as o3d
import yaml

# Check for pycolmap availability
try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# ---------------------------------------------------------------------------
# I/O Helpers & Parsers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load configuration YAML file."""
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_cameras_txt(path: Path) -> dict:
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


def parse_images_txt(path: Path) -> dict:
    """Parse COLMAP images.txt (2 lines per registered image)."""
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


def parse_points3d_txt(path: Path) -> dict:
    """Parse COLMAP points3D.txt."""
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
        for cam_id, cam in sorted(cameras.items()):
            params_str = " ".join(repr(p) for p in cam["params"])
            f.write(f"{cam_id} {cam['model']} {cam['width']} {cam['height']} {params_str}\n")


def write_images_txt(images: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, img in sorted(images.items()):
            f.write(
                f"{img_id}"
                f" {img['qw']:.10f} {img['qx']:.10f}"
                f" {img['qy']:.10f} {img['qz']:.10f}"
                f" {img['tx']:.10f} {img['ty']:.10f} {img['tz']:.10f}"
                f" {img['camera_id']} {img['name']}\n"
            )
            f.write(img["point2d_str"] + "\n")


def write_points3d_txt(points: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for pt_id, pt in sorted(points.items()):
            xyz = pt["xyz"]
            r, g, b = pt["rgb"]
            f.write(
                f"{pt_id} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
                f" {r} {g} {b} {pt['error']:.6f} {pt['track']}\n"
            )


# ---------------------------------------------------------------------------
# 3D Geometry Utilities
# ---------------------------------------------------------------------------

def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
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


def rotation_align(v_src: np.ndarray, v_dst: np.ndarray) -> np.ndarray:
    """
    Computes a proper 3D rotation matrix (det = +1) mapping unit vector v_src
    to unit vector v_dst via Rodrigues' formula.
    """
    v_src = v_src / np.linalg.norm(v_src)
    v_dst = v_dst / np.linalg.norm(v_dst)
    cross = np.cross(v_src, v_dst)
    dot = float(np.dot(v_src, v_dst))
    if abs(dot + 1.0) < 1e-8:  # 180-degree case
        perp = np.array([1., 0., 0.]) if abs(v_src[0]) < 0.9 else np.array([0., 1., 0.])
        axis = np.cross(v_src, perp)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    s = np.linalg.norm(cross)
    kmat = np.array([
        [0., -cross[2], cross[1]],
        [cross[2], 0., -cross[0]],
        [-cross[1], cross[0], 0.],
    ])
    return np.eye(3) + kmat + kmat @ kmat * (1.0 - dot) / (s * s)


def camera_centre(img: dict) -> np.ndarray:
    """Compute camera center in world coordinates from COLMAP pose."""
    R = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
    t = np.array([img["tx"], img["ty"], img["tz"]])
    return -(R.T @ t)


def frame_number(name: str) -> int:
    """Extract integer frame number from filename."""
    digits = "".join(filter(str.isdigit, name))
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# Floor Plane Detection (Iterative Open3D RANSAC)
# ---------------------------------------------------------------------------

def detect_floor_plane_iterative(
    pts_xyz: np.ndarray,
    cam_centres: np.ndarray,
    mean_cam_up: np.ndarray,
    dist_thresh: float = 0.08,
    max_angle_deg: float = 8.0,
    min_inliers: int = 150,
    max_iterations: int = 25,
) -> dict:
    """
    Detects the floor plane via iterative Open3D segment_plane RANSAC.
    
    Selection Criteria:
      1. Normal agrees with mean camera up vector within max_angle_deg.
      2. Normal is oriented so cameras are above the floor (n @ cam_c + d > 0).
      3. Large plane: meets min_inliers threshold.
      4. Lowest plane: selects the plane with the lowest elevation (floor, not
         tabletops, chair tops, or ceiling).
    """
    candidates = []

    # 1. Iterative search on points below median camera height
    # Floor geometry is always situated below cameras.
    med_cam_h = float(np.median(cam_centres @ mean_cam_up))
    pts_below = pts_xyz[(pts_xyz @ mean_cam_up) < med_cam_h] if len(pts_xyz) > 0 else pts_xyz

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_below)
    curr_pcd = pcd

    for i in range(max_iterations):
        if len(curr_pcd.points) < 100:
            break
        plane_model, inliers = curr_pcd.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=2000,
        )
        if len(inliers) < 80:
            curr_pcd = curr_pcd.select_by_index(inliers, invert=True)
            continue

        nrm = np.array(plane_model[:3])
        d = float(plane_model[3])
        n_len = np.linalg.norm(nrm)
        if n_len < 1e-7:
            curr_pcd = curr_pcd.select_by_index(inliers, invert=True)
            continue
        nrm /= n_len
        d /= n_len

        # Orient normal along camera up direction
        if nrm @ mean_cam_up < 0:
            nrm = -nrm
            d = -d

        angle_deg = float(np.degrees(np.arccos(np.clip(nrm @ mean_cam_up, -1.0, 1.0))))

        # Check if plane is horizontal
        if angle_deg <= max_angle_deg:
            cam_dists = [float(nrm @ c + d) for c in cam_centres]
            frac_above = float(np.mean([cd > 0 for cd in cam_dists]))
            # Verify cameras are above the plane
            if frac_above >= 0.80:
                inlier_pts = np.asarray(curr_pcd.select_by_index(inliers).points)
                mean_h = float(np.mean(inlier_pts @ mean_cam_up))
                # Count total inliers across full point cloud
                all_inliers = int(np.sum(np.abs(pts_xyz @ nrm + d) < dist_thresh))
                candidates.append({
                    "iteration": i,
                    "normal": nrm,
                    "d": d,
                    "angle_deg": angle_deg,
                    "height": mean_h,
                    "cam_dist": float(np.mean(cam_dists)),
                    "inliers": all_inliers,
                })

        curr_pcd = curr_pcd.select_by_index(inliers, invert=True)

    # 2. Targeted search in lowest 30% of heights to avoid missing floor
    h_pts = pts_xyz @ mean_cam_up
    p30 = np.percentile(h_pts, 30)
    floor_slice = pts_xyz[h_pts < p30]
    if len(floor_slice) >= 150:
        pcd_slice = o3d.geometry.PointCloud()
        pcd_slice.points = o3d.utility.Vector3dVector(floor_slice)
        curr_slice = pcd_slice
        for j in range(5):
            if len(curr_slice.points) < 100:
                break
            plane_model, inliers = curr_slice.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=3,
                num_iterations=2000,
            )
            nrm = np.array(plane_model[:3])
            d = float(plane_model[3])
            n_len = np.linalg.norm(nrm)
            if n_len > 1e-7:
                nrm /= n_len
                d /= n_len
                if nrm @ mean_cam_up < 0:
                    nrm = -nrm
                    d = -d
                angle_deg = float(np.degrees(np.arccos(np.clip(nrm @ mean_cam_up, -1.0, 1.0))))
                if angle_deg <= max_angle_deg:
                    cam_dists = [float(nrm @ c + d) for c in cam_centres]
                    if float(np.mean([cd > 0 for cd in cam_dists])) >= 0.80:
                        inlier_pts = np.asarray(curr_slice.select_by_index(inliers).points)
                        mean_h = float(np.mean(inlier_pts @ mean_cam_up))
                        all_inliers = int(np.sum(np.abs(pts_xyz @ nrm + d) < dist_thresh))
                        candidates.append({
                            "iteration": f"slice_{j}",
                            "normal": nrm,
                            "d": d,
                            "angle_deg": angle_deg,
                            "height": mean_h,
                            "cam_dist": float(np.mean(cam_dists)),
                            "inliers": all_inliers,
                        })
            curr_slice = curr_slice.select_by_index(inliers, invert=True)

    # 3. Robust Floor Plane Selection & Joint PCA Refinement:
    # Indoor rooms feature multiple horizontal planes: tabletops, chair seats, and the floor.
    # The floor is the lowest physical plane situated below furniture (cam_dist >= 3.5 units).
    valid = [c for c in candidates if c["inliers"] >= min_inliers]
    if not valid:
        if candidates:
            candidates.sort(key=lambda c: -c["inliers"])
            valid = [candidates[0]]
        else:
            return None

    # Compute camera trajectory plane normal (user walked a level loop)
    mu_cams = np.mean(cam_centres, axis=0)
    _, _, Vt_cams = np.linalg.svd(cam_centres - mu_cams)
    n_cams = Vt_cams[-1]
    if n_cams @ mean_cam_up < 0:
        n_cams = -n_cams
    n_cams /= np.linalg.norm(n_cams)

    # Deep floor candidates (situated below tables and chairs, cam_dist >= 3.5)
    deep_candidates = [c for c in valid if c["cam_dist"] >= 3.5]
    if deep_candidates:
        deep_candidates.sort(key=lambda c: -c["inliers"])
        best_cand = deep_candidates[0]
    else:
        valid.sort(key=lambda c: c["height"])
        best_cand = valid[0]

    # PCA refinement of floor inliers + joint gravity alignment with camera trajectory
    n_init = best_cand["normal"]
    d_init = best_cand["d"]
    inlier_mask = np.abs(pts_xyz @ n_init + d_init) < dist_thresh
    floor_pts = pts_xyz[inlier_mask]

    if len(floor_pts) >= 50:
        mu_floor = np.mean(floor_pts, axis=0)
        _, _, Vt_f = np.linalg.svd(floor_pts - mu_floor)
        n_floor = Vt_f[-1]
        if n_floor @ mean_cam_up < 0:
            n_floor = -n_floor
        n_floor /= np.linalg.norm(n_floor)

        # Joint normal balancing dense floor surface and level camera walk
        n_joint = n_floor + n_cams
        n_joint /= np.linalg.norm(n_joint)
        d_joint = -float(np.dot(n_joint, mu_floor))
        angle_joint = float(np.degrees(np.arccos(np.clip(n_joint @ mean_cam_up, -1.0, 1.0))))
        cam_dists_joint = [float(n_joint @ c + d_joint) for c in cam_centres]

        return {
            "iteration": f"{best_cand['iteration']}_joint_pca",
            "normal": n_joint,
            "d": d_joint,
            "angle_deg": angle_joint,
            "height": float(np.mean(floor_pts @ mean_cam_up)),
            "cam_dist": float(np.mean(cam_dists_joint)),
            "inliers": int(np.sum(np.abs(pts_xyz @ n_joint + d_joint) < dist_thresh)),
        }

    return best_cand


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
        print(f"[WARN] No cameras or images to export to {output_path}")
        return

    # Use primary camera for intrinsics
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
    k1 = 0.0
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    if model in ("SIMPLE_RADIAL", "RADIAL"):
        fl_x = params[0]
        fl_y = params[0]
        cx = params[1]
        cy = params[2]
        k1 = params[3]
    elif model in ("PINHOLE",):
        fl_x = params[0]
        fl_y = params[1]
        cx = params[2]
        cy = params[3]
    elif model in ("OPENCV", "OPENCV_FISHEYE"):
        fl_x = params[0]
        fl_y = params[1]
        cx = params[2]
        cy = params[3]
        k1 = params[4]
        k2 = params[5]
        p1 = params[6]
        p2 = params[7]

    frames_list = []
    # Sort frames in natural temporal order
    sorted_img_ids = sorted(images.keys(), key=lambda i: frame_number(images[i]["name"]))

    for img_id in sorted_img_ids:
        img = images[img_id]
        R_w2c = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
        t_w2c = np.array([img["tx"], img["ty"], img["tz"]])
        C = -R_w2c.T @ t_w2c
        R_c2w = R_w2c.T  # [r_x, r_y, r_z]

        # Convert OpenCV camera axes to OpenGL camera axes:
        # Flip column 1 (Y: down -> up) and column 2 (Z: forward -> back)
        c2w_opengl = np.eye(4)
        c2w_opengl[:3, 0] = R_c2w[:, 0]   # +X Right
        c2w_opengl[:3, 1] = -R_c2w[:, 1]  # +Y Up
        c2w_opengl[:3, 2] = -R_c2w[:, 2]  # +Z Backwards
        c2w_opengl[:3, 3] = C             # Camera origin in aligned world coords

        # File path relative to scene root
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
# Main Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3: Metric, Gravity-Aligned, Y-up Floor Alignment."
    )
    ap.add_argument("--scene", required=True, help="Scene name (subfolder of data/)")
    ap.add_argument("--config", default="configs/default.yaml", help="Configuration YAML path")
    ap.add_argument("--sparse-dir", default=None, help="Input COLMAP sparse model directory")
    ap.add_argument("--output-dir", default=None, help="Output aligned COLMAP directory")
    ap.add_argument("--measurements", default=None, help="Path to measurements JSON file")
    ap.add_argument(
        "--scale-mode",
        choices=["phone_height", "points"],
        default="phone_height",
        help="Scale mode: 'phone_height' (assume camera height) or 'points' (known distance)",
    )
    ap.add_argument(
        "--phone-height",
        type=float,
        default=None,
        help="Target mean camera height above floor in metres (default: 1.45 m)",
    )
    ap.add_argument(
        "--point-ids",
        nargs=2,
        type=int,
        default=None,
        help="Two point IDs in points3D.txt for metric distance scaling",
    )
    ap.add_argument(
        "--point-coords",
        nargs=6,
        type=float,
        default=None,
        help="Coordinates x1 y1 z1 x2 y2 z2 for metric distance scaling",
    )
    ap.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Known metric distance in metres between points for scale mode (b)",
    )
    ap.add_argument(
        "--dist-thresh",
        type=float,
        default=0.08,
        help="RANSAC distance threshold for plane fitting",
    )
    ap.add_argument(
        "--max-floor-angle",
        type=float,
        default=8.0,
        help="Maximum angle (degrees) between floor normal and camera-up vector",
    )
    ap.add_argument(
        "--min-inliers",
        type=int,
        default=150,
        help="Minimum inliers required for valid floor plane",
    )
    ap.add_argument(
        "--max-planes",
        type=int,
        default=25,
        help="Maximum plane candidates to evaluate in iterative RANSAC",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    scene = args.scene
    data_dir = Path("data") / scene

    # 1. Resolve sparse input directory
    if args.sparse_dir:
        sparse_dir = Path(args.sparse_dir)
    else:
        candidates = [
            data_dir / "colmap" / "sparse" / "0",
            data_dir / "sparse" / "0",
            data_dir / "colmap" / "sparse",
        ]
        sparse_dir = next((c for c in candidates if (c / "cameras.txt").is_file() or (c / "cameras.bin").is_file()), candidates[0])

    if not sparse_dir.is_dir():
        print(f"[FAIL] Sparse reconstruction directory not found: {sparse_dir}", file=sys.stderr)
        print("  Actionable advice: Run Stage 2 SfM (pipeline/run_sfm.py) first to generate sparse/0.", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve output directories
    # Primary output: data/<scene>/colmap_aligned/
    output_dir = Path(args.output_dir) if args.output_dir else (data_dir / "colmap_aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Secondary backward-compatible directory: data/<scene>/colmap/aligned/
    legacy_aligned_dir = data_dir / "colmap" / "aligned"
    legacy_aligned_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolve target phone height from CLI or config
    target_phone_height = args.phone_height
    if target_phone_height is None:
        target_phone_height = float(cfg.get("alignment", {}).get("phone_height_m", 1.45))

    # 4. Resolve measurements
    meas_path = Path(args.measurements) if args.measurements else (data_dir / "measurements.json")
    meas_data = {}
    if meas_path.is_file():
        try:
            with open(meas_path, "r", encoding="utf-8") as f:
                meas_data = json.load(f)
        except Exception:
            pass

    print("=" * 64)
    print("  NAMMA SPACE -- STAGE 3: METRIC Y-UP SCENE ALIGNMENT")
    print("=" * 64)
    print(f"  Scene             : {scene}")
    print(f"  Input Sparse Dir  : {sparse_dir}")
    print(f"  Output Aligned Dir: {output_dir}")
    print(f"  Scale Mode        : {args.scale_mode}")
    if args.scale_mode == "phone_height":
        print(f"  Target Phone H    : {target_phone_height:.2f} m")
    print("-" * 64)

    # -----------------------------------------------------------------------
    # Step 1: Load Reconstruction
    # -----------------------------------------------------------------------
    print("\n[Step 1/5] Loading sparse reconstruction...")
    txt_cameras = sparse_dir / "cameras.txt"
    txt_images = sparse_dir / "images.txt"
    txt_points = sparse_dir / "points3D.txt"

    if not (txt_cameras.is_file() and txt_images.is_file() and txt_points.is_file()):
        print(f"[FAIL] Missing COLMAP text model files in {sparse_dir}", file=sys.stderr)
        print("  Actionable advice: Run COLMAP model_converter to export sparse/0 to TXT.", file=sys.stderr)
        sys.exit(1)

    cameras = parse_cameras_txt(txt_cameras)
    images = parse_images_txt(txt_images)
    points = parse_points3d_txt(txt_points)
    print(f"  Loaded: {len(cameras)} cameras, {len(images)} images, {len(points)} 3D points")

    if len(images) == 0 or len(points) == 0:
        print("[FAIL] Sparse reconstruction is empty (0 images or 0 points).", file=sys.stderr)
        sys.exit(1)

    # Filter degenerate poses (|C| >= 1000)
    DEGENERATE_NORM = 1000.0
    inlier_iids = [
        iid for iid, img in images.items()
        if np.linalg.norm(camera_centre(img)) < DEGENERATE_NORM
    ]
    print(f"  Inlier camera poses: {len(inlier_iids)} / {len(images)}")

    # -----------------------------------------------------------------------
    # Step 2: Mean Camera Up Vector & Floor Plane Detection
    # -----------------------------------------------------------------------
    print("\n[Step 2/5] Detecting floor plane (Iterative Open3D RANSAC)...")
    cam_ups = []
    cam_centres = []
    for iid in inlier_iids:
        img = images[iid]
        R = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
        # In COLMAP camera frame, -Y is Up
        cam_ups.append(R.T @ np.array([0., -1., 0.]))
        cam_centres.append(camera_centre(img))

    cam_ups = np.array(cam_ups)
    cam_centres = np.array(cam_centres)

    # Mean camera up vector via SVD
    _, _, Vt = np.linalg.svd(cam_ups)
    mean_up = Vt[0]
    if np.mean(cam_ups @ mean_up) < 0:
        mean_up = -mean_up
    mean_up /= np.linalg.norm(mean_up)
    print(f"  Mean camera-up vector: [{mean_up[0]:+.4f} {mean_up[1]:+.4f} {mean_up[2]:+.4f}]")

    pts_xyz = np.array([p["xyz"] for p in points.values()])

    floor_result = detect_floor_plane_iterative(
        pts_xyz=pts_xyz,
        cam_centres=cam_centres,
        mean_cam_up=mean_up,
        dist_thresh=args.dist_thresh,
        max_angle_deg=args.max_floor_angle,
        min_inliers=args.min_inliers,
        max_iterations=args.max_planes,
    )

    if floor_result is None or floor_result["inliers"] < args.min_inliers:
        found_inliers = floor_result["inliers"] if floor_result else 0
        print(
            f"[FAIL] Floor detection failed: insufficient floor inliers ({found_inliers} < {args.min_inliers}) "
            f"or no horizontal plane found within {args.max_floor_angle}° of camera up.",
            file=sys.stderr,
        )
        print("  Actionable advice: Verify scene capture has floor coverage, or increase --max-floor-angle / decrease --min-inliers.", file=sys.stderr)
        sys.exit(1)

    floor_normal = floor_result["normal"]
    angle_to_up = floor_result["angle_deg"]
    print(f"  Floor normal candidate: [{floor_normal[0]:+.4f} {floor_normal[1]:+.4f} {floor_normal[2]:+.4f}]")
    print(f"  Angle to camera-up    : {angle_to_up:.2f}° (threshold: <= {args.max_floor_angle:.1f}°)")
    print(f"  Floor inlier count    : {floor_result['inliers']} points")
    print(f"  Mean camera clearance : {floor_result['cam_dist']:.3f} units above floor")

    # -----------------------------------------------------------------------
    # Step 3: Rigid Rotation (+Y Up, Wall Alignment) & Floor Snap (y = 0)
    # -----------------------------------------------------------------------
    print("\n[Step 3/5] Rotating floor normal to +Y and snapping floor to y = 0.0...")

    # R_floor rotates floor normal to [0, 1, 0]
    R_floor = rotation_align(floor_normal, np.array([0., 1., 0.]))

    # Yaw rotation around Y: align dominant walls with coordinate axes
    pts_temp = pts_xyz @ R_floor.T
    xz = pts_temp[:, [0, 2]]
    p05 = np.percentile(xz, 5, axis=0)
    p95 = np.percentile(xz, 95, axis=0)
    xz_filt = xz[(xz[:, 0] >= p05[0]) & (xz[:, 0] <= p95[0]) & (xz[:, 1] >= p05[1]) & (xz[:, 1] <= p95[1])]

    cov = np.cov(xz_filt.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    v_wall = eigenvecs[:, 1]
    wall_phi = math.atan2(v_wall[1], v_wall[0])

    # Direct Leg 1 towards North (-Z)
    sorted_iids = sorted(inlier_iids, key=lambda i: frame_number(images[i]["name"]))
    c_start = camera_centre(images[sorted_iids[0]]) @ R_floor.T
    c_leg1_end = camera_centre(images[sorted_iids[min(20, len(sorted_iids)-1)]]) @ R_floor.T
    v_leg1 = c_leg1_end - c_start

    best_angle_deg = 0.0
    best_score = -1e9
    for k in range(4):
        deg = math.degrees(-wall_phi) + k * 90.0
        rad = math.radians(deg)
        R2 = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        leg1_rot = R2 @ np.array([v_leg1[0], v_leg1[2]])
        score = -leg1_rot[1]
        if score > best_score:
            best_score = score
            best_angle_deg = deg

    rad_yaw = math.radians(best_angle_deg)
    cos_y, sin_y = math.cos(rad_yaw), math.sin(rad_yaw)
    R_yaw = np.array([
        [ cos_y, 0.0, -sin_y],
        [   0.0, 1.0,    0.0],
        [ sin_y, 0.0,  cos_y],
    ])

    # Total rotation matrix: R_total = R_yaw @ R_floor (det = +1)
    R_total = R_yaw @ R_floor
    print(f"  Canonical wall yaw rotation: {best_angle_deg:.2f}° (Leg 1 points North -Z)")

    # Rotate 3D points
    for pt in points.values():
        pt["xyz"] = R_total @ pt["xyz"]

    # Rotate camera poses: R_new = R_old @ R_total.T
    R_total_T = R_total.T
    for img in images.values():
        R_cur = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"]) @ R_total_T
        qw, qx, qy, qz = rot_to_quat(R_cur)
        img["qw"], img["qx"], img["qy"], img["qz"] = qw, qx, qy, qz

    # Floor snap: calculate median Y of floor inliers in rotated frame
    all_dists = np.abs(pts_xyz @ floor_normal + floor_result["d"])
    floor_inliers_orig = pts_xyz[all_dists < args.dist_thresh]
    floor_inliers_rot = floor_inliers_orig @ R_total.T
    floor_y_unscaled = float(np.median(floor_inliers_rot[:, 1]))
    print(f"  Unscaled floor plane Y level: {floor_y_unscaled:.4f} units")

    # Translate so floor plane is at y = 0.0
    t_snap = np.array([0., -floor_y_unscaled, 0.])
    for pt in points.values():
        pt["xyz"] += t_snap

    for img in images.values():
        R_cur = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
        t_old = np.array([img["tx"], img["ty"], img["tz"]])
        t_new = t_old - R_cur @ t_snap
        img["tx"], img["ty"], img["tz"] = float(t_new[0]), float(t_new[1]), float(t_new[2])

    # -----------------------------------------------------------------------
    # Step 4: Metric Scale Alignment
    # -----------------------------------------------------------------------
    print("\n[Step 4/5] Metric scale alignment...")
    unscaled_centres = [camera_centre(images[i]) for i in inlier_iids]
    unscaled_cam_heights = [float(c[1]) for c in unscaled_centres]
    mean_unscaled_h = float(np.mean(unscaled_cam_heights))

    scale_method = args.scale_mode
    scale_m_per_unit = 1.0

    if args.scale_mode == "points":
        if args.point_ids and args.distance:
            p1_id, p2_id = args.point_ids
            if p1_id in points and p2_id in points:
                dist_unscaled = float(np.linalg.norm(points[p1_id]["xyz"] - points[p2_id]["xyz"]))
                scale_m_per_unit = args.distance / dist_unscaled
                print(f"  Mode (b) Point IDs [{p1_id}, {p2_id}]: unscaled dist = {dist_unscaled:.4f} u, target = {args.distance:.2f} m")
                print(f"  --> Metric scale: {scale_m_per_unit:.5f} m/unit")
            else:
                print(f"[WARN] Point IDs [{p1_id}, {p2_id}] not found in points3D.txt; falling back to phone_height mode.")
                scale_method = "phone_height"
        elif args.point_coords and args.distance:
            p1 = np.array(args.point_coords[:3])
            p2 = np.array(args.point_coords[3:])
            dist_unscaled = float(np.linalg.norm(p1 - p2))
            scale_m_per_unit = args.distance / dist_unscaled
            print(f"  Mode (b) Coordinates: unscaled dist = {dist_unscaled:.4f} u, target = {args.distance:.2f} m")
            print(f"  --> Metric scale: {scale_m_per_unit:.5f} m/unit")
        else:
            print("[WARN] Incomplete points arguments for mode (b); falling back to phone_height mode.")
            scale_method = "phone_height"

    if scale_method == "phone_height":
        scale_m_per_unit = target_phone_height / mean_unscaled_h if mean_unscaled_h > 0.01 else 0.35
        print(f"  Mode (a) Phone Height: unscaled mean height = {mean_unscaled_h:.4f} u, target = {target_phone_height:.2f} m")
        print(f"  --> Metric scale: {scale_m_per_unit:.5f} m/unit")

    # Apply metric scale
    for pt in points.values():
        pt["xyz"] *= scale_m_per_unit

    for img in images.values():
        img["tx"] *= scale_m_per_unit
        img["ty"] *= scale_m_per_unit
        img["tz"] *= scale_m_per_unit

    # -----------------------------------------------------------------------
    # Sanity Checks & Statistics
    # -----------------------------------------------------------------------
    aligned_pts = np.array([p["xyz"] for p in points.values()])
    aligned_cams = np.array([camera_centre(images[i]) for i in inlier_iids])
    aligned_cam_heights = aligned_cams[:, 1]

    mean_h_m = float(np.mean(aligned_cam_heights))
    std_h_m = float(np.std(aligned_cam_heights))
    med_h_m = float(np.median(aligned_cam_heights))
    min_h_m = float(np.min(aligned_cam_heights))
    max_h_m = float(np.max(aligned_cam_heights))

    x_span_m = float(np.percentile(aligned_pts[:, 0], 98) - np.percentile(aligned_pts[:, 0], 2))
    z_span_m = float(np.percentile(aligned_pts[:, 2], 98) - np.percentile(aligned_pts[:, 2], 2))
    ceiling_h_m = float(np.percentile(aligned_pts[:, 1], 98))

    print("\n" + "=" * 64)
    print("  METRIC SCENE ALIGNMENT SANITY CHECK")
    print("=" * 64)
    print(f"  Camera Height Mean : {mean_h_m:.3f} m (std: ±{std_h_m:.3f} m)")
    print(f"  Camera Height Range: [{min_h_m:.2f} m, {max_h_m:.2f} m] (median: {med_h_m:.2f} m)")
    print(f"  Room Extents (X x Z): {x_span_m:.2f} m (width) x {z_span_m:.2f} m (length)")
    print(f"  Estimated Ceiling  : {ceiling_h_m:.2f} m above floor")

    # Compare against measurements.json if present
    if meas_data:
        room_meas = meas_data.get("room", meas_data)
        m_len = room_meas.get("length_m", room_meas.get("length"))
        m_wid = room_meas.get("width_m", room_meas.get("width"))
        m_cei = room_meas.get("ceiling_height_m", room_meas.get("ceiling"))
        print("\n  Comparison against measurements.json:")
        if m_wid is not None:
            print(f"    Width  (X): Reconstructed {x_span_m:.2f} m vs Measured {float(m_wid):.2f} m")
        if m_len is not None:
            print(f"    Length (Z): Reconstructed {z_span_m:.2f} m vs Measured {float(m_len):.2f} m")
        if m_cei is not None:
            print(f"    Ceiling   : Reconstructed {ceiling_h_m:.2f} m vs Measured {float(m_cei):.2f} m")
    print("-" * 64)

    # -----------------------------------------------------------------------
    # Step 5: Save Outputs
    # -----------------------------------------------------------------------
    print("\n[Step 5/5] Writing outputs...")

    # A. COLMAP text model -> data/<scene>/colmap_aligned/
    write_cameras_txt(cameras, output_dir / "cameras.txt")
    write_images_txt(images, output_dir / "images.txt")
    write_points3d_txt(points, output_dir / "points3D.txt")
    print(f"  [+] Saved {output_dir / 'cameras.txt'}")
    print(f"  [+] Saved {output_dir / 'images.txt'}")
    print(f"  [+] Saved {output_dir / 'points3D.txt'}")

    # Also keep legacy directory updated
    if legacy_aligned_dir != output_dir:
        write_cameras_txt(cameras, legacy_aligned_dir / "cameras.txt")
        write_images_txt(images, legacy_aligned_dir / "images.txt")
        write_points3d_txt(points, legacy_aligned_dir / "points3D.txt")

    # B. Nerfstudio transforms.json
    transforms_path_scene = data_dir / "transforms.json"
    transforms_path_aligned = output_dir / "transforms.json"
    export_nerfstudio_transforms(cameras, images, transforms_path_scene, image_rel_prefix="images")
    export_nerfstudio_transforms(cameras, images, transforms_path_aligned, image_rel_prefix="../images")
    print(f"  [+] Saved {transforms_path_scene} (Nerfstudio OpenGL aligned format)")

    # C. alignment.json
    alignment_data = {
        "scene": scene,
        "convention": "right-handed Y-up metric (floor at y=0.0)",
        "R": R_total.tolist(),
        "t": (t_snap * scale_m_per_unit).tolist(),
        "s": float(scale_m_per_unit),
        "stats": {
            "scale_mode": scale_method,
            "target_phone_height_m": float(target_phone_height) if scale_method == "phone_height" else None,
            "mean_camera_height_m": float(mean_h_m),
            "std_camera_height_m": float(std_h_m),
            "median_camera_height_m": float(med_h_m),
            "min_camera_height_m": float(min_h_m),
            "max_camera_height_m": float(max_h_m),
            "plane_inliers": int(floor_result["inliers"]),
            "floor_plane_angle_to_cam_up_deg": float(angle_to_up),
            "floor_y_unscaled": float(floor_y_unscaled),
            "room_extents_m": {
                "x_width": round(x_span_m, 3),
                "z_length": round(z_span_m, 3),
                "y_ceiling": round(ceiling_h_m, 3),
            },
            "num_cameras": len(images),
            "num_points": len(points),
        },
    }

    alignment_path_scene = data_dir / "alignment.json"
    alignment_path_aligned = output_dir / "alignment.json"
    with open(alignment_path_scene, "w", encoding="utf-8") as f:
        json.dump(alignment_data, f, indent=2)
    with open(alignment_path_aligned, "w", encoding="utf-8") as f:
        json.dump(alignment_data, f, indent=2)
    print(f"  [+] Saved {alignment_path_scene}")

    # Also save legacy align_report.json
    pass_split = 201
    frames_rep_path = data_dir / "frames_report.json"
    if frames_rep_path.is_file():
        try:
            with open(frames_rep_path, "r", encoding="utf-8") as fr_f:
                vids = json.load(fr_f).get("videos", [])
                if vids:
                    pass_split = vids[0].get("frames_kept", 201)
        except Exception:
            pass

    p1_hs = [c[1] for i, c in zip(inlier_iids, aligned_cams) if frame_number(images[i]["name"]) <= pass_split]
    p2_hs = [c[1] for i, c in zip(inlier_iids, aligned_cams) if frame_number(images[i]["name"]) > pass_split]
    p1_med = float(np.median(p1_hs)) if p1_hs else float(mean_h_m)
    p2_med = float(np.median(p2_hs)) if p2_hs else float(mean_h_m)

    rpt = {
        "scene": scene,
        "scale_m_per_unit": scale_m_per_unit,
        "scale_method": scale_method,
        "up_vector": floor_normal.tolist(),
        "floor_angle_to_cam_up_deg": float(angle_to_up),
        "yaw_rotation_deg": float(best_angle_deg),
        "floor_y_colmap": float(floor_y_unscaled),
        "pass_1_height_m": p1_med,
        "pass_2_height_m": p2_med,
        "ceiling_height_m": float(ceiling_h_m),
        "n_images": len(images),
        "n_points": len(points),
    }
    with open(data_dir / "align_report.json", "w", encoding="utf-8") as f:
        json.dump(rpt, f, indent=2)

    print("\nStage 3 Alignment Complete.")


if __name__ == "__main__":
    main()
