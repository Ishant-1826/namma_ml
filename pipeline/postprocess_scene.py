#!/usr/bin/env python3
"""
pipeline/postprocess_scene.py  --  Phase 6: Gaussian Splat Post-Processing, Occupancy Grid & Compression.

Loads trained 3D Gaussian Splat model from outputs/<scene>/scene_raw.ply, applies:
1. Crop margins:
   - Margin from config crop_margin_m (default 4.0 m) around camera-path bounding box.
   - Intersected with robust extents of reference points (p0.5-p99.5, track length >= 4,
     reprojection error <= 1.0 px) + 1.0 m.
   - Y clamped to [floor_y - 0.3, ceiling_y + 0.3], where floor_y = 0.0 and ceiling_y is
     loaded from measurements.json or config.
   - Open3D statistical outlier removal on splat centres (nb_neighbors 20, std_ratio 2.5).
   - Splat counts printed after every stage in the filter table.
2. Floater pruning:
   - Low opacity pruning (sigmoid(opacity) < min_opacity).
   - Degenerate scale pruning (max_scale > max_scale_m).
3. Metadata & Scene Definition (outputs/<scene>/scene.json):
   - Bounds (min and max x, y, z in metres)
   - floor_y, ceiling_y
   - Spawn position (first camera projected to eye height 1.45 m above floor_y)
   - Spawn yaw along travel direction over first 10 cameras (Three.js convention)
   - splat_count, units ("metres"), splat file names, splat_rotation_euler ([0, 0, 0])
4. 2D Occupancy Grid (outputs/<scene>/occupancy.json & occupancy.png):
   - Centred splat points (opacity >= 0.3) in height band [0.15 m, 1.80 m] above floor.
   - Assert >= 98% of camera positions and spawn fall in FREE cells.
     If not, test inflation 0.20 m, then min point count 8; save working settings to configs/default.yaml.
   - Walkable array within walkable_radius_m (default 1.5 m) of the camera path.
   - Print percentage of occupied, free, and walkable cells.
   - occupancy.png draws Z DOWN, overlays camera trajectory in colour, shows spawn point & arrow,
     and outlines the walkable region.
5. Compression Quality:
   - Exports 32-byte web standard binary outputs/<scene>/scene.splat.
   - Converts to .spz using @playcanvas/splat-transform CLI.
   - Converts scene.spz back to PLY using splat-transform and reports max/mean centre error,
     mean opacity error, splat count difference, and format sizes.
6. Web Asset Deployment:
   - Copies scene.json, scene.splat, scene.spz, occupancy.json, occupancy.png to viewer/public/scene/.
   - Removes any duplicate legacy viewer/public/scene.splat.
7. Verification:
   - Runs scripts/verify_splat_frame.py on the new scene_clean.ply.

Usage:
  python pipeline/postprocess_scene.py --scene home
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.spatial import KDTree
import yaml

try:
    from pipeline.utils.point_cloud import o3d
except ImportError:
    try:
        import open3d as o3d
    except ImportError:
        o3d = None


SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_ply_header(path: Path) -> Tuple[int, List[Tuple[str, str]], int]:
    """Parses binary PLY header and returns (vertex_count, properties, header_offset)."""
    type_map = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    props = []
    n = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("latin1").strip()
            if not line:
                break
            parts = line.split()
            if line.startswith("element vertex"):
                n = int(parts[2])
            elif line.startswith("property") and not line.startswith("property list"):
                props.append((parts[2], type_map.get(parts[1], "<f4")))
            elif line == "end_header":
                break
        offset = f.tell()
    return n, props, offset


def load_ply_all(ply_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads PLY properties:
    Returns (xyz, sigmoid_opacity, max_real_scale).
    """
    n, props, offset = parse_ply_header(ply_path)
    dt = np.dtype(props)
    data = np.fromfile(ply_path, dtype=dt, count=n, offset=offset)

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

    if "opacity" in data.dtype.names:
        raw_op = data["opacity"].astype(np.float64)
        sig_op = 1.0 / (1.0 + np.exp(-raw_op))
    else:
        sig_op = np.ones(len(xyz), dtype=np.float64)

    scale_names = [f"scale_{i}" for i in range(3)]
    if all(s in data.dtype.names for s in scale_names):
        log_scales = np.column_stack([data[s] for s in scale_names]).astype(np.float64)
        max_scale = np.max(np.exp(log_scales), axis=1)
    else:
        max_scale = np.zeros(len(xyz), dtype=np.float64)

    return xyz, sig_op, max_scale


def load_robust_reference_extents(
    points3d_path: Path,
    min_track_len: int = 4,
    max_reproj_error: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Computes robust extents (p0.5 and p99.5) of reference points with
    track length >= min_track_len and reprojection error <= max_reproj_error.
    """
    if not points3d_path.is_file():
        return None
    robust_pts = []
    with open(points3d_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            err = float(parts[7])
            track_tokens = parts[8:]
            track_len = len(track_tokens) // 2
            if track_len >= min_track_len and err <= max_reproj_error:
                robust_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if len(robust_pts) < 10:
        return None

    pts_arr = np.array(robust_pts, dtype=np.float64)
    p005 = np.percentile(pts_arr, 0.5, axis=0)
    p995 = np.percentile(pts_arr, 99.5, axis=0)
    return p005, p995, len(pts_arr)


def write_clean_ply(
    output_path: Path,
    data_arr: np.ndarray,
    props: List[Tuple[str, str]],
) -> None:
    """Writes filtered structured numpy array to binary little-endian PLY."""
    n = len(data_arr)
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Namma Space clean splat export",
        "comment Vertical axis: y",
        f"element vertex {n}",
    ]
    for name, dtype_str in props:
        if "f4" in dtype_str:
            p_type = "float"
        elif "f8" in dtype_str:
            p_type = "double"
        elif "u1" in dtype_str:
            p_type = "uchar"
        elif "i4" in dtype_str:
            p_type = "int"
        else:
            p_type = "float"
        header_lines.append(f"property {p_type} {name}")
    header_lines.append("end_header\n")

    header = "\n".join(header_lines).encode("ascii")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(data_arr.tobytes())


def write_splat_binary(
    output_path: Path,
    xyz: np.ndarray,
    scales: np.ndarray,
    sh_dc: np.ndarray,
    opacities: np.ndarray,
    rotations: np.ndarray,
) -> int:
    """
    Vectorized export to 32-byte .splat binary format (Antimatter15 standard):
    - position : 3 x float32 (12 bytes: x, y, z)
    - scale    : 3 x float32 (12 bytes: exp(scale_0), exp(scale_1), exp(scale_2))
    - color    : 4 x uint8   (4 bytes: r, g, b, a)
    - rotation : 4 x uint8   (4 bytes: q_w, q_x, q_y, q_z normalized to 0..255)
    Total: 32 bytes per Gaussian.
    """
    n = len(xyz)
    dt_splat = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
        ("rot_0", "u1"), ("rot_1", "u1"), ("rot_2", "u1"), ("rot_3", "u1"),
    ])

    splat_arr = np.empty(n, dtype=dt_splat)

    # 1. Position
    splat_arr["x"] = xyz[:, 0].astype(np.float32)
    splat_arr["y"] = xyz[:, 1].astype(np.float32)
    splat_arr["z"] = xyz[:, 2].astype(np.float32)

    # 2. Scale (real physical semi-axes = exp(log_scales))
    real_scales = np.exp(scales).astype(np.float32)
    splat_arr["sx"] = real_scales[:, 0]
    splat_arr["sy"] = real_scales[:, 1]
    splat_arr["sz"] = real_scales[:, 2]

    # 3. Colors (RGB from DC spherical harmonics + alpha from sigmoid opacity)
    rgb = 0.5 + SH_C0 * sh_dc
    rgb_clipped = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    splat_arr["r"] = rgb_clipped[:, 0]
    splat_arr["g"] = rgb_clipped[:, 1]
    splat_arr["b"] = rgb_clipped[:, 2]

    sig_op = 1.0 / (1.0 + np.exp(-opacities.astype(np.float64)))
    splat_arr["a"] = np.clip(sig_op * 255.0, 0, 255).astype(np.uint8)

    # 4. Rotation (normalized quaternion mapped to 0..255: q * 127.5 + 128)
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    rot_norm = rotations / norms
    rot_u8 = np.clip(rot_norm * 127.5 + 128.0, 0, 255).astype(np.uint8)
    splat_arr["rot_0"] = rot_u8[:, 0]
    splat_arr["rot_1"] = rot_u8[:, 1]
    splat_arr["rot_2"] = rot_u8[:, 2]
    splat_arr["rot_3"] = rot_u8[:, 3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(splat_arr.tobytes())

    return output_path.stat().st_size


def convert_to_spz(input_ply: Path, output_spz: Path) -> Tuple[bool, Optional[str]]:
    """Converts PLY to compressed .spz using @playcanvas/splat-transform CLI."""
    output_spz.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        cmd_exec = ["cmd.exe", "/c", "npx.cmd", "--yes", "@playcanvas/splat-transform", "-w", str(input_ply), str(output_spz)]
    else:
        cmd_exec = ["npx", "-y", "@playcanvas/splat-transform", "-w", str(input_ply), str(output_spz)]

    try:
        res = subprocess.run(cmd_exec, shell=False, capture_output=True, text=True, timeout=120)
        if res.returncode == 0 and output_spz.is_file() and output_spz.stat().st_size > 0:
            return True, None
        err = res.stderr or res.stdout
        return False, f"CLI exited with code {res.returncode}: {err.strip()}"
    except Exception as e:
        return False, str(e)


def evaluate_spz_quality(clean_ply_path: Path, spz_path: Path) -> Dict[str, Any]:
    """
    Requirement 4: Converts scene.spz back to PLY using splat-transform,
    and computes max/mean centre error, mean opacity error, and splat count difference.
    """
    decompressed_ply = spz_path.parent / "scene_decompressed.ply"
    if os.name == "nt":
        cmd_exec = ["cmd.exe", "/c", "npx.cmd", "--yes", "@playcanvas/splat-transform", "-w", str(spz_path), str(decompressed_ply)]
    else:
        cmd_exec = ["npx", "-y", "@playcanvas/splat-transform", "-w", str(spz_path), str(decompressed_ply)]

    res = subprocess.run(cmd_exec, shell=False, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not decompressed_ply.is_file():
        raise RuntimeError(f"Decompressing SPZ to PLY failed: {res.stderr or res.stdout}")

    clean_xyz, clean_op, _ = load_ply_all(clean_ply_path)
    decomp_xyz, decomp_op, _ = load_ply_all(decompressed_ply)

    splat_diff = len(decomp_xyz) - len(clean_xyz)

    if len(clean_xyz) == len(decomp_xyz):
        diff_centres = np.linalg.norm(clean_xyz - decomp_xyz, axis=1)
        diff_op = np.abs(clean_op - decomp_op)
    else:
        tree = KDTree(clean_xyz)
        dists, idxs = tree.query(decomp_xyz)
        diff_centres = dists
        diff_op = np.abs(clean_op[idxs] - decomp_op)

    max_centre_error_m = float(np.max(diff_centres))
    mean_centre_error_m = float(np.mean(diff_centres))
    mean_opacity_error = float(np.mean(diff_op))

    return {
        "decompressed_ply": str(decompressed_ply),
        "splat_count_clean": len(clean_xyz),
        "splat_count_decompressed": len(decomp_xyz),
        "splat_count_diff": splat_diff,
        "max_centre_error_m": round(max_centre_error_m, 6),
        "max_centre_error_mm": round(max_centre_error_m * 1000.0, 3),
        "mean_centre_error_m": round(mean_centre_error_m, 6),
        "mean_centre_error_mm": round(mean_centre_error_m * 1000.0, 3),
        "mean_opacity_error": round(mean_opacity_error, 6),
    }


def update_config_occupancy_settings(config_path: str, inflation_m: float, min_points: int) -> None:
    """Updates configs/default.yaml with working occupancy settings."""
    cfg_p = Path(config_path)
    if not cfg_p.is_file():
        return
    try:
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "postprocess" not in cfg:
            cfg["postprocess"] = {}
        cfg["postprocess"]["occupancy_inflation_m"] = inflation_m
        cfg["postprocess"]["occupancy_min_points"] = min_points
        with open(cfg_p, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"  [!] Notice updating config: {e}")


def check_free_cells(
    occ_grid: np.ndarray,
    min_x: float,
    min_z: float,
    cell_size: float,
    width: int,
    height: int,
    cams: np.ndarray,
    spawn: np.ndarray,
) -> Tuple[float, int, bool]:
    """Tests what percentage of camera positions and spawn fall in FREE (0) cells."""
    cam_cols = np.clip(((cams[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    cam_rows = np.clip(((cams[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    cam_free = (occ_grid[cam_rows, cam_cols] == 0)
    free_pct = float(np.mean(cam_free) * 100.0)
    free_cnt = int(np.sum(cam_free))

    sp_col = int(np.clip((spawn[0] - min_x) / cell_size, 0, width - 1))
    sp_row = int(np.clip((spawn[2] - min_z) / cell_size, 0, height - 1))
    sp_free = bool(occ_grid[sp_row, sp_col] == 0)
    return free_pct, free_cnt, sp_free


def build_and_save_occupancy(
    output_dir: Path,
    clean_xyz: np.ndarray,
    clean_sig_op: np.ndarray,
    cams: np.ndarray,
    c0: np.ndarray,
    spawn_yaw: float,
    scene: str,
    pass_split_idx: int = 222,
    cell_size: float = 0.10,
    min_points: int = 5,
    inflation_radius_m: float = 0.25,
    walkable_radius_m: float = 1.50,
    config_path: str = "configs/default.yaml",
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Requirement 3:
    - Filters splat centres (opacity >= 0.3, y in [0.15, 1.80] m).
    - Tests camera and spawn free cell assertion (>= 98.0% cameras in free cells, spawn in free cell).
      If not satisfied, tests inflation 0.20 m, then min points 8, reports setting that worked,
      and updates configs/default.yaml.
    - Adds 'walkable' array to occupancy.json: cells within walkable_radius_m of camera path.
    - Prints percentage of cells occupied, free, and walkable.
    - Updates occupancy.png showing walkable region as a light outline.
    """
    mask_band = (clean_sig_op >= 0.30) & (clean_xyz[:, 1] >= 0.15) & (clean_xyz[:, 1] <= 1.80)
    pts_cand = clean_xyz[mask_band]
    if len(pts_cand) == 0:
        pts_cand = clean_xyz

    min_x = float(math.floor(np.min(clean_xyz[:, 0]) / cell_size) * cell_size)
    max_x = float(math.ceil(np.max(clean_xyz[:, 0]) / cell_size) * cell_size)
    min_z = float(math.floor(np.min(clean_xyz[:, 2]) / cell_size) * cell_size)
    max_z = float(math.ceil(np.max(clean_xyz[:, 2]) / cell_size) * cell_size)

    width = int(round((max_x - min_x) / cell_size))
    height = int(round((max_z - min_z) / cell_size))
    origin = [min_x, min_z]
    total_cells = width * height

    counts, _, _ = np.histogram2d(
        pts_cand[:, 2], pts_cand[:, 0],
        bins=[height, width],
        range=[[min_z, max_z], [min_x, max_x]],
    )

    spawn = np.array([float(c0[0]), 1.45, float(c0[2])])

    # Sequence of candidate occupancy parameter configurations (Requirement 3a)
    eval_candidates = [
        (inflation_radius_m, min_points, f"initial config (inflation={inflation_radius_m:.2f}m, min_pts={min_points})"),
        (0.20, min_points, f"fallback 1: inflation 0.20 m (min_pts={min_points})"),
        (0.20, 8, "fallback 2: inflation 0.20 m and min point count 8"),
        (inflation_radius_m, 8, f"fallback 3: min point count 8 (inflation={inflation_radius_m:.2f}m)"),
    ]

    chosen_inflation = inflation_radius_m
    chosen_min_pts = min_points
    chosen_raw_occ = None
    chosen_inflated_occ = None
    passed_candidate_name = None
    final_cam_pct = 0.0
    final_cam_cnt = 0
    final_sp_free = False

    for cur_infl, cur_pts, cand_name in eval_candidates:
        r_occ = (counts >= cur_pts).astype(np.uint8)
        d_cells = scipy.ndimage.distance_transform_edt(r_occ == 0)
        i_occ = (d_cells * cell_size <= cur_infl).astype(np.uint8)

        pct, cnt, sp_f = check_free_cells(i_occ, min_x, min_z, cell_size, width, height, cams, spawn)
        if pct >= 98.0 and sp_f:
            chosen_inflation = cur_infl
            chosen_min_pts = cur_pts
            chosen_raw_occ = r_occ
            chosen_inflated_occ = i_occ
            passed_candidate_name = cand_name
            final_cam_pct = pct
            final_cam_cnt = cnt
            final_sp_free = sp_f
            break

    if chosen_inflated_occ is None:
        # Fallback to candidate with highest free camera percentage
        chosen_inflation = 0.20
        chosen_min_pts = 8
        chosen_raw_occ = (counts >= 8).astype(np.uint8)
        d_cells = scipy.ndimage.distance_transform_edt(chosen_raw_occ == 0)
        chosen_inflated_occ = (d_cells * cell_size <= 0.20).astype(np.uint8)
        final_cam_pct, final_cam_cnt, final_sp_free = check_free_cells(
            chosen_inflated_occ, min_x, min_z, cell_size, width, height, cams, spawn
        )
        passed_candidate_name = "best effort (inflation 0.20m, min points 8)"

    # Assert Requirement 3a: at least 98% of camera positions and spawn fall in FREE cells
    assert final_cam_pct >= 98.0 and final_sp_free, (
        f"Assertion failed: Only {final_cam_pct:.2f}% cameras in free cells (spawn free: {final_sp_free}) < 98.0% threshold."
    )

    print(
        f"  Occupancy free cell check: {final_cam_cnt}/{len(cams)} ({final_cam_pct:.2f}%) cameras in FREE cells, "
        f"Spawn in free cell: {final_sp_free} [Fixed/Passed via: {passed_candidate_name}]"
    )

    # Save the values that worked in configs/default.yaml
    update_config_occupancy_settings(config_path, chosen_inflation, chosen_min_pts)

    # Compute Walkable Region (Requirement 3b)
    # Continuous interpolated path along camera poses
    dense_path = []
    xz = cams[:, [0, 2]]
    for i in range(len(xz) - 1):
        if i == pass_split_idx - 1:
            dense_path.append(xz[i])
            continue
        pA, pB = xz[i], xz[i + 1]
        seg_len = float(np.linalg.norm(pB - pA))
        n_samples = max(1, int(math.ceil(seg_len / 0.05)))
        for t in np.linspace(0.0, 1.0, n_samples, endpoint=False):
            dense_path.append(pA * (1.0 - t) + pB * t)
    dense_path.append(xz[-1])
    dense_arr = np.array(dense_path)

    cam_tree = KDTree(dense_arr)
    col_coords = min_x + (np.arange(width) + 0.5) * cell_size
    row_coords = min_z + (np.arange(height) + 0.5) * cell_size
    grid_z, grid_x = np.meshgrid(row_coords, col_coords, indexing="ij")
    cell_centers_xz = np.column_stack([grid_x.ravel(), grid_z.ravel()])
    dists, _ = cam_tree.query(cell_centers_xz)
    walkable_grid = (dists <= walkable_radius_m).astype(np.uint8).reshape(height, width)

    # Calculate percentages (Requirement 3c)
    occ_count = int(chosen_inflated_occ.sum())
    free_count = int(total_cells - occ_count)
    walkable_count = int(walkable_grid.sum())

    occ_pct = (occ_count / total_cells) * 100.0
    free_pct = (free_count / total_cells) * 100.0
    walkable_pct = (walkable_count / total_cells) * 100.0

    print("\n  2D Occupancy Grid Statistics:")
    print(f"    - Grid Resolution  : {width} cols x {height} rows ({total_cells:,} total cells, {cell_size:.2f} m/cell)")
    print(f"    - Occupied Cells   : {occ_count:,} ({occ_pct:.2f}%)")
    print(f"    - Free Cells       : {free_count:,} ({free_pct:.2f}%)")
    print(f"    - Walkable Cells   : {walkable_count:,} ({walkable_pct:.2f}% within {walkable_radius_m:.2f} m of camera path)")

    # Save occupancy.json with walkable array (Requirement 3b)
    occupancy_data = {
        "origin": origin,
        "cell_size": cell_size,
        "width": width,
        "height": height,
        "axes": "x = X, row = Z",
        "axes_convention": {
            "col_axis": "X (East/West)",
            "row_axis": "Z (North/South)",
            "data_ordering": "row-major (index = row * width + col)",
        },
        "min_points_per_cell": chosen_min_pts,
        "inflation_radius_m": chosen_inflation,
        "walkable_radius_m": walkable_radius_m,
        "height_band_m": [0.15, 1.80],
        "occupied_cells_raw": int(chosen_raw_occ.sum()),
        "occupied_cells_inflated": occ_count,
        "free_cells": free_count,
        "walkable_cells": walkable_count,
        "total_cells": total_cells,
        "occupied_pct": round(occ_pct, 2),
        "free_pct": round(free_pct, 2),
        "walkable_pct": round(walkable_pct, 2),
        "camera_free_pct": round(final_cam_pct, 2),
        "spawn_free": final_sp_free,
        "data": chosen_inflated_occ.flatten().tolist(),
        "walkable": walkable_grid.flatten().tolist(),
    }

    json_path = output_dir / "occupancy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(occupancy_data, f, indent=2)

    # Render and save occupancy.png with walkable light outline (Requirement 3d)
    fig, ax = plt.subplots(figsize=(8, 10), dpi=180)
    cmap = matplotlib.colors.ListedColormap(["#12151B", "#E63946"])
    # origin='upper' with extent=[min_x, max_x, max_z, min_z]: Z increases strictly DOWN
    ax.imshow(chosen_inflated_occ, extent=[min_x, max_x, max_z, min_z], origin="upper", cmap=cmap, alpha=0.90)

    # Walkable area light outline (Requirement 3d)
    ax.contour(
        walkable_grid,
        levels=[0.5],
        extent=[min_x, max_x, max_z, min_z],
        origin="upper",
        colors=["#70D6FF"],
        linewidths=[1.6],
        linestyles=["--"],
        zorder=3,
    )
    ax.plot([], [], color="#70D6FF", linestyle="--", linewidth=1.6, label=f"Walkable Area (radius <= {walkable_radius_m:.1f}m)")

    # Overlay camera path in colour (Pass 1 eye in Cyan, Pass 2 chest in Magenta)
    if pass_split_idx > 0 and pass_split_idx < len(cams):
        ax.plot(cams[:pass_split_idx, 0], cams[:pass_split_idx, 2], color="#00E5FF", linewidth=2.0, label="Pass 1: Eye level (~1.45m)", zorder=4)
        ax.plot(cams[pass_split_idx:, 0], cams[pass_split_idx:, 2], color="#FF007F", linewidth=2.0, label="Pass 2: Chest level (~1.02m)", zorder=4)
    elif len(cams) > 0:
        ax.plot(cams[:, 0], cams[:, 2], color="#00E5FF", linewidth=2.0, label="Camera Trajectory", zorder=4)

    # Mark spawn point
    spawn_x, spawn_z = float(c0[0]), float(c0[2])
    ax.scatter(
        [spawn_x], [spawn_z],
        color="#00FF66", s=180, marker="*",
        zorder=10, edgecolor="black", linewidth=1.2,
        label=f"Spawn: ({spawn_x:.2f}, 1.45, {spawn_z:.2f})",
    )
    # Spawn heading arrow
    arrow_len = 0.55
    ax.arrow(
        spawn_x, spawn_z,
        arrow_len * -math.sin(spawn_yaw), arrow_len * -math.cos(spawn_yaw),
        color="#00FF66", width=0.04, head_width=0.18, head_length=0.15,
        zorder=11, length_includes_head=True,
    )

    # Coordinate styling: Z points strictly DOWN the image
    ax.set_ylim(max_z, min_z)
    ax.set_xlim(min_x, max_x)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.25, color="white")
    ax.set_facecolor("#111318")
    fig.patch.set_facecolor("#111318")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax.set_title(
        f"Namma Space — 2D Occupancy Grid ({scene})\nZ points DOWN the image | 0.1m cell, {chosen_inflation:.2f}m inflation | Spawn Yaw: {math.degrees(spawn_yaw):+.1f}°",
        color="white", fontsize=11, pad=12,
    )
    ax.set_xlabel("X (m) [East →]", color="white", fontsize=10)
    ax.set_ylabel("Z (m) [South ↓]", color="white", fontsize=10)
    ax.legend(loc="upper right", facecolor="#1E222A", edgecolor="#444", labelcolor="white", fontsize=8.5)

    plt.tight_layout()
    png_path = output_dir / "occupancy.png"
    plt.savefig(png_path, facecolor=fig.get_facecolor(), dpi=180)
    plt.close(fig)

    stats = {
        "width": width,
        "height": height,
        "total_cells": total_cells,
        "occupied_cells": occ_count,
        "free_cells": free_count,
        "walkable_cells": walkable_count,
        "occupied_pct": occ_pct,
        "free_pct": free_pct,
        "walkable_pct": walkable_pct,
        "camera_free_pct": final_cam_pct,
        "spawn_free": final_sp_free,
        "inflation_radius_m": chosen_inflation,
        "min_points_per_cell": chosen_min_pts,
        "walkable_radius_m": walkable_radius_m,
    }

    return json_path, png_path, stats


def postprocess_scene(
    scene: str,
    input_ply: Path,
    output_dir: Path,
    config_path: str = "configs/default.yaml",
    max_splats: int = 0,
    crop_margin_m: float = 4.0,
    walkable_radius_m: float = 1.5,
    min_opacity: float = 0.05,
    max_scale_m: float = 0.50,
    occupancy_cell_size: float = 0.10,
    occupancy_min_points: int = 5,
    occupancy_inflation_m: float = 0.25,
    copy_to_viewer: bool = True,
    run_compression: bool = True,
) -> bool:
    print("=" * 78)
    print("  NAMMA SPACE -- STAGE 5: POST-PROCESSING, OCCUPANCY & COMPRESSION")
    print("=" * 78)
    print(f"  Scene               : {scene}")
    print(f"  Input Raw Model     : {input_ply}")
    print(f"  Output Directory    : {output_dir}")
    print(f"  Max Splats Limit    : {max_splats:,} (0 = unlimited)" if max_splats > 0 else "  Max Splats Limit    : unlimited (0 = keep all)")
    print(f"  Configured Margin   : +{crop_margin_m:.2f} m")
    print(f"  Walkable Radius     : {walkable_radius_m:.2f} m")
    print(f"  Min Opacity         : {min_opacity:.2f}")
    print(f"  Max Splat Scale     : {max_scale_m:.2f} m")
    print("-" * 78)

    if not input_ply.is_file():
        print(f"[FAIL] Raw splat file not found: {input_ply}", file=sys.stderr)
        print("  Actionable advice: Run pipeline/train_splat.py --scene <scene> first.", file=sys.stderr)
        return False

    cfg = load_config(config_path)

    # 1. Determine Floor and Ceiling Elevations from measurements.json
    floor_y = 0.0
    meas_path = Path("data") / scene / "measurements.json"
    meas_data = {}
    if meas_path.is_file():
        try:
            with open(meas_path, "r", encoding="utf-8") as mf:
                meas_data = json.load(mf)
        except Exception as e:
            print(f"  [!] Warning loading measurements.json: {e}")

    ceiling_y = float(meas_data.get("ceiling_height_m", cfg.get("alignment", {}).get("ceiling_height_m", 4.10)))
    y_min_crop = floor_y - 0.30  # -0.30 m: preserves floor layers (Requirement 1c)
    y_max_crop = ceiling_y + 0.30  # ceiling_y + 0.30 m: preserves ceiling layers (Requirement 1c)

    print(f"  Floor Level (y=0)   : {floor_y:+.2f} m (Crop min Y: {y_min_crop:+.2f} m)")
    print(f"  Ceiling Level       : {ceiling_y:+.2f} m (Crop max Y: {y_max_crop:+.2f} m)")

    # 2. Load Camera Trajectory for Bounding Box Crop & Spawn Calculation
    transforms_path = Path("data") / scene / "colmap_aligned" / "transforms.json"
    if not transforms_path.is_file():
        transforms_path = Path("data") / scene / "transforms.json"

    if not transforms_path.is_file():
        print(f"[FAIL] Aligned transforms.json not found at {transforms_path}", file=sys.stderr)
        print("  Actionable advice: Run pipeline/align_scene.py --scene <scene> first.", file=sys.stderr)
        return False

    with open(transforms_path, "r", encoding="utf-8") as tf_f:
        transforms_data = json.load(tf_f)

    cams_list = []
    p1_cams_cnt = 0
    p2_cams_cnt = 0
    pass_split_idx = 222

    frames_rep_path = Path("data") / scene / "frames_report.json"
    if frames_rep_path.is_file():
        try:
            with open(frames_rep_path, "r", encoding="utf-8") as fr_f:
                vids = json.load(fr_f).get("videos", [])
                if vids:
                    pass_split_idx = vids[0].get("kept_after_duplicate", vids[0].get("raw_extracted", 222))
        except Exception:
            pass

    for frame in transforms_data.get("frames", []):
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        cams_list.append(mat[:3, 3])
        fpath = frame["file_path"].lower()
        fname = Path(fpath).name
        fnum = int("".join(filter(str.isdigit, fname))) if any(c.isdigit() for c in fname) else 0
        if "video2" in fpath or ("video1" not in fpath and fnum > pass_split_idx):
            p2_cams_cnt += 1
        else:
            p1_cams_cnt += 1

    cams = np.array(cams_list, dtype=np.float64)
    if len(cams) == 0:
        print("[FAIL] No camera frames found in transforms.json.", file=sys.stderr)
        return False

    pass_split = p1_cams_cnt if p1_cams_cnt > 0 else pass_split_idx

    # Requirement 1: Camera path bounding box + crop_margin_m
    cam_min = np.min(cams, axis=0)
    cam_max = np.max(cams, axis=0)
    cam_crop_x_min = float(cam_min[0] - crop_margin_m)
    cam_crop_x_max = float(cam_max[0] + crop_margin_m)
    cam_crop_z_min = float(cam_min[2] - crop_margin_m)
    cam_crop_z_max = float(cam_max[2] + crop_margin_m)

    # Requirement 1b: Intersected with robust extents of reference points (p0.5-p99.5, track >= 4, error <= 1.0) + 1.0 m
    points3d_path = Path("data") / scene / "colmap_aligned" / "points3D.txt"
    if not points3d_path.is_file():
        points3d_path = Path("data") / scene / "points3D.txt"

    ref_extents = load_robust_reference_extents(points3d_path)
    if ref_extents is not None:
        ref_p005, ref_p995, n_robust = ref_extents
        ref_crop_x_min = float(ref_p005[0] - 1.0)
        ref_crop_x_max = float(ref_p995[0] + 1.0)
        ref_crop_z_min = float(ref_p005[2] - 1.0)
        ref_crop_z_max = float(ref_p995[2] + 1.0)

        # Intersection
        crop_x_min = max(cam_crop_x_min, ref_crop_x_min)
        crop_x_max = min(cam_crop_x_max, ref_crop_x_max)
        crop_z_min = max(cam_crop_z_min, ref_crop_z_min)
        crop_z_max = min(cam_crop_z_max, ref_crop_z_max)
        print(f"  Camera Footprint      : X=[{cam_min[0]:+.2f}, {cam_max[0]:+.2f}], Z=[{cam_min[2]:+.2f}, {cam_max[2]:+.2f}] m")
        print(f"  Cam BBox Margin (+{crop_margin_m:.1f}m): X=[{cam_crop_x_min:+.2f}, {cam_crop_x_max:+.2f}], Z=[{cam_crop_z_min:+.2f}, {cam_crop_z_max:+.2f}] m")
        print(f"  Robust Ref Extents (+1m): X=[{ref_crop_x_min:+.2f}, {ref_crop_x_max:+.2f}], Z=[{ref_crop_z_min:+.2f}, {ref_crop_z_max:+.2f}] m ({n_robust:,} points)")
        print(f"  Intersected Crop Box  : X=[{crop_x_min:+.2f}, {crop_x_max:+.2f}], Z=[{crop_z_min:+.2f}, {crop_z_max:+.2f}] m")
    else:
        crop_x_min, crop_x_max = cam_crop_x_min, cam_crop_x_max
        crop_z_min, crop_z_max = cam_crop_z_min, cam_crop_z_max
        print(f"  Camera BBox Margin (+{crop_margin_m:.1f}m): X=[{crop_x_min:+.2f}, {crop_x_max:+.2f}], Z=[{crop_z_min:+.2f}, {crop_z_max:+.2f}] m")

    # 3. Load Raw PLY Model
    print("\n[Step 1/6] Loading raw trained Gaussian Splat...")
    t0 = time.time()
    raw_bytes = input_ply.stat().st_size
    raw_mb = raw_bytes / (1024 * 1024)
    n_raw, props, offset = parse_ply_header(input_ply)
    dt = np.dtype(props)
    arr = np.fromfile(input_ply, dtype=dt, count=n_raw, offset=offset)
    print(f"  Loaded {n_raw:,} Gaussians ({raw_mb:.1f} MB) in {time.time() - t0:.2f}s")

    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)

    if "opacity" in arr.dtype.names:
        raw_op = arr["opacity"].astype(np.float64)
        sig_op = 1.0 / (1.0 + np.exp(-raw_op))
    else:
        sig_op = np.ones(n_raw)
        raw_op = np.zeros(n_raw)

    scale_names = ["scale_0", "scale_1", "scale_2"]
    if all(s in arr.dtype.names for s in scale_names):
        log_scales = np.column_stack([arr[s] for s in scale_names]).astype(np.float64)
        max_scales = np.max(np.exp(log_scales), axis=1)
    else:
        log_scales = np.zeros((n_raw, 3))
        max_scales = np.zeros(n_raw)

    # 4. Multi-Stage Filtering with Splat Counts Printed After Every Stage (Requirement 1)
    print("\n[Step 2/6] Applying sequential filters (Crop margins, Opacity, Scale, SOR)...")
    print(f"  {'Filter Stage':<38} | {'Before':<10} | {'After':<10} | {'Pruned':<10} | {'Pruned %'}")
    print("  " + "-" * 78)

    # Stage 1: Y Height Clamp [floor_y - 0.3, ceiling_y + 0.3]
    mask_y = (xyz[:, 1] >= y_min_crop) & (xyz[:, 1] <= y_max_crop)
    count_after_y = int(mask_y.sum())
    pruned_y = n_raw - count_after_y
    pct_y = (pruned_y / n_raw) * 100
    print(f"  {f'1. Y Clamp [{y_min_crop:+.2f}m, {y_max_crop:+.2f}m]':<38} | {n_raw:<10,} | {count_after_y:<10,} | {pruned_y:<10,} | {pct_y:5.1f}%")

    # Stage 2: Intersected Camera Path & Robust Ref XZ Crop
    mask_xz = (
        (xyz[:, 0] >= crop_x_min) & (xyz[:, 0] <= crop_x_max) &
        (xyz[:, 2] >= crop_z_min) & (xyz[:, 2] <= crop_z_max)
    )
    mask_stage2 = mask_y & mask_xz
    count_after_xz = int(mask_stage2.sum())
    pruned_xz = count_after_y - count_after_xz
    pct_xz = (pruned_xz / count_after_y) * 100 if count_after_y else 0
    print(f"  {'2. Intersected XZ BBox Crop':<38} | {count_after_y:<10,} | {count_after_xz:<10,} | {pruned_xz:<10,} | {pct_xz:5.1f}%")

    # Stage 3: Opacity Filter (sig_op >= min_opacity)
    mask_op = sig_op >= min_opacity
    mask_stage3 = mask_stage2 & mask_op
    count_after_op = int(mask_stage3.sum())
    pruned_op = count_after_xz - count_after_op
    pct_op = (pruned_op / count_after_xz) * 100 if count_after_xz else 0
    print(f"  {f'3. Min Opacity (>= {min_opacity:.2f})':<38} | {count_after_xz:<10,} | {count_after_op:<10,} | {pruned_op:<10,} | {pct_op:5.1f}%")

    # Stage 4: Scale Filter (max_scales <= max_scale_m)
    mask_sc = max_scales <= max_scale_m
    mask_stage4 = mask_stage3 & mask_sc
    count_after_sc = int(mask_stage4.sum())
    pruned_sc = count_after_op - count_after_sc
    pct_sc = (pruned_sc / count_after_op) * 100 if count_after_op else 0
    print(f"  {f'4. Max Scale (<= {max_scale_m:.2f}m)':<38} | {count_after_op:<10,} | {count_after_sc:<10,} | {pruned_sc:<10,} | {pct_sc:5.1f}%")

    # Stage 5: Open3D Statistical Outlier Removal on Splat Centres (nb_neighbors 20, std_ratio 2.5)
    stage4_indices = np.where(mask_stage4)[0]
    stage4_xyz = xyz[mask_stage4]
    if o3d is not None and len(stage4_xyz) > 20:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stage4_xyz)
        _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        clean_indices = stage4_indices[inlier_indices]
    else:
        clean_indices = stage4_indices

    n_after_sor = len(clean_indices)
    pruned_sor = count_after_sc - n_after_sor
    pct_sor = (pruned_sor / count_after_sc) * 100 if count_after_sc else 0
    print(f"  {'5. Statistical Outlier Removal (SOR)':<38} | {count_after_sc:<10,} | {n_after_sor:<10,} | {pruned_sor:<10,} | {pct_sor:5.1f}%")

    # Stage 6: Max Splats Importance Ranking / Budget
    # importance = sigmoid(opacity) * volume, where volume = exp(sum of the three log-scales)
    # Never drop by position, so the scene is not cut off anywhere.
    count_before_budget = n_after_sor
    if max_splats > 0 and n_after_sor > max_splats:
        sig_op_retained = sig_op[clean_indices]
        log_scales_retained = log_scales[clean_indices]
        sum_log_scales = np.sum(log_scales_retained, axis=1)
        volumes = np.exp(sum_log_scales)
        importance = sig_op_retained * volumes
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

        # Rank in descending order of importance: keep top N
        top_k_retained_idx = np.argsort(-importance)[:max_splats]
        clean_indices = np.sort(clean_indices[top_k_retained_idx])

        n_clean = len(clean_indices)
        pruned_budget = count_before_budget - n_clean
        pct_budget = (pruned_budget / count_before_budget) * 100
        print(f"  {f'6. Importance Ranking (top {max_splats:,})':<38} | {count_before_budget:<10,} | {n_clean:<10,} | {pruned_budget:<10,} | {pct_budget:5.1f}%")
        print(f"  [Max Splats Budget Filter] Count before: {count_before_budget:,} -> Count after: {n_clean:,} (pruned {pruned_budget:,} lowest-importance splats)")
    else:
        n_clean = n_after_sor
        label = "6. Importance Ranking (unlimited)" if max_splats == 0 else f"6. Importance Ranking (budget {max_splats:,} >= {n_after_sor:,})"
        print(f"  {label:<38} | {count_before_budget:<10,} | {n_clean:<10,} | {'0':<10} |   0.0%")
        if max_splats > 0:
            print(f"  [Max Splats Budget Filter] Count before: {count_before_budget:,} <= max_splats {max_splats:,} (retained all {n_clean:,})")
        else:
            print(f"  [Max Splats Budget Filter] max_splats = 0 (unlimited): retained all {n_clean:,} splats")
    print("  " + "-" * 78)

    total_pruned = n_raw - n_clean
    total_reduction_pct = (total_pruned / n_raw) * 100
    print(f"  Retained High-Fidelity Gaussians     : {n_clean:,} ({n_clean/n_raw*100:.1f}%)")
    print(f"  Total Splats Pruned                  : {total_pruned:,} ({total_reduction_pct:.1f}% reduction)")

    clean_mask = np.zeros(n_raw, dtype=bool)
    clean_mask[clean_indices] = True

    clean_arr = arr[clean_mask]
    clean_xyz = xyz[clean_mask]
    clean_sig_op = sig_op[clean_mask]

    # Floor layer counts audit
    floor_layer_raw = int(np.sum((xyz[:, 1] >= -0.15) & (xyz[:, 1] <= 0.15)))
    floor_layer_clean = int(np.sum((clean_xyz[:, 1] >= -0.15) & (clean_xyz[:, 1] <= 0.15)))
    subfloor_band_raw = int(np.sum((xyz[:, 1] >= -0.30) & (xyz[:, 1] < -0.05)))
    subfloor_band_clean = int(np.sum((clean_xyz[:, 1] >= -0.30) & (clean_xyz[:, 1] < -0.05)))

    print("\n  Floor Layer Splat Counts:")
    print(f"    - Height band [-0.15, +0.15] m : Raw = {floor_layer_raw:,} | Clean = {floor_layer_clean:,}")
    print(f"    - Sub-floor band [-0.30, -0.05)m: Raw = {subfloor_band_raw:,} | Clean = {subfloor_band_clean:,} (Preserved)")

    # 5. Export Clean PLY
    print("\n[Step 3/6] Exporting scene_clean.ply and web scene.splat...")
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_ply_path = output_dir / "scene_clean.ply"
    write_clean_ply(clean_ply_path, clean_arr, props)
    clean_bytes = clean_ply_path.stat().st_size
    clean_mb = clean_bytes / (1024 * 1024)
    print(f"  [+] Saved clean PLY : {clean_ply_path} ({clean_mb:.1f} MB)")

    # 6. Export Web .splat Binary (32 bytes per Gaussian)
    sh_dc_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    if all(s in arr.dtype.names for s in sh_dc_names):
        sh_dc = np.column_stack([clean_arr[s] for s in sh_dc_names]).astype(np.float32)
    else:
        sh_dc = np.zeros((n_clean, 3), dtype=np.float32)

    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(r in arr.dtype.names for r in rot_names):
        rotations = np.column_stack([clean_arr[r] for r in rot_names]).astype(np.float32)
    else:
        rotations = np.zeros((n_clean, 4), dtype=np.float32)
        rotations[:, 0] = 1.0

    splat_bin_path = output_dir / "scene.splat"
    splat_bytes = write_splat_binary(
        output_path=splat_bin_path,
        xyz=clean_xyz,
        scales=log_scales[clean_mask],
        sh_dc=sh_dc,
        opacities=raw_op[clean_mask],
        rotations=rotations,
    )
    splat_mb = splat_bytes / (1024 * 1024)
    total_compression_ratio = raw_bytes / splat_bytes
    print(f"  [+] Saved Web .splat: {splat_bin_path} ({splat_mb:.2f} MB, 32 bytes/splat)")
    print(f"      Size Reduction  : {raw_mb:.1f} MB -> {splat_mb:.1f} MB ({total_compression_ratio:.1f}x compression)")

    # 7. Convert to Compressed .spz and verify quality (Requirement 4)
    spz_path = output_dir / "scene.spz"
    spz_success = False
    spz_mb = 0.0
    compression_metrics = {}
    if run_compression:
        print("\n[Step 4/6] Compressing Gaussian splat with @playcanvas/splat-transform...")
        spz_ok, spz_err = convert_to_spz(clean_ply_path, spz_path)
        if spz_ok and spz_path.is_file():
            spz_success = True
            spz_bytes = spz_path.stat().st_size
            spz_mb = spz_bytes / (1024 * 1024)
            spz_ratio = raw_bytes / spz_bytes
            print(f"  [+] Saved Web .spz  : {spz_path} ({spz_mb:.2f} MB)")
            print(f"      Compression     : {raw_mb:.1f} MB -> {spz_mb:.2f} MB ({spz_ratio:.1f}x compression vs raw PLY, {splat_mb/spz_mb:.1f}x vs .splat)")

            # Requirement 4: Evaluate compression quality by converting back to PLY
            print("\n  Evaluating .spz compression quality (converting back to PLY)...")
            try:
                compression_metrics = evaluate_spz_quality(clean_ply_path, spz_path)
                print(f"    - Splat count difference : {compression_metrics['splat_count_diff']}")
                print(f"    - Max centre error       : {compression_metrics['max_centre_error_mm']:.3f} mm ({compression_metrics['max_centre_error_m']:.6f} m)")
                print(f"    - Mean centre error      : {compression_metrics['mean_centre_error_mm']:.3f} mm ({compression_metrics['mean_centre_error_m']:.6f} m)")
                print(f"    - Mean opacity error     : {compression_metrics['mean_opacity_error']:.6f}")
                print("\n  Format Sizes & Compression Quality Summary:")
                print(f"    - Raw input PLY    : {raw_mb:.2f} MB ({n_raw:,} splats)")
                print(f"    - Clean filtered PLY: {clean_mb:.2f} MB ({n_clean:,} splats, {raw_mb/clean_mb:.2f}x compression)")
                print(f"    - Web .splat binary : {splat_mb:.2f} MB ({raw_mb/splat_mb:.2f}x compression, 32 bytes/splat)")
                print(f"    - Compressed .spz   : {spz_mb:.2f} MB ({raw_mb/spz_mb:.2f}x compression vs raw PLY, {splat_mb/spz_mb:.2f}x vs .splat)")
            except Exception as eq_err:
                print(f"    [!] Error during SPZ quality evaluation: {eq_err}")
        else:
            print(f"  [!] .spz conversion notice: {spz_err}")
            print("      Keeping standard 32-byte scene.splat.")

    # 8. Compute Spawn Position & Direction of Travel
    c0 = cams[0]
    spawn_pos = [float(c0[0]), float(floor_y + 1.45), float(c0[2])]

    # Direction of travel over first 10 cameras (c0 to c9)
    n_sample_cams = min(10, len(cams))
    disp_travel = cams[n_sample_cams - 1] - cams[0]
    # Three.js yaw convention: rotation about +Y, CCW seen from above, yaw 0 looks along -Z
    spawn_yaw = float(math.atan2(-disp_travel[0], -disp_travel[2]))
    spawn_yaw_deg = float(math.degrees(spawn_yaw))

    # 9. Compute Bounding Box of Cleaned Splats
    bounds = {
        "min": [float(np.min(clean_xyz[:, 0])), float(np.min(clean_xyz[:, 1])), float(np.min(clean_xyz[:, 2]))],
        "max": [float(np.max(clean_xyz[:, 0])), float(np.max(clean_xyz[:, 1])), float(np.max(clean_xyz[:, 2]))],
    }

    # 10. Generate outputs/<scene>/scene.json
    scene_json_data = {
        "scene": scene,
        "units": "metres",
        "coordinate_convention": "Right-handed, Y-up, metric (floor at y = 0.0)",
        "bounds": bounds,
        "floor_y": floor_y,
        "ceiling_y": ceiling_y,
        "spawn_position": spawn_pos,
        "spawn_yaw": round(spawn_yaw, 4),
        "spawn_yaw_deg": round(spawn_yaw_deg, 2),
        "spawn_yaw_convention": "Rotation about +Y in radians, counter-clockwise seen from above; yaw 0 looks along -Z (Three.js convention)",
        "splat_count": n_clean,
        "splat_file": "scene.splat",
        "splat_files": {
            "splat": "scene.splat",
            "spz": "scene.spz" if spz_success else None,
        },
        "splat_rotation_euler": [0.0, 0.0, 0.0],
        "splat_rotation_note": "The data is already Y-up and metric, so start with [0, 0, 0]; viewer must verify it.",
    }

    scene_json_path = output_dir / "scene.json"
    with open(scene_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_json_data, f, indent=2)
    print(f"  [+] Saved scene manifest: {scene_json_path}")

    # 11. Generate 2D Occupancy Grid (Requirement 3)
    print("\n[Step 5/6] Building 2D occupancy grid and top-down map...")
    occ_json_path, occ_png_path, occ_stats = build_and_save_occupancy(
        output_dir=output_dir,
        clean_xyz=clean_xyz,
        clean_sig_op=clean_sig_op,
        cams=cams,
        c0=c0,
        spawn_yaw=spawn_yaw,
        scene=scene,
        pass_split_idx=pass_split,
        cell_size=occupancy_cell_size,
        min_points=occupancy_min_points,
        inflation_radius_m=occupancy_inflation_m,
        walkable_radius_m=walkable_radius_m,
        config_path=config_path,
    )
    print(f"  [+] Saved occupancy JSON: {occ_json_path}")
    print(f"  [+] Saved occupancy PNG : {occ_png_path}")

    # 12. Copy Web Assets to viewer/public/scene/
    if copy_to_viewer:
        viewer_scene_dir = Path("viewer") / "public" / "scene"
        viewer_scene_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(scene_json_path, viewer_scene_dir / "scene.json")
        shutil.copy2(splat_bin_path, viewer_scene_dir / "scene.splat")
        if spz_success and spz_path.is_file():
            shutil.copy2(spz_path, viewer_scene_dir / "scene.spz")
        shutil.copy2(occ_json_path, viewer_scene_dir / "occupancy.json")
        shutil.copy2(occ_png_path, viewer_scene_dir / "occupancy.png")
        print(f"  [+] Copied web assets to: {viewer_scene_dir}")

        legacy_splat = Path("viewer") / "public" / "scene.splat"
        if legacy_splat.is_file():
            legacy_splat.unlink()
            print(f"  [-] Removed legacy duplicate: {legacy_splat}")

    # 13. Verify Clean Model Coordinate Frame
    print("\n[Step 6/6] Verifying coordinate integrity on clean model...")
    verify_cmd = [
        sys.executable, "scripts/verify_splat_frame.py",
        "--scene", scene,
        "--splat-ply", str(clean_ply_path),
        "--scene-json", str(scene_json_path),
        "--span-tolerance", "15.0",
        "--output-json", str(output_dir / "clean_frame_verification.json"),
    ]
    verify_res = subprocess.run(verify_cmd)

    postprocess_report = {
        "scene": scene,
        "input_ply": str(input_ply.resolve()),
        "max_splats": max_splats,
        "raw_gaussian_count": n_raw,
        "clean_gaussian_count": n_clean,
        "splats_pruned": total_pruned,
        "pruning_reduction_pct": round(total_reduction_pct, 2),
        "raw_ply_mb": round(raw_mb, 2),
        "clean_ply_mb": round(clean_mb, 2),
        "web_splat_mb": round(splat_mb, 2),
        "spz_mb": round(spz_mb, 2) if spz_success else None,
        "compression_factor_splat": round(total_compression_ratio, 2),
        "compression_factor_spz": round(raw_bytes / spz_path.stat().st_size, 2) if spz_success else None,
        "compression_quality": compression_metrics,
        "floor_crop_m": y_min_crop,
        "ceiling_crop_m": y_max_crop,
        "crop_margin_m": crop_margin_m,
        "crop_bounds": {
            "x": [round(crop_x_min, 3), round(crop_x_max, 3)],
            "y": [round(y_min_crop, 3), round(y_max_crop, 3)],
            "z": [round(crop_z_min, 3), round(crop_z_max, 3)],
        },
        "min_opacity": min_opacity,
        "max_scale_m": max_scale_m,
        "occupancy_stats": occ_stats,
        "floor_layer_splats_raw": floor_layer_raw,
        "floor_layer_splats_clean": floor_layer_clean,
        "subfloor_band_splats_clean": subfloor_band_clean,
        "verification_exit_code": verify_res.returncode,
    }

    report_path = output_dir / "postprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(postprocess_report, f, indent=2)
    print(f"\n  [+] Saved post-processing report: {report_path}")

    print("=" * 78)
    print("  POST-PROCESSING COMPLETE: CLEAN ASSETS READY IN outputs/ AND viewer/public/scene/")
    print(f"  Raw PLY     : {raw_mb:.1f} MB ({n_raw:,} splats)")
    print(f"  Clean PLY   : {clean_mb:.1f} MB ({n_clean:,} splats)")
    print(f"  Web .splat  : {splat_mb:.2f} MB (32 bytes/splat)")
    if spz_success:
        print(f"  Web .spz    : {spz_mb:.2f} MB ({raw_mb/spz_mb:.1f}x compression)")
    print("=" * 78)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 6: Gaussian Splat Post-Processing, Occupancy Grid & Compression."
    )
    ap.add_argument("--scene", default="home", help="Scene name (subfolder of data/)")
    ap.add_argument("--config", default="configs/default.yaml", help="Configuration YAML path")
    ap.add_argument("--input", default=None, help="Input raw PLY model path")
    ap.add_argument("--output-dir", default=None, help="Output directory for cleaned models")
    ap.add_argument("--crop-margin", type=float, default=None, help="Bounding box margin in metres (default from config: 4.0)")
    ap.add_argument("--cam-margin", type=float, default=None, help="Alias for --crop-margin")
    ap.add_argument("--walkable-radius", type=float, default=None, help="Radius around camera path for walkable area (default: 1.5)")
    ap.add_argument("--min-opacity", type=float, default=None, help="Minimum opacity threshold (default: 0.05)")
    ap.add_argument("--max-scale", type=float, default=None, help="Maximum splat semi-axis scale (default: 0.50)")
    ap.add_argument("--cell-size", type=float, default=None, help="Occupancy grid cell size in metres (default: 0.10)")
    ap.add_argument("--min-points", type=int, default=None, help="Min points for occupied cell (default: 5)")
    ap.add_argument("--inflation", type=float, default=None, help="Obstacle inflation radius in metres (default: 0.25)")
    ap.add_argument("--max-splats", type=int, default=None, help="Maximum number of splats to keep after filtering (default from config: 0 = unlimited)")
    ap.add_argument("--no-copy-viewer", action="store_true", help="Do not copy web assets to viewer/public/scene/")
    ap.add_argument("--no-spz", action="store_true", help="Skip .spz conversion")

    args = ap.parse_args()
    cfg = load_config(args.config)
    pp_cfg = cfg.get("postprocess", {})

    scene = args.scene
    input_ply = Path(args.input) if args.input else (Path("outputs") / scene / "scene_raw.ply")
    output_dir = Path(args.output_dir) if args.output_dir else (Path("outputs") / scene)

    # Resolution of margins: --crop-margin takes precedence, then --cam-margin, then config crop_margin_m, then default 4.0
    crop_margin = args.crop_margin
    if crop_margin is None:
        crop_margin = args.cam_margin
    if crop_margin is None:
        crop_margin = float(pp_cfg.get("crop_margin_m", pp_cfg.get("cam_margin_m", 4.0)))

    walkable_radius = args.walkable_radius if args.walkable_radius is not None else float(pp_cfg.get("walkable_radius_m", 1.5))
    min_opacity = args.min_opacity if args.min_opacity is not None else float(pp_cfg.get("min_opacity", 0.05))
    max_scale = args.max_scale if args.max_scale is not None else float(pp_cfg.get("max_scale_m", 0.50))
    cell_size = args.cell_size if args.cell_size is not None else float(pp_cfg.get("occupancy_cell_size_m", 0.10))
    min_points = args.min_points if args.min_points is not None else int(pp_cfg.get("occupancy_min_points", 5))
    inflation = args.inflation if args.inflation is not None else float(pp_cfg.get("occupancy_inflation_m", 0.25))
    max_splats = args.max_splats if args.max_splats is not None else int(pp_cfg.get("max_splats", 0))

    success = postprocess_scene(
        scene=scene,
        input_ply=input_ply,
        output_dir=output_dir,
        config_path=args.config,
        max_splats=max_splats,
        crop_margin_m=crop_margin,
        walkable_radius_m=walkable_radius,
        min_opacity=min_opacity,
        max_scale_m=max_scale,
        occupancy_cell_size=cell_size,
        occupancy_min_points=min_points,
        occupancy_inflation_m=inflation,
        copy_to_viewer=(not args.no_copy_viewer),
        run_compression=(not args.no_spz),
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
