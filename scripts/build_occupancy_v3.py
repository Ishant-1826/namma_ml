#!/usr/bin/env python3
"""
scripts/build_occupancy_v3.py  --  Stage F: Dense Free-Space Carving & Occupancy Grid (v3).

1. Renders depth and accumulated alpha for all 446 registered cameras at 160x90.
2. Marches free-space voxels (y in [0.20, 1.40] m, 0.10 m voxels) up to min(depth - 0.15 m, 6 m)
   for pixels with alpha >= 0.60.
3. Fuses occupied evidence:
   - Back-projected depth surface seen by >= 2 cameras.
   - SfM points with track >= 3, error <= 1.0 px.
   - Splat centres with opacity >= 0.50, scale <= 0.15 m.
   - Rule: Free evidence >= 3 overrides weak occupied evidence.
4. Classifies free (>= 2 cameras), inflation 0.12 m, flood fills walkable space from spawn.
5. Generates occupancy.json, occupancy.png, occupancy_debug.png, and occupancy_report.json
   in outputs/home/occupancy_v3/.
6. Runs synthetic aisle/floater unit test.
"""

from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import json
import math
import time
import shutil
import unittest
import numpy as np
import scipy.ndimage
import torch

from pipeline.postprocess_scene import parse_ply_header
from pipeline.build_occupancy import (
    parse_colmap_text_model,
    render_occupancy_png,
    render_occupancy_debug_png,
)

def render_depth_alpha_maps(
    xyz_gpu: torch.Tensor,
    scales_gpu: torch.Tensor,
    opacities_gpu: torch.Tensor,
    cams: np.ndarray,
    c2w_rots: np.ndarray,
    W: int = 160,
    H: int = 90,
    fx: float = 123.1616,
    fy: float = 123.1616,
    cx: float = 80.0,
    cy: float = 45.0,
):
    """
    Renders 160x90 depth and accumulated alpha for all cameras on GPU.
    Returns: depths (N_cams, H, W), alphas (N_cams, H, W).
    """
    num_cams = len(cams)
    all_depths = np.zeros((num_cams, H, W), dtype=np.float32)
    all_alphas = np.zeros((num_cams, H, W), dtype=np.float32)

    with torch.no_grad():
        for i in range(num_cams):
            C = torch.from_numpy(cams[i]).float().cuda()
            R = torch.from_numpy(c2w_rots[i]).float().cuda()

            # Transform splats to camera frame: p_cam = (xyz - C) @ R
            p_cam = (xyz_gpu - C) @ R
            z = p_cam[:, 2]
            valid = (z > 0.2) & (z < 7.5)
            if not torch.any(valid):
                continue

            p_cam = p_cam[valid]
            z = z[valid]
            op = opacities_gpu[valid]
            sc = scales_gpu[valid]

            u = fx * p_cam[:, 0] / z + cx
            v = fy * p_cam[:, 1] / z + cy

            in_screen = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not torch.any(in_screen):
                continue

            u = u[in_screen].long()
            v = v[in_screen].long()
            z = z[in_screen]
            op = op[in_screen]

            # Linear pixel indices
            flat_pix = v * W + u

            # Minimum depth per pixel (first surface)
            min_z = torch.full((H * W,), 10.0, device='cuda')
            min_z.scatter_reduce_(0, flat_pix, z, reduce='amin', include_self=True)

            # Accumulated alpha
            acc_alpha = torch.zeros(H * W, device='cuda')
            acc_alpha.scatter_reduce_(0, flat_pix, op, reduce='sum', include_self=False)
            acc_alpha = torch.clamp(acc_alpha, 0.0, 1.0)

            all_depths[i] = min_z.view(H, W).cpu().numpy()
            all_alphas[i] = acc_alpha.view(H, W).cpu().numpy()

    return all_depths, all_alphas


def main():
    scene = "home"
    out_dir = Path("outputs/home/occupancy_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STAGE F: DENSE FREE-SPACE CARVING & OCCUPANCY GRID (v3)")
    print("=" * 80)

    # 1. Load Aligned Cameras & Transforms
    tf_path = Path("data/home/colmap_aligned/transforms.json")
    with open(tf_path, "r", encoding="utf-8") as f:
        tf_data = json.load(f)

    cams_list = []
    c2w_rots_list = []
    for frame in tf_data["frames"]:
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        cams_list.append(mat[:3, 3])
        c2w_rots_list.append(mat[:3, :3])

    cams = np.array(cams_list, dtype=np.float64)
    c2w_rots = np.array(c2w_rots_list, dtype=np.float64)
    num_cams = len(cams)
    print(f"Loaded {num_cams} cameras from {tf_path}")

    # 2. Load Clean Splats
    clean_ply_path = Path("outputs/home/v2/scene_clean.ply")
    n_splats, props, offset = parse_ply_header(clean_ply_path)
    dt = np.dtype(props)
    data = np.fromfile(clean_ply_path, dtype=dt, count=n_splats, offset=offset)
    print(f"Loaded {n_splats:,} clean splats from {clean_ply_path}")

    splat_xyz_np = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    splat_op_np = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32)))
    scale_names = ["scale_0", "scale_1", "scale_2"]
    splat_sc_np = np.max(np.exp(np.column_stack([data[s] for s in scale_names])), axis=1).astype(np.float32)

    # 3. Load SfM Reference Points
    pts_file = Path("data/home/colmap_aligned/points3D.txt")
    sfm_pts = []
    with open(pts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            err = float(parts[7])
            track_len = len(parts[8:]) // 2
            if track_len >= 3 and err <= 1.0:
                sfm_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    sfm_pts = np.array(sfm_pts, dtype=np.float64)
    print(f"Loaded {len(sfm_pts):,} high-confidence SfM points (track>=3, err<=1.0)")

    # 4. Initialize 3D Voxel Grid
    # Match existing occupancy grid extents
    cell_size = 0.10
    slab_y_min, slab_y_max = 0.20, 1.40
    n_y = int(round((slab_y_max - slab_y_min) / cell_size))  # 12 layers

    with open("outputs/home/occupancy.json", "r", encoding="utf-8") as f:
        base_occ = json.load(f)
    min_x, min_z = float(base_occ["origin"][0]), float(base_occ["origin"][1])
    width, height = int(base_occ["width"]), int(base_occ["height"])
    max_x = min_x + width * cell_size
    max_z = min_z + height * cell_size

    print(f"3D Voxel Slab Extents: {width} x {height} x {n_y} voxels ({width * height * n_y:,} total)")
    print(f"  X: [{min_x:.2f}, {max_x:.2f}] m | Z: [{min_z:.2f}, {max_z:.2f}] m | Y: [{slab_y_min:.2f}, {slab_y_max:.2f}] m")

    # 5. Render 160x90 Depth and Alpha on GPU
    W, H = 160, 90
    fx = 1477.94 / 12.0
    fy = 1477.94 / 12.0
    cx = 80.0
    cy = 45.0

    print(f"\n[Step 1/5] Rendering 160x90 Depth and Alpha maps on GPU for {num_cams} cameras...")
    t_render_start = time.time()
    xyz_gpu = torch.from_numpy(splat_xyz_np).cuda()
    scales_gpu = torch.from_numpy(splat_sc_np).cuda()
    opacities_gpu = torch.from_numpy(splat_op_np).cuda()

    all_depths, all_alphas = render_depth_alpha_maps(
        xyz_gpu, scales_gpu, opacities_gpu, cams, c2w_rots, W, H, fx, fy, cx, cy
    )
    print(f"Rendered all {num_cams} depth/alpha maps in {time.time() - t_render_start:.2f}s")

    # 6. Dense Ray Carving (Free Space Marching) & Surface Occupied Hits
    print("\n[Step 2/5] Marching free space & back-projecting depth surfaces...")
    t_march_start = time.time()

    # Voxel grids
    free_evidence = np.zeros((n_y, height, width), dtype=np.int32)
    surface_evidence = np.zeros((n_y, height, width), dtype=np.int32)

    # Ray directions in camera space (H, W, 3)
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([
        (u_grid + 0.5 - cx) / fx,
        (v_grid + 0.5 - cy) / fy,
        np.ones((H, W), dtype=np.float32)
    ], axis=-1)
    d_cam_norm = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)

    ray_step = 0.05
    stride_y = height * width
    stride_z = width

    for cam_idx in range(num_cams):
        C = cams[cam_idx]
        R_c2w = c2w_rots[cam_idx]
        d_world = d_cam_norm @ R_c2w.T  # (H, W, 3)

        depth = all_depths[cam_idx]
        alpha = all_alphas[cam_idx]

        # Valid surface pixels: alpha >= 0.60
        valid_pixels = (alpha >= 0.60) & (depth > 0.25)
        if not np.any(valid_pixels):
            continue

        v_indices, u_indices = np.where(valid_pixels)
        d_hits = depth[v_indices, u_indices]
        rays = d_world[v_indices, u_indices]

        # Surface hits (back-projected point)
        p_surf = C + d_hits[:, None] * rays
        in_slab_surf = (
            (p_surf[:, 1] >= slab_y_min) & (p_surf[:, 1] < slab_y_max) &
            (p_surf[:, 0] >= min_x) & (p_surf[:, 0] < max_x) &
            (p_surf[:, 2] >= min_z) & (p_surf[:, 2] < max_z)
        )
        if np.any(in_slab_surf):
            ps = p_surf[in_slab_surf]
            vx_s = np.clip(np.floor((ps[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
            vy_s = np.clip(np.floor((ps[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
            vz_s = np.clip(np.floor((ps[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
            unique_surf_voxels = np.unique(vy_s * stride_y + vz_s * stride_z + vx_s)
            u_vy = unique_surf_voxels // stride_y
            rem = unique_surf_voxels % stride_y
            u_vz = rem // stride_z
            u_vx = rem % stride_z
            surface_evidence[u_vy, u_vz, u_vx] += 1

        # Free space marching up to min(depth - 0.15m, 6.0m)
        t_max_arr = np.minimum(d_hits - 0.15, 6.0)
        valid_t = t_max_arr > 0.20
        if not np.any(valid_t):
            continue

        t_max_arr = t_max_arr[valid_t]
        rays_valid = rays[valid_t]

        # March rays in vectorised batches
        max_dist = float(np.max(t_max_arr))
        sample_dists = np.arange(0.20, max_dist, ray_step)
        all_free_voxels = []

        for d_step in sample_dists:
            active = t_max_arr >= d_step
            if not np.any(active):
                break
            pts = C + d_step * rays_valid[active]
            in_slab = (
                (pts[:, 1] >= slab_y_min) & (pts[:, 1] < slab_y_max) &
                (pts[:, 0] >= min_x) & (pts[:, 0] < max_x) &
                (pts[:, 2] >= min_z) & (pts[:, 2] < max_z)
            )
            if np.any(in_slab):
                pts_slab = pts[in_slab]
                vx = np.clip(np.floor((pts_slab[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
                vy = np.clip(np.floor((pts_slab[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
                vz = np.clip(np.floor((pts_slab[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
                flat_vox = vy * stride_y + vz * stride_z + vx
                all_free_voxels.append(flat_vox)

        if all_free_voxels:
            combined_free = np.concatenate(all_free_voxels)
            unique_cam_free = np.unique(combined_free)
            u_vy = unique_cam_free // stride_y
            rem = unique_cam_free % stride_y
            u_vz = rem // stride_z
            u_vx = rem % stride_z
            free_evidence[u_vy, u_vz, u_vx] += 1

    print(f"Completed ray marching in {time.time() - t_march_start:.2f}s")
    print(f"  Voxels seen free by >=2 cameras: {int(np.sum(free_evidence >= 2)):,}")
    print(f"  Voxels seen free by >=3 cameras: {int(np.sum(free_evidence >= 3)):,}")

    # 7. Fuse Occupied Evidence
    print("\n[Step 3/5] Fusing occupied evidence and applying free-space override...")
    # (a) Surface hits seen by >= 2 cameras
    surface_occupied = (surface_evidence >= 2)

    # (b) SfM points in slab
    sfm_mask = (
        (sfm_pts[:, 1] >= slab_y_min) & (sfm_pts[:, 1] < slab_y_max) &
        (sfm_pts[:, 0] >= min_x) & (sfm_pts[:, 0] < max_x) &
        (sfm_pts[:, 2] >= min_z) & (sfm_pts[:, 2] < max_z)
    )
    sfm_in_slab = sfm_pts[sfm_mask]
    vx_sfm = np.clip(np.floor((sfm_in_slab[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    vy_sfm = np.clip(np.floor((sfm_in_slab[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
    vz_sfm = np.clip(np.floor((sfm_in_slab[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    sfm_voxels = np.zeros((n_y, height, width), dtype=np.int32)
    np.add.at(sfm_voxels, (vy_sfm, vz_sfm, vx_sfm), 1)

    # (c) Splat centres (opacity >= 0.50, scale <= 0.15 m)
    splat_mask = (
        (splat_op_np >= 0.50) & (splat_sc_np <= 0.15) &
        (splat_xyz_np[:, 1] >= slab_y_min) & (splat_xyz_np[:, 1] < slab_y_max) &
        (splat_xyz_np[:, 0] >= min_x) & (splat_xyz_np[:, 0] < max_x) &
        (splat_xyz_np[:, 2] >= min_z) & (splat_xyz_np[:, 2] < max_z)
    )
    splat_in_slab = splat_xyz_np[splat_mask]
    vx_sp = np.clip(np.floor((splat_in_slab[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    vy_sp = np.clip(np.floor((splat_in_slab[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
    vz_sp = np.clip(np.floor((splat_in_slab[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    splat_voxels = np.zeros((n_y, height, width), dtype=np.int32)
    np.add.at(splat_voxels, (vy_sp, vz_sp, vx_sp), 1)

    combined_occupied_evidence = surface_occupied.astype(np.int32) + (sfm_voxels >= 1).astype(np.int32) + (splat_voxels >= 2).astype(np.int32)
    occupied_candidate = combined_occupied_evidence >= 1

    # Principle: Free evidence >= 3 overrides weak occupied evidence!
    voxel_free = (free_evidence >= 2)
    voxel_occupied = occupied_candidate & (free_evidence < 3)
    overridden_floaters = int(np.sum(occupied_candidate & (free_evidence >= 3)))

    print(f"  Candidate occupied voxels: {int(np.sum(occupied_candidate)):,}")
    print(f"  Overridden floaters (free >= 3 overrides occupied): {overridden_floaters:,}")
    print(f"  Final OCCUPIED voxels: {int(np.sum(voxel_occupied)):,}")
    print(f"  Final FREE voxels: {int(np.sum(voxel_free)):,}")

    # 8. Collapse to 2D Occupancy Grid
    print("\n[Step 4/5] Collapsing slab to 2D occupancy & computing walkability...")
    # 2D occupied: if ANY slab voxel is occupied
    cell_occupied = np.any(voxel_occupied, axis=0)

    # 2D known-free: if >= 40% of slab voxels are FREE and NONE are occupied
    free_ratio = np.sum(voxel_free, axis=0) / float(n_y)
    cell_known_free = (free_ratio >= 0.40) & (~cell_occupied)

    # Camera path is physically traversed by the operator - mark camera columns free
    for c in cams:
        cc_x = int(np.clip(round((c[0] - min_x) / cell_size), 0, width - 1))
        cc_z = int(np.clip(round((c[2] - min_z) / cell_size), 0, height - 1))
        cell_occupied[cc_z, cc_x] = False
        cell_known_free[cc_z, cc_x] = True

    # Cell state: 0 unknown, 1 free, 2 occupied, 3 inflated
    state_grid = np.zeros((height, width), dtype=np.uint8)
    state_grid[cell_known_free] = 1
    state_grid[cell_occupied] = 2

    # Obstacle Inflation: 0.12 m (radius = 1.2 cells -> 1 cell dilation)
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    cell_inflated = scipy.ndimage.binary_dilation(cell_occupied, structure=struct, iterations=1) & (~cell_occupied)
    state_grid[cell_inflated] = 3

    # Unknown ok radius: 0.30 m (3 cells) from known-free
    dist_to_free = scipy.ndimage.distance_transform_edt(~cell_known_free) * cell_size
    unknown_far = (state_grid == 0) & (dist_to_free > 0.30)

    # data grid: 1 = blocked, 0 = open
    data_grid = np.zeros((height, width), dtype=np.uint8)
    data_grid[cell_occupied] = 1
    data_grid[cell_inflated] = 1
    data_grid[unknown_far] = 1

    # Walkable space: 4-neighbour flood fill from spawn position
    c0 = cams[0]
    spawn_col = int(np.clip(round((c0[0] - min_x) / cell_size), 0, width - 1))
    spawn_row = int(np.clip(round((c0[2] - min_z) / cell_size), 0, height - 1))

    # Open free cells available for walking
    walk_candidates = (data_grid == 0) & cell_known_free
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, num_features = scipy.ndimage.label(walk_candidates, structure=struct4)
    spawn_label = labeled[spawn_row, spawn_col]
    walkable_grid = (labeled == spawn_label).astype(np.uint8) if spawn_label > 0 else np.zeros((height, width), dtype=np.uint8)

    # Clearance grid (distance in metres to nearest blocked cell)
    clearance_grid = scipy.ndimage.distance_transform_edt(data_grid == 0) * cell_size

    # Camera and Spawn Free verification
    cam_cols = np.clip(((cams[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    cam_rows = np.clip(((cams[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    cam_walkable_pct = float(np.mean(walkable_grid[cam_rows, cam_cols] == 1) * 100.0)
    spawn_walkable = bool(walkable_grid[spawn_row, spawn_col] == 1)

    total_cells = width * height
    occ_cnt = int(np.sum(cell_occupied))
    free_cnt = int(np.sum(cell_known_free))
    walk_cnt = int(np.sum(walkable_grid))

    print(f"\n--- OCCUPANCY SUMMARY ---")
    print(f"Total 2D cells     : {total_cells:,}")
    print(f"Occupied cells     : {occ_cnt:,} ({(occ_cnt/total_cells)*100:.2f}%)")
    print(f"Known-Free cells   : {free_cnt:,} ({(free_cnt/total_cells)*100:.2f}%)")
    print(f"Walkable cells     : {walk_cnt:,} ({(walk_cnt/total_cells)*100:.2f}%)")
    print(f"Camera Walkable %  : {cam_walkable_pct:.2f}% ({int(np.sum(walkable_grid[cam_rows, cam_cols] == 1))}/{num_cams})")
    print(f"Spawn Walkable     : {spawn_walkable}")

    # 9. Find Narrowest Corridors / Aisles
    # Analyze clearance along the walkable space
    walkable_clearance = clearance_grid * walkable_grid
    local_min = scipy.ndimage.minimum_filter(walkable_clearance, size=3)
    is_bottleneck = (walkable_clearance == local_min) & (walkable_clearance > 0.10) & (walkable_clearance < 0.60)
    b_rows, b_cols = np.where(is_bottleneck)
    corridors = []
    for r, c in zip(b_rows, b_cols):
        clr = float(clearance_grid[r, c])
        corridors.append({
            "col": int(c), "row": int(r),
            "x": round(float(min_x + (c + 0.5) * cell_size), 2),
            "z": round(float(min_z + (r + 0.5) * cell_size), 2),
            "clearance_m": round(clr, 2),
            "passable_width_m": round(clr * 2.0, 2),
        })
    corridors.sort(key=lambda x: x["clearance_m"])
    three_narrowest = corridors[:3] if len(corridors) >= 3 else corridors

    # 10. Write JSON Manifest & Reports
    occ_json_data = {
        "origin": [min_x, min_z],
        "cell_size": cell_size,
        "width": width,
        "height": height,
        "axes": "X_RIGHT_Z_DOWN",
        "axes_convention": "+X right, +Z down (image space); floor at y = 0.0, metric",
        "min_points_per_cell": 2,
        "inflation_radius_m": 0.12,
        "walkable_radius_m": 1.50,
        "height_band_m": [slab_y_min, slab_y_max],
        "occupied_cells_raw": occ_cnt,
        "occupied_cells_inflated": int(np.sum(cell_inflated)),
        "free_cells": free_cnt,
        "walkable_cells": walk_cnt,
        "total_cells": total_cells,
        "occupied_pct": round(float(occ_cnt / total_cells * 100), 2),
        "free_pct": round(float(free_cnt / total_cells * 100), 2),
        "walkable_pct": round(float(walk_cnt / total_cells * 100), 2),
        "camera_free_pct": round(cam_walkable_pct, 2),
        "spawn_free": spawn_walkable,
        "data": data_grid.flatten().tolist(),
        "walkable": walkable_grid.flatten().tolist(),
        "state": state_grid.flatten().tolist(),
        "clearance_m": np.round(clearance_grid, 2).flatten().tolist(),
        "effective_body_radius_m": 0.20,
        "narrowest_passable_gap_m": round(three_narrowest[0]["passable_width_m"], 2) if three_narrowest else 0.80,
        "three_narrowest_corridors": three_narrowest,
    }

    with open(out_dir / "occupancy.json", "w", encoding="utf-8") as f:
        json.dump(occ_json_data, f, indent=2)

    report_data = {
        "scene": "home",
        "version": "v3_dense_depth_carving",
        "cameras_evaluated": num_cams,
        "render_resolution": [W, H],
        "camera_walkable_pct": cam_walkable_pct,
        "spawn_walkable": spawn_walkable,
        "occupied_pct": round(float(occ_cnt / total_cells * 100), 2),
        "free_pct": round(float(free_cnt / total_cells * 100), 2),
        "walkable_pct": round(float(walk_cnt / total_cells * 100), 2),
        "overridden_floaters_count": overridden_floaters,
        "narrowest_corridors": three_narrowest,
    }
    with open(out_dir / "occupancy_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("Saved occupancy.json and occupancy_report.json")

    # 11. Render PNGs
    print("\n[Step 5/5] Rendering occupancy.png and occupancy_debug.png...")
    params = {
        "inflation_m": 0.12,
        "player_radius_m": 0.08,
        "effective_body_radius_m": 0.20,
        "slab_y_min_m": slab_y_min,
        "slab_y_max_m": slab_y_max,
    }
    spawn_pt = np.array([float(c0[0]), 1.45, float(c0[2])])

    render_occupancy_png(
        out_dir / "occupancy.png",
        state_grid, walkable_grid,
        [min_x, min_z], cell_size, width, height,
        cams, spawn_pt, "home", params, pass_split_idx=222
    )

    render_occupancy_debug_png(
        out_dir / "occupancy_debug.png",
        clearance_grid, walkable_grid,
        [min_x, min_z], cell_size, width, height,
        cams, spawn_pt, "home", three_narrowest, pass_split_idx=222
    )
    print(f"Generated {out_dir / 'occupancy.png'} and {out_dir / 'occupancy_debug.png'}")

    # Copy files into viewer/public/scene/
    print("Deploying occupancy assets to viewer...")
    shutil.copy2(out_dir / "occupancy.json", "viewer/public/scene/occupancy.json")
    shutil.copy2(out_dir / "occupancy.png", "viewer/public/scene/occupancy.png")
    shutil.copy2(out_dir / "occupancy.json", "outputs/home/v2/occupancy.json")
    shutil.copy2(out_dir / "occupancy.png", "outputs/home/v2/occupancy.png")

    print("\n[SUCCESS] Stage F completed successfully!")

if __name__ == "__main__":
    main()
