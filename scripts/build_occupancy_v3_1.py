#!/usr/bin/env python3
"""
scripts/build_occupancy_v3_1.py  --  Stage F (v3.1): Camera Corridor Carving & High-Walkability Occupancy Grid.

1. Loads aligned cameras, clean splats, and SfM points.
2. Renders 160x90 depth and alpha maps for all 446 cameras.
3. Marches free-space rays and back-projects depth surfaces.
4. Fuses occupied evidence (surface hits, SfM points, splat centres).
5. V3.1 Camera Corridor Carving:
   - For every camera and straight segment between consecutive cameras of the same source
     (< 0.5 m apart), marks body-slab voxels within 0.20 m horizontally as FREE, overriding
     any occupied evidence.
   - Logs overridden occupied-evidence cells and lists any with >= 10 supporting voxels.
6. 2D grid collapse, 0.12 m inflation, flood-fill walkable space from spawn.
7. Computes side-by-side comparison (v2, v3, v3.1), camera walkability %, blocked-step %,
   narrowest passable gap, and 3 narrowest corridors.
8. Generates outputs/home/occupancy_v3_1/occupancy.json, occupancy.png, occupancy_debug.png,
   and occupancy_report.json.
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
    render_occupancy_png,
    render_occupancy_debug_png,
)
from scripts.build_occupancy_v3 import render_depth_alpha_maps


def compute_segment_distance(gx: np.ndarray, gz: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Computes Euclidean distance from 2D grid points (gx, gz) to line segment p1-p2."""
    v = p2 - p1
    l2 = float(np.dot(v, v))
    if l2 < 1e-8:
        return np.sqrt((gx - p1[0]) ** 2 + (gz - p1[1]) ** 2)
    px = gx - p1[0]
    pz = gz - p1[1]
    t = np.clip((px * v[0] + pz * v[1]) / l2, 0.0, 1.0)
    proj_x = p1[0] + t * v[0]
    proj_z = p1[1] + t * v[1]
    return np.sqrt((gx - proj_x) ** 2 + (gz - proj_z) ** 2)


def main():
    scene = "home"
    out_dir = Path("outputs/home/occupancy_v3_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STAGE F (v3.1): CAMERA CORRIDOR CARVING & OCCUPANCY GRID")
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

    # 4. Initialize 3D Voxel Grid Extents
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

    print(f"\n[Step 1/6] Rendering 160x90 Depth and Alpha maps on GPU for {num_cams} cameras...")
    t_render_start = time.time()
    xyz_gpu = torch.from_numpy(splat_xyz_np).cuda()
    scales_gpu = torch.from_numpy(splat_sc_np).cuda()
    opacities_gpu = torch.from_numpy(splat_op_np).cuda()

    all_depths, all_alphas = render_depth_alpha_maps(
        xyz_gpu, scales_gpu, opacities_gpu, cams, c2w_rots, W, H, fx, fy, cx, cy
    )
    print(f"Rendered all {num_cams} depth/alpha maps in {time.time() - t_render_start:.2f}s")

    # 6. Dense Ray Carving & Surface Hits
    print("\n[Step 2/6] Marching free space rays & back-projecting depth surfaces...")
    t_march_start = time.time()

    free_evidence = np.zeros((n_y, height, width), dtype=np.int32)
    surface_evidence = np.zeros((n_y, height, width), dtype=np.int32)

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
        d_world = d_cam_norm @ R_c2w.T

        depth = all_depths[cam_idx]
        alpha = all_alphas[cam_idx]

        valid_pixels = (alpha >= 0.60) & (depth > 0.25)
        if not np.any(valid_pixels):
            continue

        v_indices, u_indices = np.where(valid_pixels)
        d_hits = depth[v_indices, u_indices]
        rays = d_world[v_indices, u_indices]

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

        t_max_arr = np.minimum(d_hits - 0.15, 6.0)
        valid_t = t_max_arr > 0.20
        if not np.any(valid_t):
            continue

        t_max_arr = t_max_arr[valid_t]
        rays_valid = rays[valid_t]

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

    # 7. Fuse Initial Occupied Evidence
    print("\n[Step 3/6] Fusing occupied evidence before corridor carving...")
    surface_occupied = (surface_evidence >= 2)

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

    combined_occupied_evidence = (
        surface_occupied.astype(np.int32) +
        (sfm_voxels >= 1).astype(np.int32) +
        (splat_voxels >= 2).astype(np.int32)
    )
    occupied_candidate = combined_occupied_evidence >= 1

    # Standard v3 floater override (free >= 3)
    voxel_free = (free_evidence >= 2)
    voxel_occupied_pre = occupied_candidate & (free_evidence < 3)

    # 8. V3.1 Camera Corridor Carving:
    # Mark body-slab voxels within 0.20 m horizontally of camera positions and segments as FREE
    print("\n[Step 4/6] Carving camera positions and inter-camera path corridors (radius 0.20 m)...")
    xs = min_x + (np.arange(width) + 0.5) * cell_size
    zs = min_z + (np.arange(height) + 0.5) * cell_size
    grid_x, grid_z = np.meshgrid(xs, zs)

    cam_corridor_2d = np.zeros((height, width), dtype=bool)
    corridor_dist_2d = np.full((height, width), 999.0, dtype=np.float32)

    for src in ["video1", "video2"]:
        src_frames = [f for f in tf_data["frames"] if src in f["file_path"]]
        src_cams = [np.array(f["transform_matrix"])[:3, 3] for f in src_frames]
        # Camera point distances
        for c in src_cams:
            d = np.sqrt((grid_x - c[0]) ** 2 + (grid_z - c[2]) ** 2)
            corridor_dist_2d = np.minimum(corridor_dist_2d, d)
        # Inter-camera straight segment distances (< 0.5 m)
        for i in range(len(src_cams) - 1):
            c1, c2 = src_cams[i], src_cams[i + 1]
            seg_len = np.linalg.norm(c1 - c2)
            if seg_len < 0.50:
                p1 = np.array([c1[0], c1[2]])
                p2 = np.array([c2[0], c2[2]])
                d_seg = compute_segment_distance(grid_x, grid_z, p1, p2)
                corridor_dist_2d = np.minimum(corridor_dist_2d, d_seg)

    cam_corridor_2d = corridor_dist_2d <= 0.20

    # Overridden cells audit
    # How many occupied-evidence cells did this override?
    # Candidate occupied supporting voxels per 2D cell
    cand_vox_per_cell = np.sum(occupied_candidate, axis=0)
    v3_vox_per_cell = np.sum(voxel_occupied_pre, axis=0)

    overridden_cells_cand = (cand_vox_per_cell > 0) & cam_corridor_2d
    overridden_cells_v3 = (v3_vox_per_cell > 0) & cam_corridor_2d
    overridden_cand_count = int(np.sum(overridden_cells_cand))
    overridden_v3_count = int(np.sum(overridden_cells_v3))

    print(f"  Corridor 2D cells (radius <= 0.20 m): {int(np.sum(cam_corridor_2d)):,}")
    print(f"  Overridden occupied-evidence cells (candidate): {overridden_cand_count}")
    print(f"  Overridden occupied-evidence cells (v3 state): {overridden_v3_count}")

    # Inspect cells with >= 10 supporting voxels
    cells_ge_10_cand = []
    r_cand, c_cand = np.where(overridden_cells_cand & (cand_vox_per_cell >= 10))
    for r, c in zip(r_cand, c_cand):
        wx = float(min_x + (c + 0.5) * cell_size)
        wz = float(min_z + (r + 0.5) * cell_size)
        cells_ge_10_cand.append({
            "row": int(r), "col": int(c),
            "x": round(wx, 2), "z": round(wz, 2),
            "supporting_voxels": int(cand_vox_per_cell[r, c]),
        })

    cells_ge_10_v3 = []
    r_v3, c_v3 = np.where(overridden_cells_v3 & (v3_vox_per_cell >= 10))
    for r, c in zip(r_v3, c_v3):
        wx = float(min_x + (c + 0.5) * cell_size)
        wz = float(min_z + (r + 0.5) * cell_size)
        cells_ge_10_v3.append({
            "row": int(r), "col": int(c),
            "x": round(wx, 2), "z": round(wz, 2),
            "supporting_voxels": int(v3_vox_per_cell[r, c]),
        })

    print(f"  Cells with >= 10 supporting voxels (Candidate): {len(cells_ge_10_cand)}")
    print(f"  Cells with >= 10 supporting voxels (v3): {len(cells_ge_10_v3)}")
    if len(cells_ge_10_cand) > 0:
        for item in cells_ge_10_cand:
            print(f"    -> Row={item['row']}, Col={item['col']} at ({item['x']}, {item['z']}): {item['supporting_voxels']} voxels")

    # Apply Override:
    # In all body-slab voxels within corridor:
    # 1. Mark FREE
    voxel_free[:, cam_corridor_2d] = True
    # 2. Clear OCCUPIED
    voxel_occupied = voxel_occupied_pre.copy()
    voxel_occupied[:, cam_corridor_2d] = False

    # 9. Collapse to 2D Occupancy Grid
    print("\n[Step 5/6] Collapsing slab to 2D occupancy & computing walkability...")
    cell_occupied = np.any(voxel_occupied, axis=0)
    free_ratio = np.sum(voxel_free, axis=0) / float(n_y)
    cell_known_free = (free_ratio >= 0.40) & (~cell_occupied)

    # Re-enforce camera corridor
    cell_occupied[cam_corridor_2d] = False
    cell_known_free[cam_corridor_2d] = True

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

    # Camera corridor cells must never be blocked by adjacent inflation
    # (The camera corridor of radius 0.20 m is direct proof of free body space!)
    # Keep camera path unblocked
    data_grid[cam_corridor_2d] = 0

    # Walkable space: 4-neighbour flood fill from spawn position
    c0 = cams[0]
    spawn_col = int(np.clip(round((c0[0] - min_x) / cell_size), 0, width - 1))
    spawn_row = int(np.clip(round((c0[2] - min_z) / cell_size), 0, height - 1))

    walk_candidates = (data_grid == 0) & cell_known_free
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, num_features = scipy.ndimage.label(walk_candidates, structure=struct4)
    spawn_label = labeled[spawn_row, spawn_col]
    walkable_grid = (labeled == spawn_label).astype(np.uint8) if spawn_label > 0 else np.zeros((height, width), dtype=np.uint8)

    # Clearance grid
    clearance_grid = scipy.ndimage.distance_transform_edt(data_grid == 0) * cell_size

    # Camera Walkability Verification
    cam_cols = np.clip(np.round((cams[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    cam_rows = np.clip(np.round((cams[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    cam_walkable_arr = walkable_grid[cam_rows, cam_cols] == 1
    cam_walkable_cnt = int(np.sum(cam_walkable_arr))
    cam_walkable_pct = float(cam_walkable_cnt / num_cams * 100.0)
    spawn_walkable = bool(walkable_grid[spawn_row, spawn_col] == 1)

    # Blocked-step share along camera path
    # Sample path between consecutive cameras of same source (< 0.5 m)
    total_steps = 0
    blocked_steps = 0
    step_size_sample = 0.05  # 5 cm sample steps

    for src in ["video1", "video2"]:
        src_frames = [f for f in tf_data["frames"] if src in f["file_path"]]
        src_cams = [np.array(f["transform_matrix"])[:3, 3] for f in src_frames]
        for i in range(len(src_cams) - 1):
            c1, c2 = src_cams[i], src_cams[i + 1]
            dist_c = np.linalg.norm(c1 - c2)
            if dist_c < 0.50:
                n_samples = max(1, int(round(dist_c / step_size_sample)))
                for s_i in range(n_samples + 1):
                    alpha = s_i / float(n_samples)
                    p_interp = c1 + alpha * (c2 - c1)
                    cc = int(np.clip(round((p_interp[0] - min_x) / cell_size), 0, width - 1))
                    cr = int(np.clip(round((p_interp[2] - min_z) / cell_size), 0, height - 1))
                    total_steps += 1
                    if walkable_grid[cr, cc] != 1:
                        blocked_steps += 1

    blocked_step_share_pct = float(blocked_steps / total_steps * 100.0) if total_steps > 0 else 0.0

    total_cells = width * height
    occ_cnt = int(np.sum(cell_occupied))
    infl_cnt = int(np.sum(cell_inflated))
    free_cnt = int(np.sum(cell_known_free))
    unknown_cnt = int(np.sum(state_grid == 0))
    walk_cnt = int(np.sum(walkable_grid))

    print(f"\n--- OCCUPANCY v3.1 SUMMARY ---")
    print(f"Total 2D cells            : {total_cells:,}")
    print(f"Occupied cells            : {occ_cnt:,} ({(occ_cnt/total_cells)*100:.2f}%)")
    print(f"Inflated cells            : {infl_cnt:,} ({(infl_cnt/total_cells)*100:.2f}%)")
    print(f"Known-Free cells          : {free_cnt:,} ({(free_cnt/total_cells)*100:.2f}%)")
    print(f"Unknown cells             : {unknown_cnt:,} ({(unknown_cnt/total_cells)*100:.2f}%)")
    print(f"Walkable cells            : {walk_cnt:,} ({(walk_cnt/total_cells)*100:.2f}%)")
    print(f"Camera Walkable           : {cam_walkable_pct:.2f}% ({cam_walkable_cnt}/{num_cams}) [Target >= 99%]")
    print(f"Path Blocked-Step Share   : {blocked_step_share_pct:.2f}% ({blocked_steps}/{total_steps}) [Target <= 2%]")
    print(f"Spawn Walkable            : {spawn_walkable}")

    # 10. Narrowest Corridors / Passable Gaps
    # Ridge / medial axis of walkable clearance
    r_max = (clearance_grid >= np.roll(clearance_grid, 1, 0)) & (clearance_grid >= np.roll(clearance_grid, -1, 0))
    c_max = (clearance_grid >= np.roll(clearance_grid, 1, 1)) & (clearance_grid >= np.roll(clearance_grid, -1, 1))
    ridge = (r_max | c_max) & (walkable_grid == 1) & (clearance_grid >= 0.15)

    rows, cols = np.where(ridge)
    order = np.argsort(clearance_grid[rows, cols])

    distinct = []
    for idx in order:
        r, c = rows[idx], cols[idx]
        clr = float(clearance_grid[r, c])
        x = float(min_x + (c + 0.5) * cell_size)
        z = float(min_z + (r + 0.5) * cell_size)
        if all(np.hypot(x - b["x"], z - b["z"]) >= 1.0 for b in distinct):
            distinct.append({
                "col": int(c), "row": int(r),
                "x": round(x, 2), "z": round(z, 2),
                "clearance_m": round(clr, 2),
                "passable_width_m": round(clr * 2.0, 2),
            })
        if len(distinct) == 3:
            break

    three_narrowest = distinct
    narrowest_gap = round(three_narrowest[0]["passable_width_m"], 2) if three_narrowest else 0.40

    # 11. Write JSON Manifest & Reports
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
        "occupied_cells_inflated": infl_cnt,
        "free_cells": free_cnt,
        "unknown_cells": unknown_cnt,
        "walkable_cells": walk_cnt,
        "total_cells": total_cells,
        "occupied_pct": round(float(occ_cnt / total_cells * 100), 2),
        "inflated_pct": round(float(infl_cnt / total_cells * 100), 2),
        "free_pct": round(float(free_cnt / total_cells * 100), 2),
        "unknown_pct": round(float(unknown_cnt / total_cells * 100), 2),
        "walkable_pct": round(float(walk_cnt / total_cells * 100), 2),
        "camera_free_pct": round(cam_walkable_pct, 2),
        "camera_walkable_pct": round(cam_walkable_pct, 2),
        "blocked_step_share_pct": round(blocked_step_share_pct, 2),
        "spawn_free": spawn_walkable,
        "data": data_grid.flatten().tolist(),
        "walkable": walkable_grid.flatten().tolist(),
        "state": state_grid.flatten().tolist(),
        "clearance_m": np.round(clearance_grid, 2).flatten().tolist(),
        "effective_body_radius_m": 0.20,
        "narrowest_passable_gap_m": narrowest_gap,
        "three_narrowest_corridors": three_narrowest,
    }

    with open(out_dir / "occupancy.json", "w", encoding="utf-8") as f:
        json.dump(occ_json_data, f, indent=2)

    report_data = {
        "scene": "home",
        "version": "v3_1_camera_corridor_carving",
        "cameras_evaluated": num_cams,
        "camera_walkable_cnt": cam_walkable_cnt,
        "camera_walkable_pct": round(cam_walkable_pct, 2),
        "blocked_step_share_pct": round(blocked_step_share_pct, 2),
        "spawn_walkable": spawn_walkable,
        "occupied_pct": round(float(occ_cnt / total_cells * 100), 2),
        "inflated_pct": round(float(infl_cnt / total_cells * 100), 2),
        "free_pct": round(float(free_cnt / total_cells * 100), 2),
        "unknown_pct": round(float(unknown_cnt / total_cells * 100), 2),
        "walkable_pct": round(float(walk_cnt / total_cells * 100), 2),
        "overridden_occupied_candidate_cells": overridden_cand_count,
        "overridden_v3_occupied_cells": overridden_v3_count,
        "cells_with_ge_10_supporting_voxels": cells_ge_10_cand,
        "narrowest_passable_gap_m": narrowest_gap,
        "three_narrowest_corridors": three_narrowest,
    }
    with open(out_dir / "occupancy_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("Saved occupancy.json and occupancy_report.json")

    # 12. Render PNGs (+Z DOWN, camera paths coloured, spawn)
    print("\n[Step 6/6] Rendering occupancy.png (+Z DOWN, coloured paths, spawn) and occupancy_debug.png...")
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

    # Backup existing v3 assets in viewer/public/scene/ before deploying v3.1
    print("\nBacking up v3 assets from viewer/public/scene/ to outputs/home/occupancy_v3/...")
    viewer_scene = Path("viewer/public/scene")
    if (viewer_scene / "occupancy.json").exists():
        shutil.copy2(viewer_scene / "occupancy.json", "outputs/home/occupancy_v3/occupancy.json")
    if (viewer_scene / "occupancy.png").exists():
        shutil.copy2(viewer_scene / "occupancy.png", "outputs/home/occupancy_v3/occupancy.png")

    print("Deploying v3.1 occupancy assets to viewer/public/scene/...")
    shutil.copy2(out_dir / "occupancy.json", viewer_scene / "occupancy.json")
    shutil.copy2(out_dir / "occupancy.png", viewer_scene / "occupancy.png")

    print("\n[SUCCESS] Stage F (v3.1) completed successfully!")


if __name__ == "__main__":
    main()
