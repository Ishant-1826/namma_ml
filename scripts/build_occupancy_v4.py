#!/usr/bin/env python3
"""
scripts/build_occupancy_v4.py  --  Stage Q3: Occupancy v4 from Q2 Cleaned Splats.

1. Loads aligned cameras, Q2 clean splats (outputs/home/v3/scene_clean.ply), and SfM points.
2. Renders 160x90 depth and alpha maps on GPU.
3. Marches free-space rays and back-projects depth surfaces (slab y in [0.20, 1.40] m).
4. Fuses occupied evidence:
   - Depth surface hits seen by >= 2 cameras.
   - SfM points with track >= 3, err <= 1.0 px.
   - Q2 splat centres with opacity >= 0.50, scale <= 0.15 m.
5. Camera Corridor Carving:
   - Camera positions & inter-camera segments (<0.5m) carve 0.20m horizontal cylinder as FREE.
   - Overrides occupied evidence and logs any with >= 10 supporting voxels.
6. Noise cleanup:
   - Removes occupied components < 4 cells OR thinner than 2 cells wide (morphological opening),
     unless they touch a component of >= 20 cells.
7. Inflation & Total Body Radius:
   - Inflation: 0.07 m.
   - Player circle: 0.08 m.
   - Total body radius: 0.15 m.
8. Computes side-by-side comparison (v2, v3, v4), camera walkability %, blocked-step %,
   narrowest passable gap, and 3 narrowest corridors.
9. Exports to outputs/home/occupancy_v4/:
   - occupancy.json, occupancy.png, occupancy_debug.png, occupancy_report.json.
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
from scripts.build_occupancy_v3_1 import compute_segment_distance


def main():
    scene = "home"
    out_dir = Path("outputs/home/occupancy_v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STAGE Q3: OCCUPANCY v4 FROM Q2 CLEANED SPLATS")
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
        R_gl = mat[:3, :3]
        R_cv = np.column_stack([R_gl[:, 0], -R_gl[:, 1], -R_gl[:, 2]])
        c2w_rots_list.append(R_cv)

    cams = np.array(cams_list, dtype=np.float64)
    c2w_rots = np.array(c2w_rots_list, dtype=np.float64)
    num_cams = len(cams)
    print(f"Loaded {num_cams} cameras from {tf_path}")

    # 2. Load Q2 Cleaned Splats (outputs/home/v3/scene_clean.ply)
    clean_ply_path = Path("outputs/home/v3/scene_clean.ply")
    if not clean_ply_path.is_file():
        print(f"[ERROR] Q2 splats missing at {clean_ply_path}")
        sys.exit(1)

    n_splats, props, offset = parse_ply_header(clean_ply_path)
    dt = np.dtype(props)
    data = np.fromfile(clean_ply_path, dtype=dt, count=n_splats, offset=offset)
    print(f"Loaded {n_splats:,} Q2 clean splats from {clean_ply_path}")

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
    print(f"Loaded {len(sfm_pts):,} high-confidence SfM points (track>=3, err<=1.0 px)")

    # 4. Initialize 3D Voxel Grid
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

    # 6. Dense Ray Carving & Surface Occupied Hits
    print("\n[Step 2/5] Marching free space & back-projecting depth surfaces...")
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

        # Surface hits
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

    # 7. Fuse Occupied Evidence
    print("\n[Step 3/5] Fusing occupied evidence & applying floater override...")
    # (a) Surface hits seen by >= 2 cameras
    surface_occupied = (surface_evidence >= 2)

    # (b) SfM points in slab
    sfm_occupied = np.zeros((n_y, height, width), dtype=bool)
    in_slab_sfm = (
        (sfm_pts[:, 1] >= slab_y_min) & (sfm_pts[:, 1] < slab_y_max) &
        (sfm_pts[:, 0] >= min_x) & (sfm_pts[:, 0] < max_x) &
        (sfm_pts[:, 2] >= min_z) & (sfm_pts[:, 2] < max_z)
    )
    if np.any(in_slab_sfm):
        pts_s = sfm_pts[in_slab_sfm]
        vx_sfm = np.clip(np.floor((pts_s[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
        vy_sfm = np.clip(np.floor((pts_s[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
        vz_sfm = np.clip(np.floor((pts_s[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
        sfm_occupied[vy_sfm, vz_sfm, vx_sfm] = True

    # (c) Q2 Clean Splat centres (opacity >= 0.50, scale <= 0.15m)
    splat_filter_mask = (splat_op_np >= 0.50) & (splat_sc_np <= 0.15)
    splat_pts_filt = splat_xyz_np[splat_filter_mask]

    splat_occupied = np.zeros((n_y, height, width), dtype=bool)
    in_slab_splat = (
        (splat_pts_filt[:, 1] >= slab_y_min) & (splat_pts_filt[:, 1] < slab_y_max) &
        (splat_pts_filt[:, 0] >= min_x) & (splat_pts_filt[:, 0] < max_x) &
        (splat_pts_filt[:, 2] >= min_z) & (splat_pts_filt[:, 2] < max_z)
    )
    if np.any(in_slab_splat):
        pts_sp = splat_pts_filt[in_slab_splat]
        vx_sp = np.clip(np.floor((pts_sp[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
        vy_sp = np.clip(np.floor((pts_sp[:, 1] - slab_y_min) / cell_size).astype(int), 0, n_y - 1)
        vz_sp = np.clip(np.floor((pts_sp[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
        splat_occupied[vy_sp, vz_sp, vx_sp] = True

    # Combined candidate occupied voxels
    occupied_candidate = surface_occupied | sfm_occupied | splat_occupied
    print(f"  Candidate occupied voxels: {int(np.sum(occupied_candidate)):,}")

    # Free Space Override Rule: Free evidence >= 3 cameras overrides weak occupied evidence
    voxel_free = free_evidence >= 2
    strong_free = free_evidence >= 3
    floater_overrides = occupied_candidate & strong_free
    voxel_occupied_pre = occupied_candidate & (~floater_overrides)

    print(f"  Floaters overridden by >=3 camera line-of-sight: {int(np.sum(floater_overrides)):,}")
    print(f"  Pre-corridor occupied voxels: {int(np.sum(voxel_occupied_pre)):,}")

    # 8. Camera Corridor Carving (radius 0.20 m horizontal cylinder)
    print("\n[Step 4/5] Carving camera path & consecutive inter-camera corridors (r=0.20 m)...")
    cols_2d, rows_2d = np.meshgrid(np.arange(width), np.arange(height))
    grid_x = min_x + (cols_2d + 0.5) * cell_size
    grid_z = min_z + (rows_2d + 0.5) * cell_size

    # Distance to camera positions
    cam_dists_2d = np.full((height, width), 999.0, dtype=np.float64)
    for c in cams:
        d = np.sqrt((grid_x - c[0]) ** 2 + (grid_z - c[2]) ** 2)
        cam_dists_2d = np.minimum(cam_dists_2d, d)

    # Distance to inter-camera segments (< 0.5 m)
    corridor_dist_2d = cam_dists_2d.copy()
    for src in ["video1", "video2"]:
        src_frames = [f for f in tf_data["frames"] if src in f["file_path"]]
        src_cams = [np.array(f["transform_matrix"])[:3, 3] for f in src_frames]
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
    cand_vox_per_cell = np.sum(occupied_candidate, axis=0)
    pre_vox_per_cell = np.sum(voxel_occupied_pre, axis=0)

    overridden_cells_cand = (cand_vox_per_cell > 0) & cam_corridor_2d
    overridden_cells_pre = (pre_vox_per_cell > 0) & cam_corridor_2d
    overridden_cand_count = int(np.sum(overridden_cells_cand))
    overridden_pre_count = int(np.sum(overridden_cells_pre))

    print(f"  Corridor 2D cells (radius <= 0.20 m): {int(np.sum(cam_corridor_2d)):,}")
    print(f"  Overridden occupied-evidence cells (candidate): {overridden_cand_count}")
    print(f"  Overridden occupied-evidence cells (after floater override): {overridden_pre_count}")

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

    print(f"  Cells overridden by camera path with >= 10 supporting voxels: {len(cells_ge_10_cand)}")
    if len(cells_ge_10_cand) > 0:
        for item in cells_ge_10_cand:
            print(f"    -> Row={item['row']}, Col={item['col']} at ({item['x']}, {item['z']}): {item['supporting_voxels']} voxels")

    # Apply camera path override in body slab
    voxel_free[:, cam_corridor_2d] = True
    voxel_occupied = voxel_occupied_pre.copy()
    voxel_occupied[:, cam_corridor_2d] = False

    # 9. Collapse to 2D Occupancy Grid
    print("\n[Step 5/5] Collapsing slab to 2D occupancy & applying noise removal...")
    cell_occupied_raw = np.any(voxel_occupied, axis=0)
    free_ratio = np.sum(voxel_free, axis=0) / float(n_y)
    cell_known_free = (free_ratio >= 0.40) & (~cell_occupied_raw)

    # Re-enforce camera corridor
    cell_occupied_raw[cam_corridor_2d] = False
    cell_known_free[cam_corridor_2d] = True

    # Noise removal on occupied cells:
    # "remove occupied components smaller than 4 cells or thinner than 2 cells wide (morphological opening)
    # unless they touch a component of at least 20 cells."
    struct8 = np.ones((3, 3), dtype=bool)
    labeled_occ, num_comps = scipy.ndimage.label(cell_occupied_raw, structure=struct8)
    comp_sizes = np.bincount(labeled_occ.ravel())

    # Large components: >= 20 cells
    large_comp_ids = set(np.where(comp_sizes >= 20)[0])
    large_comp_ids.discard(0)  # 0 is background
    large_mask = np.isin(labeled_occ, list(large_comp_ids))

    cell_occupied = np.zeros_like(cell_occupied_raw)
    cell_occupied[large_mask] = True

    struct2x2 = np.ones((2, 2), dtype=bool)
    pruned_noise_cells = 0
    kept_touching_cells = 0

    for comp_id in range(1, num_comps + 1):
        if comp_id in large_comp_ids:
            continue
        comp_mask = (labeled_occ == comp_id)
        size = comp_sizes[comp_id]

        # Check if it touches a large component (1-cell neighbor)
        dilated_comp = scipy.ndimage.binary_dilation(comp_mask, structure=struct8)
        touches_large = np.any(dilated_comp & large_mask)
        if touches_large:
            cell_occupied[comp_mask] = True
            kept_touching_cells += size
            continue

        # If isolated from large components:
        # Check size < 4
        if size < 4:
            pruned_noise_cells += size
            continue

        # Check if thinner than 2 cells wide
        opened = scipy.ndimage.binary_opening(comp_mask, structure=struct2x2)
        if not np.any(opened):
            # Thinner than 2 cells wide throughout: remove!
            pruned_noise_cells += size
            continue

        # Keep solid components of >= 4 cells
        cell_occupied[comp_mask] = True

    print(f"  Occupied connected components: {num_comps}")
    print(f"  Large components (>=20 cells): {len(large_comp_ids)} ({int(np.sum(large_mask))} cells)")
    print(f"  Pruned noise cells (<4 cells or <2 wide isolated): {pruned_noise_cells} cells")
    print(f"  Kept touching sub-components: {kept_touching_cells} cells")
    print(f"  Occupied cells: {int(np.sum(cell_occupied_raw)):,} -> {int(np.sum(cell_occupied)):,}")

    # Re-evaluate known-free (pruned noise cells become known-free if free_ratio >= 0.40)
    cell_known_free = (free_ratio >= 0.40) & (~cell_occupied)
    cell_known_free[cam_corridor_2d] = True

    # 10. Inflation: 0.07 m
    inflation_m = 0.07
    player_radius_m = 0.08
    total_body_radius_m = round(inflation_m + player_radius_m, 2)  # 0.15 m

    # 4-connected single-cell dilation matches 0.07m buffer on 0.10m grid
    struct_infl = scipy.ndimage.generate_binary_structure(2, 1)
    cell_inflated = scipy.ndimage.binary_dilation(cell_occupied, structure=struct_infl, iterations=1) & (~cell_occupied)
    cell_inflated[cam_corridor_2d] = False

    # Cell state: 0 unknown, 1 free, 2 occupied, 3 inflated
    state_grid = np.zeros((height, width), dtype=np.uint8)
    state_grid[cell_known_free] = 1
    state_grid[cell_inflated] = 3
    state_grid[cell_occupied] = 2

    # Unknown ok radius: 0.30 m
    dist_to_free = scipy.ndimage.distance_transform_edt(~cell_known_free) * cell_size
    unknown_far = (state_grid == 0) & (dist_to_free > 0.30)

    # data grid: 1 = blocked, 0 = open
    data_grid = np.zeros((height, width), dtype=np.uint8)
    data_grid[cell_occupied] = 1
    data_grid[cell_inflated] = 1
    data_grid[unknown_far] = 1

    # Camera corridor cells must never be blocked by adjacent inflation
    data_grid[cam_corridor_2d] = 0

    # Walkable space: 4-neighbour flood fill from spawn position
    c0 = cams[0]
    spawn_col = int(np.clip(round((c0[0] - min_x) / cell_size), 0, width - 1))
    spawn_row = int(np.clip(round((c0[2] - min_z) / cell_size), 0, height - 1))

    walk_candidates = (data_grid == 0) & cell_known_free
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled_walk, num_features = scipy.ndimage.label(walk_candidates, structure=struct4)
    spawn_label = labeled_walk[spawn_row, spawn_col]
    walkable_grid = (labeled_walk == spawn_label).astype(np.uint8) if spawn_label > 0 else np.zeros((height, width), dtype=np.uint8)

    # Clearance grid (distance to blocked cells: occupied or inflated)
    blocked_cells = (cell_occupied | cell_inflated)
    clearance_grid = scipy.ndimage.distance_transform_edt(~blocked_cells) * cell_size

    # Camera walkability verification
    cam_cols = np.clip(np.round((cams[:, 0] - min_x) / cell_size).astype(int), 0, width - 1)
    cam_rows = np.clip(np.round((cams[:, 2] - min_z) / cell_size).astype(int), 0, height - 1)
    cam_walkable_arr = walkable_grid[cam_rows, cam_cols] == 1
    cam_walkable_cnt = int(np.sum(cam_walkable_arr))
    cam_walkable_pct = float(cam_walkable_cnt / num_cams * 100.0)
    spawn_walkable = bool(walkable_grid[spawn_row, spawn_col] == 1)

    # Blocked-step share along camera path
    total_steps = 0
    blocked_steps = 0
    step_size_sample = 0.05

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

    # Narrowest Passable Gap and 3 Narrowest Corridors
    # Walkable cells clearance >= player_radius_m (0.08m)
    walkable_clearance = np.where(walkable_grid == 1, clearance_grid, 0.0)
    passable_clearances = walkable_clearance[walkable_clearance >= player_radius_m]

    narrowest_passable_gap_m = float(np.min(passable_clearances) * 2.0) if len(passable_clearances) > 0 else 0.0

    # Find the 3 narrowest corridors using directional clearance ridges
    # A corridor ridge cell is a local maximum perpendicular to the passage direction
    # with minimum clearance along the passage
    ridge_candidates = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if walkable_grid[r, c] == 1:
                cl = clearance_grid[r, c]
                if cl < player_radius_m or cl > 0.80:
                    continue
                # Check horizontal ridge: local max across columns
                is_horiz_ridge = (cl >= clearance_grid[r, c - 1]) and (cl >= clearance_grid[r, c + 1])
                # Check vertical ridge: local max across rows
                is_vert_ridge = (cl >= clearance_grid[r - 1, c]) and (cl >= clearance_grid[r + 1, c])

                if is_horiz_ridge or is_vert_ridge:
                    wx = float(min_x + (c + 0.5) * cell_size)
                    wz = float(min_z + (r + 0.5) * cell_size)
                    passage_width_m = cl * 2.0
                    ridge_candidates.append((cl, passage_width_m, wx, wz, r, c))

    ridge_candidates.sort(key=lambda item: item[0])

    # Cluster candidates with >= 1.0 m spacing to pick 3 distinct corridors
    distinct_corridors = []
    for cl, w_m, wx, wz, r, c in ridge_candidates:
        too_close = False
        for _, _, ex_x, ex_z, _, _ in distinct_corridors:
            if math.hypot(wx - ex_x, wz - ex_z) < 1.0:
                too_close = True
                break
        if not too_close:
            distinct_corridors.append((cl, w_m, wx, wz, r, c))
            if len(distinct_corridors) == 3:
                break

    # If fewer than 3, pad with top candidates
    if len(distinct_corridors) < 3 and len(ridge_candidates) >= 3:
        for item in ridge_candidates:
            if item not in distinct_corridors:
                distinct_corridors.append(item)
                if len(distinct_corridors) == 3:
                    break

    print(f"\nNarrowest Passable Gap: {narrowest_passable_gap_m:.2f} m")
    print("Three Narrowest Passable Corridors:")
    for idx, (cl, w_m, wx, wz, r, c) in enumerate(distinct_corridors, 1):
        print(f"  {idx}. Position: ({wx:+.2f}, {wz:+.2f}) m | Clearance: {cl:.2f} m | Passage Width: {w_m:.2f} m (Row {r}, Col {c})")

    # 11. Side-by-Side Comparison (v2, v3, v4)
    # Load v2 & v3 numbers
    v2_stats = {"occ": 8.0, "infl": 13.5, "free": 43.4, "unknown": 48.6, "walk": 29.8, "cam_walk": 92.6, "blocked_step": 6.8}
    v3_stats = {"occ": 30.06, "infl": 28.61, "free": 31.42, "unknown": 22.92, "walk": 18.32, "cam_walk": 91.70, "blocked_step": 3.73}
    v3_1_stats = {"occ": 29.76, "infl": 28.50, "free": 31.71, "unknown": 22.63, "walk": 18.48, "cam_walk": 100.00, "blocked_step": 0.00}

    v4_stats = {
        "occ": round(float(occ_cnt / total_cells * 100.0), 2),
        "infl": round(float(infl_cnt / total_cells * 100.0), 2),
        "free": round(float(free_cnt / total_cells * 100.0), 2),
        "unknown": round(float(unknown_cnt / total_cells * 100.0), 2),
        "walk": round(float(walk_cnt / total_cells * 100.0), 2),
        "cam_walk": round(cam_walkable_pct, 2),
        "blocked_step": round(blocked_step_share_pct, 2),
    }

    print("\n" + "=" * 80)
    print("  SIDE-BY-SIDE OCCUPANCY COMPARISON (v2 vs v3 vs v4)")
    print("=" * 80)
    print(f"{'Metric':<32} | {'v2 (Legacy)':<12} | {'v3 (Dense)':<12} | {'v4 (Cleaned)':<12} | {'Target':<10}")
    print("-" * 80)
    print(f"{'Occupied Cells (%)':<32} | {v2_stats['occ']:<12.2f} | {v3_stats['occ']:<12.2f} | {v4_stats['occ']:<12.2f} | {'-':<10}")
    print(f"{'Inflated Cells (%)':<32} | {v2_stats['infl']:<12.2f} | {v3_stats['infl']:<12.2f} | {v4_stats['infl']:<12.2f} | {'-':<10}")
    print(f"{'Known-Free Cells (%)':<32} | {v2_stats['free']:<12.2f} | {v3_stats['free']:<12.2f} | {v4_stats['free']:<12.2f} | {'-':<10}")
    print(f"{'Unknown Cells (%)':<32} | {v2_stats['unknown']:<12.2f} | {v3_stats['unknown']:<12.2f} | {v4_stats['unknown']:<12.2f} | {'-':<10}")
    print(f"{'Walkable Space (%)':<32} | {v2_stats['walk']:<12.2f} | {v3_stats['walk']:<12.2f} | {v4_stats['walk']:<12.2f} | {'-':<10}")
    print(f"{'Camera Walkability (%)':<32} | {v2_stats['cam_walk']:<12.2f} | {v3_stats['cam_walk']:<12.2f} | {v4_stats['cam_walk']:<12.2f} | {'>= 99.00%':<10}")
    print(f"{'Camera Path Blocked-Step (%)':<32} | {v2_stats['blocked_step']:<12.2f} | {v3_stats['blocked_step']:<12.2f} | {v4_stats['blocked_step']:<12.2f} | {'<= 2.00%':<10}")
    print("=" * 80)

    # 12. Export JSON, PNGs, and Report
    print("\nExporting occupancy_v4 assets...")

    occ_data = {
        "scene": "home",
        "units": "metres",
        "origin": [round(min_x, 3), round(min_z, 3)],
        "cell_size": cell_size,
        "width": width,
        "height": height,
        "data": data_grid.ravel().tolist(),
        "walkable": walkable_grid.ravel().tolist(),
        "state": state_grid.ravel().tolist(),
        "clearance_m": [round(float(val), 3) for val in clearance_grid.ravel()],
        "params": {
            "version": "v4",
            "slab_y_min_m": slab_y_min,
            "slab_y_max_m": slab_y_max,
            "voxel_size_m": cell_size,
            "inflation_m": inflation_m,
            "player_radius_m": player_radius_m,
            "total_body_radius_m": total_body_radius_m,
            "narrowest_passable_gap_m": round(narrowest_passable_gap_m, 2),
            "narrowest_corridors": [
                {"rank": i, "position": [round(wx, 2), round(wz, 2)], "clearance_m": round(cl, 2), "width_m": round(w_m, 2)}
                for i, (cl, w_m, wx, wz, r, c) in enumerate(distinct_corridors, 1)
            ],
            "camera_walkable_share": round(cam_walkable_pct / 100.0, 4),
            "camera_path_blocked_step_share": round(blocked_step_share_pct / 100.0, 4),
        }
    }

    occ_json_path = out_dir / "occupancy.json"
    with open(occ_json_path, "w", encoding="utf-8") as f:
        json.dump(occ_data, f)
    print(f"  Exported {occ_json_path} ({occ_json_path.stat().st_size / 1024:.1f} KB)")

    # Render occupancy.png (+Z DOWN, camera paths in colour, spawn)
    occ_png_path = out_dir / "occupancy.png"
    spawn_pt = np.array([float(c0[0]), 1.45, float(c0[2])])
    params = {
        "inflation_m": inflation_m,
        "player_radius_m": player_radius_m,
        "total_body_radius_m": total_body_radius_m,
        "slab_y_min_m": slab_y_min,
        "slab_y_max_m": slab_y_max,
    }
    render_occupancy_png(
        occ_png_path,
        state_grid,
        walkable_grid,
        [min_x, min_z],
        cell_size,
        width,
        height,
        cams,
        spawn_pt,
        "home",
        params,
        pass_split_idx=222
    )
    print(f"  Exported {occ_png_path}")

    # Render occupancy_debug.png
    occ_debug_png_path = out_dir / "occupancy_debug.png"
    corridors_for_debug = [{"x": wx, "z": wz, "passable_width_m": w_m, "clearance_m": cl} for cl, w_m, wx, wz, r, c in distinct_corridors]
    render_occupancy_debug_png(
        occ_debug_png_path,
        clearance_grid,
        walkable_grid,
        [min_x, min_z],
        cell_size,
        width,
        height,
        cams,
        spawn_pt,
        "home",
        corridors_for_debug,
        pass_split_idx=222
    )
    print(f"  Exported {occ_debug_png_path}")

    # Write report
    report_data = {
        "version": "v4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splats_input": str(clean_ply_path),
        "total_cells": total_cells,
        "occupied_cells": occ_cnt,
        "occupied_pct": v4_stats["occ"],
        "inflated_cells": infl_cnt,
        "inflated_pct": v4_stats["infl"],
        "known_free_cells": free_cnt,
        "known_free_pct": v4_stats["free"],
        "unknown_cells": unknown_cnt,
        "unknown_pct": v4_stats["unknown"],
        "walkable_cells": walk_cnt,
        "walkable_pct": v4_stats["walk"],
        "camera_walkable_pct": cam_walkable_pct,
        "camera_walkable_target_met": cam_walkable_pct >= 99.0,
        "camera_path_blocked_step_pct": blocked_step_share_pct,
        "camera_path_blocked_target_met": blocked_step_share_pct <= 2.0,
        "narrowest_passable_gap_m": round(narrowest_passable_gap_m, 2),
        "narrowest_corridors": [
            {"rank": i, "position": [round(wx, 2), round(wz, 2)], "clearance_m": round(cl, 2), "width_m": round(w_m, 2)}
            for i, (cl, w_m, wx, wz, r, c) in enumerate(distinct_corridors, 1)
        ],
        "overridden_corridor_cells_candidate": overridden_cand_count,
        "overridden_corridor_cells_ge_10_voxels": cells_ge_10_cand,
        "side_by_side": {
            "v2": v2_stats,
            "v3": v3_stats,
            "v3_1": v3_1_stats,
            "v4": v4_stats,
        }
    }

    report_json_path = out_dir / "occupancy_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Exported {report_json_path}")

    print("\n[SUCCESS] Stage Q3: Occupancy v4 Generation Completed Successfully!")

if __name__ == "__main__":
    main()
