#!/usr/bin/env python3
"""
scripts/build_q2_splats.py -- Q2: Floater Removal from Free Space on Raw Splats (outputs/home/v3/)

1. Loads raw splats (outputs/home/v2/scene_raw.ply) and aligned COLMAP cameras & SfM points.
2. Carves free space (0.10m voxels, y in [0.10, 2.20] m above floor) with:
   (i) Rays from each camera to its observed SfM points (stop 0.15m before point).
   (ii) Rays through rendered depth (alpha >= 0.60, up to min(depth - 0.15, 6m)).
   A camera counts once per voxel.
3. Removes floaters:
   - Rule 1 (Camera proximity): Splat centre within 0.25 m of any camera centre or inter-camera segment (<0.5m).
   - Rule 2 (Free space floater): Splat centre in voxel with free_cams >= 4 AND (max_scale > 0.08m OR opacity < 0.70).
4. Re-applies v2 filters:
   - Opacity >= 0.05
   - Max scale <= 0.25 m
   - Needle filter: ratio <= 12 OR max_scale <= 0.08 m
   - Y clamp: [-0.30, 4.40] m
   - Intersected XZ bounding box crop
   - Open3D Statistical Outlier Removal (SOR, nb=20, std=2.5)
5. Exports:
   - outputs/home/v3/scene_clean.ply
   - outputs/home/v3/scene.splat
   - outputs/home/v3/scene.json
   - outputs/home/v3/scene.spz (via splat-transform)
6. Runs frame verification and SPZ evaluation.
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
import subprocess
import numpy as np
import scipy.ndimage
from scipy.spatial import KDTree
import torch

try:
    import open3d as o3d
except ImportError:
    o3d = None

from pipeline.postprocess_scene import (
    parse_ply_header,
    load_robust_reference_extents,
    write_clean_ply,
    write_splat_binary,
    convert_to_spz,
    evaluate_spz_quality,
)
from scripts.build_occupancy_v3 import render_depth_alpha_maps

def main():
    print("=" * 80)
    print("  Q2: FLOATER REMOVAL FROM FREE SPACE ON RAW SPLATS (outputs/home/v3/)")
    print("=" * 80)

    scene = "home"
    raw_ply = Path("outputs/home/v2/scene_raw.ply")
    v3_dir = Path("outputs/home/v3")
    v3_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Camera Trajectory & Aligned Reference Points
    transforms_path = Path("data/home/colmap_aligned/transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        tf_data = json.load(f)

    frames = tf_data.get("frames", [])
    num_cams = len(frames)
    cams = []
    c2w_rots = []
    cam_sources = []

    for fr in frames:
        mat = np.array(fr["transform_matrix"], dtype=np.float64)
        cams.append(mat[:3, 3])
        # Three.js / OpenGL to OpenCV: invert Y and Z columns for ray directions
        R_gl = mat[:3, :3]
        R_cv = np.column_stack([R_gl[:, 0], -R_gl[:, 1], -R_gl[:, 2]])
        c2w_rots.append(R_cv)
        fp = fr.get("file_path", "")
        src = "video1" if "video1" in fp or "frame_0000" in fp and int(fp.split("frame_")[1][:6]) <= 222 else "video2"
        cam_sources.append(src)

    cams = np.array(cams, dtype=np.float64)
    c2w_rots = np.array(c2w_rots, dtype=np.float64)
    print(f"Loaded {num_cams} cameras from {transforms_path}")

    # Build straight line segments between consecutive cameras of the same source (<0.5m apart)
    cam_segments = []
    for i in range(num_cams - 1):
        if cam_sources[i] == cam_sources[i + 1]:
            dist = float(np.linalg.norm(cams[i + 1] - cams[i]))
            if dist < 0.50:
                cam_segments.append((cams[i], cams[i + 1]))
    print(f"Constructed {len(cam_segments)} consecutive inter-camera segments (<0.5 m apart)")

    # 2. Load SfM Points & Observations from images.txt & points3D.txt
    points3d_path = Path("data/home/colmap_aligned/points3D.txt")
    pts_dict = {}
    with open(points3d_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            pts_dict[pid] = xyz

    # Load images.txt to get which 3D points each camera observes
    images_txt_path = Path("data/home/colmap_aligned/images.txt")
    cam_observed_pts = []
    img_name_to_pids = {}
    with open(images_txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        parts = line.split()
        name = parts[9].replace("\\", "/")
        idx += 1
        pt_line = lines[idx].strip()
        p_parts = pt_line.split()
        pids = [int(p_parts[k + 2]) for k in range(0, len(p_parts), 3) if int(p_parts[k + 2]) != -1]
        img_name_to_pids[name] = pids
        idx += 1

    # Map transforms frames to observed points
    for fr in frames:
        fp = fr["file_path"].replace("\\", "/")
        matched_pids = None
        for k, v in img_name_to_pids.items():
            if k in fp or fp.endswith(k) or k.endswith(Path(fp).name):
                matched_pids = v
                break
        cam_observed_pts.append(matched_pids or [])

    total_obs = sum(len(p) for p in cam_observed_pts)
    print(f"Loaded {len(pts_dict):,} SfM points; mapped {total_obs:,} camera-point observations")

    # 3. Load Raw PLY Splats
    n_raw, props, offset = parse_ply_header(raw_ply)
    dt = np.dtype(props)
    data = np.fromfile(raw_ply, dtype=dt, count=n_raw, offset=offset)
    print(f"Loaded raw PLY: {n_raw:,} splats from {raw_ply}")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    raw_op = data["opacity"].astype(np.float64)
    sig_op = 1.0 / (1.0 + np.exp(-raw_op))

    scale_names = ["scale_0", "scale_1", "scale_2"]
    log_scales = np.column_stack([data[s] for s in scale_names]).astype(np.float64)
    real_scales = np.exp(log_scales)
    max_scale = np.max(real_scales, axis=1)
    min_scale = np.min(real_scales, axis=1)
    scale_ratio = max_scale / np.maximum(min_scale, 1e-7)

    # 4. Set Up Free Space Voxel Grid (voxel 0.10m, y in [0.10, 2.20] m)
    voxel_size = 0.10
    slab_y_min, slab_y_max = 0.10, 2.20
    n_y = int(round((slab_y_max - slab_y_min) / voxel_size))  # 21 layers

    # Determine XZ grid extents covering room
    grid_margin = 2.0
    all_pts_xyz = np.array(list(pts_dict.values()))
    min_x = np.floor(min(np.min(all_pts_xyz[:, 0]), np.min(cams[:, 0])) - grid_margin)
    max_x = np.ceil(max(np.max(all_pts_xyz[:, 0]), np.max(cams[:, 0])) + grid_margin)
    min_z = np.floor(min(np.min(all_pts_xyz[:, 2]), np.min(cams[:, 2])) - grid_margin)
    max_z = np.ceil(max(np.max(all_pts_xyz[:, 2]), np.max(cams[:, 2])) + grid_margin)

    n_x = int(round((max_x - min_x) / voxel_size))
    n_z = int(round((max_z - min_z) / voxel_size))

    print(f"\nFree-Space Voxel Grid Extents: {n_x} (X) x {n_z} (Z) x {n_y} (Y) = {n_x * n_z * n_y:,} voxels")
    print(f"  X: [{min_x:.2f}, {max_x:.2f}] m | Z: [{min_z:.2f}, {max_z:.2f}] m | Y: [{slab_y_min:.2f}, {slab_y_max:.2f}] m")

    stride_y = n_z * n_x
    stride_z = n_x

    # 5. Render 160x90 Depth and Alpha on GPU
    W, H = 160, 90
    fx = 1477.94 / 12.0
    fy = 1477.94 / 12.0
    cx = 80.0
    cy = 45.0

    print(f"\n[Step 1/5] Rendering 160x90 Depth and Alpha maps on GPU for {num_cams} cameras...")
    t_render_start = time.time()
    xyz_gpu = torch.from_numpy(xyz.astype(np.float32)).cuda()
    scales_gpu = torch.from_numpy(max_scale.astype(np.float32)).cuda()
    opacities_gpu = torch.from_numpy(sig_op.astype(np.float32)).cuda()

    all_depths, all_alphas = render_depth_alpha_maps(
        xyz_gpu, scales_gpu, opacities_gpu, cams, c2w_rots, W, H, fx, fy, cx, cy
    )
    print(f"Rendered all {num_cams} depth/alpha maps in {time.time() - t_render_start:.2f}s")

    # 6. Carve Free Space with Rays (i) and (ii)
    print("\n[Step 2/5] Carving free-space voxels (SfM rays + Depth rays)...")
    t_carve_start = time.time()

    voxel_cam_counts = np.zeros((n_y, n_z, n_x), dtype=np.int32)

    # Ray directions in camera space (H, W, 3)
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([
        (u_grid + 0.5 - cx) / fx,
        (v_grid + 0.5 - cy) / fy,
        np.ones((H, W), dtype=np.float32)
    ], axis=-1)
    d_cam_norm = d_cam / np.linalg.norm(d_cam, axis=-1, keepdims=True)

    ray_step = 0.05

    for cam_idx in range(num_cams):
        C = cams[cam_idx]
        cam_voxel_hits = []

        # (i) Rays to observed SfM points (stop 0.15m before point)
        pids = cam_observed_pts[cam_idx]
        if pids:
            pt_coords = [pts_dict[pid] for pid in pids if pid in pts_dict]
            if pt_coords:
                pt_coords = np.array(pt_coords, dtype=np.float64)
                vecs = pt_coords - C
                dists = np.linalg.norm(vecs, axis=1)
                valid_sfm = dists > 0.20
                if np.any(valid_sfm):
                    vecs = vecs[valid_sfm]
                    dists = dists[valid_sfm]
                    dirs = vecs / dists[:, None]
                    max_t_sfm = np.maximum(dists - 0.15, 0.0)
                    active_sfm = max_t_sfm > 0.15
                    if np.any(active_sfm):
                        max_t_sfm = max_t_sfm[active_sfm]
                        dirs = dirs[active_sfm]
                        max_d = float(np.max(max_t_sfm))
                        steps = np.arange(0.15, max_d, ray_step)
                        for st in steps:
                            act = max_t_sfm >= st
                            if not np.any(act):
                                break
                            pts = C + st * dirs[act]
                            in_vox = (
                                (pts[:, 1] >= slab_y_min) & (pts[:, 1] < slab_y_max) &
                                (pts[:, 0] >= min_x) & (pts[:, 0] < max_x) &
                                (pts[:, 2] >= min_z) & (pts[:, 2] < max_z)
                            )
                            if np.any(in_vox):
                                pv = pts[in_vox]
                                vx = np.floor((pv[:, 0] - min_x) / voxel_size).astype(int)
                                vy = np.floor((pv[:, 1] - slab_y_min) / voxel_size).astype(int)
                                vz = np.floor((pv[:, 2] - min_z) / voxel_size).astype(int)
                                cam_voxel_hits.append(vy * stride_y + vz * stride_z + vx)

        # (ii) Rays through rendered depth (alpha >= 0.60, min(depth - 0.15, 6m))
        R_c2w = c2w_rots[cam_idx]
        d_world = d_cam_norm @ R_c2w.T
        depth = all_depths[cam_idx]
        alpha = all_alphas[cam_idx]

        valid_pix = (alpha >= 0.60) & (depth > 0.25)
        if np.any(valid_pix):
            v_idx, u_idx = np.where(valid_pix)
            d_hit = depth[v_idx, u_idx]
            rays = d_world[v_idx, u_idx]

            t_max_depth = np.minimum(d_hit - 0.15, 6.0)
            valid_td = t_max_depth > 0.20
            if np.any(valid_td):
                t_max_depth = t_max_depth[valid_td]
                rays = rays[valid_td]
                max_d_d = float(np.max(t_max_depth))
                steps_d = np.arange(0.20, max_d_d, ray_step)
                for st in steps_d:
                    act = t_max_depth >= st
                    if not np.any(act):
                        break
                    pts = C + st * rays[act]
                    in_vox = (
                        (pts[:, 1] >= slab_y_min) & (pts[:, 1] < slab_y_max) &
                        (pts[:, 0] >= min_x) & (pts[:, 0] < max_x) &
                        (pts[:, 2] >= min_z) & (pts[:, 2] < max_z)
                    )
                    if np.any(in_vox):
                        pv = pts[in_vox]
                        vx = np.floor((pv[:, 0] - min_x) / voxel_size).astype(int)
                        vy = np.floor((pv[:, 1] - slab_y_min) / voxel_size).astype(int)
                        vz = np.floor((pv[:, 2] - min_z) / voxel_size).astype(int)
                        cam_voxel_hits.append(vy * stride_y + vz * stride_z + vx)

        # Camera counts ONCE per voxel:
        if cam_voxel_hits:
            combined_hits = np.concatenate(cam_voxel_hits)
            unique_voxels = np.unique(combined_hits)
            u_vy = unique_voxels // stride_y
            rem = unique_voxels % stride_y
            u_vz = rem // stride_z
            u_vx = rem % stride_z
            voxel_cam_counts[u_vy, u_vz, u_vx] += 1

    print(f"Carving completed in {time.time() - t_carve_start:.2f}s")
    print(f"  Voxels with free_cams >= 1: {int(np.sum(voxel_cam_counts >= 1)):,}")
    print(f"  Voxels with free_cams >= 2: {int(np.sum(voxel_cam_counts >= 2)):,}")
    print(f"  Voxels with free_cams >= 4: {int(np.sum(voxel_cam_counts >= 4)):,}")

    # 7. Apply Floater Pruning Rules
    print("\n[Step 3/5] Evaluating floater rules...")
    t_filter_start = time.time()

    # Rule 1: Lens / Trajectory Proximity (within 0.25m of camera or segment)
    # Check camera centres:
    cam_tree = KDTree(cams)
    dists_to_cam_centres, _ = cam_tree.query(xyz)
    within_cam_centre = dists_to_cam_centres <= 0.25

    # Check inter-camera segments:
    # Dense sample segments at 0.05m
    seg_samples = []
    for a, b in cam_segments:
        L = np.linalg.norm(b - a)
        n_pts = max(int(np.ceil(L / 0.05)), 1)
        for t in np.linspace(0.0, 1.0, n_pts + 1):
            seg_samples.append(a + t * (b - a))
    seg_samples = np.array(seg_samples, dtype=np.float64)
    seg_tree = KDTree(seg_samples)
    dists_to_segs, _ = seg_tree.query(xyz)
    within_cam_segment = dists_to_segs <= 0.25

    rule1_lens_floater = within_cam_centre | within_cam_segment
    count_rule1 = int(np.sum(rule1_lens_floater))
    print(f"  Rule 1 Floaters (Lens/Path Proximity <= 0.25m): {count_rule1:,} splats")

    # Rule 2: Free Space Floater (voxel free_cams >= 4 AND (max_scale > 0.08m OR opacity < 0.70))
    # Query voxel count for all splats in slab:
    in_grid_slab = (
        (xyz[:, 1] >= slab_y_min) & (xyz[:, 1] < slab_y_max) &
        (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x) &
        (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
    )

    splat_free_cams = np.zeros(n_raw, dtype=np.int32)
    slab_indices = np.where(in_grid_slab)[0]
    if len(slab_indices) > 0:
        sp_xyz = xyz[slab_indices]
        vx_sp = np.clip(np.floor((sp_xyz[:, 0] - min_x) / voxel_size).astype(int), 0, n_x - 1)
        vy_sp = np.clip(np.floor((sp_xyz[:, 1] - slab_y_min) / voxel_size).astype(int), 0, n_y - 1)
        vz_sp = np.clip(np.floor((sp_xyz[:, 2] - min_z) / voxel_size).astype(int), 0, n_z - 1)
        splat_free_cams[slab_indices] = voxel_cam_counts[vy_sp, vz_sp, vx_sp]

    rule2_free_floater = (splat_free_cams >= 4) & ((max_scale > 0.08) | (sig_op < 0.70))
    # Ensure mutually exclusive reporting
    count_rule2_only = int(np.sum(rule2_free_floater & (~rule1_lens_floater)))
    print(f"  Rule 2 Floaters (Free Voxel >=4 cams & (scale>0.08m or op<0.7)): {count_rule2_only:,} splats")

    # Combined floater mask: keep if NOT a floater
    keep_mask_floaters = ~(rule1_lens_floater | rule2_free_floater)
    cnt_after_floaters = int(np.sum(keep_mask_floaters))
    total_floaters_removed = n_raw - cnt_after_floaters

    # 8. Re-apply v2 Filters
    print("\n[Step 4/5] Re-applying v2 standard filter pipeline...")

    # Filter 3: Y height clamp [-0.30, 4.40] m
    y_min, y_max = -0.30, 4.40
    mask_y = (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    keep_3 = keep_mask_floaters & mask_y
    cnt_3 = int(np.sum(keep_3))

    # Filter 4: Intersected XZ bounding box crop
    cam_min = np.min(cams, axis=0)
    cam_max = np.max(cams, axis=0)
    crop_margin = 4.0
    cam_x_min, cam_x_max = cam_min[0] - crop_margin, cam_max[0] + crop_margin
    cam_z_min, cam_z_max = cam_min[2] - crop_margin, cam_max[2] + crop_margin

    ref_extents = load_robust_reference_extents(points3d_path)
    if ref_extents is not None:
        p005, p995, _ = ref_extents
        ref_x_min, ref_x_max = p005[0] - 1.0, p995[0] + 1.0
        ref_z_min, ref_z_max = p005[2] - 1.0, p995[2] + 1.0
        xz_x_min = max(cam_x_min, ref_x_min)
        xz_x_max = min(cam_x_max, ref_x_max)
        xz_z_min = max(cam_z_min, ref_z_min)
        xz_z_max = min(cam_z_max, ref_z_max)
    else:
        xz_x_min, xz_x_max = cam_x_min, cam_x_max
        xz_z_min, xz_z_max = cam_z_min, cam_z_max

    mask_xz = (xyz[:, 0] >= xz_x_min) & (xyz[:, 0] <= xz_x_max) & (xyz[:, 2] >= xz_z_min) & (xyz[:, 2] <= xz_z_max)
    keep_4 = keep_3 & mask_xz
    cnt_4 = int(np.sum(keep_4))

    # Filter 5: Min Opacity (sig_op >= 0.05)
    mask_op = sig_op >= 0.05
    keep_5 = keep_4 & mask_op
    cnt_5 = int(np.sum(keep_5))

    # Filter 6: Max Scale (max_scale <= 0.25 m)
    mask_scale = max_scale <= 0.25
    keep_6 = keep_5 & mask_scale
    cnt_6 = int(np.sum(keep_6))

    # Filter 7: Needle Filter (ratio <= 12 OR max_scale <= 0.08 m)
    mask_needle = ~((scale_ratio > 12.0) & (max_scale > 0.08))
    keep_7 = keep_6 & mask_needle
    cnt_7 = int(np.sum(keep_7))

    # Filter 8: Camera Coverage Filter (seen by >= 3 cameras OR within 1.0 m of camera path)
    cov_cell_size = 0.5
    cov_nx = int(np.ceil((max_x - min_x) / cov_cell_size))
    cov_nz = int(np.ceil((max_z - min_z) / cov_cell_size))
    cell_obs = [[set() for _ in range(cov_nz)] for _ in range(cov_nx)]

    # Populate observations from SfM points
    for pid, pxyz in pts_dict.items():
        col = int((pxyz[0] - min_x) / cov_cell_size)
        row = int((pxyz[2] - min_z) / cov_cell_size)
        if 0 <= col < cov_nx and 0 <= row < cov_nz:
            # find cameras observing this point
            for c_i, p_set in enumerate(cam_observed_pts):
                if pid in p_set:
                    cell_obs[col][row].add(c_i)

    cov_grid_2d = np.zeros((cov_nx, cov_nz), dtype=int)
    for i in range(cov_nx):
        for j in range(cov_nz):
            cov_grid_2d[i, j] = len(cell_obs[i][j])

    splat_cols = np.clip(((xyz[:, 0] - min_x) / cov_cell_size).astype(int), 0, cov_nx - 1)
    splat_rows = np.clip(((xyz[:, 2] - min_z) / cov_cell_size).astype(int), 0, cov_nz - 1)
    splat_cov_counts = cov_grid_2d[splat_cols, splat_rows]

    within_1m_cam = dists_to_cam_centres <= 1.0
    mask_cov = (splat_cov_counts >= 3) | within_1m_cam
    keep_8 = keep_7 & mask_cov
    cnt_8 = int(np.sum(keep_8))

    # Filter 9: Statistical Outlier Removal (SOR, nb=20, std=2.5)
    stage8_indices = np.where(keep_8)[0]
    stage8_xyz = xyz[stage8_indices]
    if o3d is not None and len(stage8_xyz) > 20:
        print("  Running Open3D Statistical Outlier Removal (SOR)...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stage8_xyz)
        _, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        clean_indices = stage8_indices[inliers]
    else:
        clean_indices = stage8_indices

    n_clean = len(clean_indices)
    cnt_9 = n_clean

    # Print Filter Count Table
    print("\n" + "=" * 88)
    print("  Q2 FILTER PROGRESSION TABLE")
    print("=" * 88)
    print(f"{'Filter Stage':<46} | {'Before':<10} | {'After':<10} | {'Pruned':<10} | {'Pruned %'}")
    print("-" * 88)
    print(f"{'0. Raw Splat Baseline':<46} | {'-':<10} | {n_raw:<10,} | {'-':<10} | {'-':<7}")
    print(f"{'1. Floaters: Lens/Trajectory (<0.25m)':<46} | {n_raw:<10,} | {n_raw - count_rule1:<10,} | {count_rule1:<10,} | {(count_rule1/n_raw)*100:5.2f}%")
    print(f"{'2. Floaters: Free Voxel (>=4 cams & weak)':<46} | {n_raw - count_rule1:<10,} | {cnt_after_floaters:<10,} | {count_rule2_only:<10,} | {(count_rule2_only/n_raw)*100:5.2f}%")
    print(f"{'3. Y Clamp [-0.30, 4.40] m':<46} | {cnt_after_floaters:<10,} | {cnt_3:<10,} | {cnt_after_floaters - cnt_3:<10,} | {((cnt_after_floaters - cnt_3)/cnt_after_floaters)*100:5.2f}%")
    print(f"{'4. Intersected XZ Crop':<46} | {cnt_3:<10,} | {cnt_4:<10,} | {cnt_3 - cnt_4:<10,} | {((cnt_3 - cnt_4)/cnt_3)*100:5.2f}%")
    print(f"{'5. Min Opacity (>= 0.05)':<46} | {cnt_4:<10,} | {cnt_5:<10,} | {cnt_4 - cnt_5:<10,} | {((cnt_4 - cnt_5)/cnt_4)*100:5.2f}%")
    print(f"{'6. Max Scale (<= 0.25 m)':<46} | {cnt_5:<10,} | {cnt_6:<10,} | {cnt_5 - cnt_6:<10,} | {((cnt_5 - cnt_6)/cnt_5)*100:5.2f}%")
    print(f"{'7. Needle Filter (ratio>12 & scale>0.08m)':<46} | {cnt_6:<10,} | {cnt_7:<10,} | {cnt_6 - cnt_7:<10,} | {((cnt_6 - cnt_7)/cnt_6)*100:5.2f}%")
    print(f"{'8. Coverage (cams>=3 or dist<=1.0m)':<46} | {cnt_7:<10,} | {cnt_8:<10,} | {cnt_7 - cnt_8:<10,} | {((cnt_7 - cnt_8)/cnt_7)*100:5.2f}%")
    print(f"{'9. Open3D SOR (nb=20, std=2.5)':<46} | {cnt_8:<10,} | {cnt_9:<10,} | {cnt_8 - cnt_9:<10,} | {((cnt_8 - cnt_9)/cnt_8)*100:5.2f}%")
    print("-" * 88)
    print(f"Total Filtered: {n_raw:,} -> {n_clean:,} splats ({((n_raw - n_clean)/n_raw)*100:.2f}% pruned)")
    print("=" * 88)

    # 9. Export Clean Splats
    print("\n[Step 5/5] Exporting clean splat assets to outputs/home/v3/...")

    clean_data = data[clean_indices]
    clean_ply_path = v3_dir / "scene_clean.ply"
    write_clean_ply(clean_ply_path, clean_data, props)
    ply_size_mb = clean_ply_path.stat().st_size / (1024 * 1024)
    print(f"  Exported scene_clean.ply: {ply_size_mb:.2f} MB")

    clean_xyz = xyz[clean_indices]
    clean_scales = log_scales[clean_indices]
    clean_opacities = raw_op[clean_indices]
    sh_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    clean_sh_dc = np.column_stack([data[s][clean_indices] for s in sh_names]).astype(np.float64)
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    clean_rot = np.column_stack([data[s][clean_indices] for s in rot_names]).astype(np.float64)

    clean_splat_path = v3_dir / "scene.splat"
    write_splat_binary(clean_splat_path, clean_xyz, clean_scales, clean_sh_dc, clean_opacities, clean_rot)
    splat_size_mb = clean_splat_path.stat().st_size / (1024 * 1024)
    print(f"  Exported scene.splat: {splat_size_mb:.2f} MB")

    # scene.json
    min_b = np.min(clean_xyz, axis=0)
    max_b = np.max(clean_xyz, axis=0)
    scene_json_data = {
        "scene": "home",
        "units": "metres",
        "splat_count": n_clean,
        "floor_y": 0.0,
        "ceiling_y": 4.10,
        "bounds": {
            "min": [round(float(min_b[0]), 3), round(float(min_b[1]), 3), round(float(min_b[2]), 3)],
            "max": [round(float(max_b[0]), 3), round(float(max_b[1]), 3), round(float(max_b[2]), 3)]
        },
        "spawn_position": [1.771, 1.450, 3.523],
        "spawn_yaw": -0.0426,
        "splat_rotation_euler": [0.0, 0.0, 0.0],
        "splat_files": {
            "spz": "scene.spz",
            "splat": "scene.splat"
        }
    }
    scene_json_path = v3_dir / "scene.json"
    with open(scene_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_json_data, f, indent=2)
    print(f"  Exported scene.json: {scene_json_path}")

    # scene.spz
    clean_spz_path = v3_dir / "scene.spz"
    print("  Converting to scene.spz via splat-transform...")
    ok, err = convert_to_spz(clean_ply_path, clean_spz_path)
    if not ok:
        print(f"[FAIL] SPZ conversion failed: {err}")
        sys.exit(1)
    spz_size_mb = clean_spz_path.stat().st_size / (1024 * 1024)
    print(f"  Exported scene.spz: {spz_size_mb:.2f} MB")

    # SPZ Evaluation
    print("\nRunning SPZ round-trip error evaluation...")
    spz_eval = evaluate_spz_quality(clean_ply_path, clean_spz_path)
    print(f"  Splat Count Diff   : {spz_eval['splat_count_diff']}")
    print(f"  Max Centre Error   : {spz_eval['max_centre_error_mm']:.3f} mm")
    print(f"  Mean Centre Error  : {spz_eval['mean_centre_error_mm']:.3f} mm")
    print(f"  Mean Opacity Error : {spz_eval['mean_opacity_error']:.6f}")
    with open(v3_dir / "spz_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(spz_eval, f, indent=2)

    # Coordinate frame verification
    print("\nVerifying splat coordinate frame (verify_splat_frame.py)...")
    verify_cmd = [
        sys.executable,
        "scripts/verify_splat_frame.py",
        "--scene", "home",
        "--splat-ply", str(clean_ply_path),
        "--reference-pts", str(points3d_path),
        "--transforms", str(transforms_path),
        "--scene-json", str(scene_json_path),
        "--output-json", str(v3_dir / "splat_frame_verification.json")
    ]
    res_ver = subprocess.run(verify_cmd, capture_output=True, text=True)
    print(res_ver.stdout)
    if res_ver.returncode != 0:
        print(f"[FAIL] Frame verification failed: {res_ver.stderr}")
        sys.exit(1)

    # Save filter summary report
    filter_report = {
        "raw_splats": n_raw,
        "rule1_lens_floaters": count_rule1,
        "rule2_free_floaters": count_rule2_only,
        "after_floaters": cnt_after_floaters,
        "after_y_clamp": cnt_3,
        "after_xz_crop": cnt_4,
        "after_min_opacity": cnt_5,
        "after_max_scale": cnt_6,
        "after_needle": cnt_7,
        "after_coverage": cnt_8,
        "final_clean_splats": n_clean,
        "ply_size_mb": round(ply_size_mb, 2),
        "splat_size_mb": round(splat_size_mb, 2),
        "spz_size_mb": round(spz_size_mb, 2),
        "spz_eval": spz_eval
    }
    with open(v3_dir / "filter_report.json", "w", encoding="utf-8") as f:
        json.dump(filter_report, f, indent=2)

    print("\n[SUCCESS] Q2 Splat Processing Finished Successfully!")

if __name__ == "__main__":
    main()
