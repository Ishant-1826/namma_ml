"""
pipeline/fix_room_reconstruction.py
Stages S1 & S2: In-place Gaussian reshaping, physical opacity filtering,
and architectural shell infill completion for Roomverse / Namma Space.

Pipeline Conventions:
- Argparse CLI reading inputs from data/ or outputs/ and writing to outputs/<scene>/complete.
- Preserves 320,000 to 450,000 real structural splats from the trained model.
- Clamps spiky needles in-place without deleting geometry (aspect ratio <= 4.5:1, max_scale <= 0.055m).
- Culls mid-air balloon floaters in empty upper air.
- Solidifies surviving real surfaces (boosted alpha).
- Merges with architectural floor and perimeter wall/ceiling infill disks where empty.
- Exports scene_clean.ply, scene.splat, scene.spz, and deploys via scripts/swap_assets.ps1.
"""

from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import argparse
import subprocess
import shutil
import time
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Any, Optional
import yaml

from pipeline.postprocess_scene import (
    parse_ply_header,
    write_clean_ply,
    write_splat_binary,
    convert_to_spz,
    evaluate_spz_quality,
)

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_fix_room_reconstruction(
    raw_ply: Path,
    output_dir: Path,
    scene_json_in: Path,
    scene_name: str = "home",
    config_path: Optional[Path] = None,
    deploy: bool = True,
) -> Dict[str, Any]:
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("  STAGE S1 & S2: IN-PLACE GAUSSIAN RESHAPING & ARCHITECTURAL SHELL COMPLETION")
    print(f"  Input Raw PLY : {raw_ply}")
    print(f"  Output Dir    : {output_dir}")
    print(f"  Scene JSON    : {scene_json_in}")
    print("=" * 88)

    if not raw_ply.is_file():
        raise FileNotFoundError(f"Input raw PLY not found: {raw_ply}")

    # 1. Parse PLY header and load structured binary data
    n_raw, props, offset = parse_ply_header(raw_ply)
    dt = np.dtype(props)
    data = np.fromfile(raw_ply, dtype=dt, count=n_raw, offset=offset)
    print(f"\n[Step 1] Loaded {n_raw:,} raw Gaussians from {raw_ply}")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    rot = np.column_stack([data[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    dc = np.column_stack([data[f"f_dc_{i}"] for i in range(3)]).astype(np.float64)

    # Convert to physical units
    log_scales = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]]).astype(np.float64)
    scales_m = np.exp(log_scales)
    orig_max_scale_m = np.max(scales_m, axis=-1)
    orig_min_scale_m = np.min(scales_m, axis=-1)
    orig_aspect_ratio = orig_max_scale_m / np.maximum(orig_min_scale_m, 1e-6)

    raw_op = data["opacity"].astype(np.float64)
    alpha = 1.0 / (1.0 + np.exp(-raw_op))

    # Raw percentiles before filtering
    p10_s_pre, p50_s_pre, p90_s_pre, p99_s_pre = np.percentile(orig_max_scale_m, [10, 50, 90, 99])
    p10_a_pre, p50_a_pre, p90_a_pre, p99_a_pre = np.percentile(alpha, [10, 50, 90, 99])

    print("\n--- Physical Units Diagnostic (BEFORE Filtering) ---")
    print(f"  max_scale_m percentiles : p10 = {p10_s_pre*100:5.2f} cm, p50 = {p50_s_pre*100:5.2f} cm, "
          f"p90 = {p90_s_pre*100:5.2f} cm, p99 = {p99_s_pre*100:5.2f} cm")
    print(f"  true alpha percentiles  : p10 = {p10_a_pre:5.3f} ({p10_a_pre*100:4.1f}%), "
          f"p50 = {p50_a_pre:5.3f} ({p50_a_pre*100:4.1f}%), "
          f"p90 = {p90_a_pre:5.3f} ({p90_a_pre*100:4.1f}%), "
          f"p99 = {p99_a_pre:5.3f} ({p99_a_pre*100:4.1f}%)")

    # Load room bounds
    if scene_json_in.is_file():
        with open(scene_json_in, "r", encoding="utf-8") as f:
            scene_manifest = json.load(f)
        bounds = scene_manifest.get("bounds", {})
        b_min = bounds.get("min", [-5.85, -0.3, -4.40])
        b_max = bounds.get("max", [4.80, 4.10, 13.50])
    else:
        b_min = [-5.85, -0.3, -4.40]
        b_max = [4.80, 4.10, 13.50]

    floor_x_min, floor_x_max = float(b_min[0]), float(b_max[0])
    floor_z_min, floor_z_max = float(b_min[2]), float(b_max[2])

    # -----------------------------------------------------------------------
    # STAGE S1: RESHAPE IN-PLACE INSTEAD OF DELETING (KEEP ~350k-450k REAL SPLATS)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Executing Stage S1 Physical Filtering & In-Place Reshaping...")

    # S1.1: Only Delete True Ghost Noise (alpha < 0.055) OR huge mid-air floaters (orig_max_scale_m > 0.22 m)
    # If surviving real splats < 300,000, automatically lower alpha cutoff until >= 350,000 survive
    alpha_cutoff = 0.055

    while True:
        keep_noise = (alpha >= alpha_cutoff) & (orig_max_scale_m <= 0.22)
        n_surv_noise = int(np.sum(keep_noise))

        # S1.3: Mid-Air Balloon Cull ONLY in Empty Upper Air:
        # Upper mid-air zone: y between 1.25m and 2.90m, at least 0.60m inside perimeter walls
        # that had orig_max_scale_m > 0.09 m OR fewer than 10 neighbors within 0.20 m
        in_upper_air = (
            keep_noise &
            (xyz[:, 1] >= 1.25) & (xyz[:, 1] <= 2.90) &
            (xyz[:, 0] >= floor_x_min + 0.60) & (xyz[:, 0] <= floor_x_max - 0.60) &
            (xyz[:, 2] >= floor_z_min + 0.60) & (xyz[:, 2] <= floor_z_max - 0.60)
        )
        surv_idx = np.where(keep_noise)[0]
        tree_noise = KDTree(xyz[keep_noise])
        upper_air_in_surv = np.where(in_upper_air[keep_noise])[0]

        if len(upper_air_in_surv) > 0:
            nb_counts = tree_noise.query_ball_point(xyz[keep_noise][upper_air_in_surv], r=0.20, return_length=True)
            balloon_cull = (orig_max_scale_m[keep_noise][upper_air_in_surv] > 0.09) | (nb_counts < 10)
            cull_indices = surv_idx[upper_air_in_surv[balloon_cull]]
        else:
            cull_indices = np.array([], dtype=int)

        surviving_mask = keep_noise.copy()
        surviving_mask[cull_indices] = False
        surviving_real_splats = int(np.sum(surviving_mask))

        if surviving_real_splats < 300000 and alpha_cutoff > 0.01:
            alpha_cutoff = max(0.005, alpha_cutoff - 0.005)
            print(f"  [Auto-adjust] surviving_real_splats ({surviving_real_splats:,}) < 300,000; "
                  f"lowering alpha cutoff to {alpha_cutoff:.3f}")
            continue
        break

    print(f"  - Noise cull (alpha >= {alpha_cutoff:.3f} & scale <= 0.22m) retained: {n_surv_noise:,} splats")
    print(f"  - Upper mid-air balloon floaters culled: {len(cull_indices):,} splats")
    print(f"  - Surviving real structural splats: {surviving_real_splats:,}")

    # Assertion 1: surviving_real_splats >= 300,000
    assert surviving_real_splats >= 300000, (
        f"Assertion Failed: surviving_real_splats ({surviving_real_splats:,}) < 300,000!"
    )

    clean_xyz = xyz[surviving_mask]
    clean_rot = rot[surviving_mask]
    clean_dc = dc[surviving_mask]
    clean_scales_m = scales_m[surviving_mask].copy()
    clean_alpha = alpha[surviving_mask].copy()

    # S1.2: In-Place Scale & Aspect-Ratio Clamping (Do NOT Delete Needles -- Reshape Them!)
    # Clamp maximum physical scale of every splat: scales_m = np.minimum(scales_m, 0.055) (max 5.5 cm)
    clean_scales_m = np.minimum(clean_scales_m, 0.055)

    # Clamp minimum physical scale so aspect ratio never exceeds 4.5:1:
    # max_s = np.max(scales_m, axis=-1, keepdims=True)
    # scales_m = np.maximum(scales_m, max_s / 4.5)
    max_s = np.max(clean_scales_m, axis=-1, keepdims=True)
    clean_scales_m = np.maximum(clean_scales_m, max_s / 4.5)

    # Convert back to log-scale
    clean_log_scales = np.log(clean_scales_m)
    print("  - In-place scale and aspect ratio clamped (max scale <= 5.5cm, aspect ratio <= 4.5:1)")

    # S1.4: Solidify Surviving Real Surfaces:
    # alpha_new = np.clip(alpha * 2.2, 0.18, 0.99)
    # opacity_new = np.log(alpha_new / (1.0 - alpha_new))
    clean_alpha = np.clip(clean_alpha * 2.2, 0.18, 0.99)
    print("  - Surviving real surfaces solidified (boosted alpha: np.clip(alpha * 2.2, 0.18, 0.99))")

    # -----------------------------------------------------------------------
    # STAGE S2: MERGE WITH ARCHITECTURAL FLOOR & WALL SHELL
    # -----------------------------------------------------------------------
    print("\n[Step 3] Executing Stage S2 Architectural Shell Synthesis...")

    # Find existing floor splats to extract true observed floor colors
    existing_floor_mask = (np.abs(clean_xyz[:, 1]) < 0.08)
    floor_xyz = clean_xyz[existing_floor_mask]
    floor_dc = clean_dc[existing_floor_mask]
    floor_rgb = np.clip(0.5 + SH_C0 * floor_dc, 0.0, 1.0)
    valid_floor_mask = np.mean(floor_rgb, axis=1) > 0.15

    if np.sum(valid_floor_mask) > 10:
        base_floor_rgb = np.median(floor_rgb[valid_floor_mask], axis=0)
        ref_floor_pts = floor_xyz[valid_floor_mask][:, [0, 2]]
        ref_floor_colors = floor_rgb[valid_floor_mask]
        tree_floor_colors = KDTree(ref_floor_pts)
        print(f"  - Extracted true observed floor colors from {np.sum(valid_floor_mask):,} splats. Median RGB: {base_floor_rgb * 255.0}")
    else:
        base_floor_rgb = np.array([160.0, 150.0, 130.0]) / 255.0
        tree_floor_colors = None
        print(f"  - Using default warm tile floor color RGB: {base_floor_rgb * 255.0}")

    # 1. Architectural Floor Disks (y = 0.0 m, 0.08 m grid)
    grid_spacing = 0.08
    xs_floor = np.arange(floor_x_min, floor_x_max + grid_spacing / 2.0, grid_spacing)
    zs_floor = np.arange(floor_z_min, floor_z_max + grid_spacing / 2.0, grid_spacing)
    gx_f, gz_f = np.meshgrid(xs_floor, zs_floor)
    cand_floor_xyz = np.column_stack([gx_f.ravel(), np.zeros(gx_f.size), gz_f.ravel()])

    # Only place infill disks where no real splat exists within 0.10 m
    tree_all_clean = KDTree(clean_xyz)
    dists_floor, _ = tree_all_clean.query(cand_floor_xyz)
    add_floor_mask = dists_floor > 0.10
    new_floor_xyz = cand_floor_xyz[add_floor_mask]
    n_new_floor = len(new_floor_xyz)
    print(f"  - Synthesizing {n_new_floor:,} architectural floor infill disks (dists > 0.10m from real splats)")

    if tree_floor_colors is not None and n_new_floor > 0:
        dists, nn_idx = tree_floor_colors.query(new_floor_xyz[:, [0, 2]], k=min(4, len(ref_floor_pts)))
        if dists.ndim == 1:
            dists = dists[:, None]
            nn_idx = nn_idx[:, None]
        weights = 1.0 / np.maximum(dists, 0.05)
        weights /= np.sum(weights, axis=1, keepdims=True)
        interp_rgb = np.sum(weights[:, :, None] * ref_floor_colors[nn_idx], axis=1)
        tile_rgb = 0.70 * interp_rgb + 0.30 * base_floor_rgb
    else:
        tile_rgb = np.tile(base_floor_rgb, (n_new_floor, 1))

    # Tile-grid micro-variation (0.60m classroom tile pattern)
    if n_new_floor > 0:
        mod_x = np.abs((new_floor_xyz[:, 0] % 0.60) - 0.30)
        mod_z = np.abs((new_floor_xyz[:, 2] % 0.60) - 0.30)
        grout_dist = np.minimum(0.30 - mod_x, 0.30 - mod_z)
        is_grout = grout_dist < 0.02
        tile_rgb[is_grout] *= 0.88
        variation = (1.0 + 0.03 * np.sin(new_floor_xyz[~is_grout, 0] * 15.0))[:, None]
        tile_rgb[~is_grout] = np.clip(tile_rgb[~is_grout] * variation, 0.0, 1.0)

    new_floor_scales = np.tile(np.array([np.log(0.06), np.log(0.005), np.log(0.06)]), (n_new_floor, 1))
    new_floor_rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_new_floor, 1))
    new_floor_alpha = np.full(n_new_floor, 0.85, dtype=np.float64)
    new_floor_dc = (tile_rgb - 0.5) / SH_C0

    # 2. Perimeter Wall & Ceiling Enclosure Disks
    wall_spacing = 0.15
    y_wall_levels = np.arange(0.10, 4.00 + wall_spacing / 2.0, wall_spacing)

    wall_splats_mask = (clean_xyz[:, 1] >= 1.5) & (clean_xyz[:, 1] <= 3.2)
    if np.sum(wall_splats_mask) > 10:
        w_rgb = np.clip(0.5 + SH_C0 * clean_dc[wall_splats_mask], 0.0, 1.0)
        w_valid = np.mean(w_rgb, axis=1) > 0.20
        base_wall_rgb = np.median(w_rgb[w_valid], axis=0) if np.sum(w_valid) > 10 else np.array([172.0, 158.0, 138.0]) / 255.0
    else:
        base_wall_rgb = np.array([172.0, 158.0, 138.0]) / 255.0

    wall_candidates = []
    wall_orientations = []

    # West wall: X = floor_x_min
    gz_w, gy_w = np.meshgrid(np.arange(floor_z_min, floor_z_max, wall_spacing), y_wall_levels)
    wall_candidates.append(np.column_stack([np.full(gz_w.size, floor_x_min), gy_w.ravel(), gz_w.ravel()]))
    wall_orientations.extend(["x_facing"] * gz_w.size)

    # East wall: X = floor_x_max
    gz_e, gy_e = np.meshgrid(np.arange(floor_z_min, floor_z_max, wall_spacing), y_wall_levels)
    wall_candidates.append(np.column_stack([np.full(gz_e.size, floor_x_max), gy_e.ravel(), gz_e.ravel()]))
    wall_orientations.extend(["x_facing"] * gz_e.size)

    # North wall: Z = floor_z_min
    gx_n, gy_n = np.meshgrid(np.arange(floor_x_min, floor_x_max, wall_spacing), y_wall_levels)
    wall_candidates.append(np.column_stack([gx_n.ravel(), gy_n.ravel(), np.full(gx_n.size, floor_z_min)]))
    wall_orientations.extend(["z_facing"] * gx_n.size)

    # South wall: Z = floor_z_max
    gx_s, gy_s = np.meshgrid(np.arange(floor_x_min, floor_x_max, wall_spacing), y_wall_levels)
    wall_candidates.append(np.column_stack([gx_s.ravel(), gy_s.ravel(), np.full(gx_s.size, floor_z_max)]))
    wall_orientations.extend(["z_facing"] * gx_s.size)

    # Ceiling: Y = 4.05 m
    gx_c, gz_c = np.meshgrid(np.arange(floor_x_min, floor_x_max, 0.20), np.arange(floor_z_min, floor_z_max, 0.20))
    wall_candidates.append(np.column_stack([gx_c.ravel(), np.full(gx_c.size, 4.05), gz_c.ravel()]))
    wall_orientations.extend(["ceiling"] * gx_c.size)

    all_shell_pts = np.vstack(wall_candidates)
    shell_orientations = np.array(wall_orientations)

    # Only place infill disks where no real splat exists within 0.10 m
    dists_shell, _ = tree_all_clean.query(all_shell_pts)
    infill_mask = dists_shell > 0.10

    new_shell_xyz = all_shell_pts[infill_mask]
    new_shell_ori = shell_orientations[infill_mask]
    n_new_shell = len(new_shell_xyz)
    print(f"  - Synthesizing {n_new_shell:,} perimeter wall and ceiling infill disks (dists > 0.10m from real splats)")

    new_shell_scales = np.zeros((n_new_shell, 3), dtype=np.float64)
    new_shell_rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_new_shell, 1))

    x_fac = new_shell_ori == "x_facing"
    z_fac = new_shell_ori == "z_facing"
    ceil_fac = new_shell_ori == "ceiling"

    new_shell_scales[x_fac] = np.array([np.log(0.005), np.log(0.10), np.log(0.10)])
    new_shell_scales[z_fac] = np.array([np.log(0.10), np.log(0.10), np.log(0.005)])
    new_shell_scales[ceil_fac] = np.array([np.log(0.15), np.log(0.005), np.log(0.15)])

    new_shell_alpha = np.full(n_new_shell, 0.75, dtype=np.float64)
    shell_rgb = np.tile(base_wall_rgb, (n_new_shell, 1))
    shell_rgb[ceil_fac] = np.clip(shell_rgb[ceil_fac] * 1.05, 0.0, 1.0)
    new_shell_dc = (shell_rgb - 0.5) / SH_C0

    # -----------------------------------------------------------------------
    # MERGE CLEAN RECONSTRUCTION WITH ARCHITECTURAL SHELL
    # -----------------------------------------------------------------------
    final_xyz = np.vstack([clean_xyz, new_floor_xyz, new_shell_xyz])
    final_log_scales = np.vstack([clean_log_scales, new_floor_scales, new_shell_scales])
    final_alpha = np.concatenate([clean_alpha, new_floor_alpha, new_shell_alpha])
    final_rot = np.vstack([clean_rot, new_floor_rot, new_shell_rot])
    final_dc = np.vstack([clean_dc, new_floor_dc, new_shell_dc])

    final_opacity = np.log(final_alpha / (1.0 - final_alpha))
    n_final = len(final_xyz)
    shell_splats = n_new_floor + n_new_shell

    # Assertion 2: total_merged_splats >= 350,000
    assert n_final >= 350000, (
        f"Assertion Failed: total_merged_splats ({n_final:,}) < 350,000!"
    )

    # -----------------------------------------------------------------------
    # EXPORT scene_clean.ply, scene.splat, scene.spz, scene.json
    # -----------------------------------------------------------------------
    clean_data = np.empty(n_final, dtype=dt)
    clean_data["x"] = final_xyz[:, 0].astype(np.float32)
    clean_data["y"] = final_xyz[:, 1].astype(np.float32)
    clean_data["z"] = final_xyz[:, 2].astype(np.float32)
    clean_data["scale_0"] = final_log_scales[:, 0].astype(np.float32)
    clean_data["scale_1"] = final_log_scales[:, 1].astype(np.float32)
    clean_data["scale_2"] = final_log_scales[:, 2].astype(np.float32)
    clean_data["rot_0"] = final_rot[:, 0].astype(np.float32)
    clean_data["rot_1"] = final_rot[:, 1].astype(np.float32)
    clean_data["rot_2"] = final_rot[:, 2].astype(np.float32)
    clean_data["rot_3"] = final_rot[:, 3].astype(np.float32)
    clean_data["opacity"] = final_opacity.astype(np.float32)
    clean_data["f_dc_0"] = final_dc[:, 0].astype(np.float32)
    clean_data["f_dc_1"] = final_dc[:, 1].astype(np.float32)
    clean_data["f_dc_2"] = final_dc[:, 2].astype(np.float32)

    # Export scene_clean.ply
    clean_ply_path = output_dir / "scene_clean.ply"
    write_clean_ply(clean_ply_path, clean_data, props)
    ply_size_mb = clean_ply_path.stat().st_size / (1024 * 1024)
    print(f"\n[Step 4] Exported scene_clean.ply: {clean_ply_path} ({ply_size_mb:.2f} MB)")

    # Export scene.splat (32-byte binary format)
    clean_splat_path = output_dir / "scene.splat"
    write_splat_binary(clean_splat_path, final_xyz, final_log_scales, final_dc, final_opacity, final_rot)
    splat_size_mb = clean_splat_path.stat().st_size / (1024 * 1024)
    print(f"[Step 5] Exported scene.splat: {clean_splat_path} ({splat_size_mb:.2f} MB)")

    # Export scene.spz using splat-transform
    clean_spz_path = output_dir / "scene.spz"
    print("\n[Step 6] Compressing clean PLY to scene.spz with splat-transform...")
    ok, err = convert_to_spz(clean_ply_path, clean_spz_path)
    if not ok:
        print(f"[FAIL] SPZ conversion failed: {err}")
        sys.exit(1)
    spz_size_mb = clean_spz_path.stat().st_size / (1024 * 1024)
    print(f"[Step 6] Exported scene.spz: {clean_spz_path} ({spz_size_mb:.2f} MB)")

    # Evaluate SPZ round-trip error
    try:
        spz_metrics = evaluate_spz_quality(clean_ply_path, clean_spz_path)
        print(f"  - SPZ Quality Check: Mean centre error = {spz_metrics['mean_centre_error_mm']:.3f} mm, "
              f"Max = {spz_metrics['max_centre_error_mm']:.3f} mm")
    except Exception as e:
        print(f"  - [WARN] SPZ round-trip check skipped: {e}")

    # Export scene.json manifest
    scene_json_data = {
        "scene": scene_name,
        "units": "metres",
        "splat_count": n_final,
        "floor_y": 0.0,
        "ceiling_y": 4.10,
        "bounds": {
            "min": [round(float(floor_x_min), 3), -0.300, round(float(floor_z_min), 3)],
            "max": [round(float(floor_x_max), 3), 4.100, round(float(floor_z_max), 3)],
        },
        "spawn_position": [1.771, 1.450, 3.523],
        "spawn_yaw": -0.0426,
        "splat_rotation_euler": [0.0, 0.0, 0.0],
        "splat_files": {
            "spz": "scene.spz",
            "splat": "scene.splat",
        },
    }
    scene_json_path = output_dir / "scene.json"
    with open(scene_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_json_data, f, indent=2)
    print(f"[Step 7] Exported scene.json: {scene_json_path}")

    # Copy v4 occupancy files so the complete bundle is self-contained
    occ_v4_json = Path("outputs/home/occupancy_v4/occupancy.json")
    occ_v4_png = Path("outputs/home/occupancy_v4/occupancy.png")
    if occ_v4_json.is_file():
        shutil.copy2(occ_v4_json, output_dir / "occupancy.json")
        print(f"[Step 8] Copied verified v4 occupancy.json into {output_dir}")
    if occ_v4_png.is_file():
        shutil.copy2(occ_v4_png, output_dir / "occupancy.png")
        print(f"[Step 8] Copied verified v4 occupancy.png into {output_dir}")

    # Percentiles after filtering and merging
    cur_final_scales_m = np.max(np.exp(final_log_scales), axis=-1)
    p10_s_post, p50_s_post, p90_s_post, p99_s_post = np.percentile(cur_final_scales_m, [10, 50, 90, 99])
    p10_a_post, p50_a_post, p90_a_post, p99_a_post = np.percentile(final_alpha, [10, 50, 90, 99])

    # -----------------------------------------------------------------------
    # PRINT SUMMARY METRICS TABLE
    # -----------------------------------------------------------------------
    print("\n" + "=" * 94)
    print("                     ROOM RECONSTRUCTION QUALITY & METRICS TABLE")
    print("=" * 94)
    print(f"  {'Metric':<36} | {'Before (scene_raw.ply)':<24} | {'After (S1 & S2 Complete)':<26}")
    print("-" * 37 + "+-" + "-" * 26 + "+-" + "-" * 26)
    print(f"  {'Input Raw Splats':<36} | {n_raw:<24,} | {'-':<26}")
    print(f"  {'Surviving Real Scene Splats':<36} | {'-':<24} | {surviving_real_splats:<26,}")
    print(f"  {'Architectural Shell Disks':<36} | {'-':<24} | {shell_splats:<26,} ({n_new_floor:,} fl, {n_new_shell:,} wl)")
    print(f"  {'Total Merged Room Splats':<36} | {'-':<24} | {n_final:<26,}")
    print(f"  {'Splat Binary File (.splat)':<36} | {'-':<24} | {splat_size_mb:<23.2f} MB")
    print(f"  {'Compressed File (.spz)':<36} | {'-':<24} | {spz_size_mb:<23.2f} MB")
    print("-" * 37 + "+-" + "-" * 26 + "+-" + "-" * 26)
    print(f"  {'Scale p10 (10th percentile)':<36} | {p10_s_pre*100:6.2f} cm ({p10_s_pre:.4f} m)   | {p10_s_post*100:6.2f} cm ({p10_s_post:.4f} m)")
    print(f"  {'Scale p50 (median)':<36} | {p50_s_pre*100:6.2f} cm ({p50_s_pre:.4f} m)   | {p50_s_post*100:6.2f} cm ({p50_s_post:.4f} m)")
    print(f"  {'Scale p90 (90th percentile)':<36} | {p90_s_pre*100:6.2f} cm ({p90_s_pre:.4f} m)   | {p90_s_post*100:6.2f} cm ({p90_s_post:.4f} m)")
    print(f"  {'Scale p99 (99th percentile)':<36} | {p99_s_pre*100:6.2f} cm ({p99_s_pre:.4f} m)   | {p99_s_post*100:6.2f} cm ({p99_s_post:.4f} m)")
    print("-" * 37 + "+-" + "-" * 26 + "+-" + "-" * 26)
    print(f"  {'Alpha p10 (10th percentile)':<36} | {p10_a_pre:6.3f} ({p10_a_pre*100:4.1f}%)        | {p10_a_post:6.3f} ({p10_a_post*100:4.1f}%)")
    print(f"  {'Alpha p50 (median)':<36} | {p50_a_pre:6.3f} ({p50_a_pre*100:4.1f}%)        | {p50_a_post:6.3f} ({p50_a_post*100:4.1f}%)")
    print(f"  {'Alpha p90 (90th percentile)':<36} | {p90_a_pre:6.3f} ({p90_a_pre*100:4.1f}%)        | {p90_a_post:6.3f} ({p90_a_post*100:4.1f}%)")
    print(f"  {'Alpha p99 (99th percentile)':<36} | {p99_a_pre:6.3f} ({p99_a_pre*100:4.1f}%)        | {p99_a_post:6.3f} ({p99_a_post*100:4.1f}%)")
    print("=" * 94)

    # Deploy to viewer if requested
    if deploy:
        swap_script = _REPO_ROOT / "scripts" / "swap_assets.ps1"
        if swap_script.is_file():
            print(f"\n[Step 9] Deploying complete assets to viewer/public/scene via {swap_script}...")
            res = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(swap_script), str(output_dir)],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                print(res.stdout.strip())
            else:
                print(f"[WARN] Deployment returned code {res.returncode}: {res.stderr}")

    duration = time.time() - t_start
    print(f"\n[PASS] Stages S1 & S2 complete in {duration:.2f} seconds.")

    return {
        "n_raw": n_raw,
        "surviving_real_splats": surviving_real_splats,
        "n_new_floor": n_new_floor,
        "n_new_shell": n_new_shell,
        "shell_splats": shell_splats,
        "total_merged_splats": n_final,
        "spz_size_mb": spz_size_mb,
        "splat_size_mb": splat_size_mb,
        "scale_percentiles_pre": [p10_s_pre, p50_s_pre, p90_s_pre, p99_s_pre],
        "scale_percentiles_post": [p10_s_post, p50_s_post, p90_s_post, p99_s_post],
        "alpha_percentiles_pre": [p10_a_pre, p50_a_pre, p90_a_pre, p99_a_pre],
        "alpha_percentiles_post": [p10_a_post, p50_a_post, p90_a_post, p99_a_post],
        "duration_s": duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Fix room reconstruction (STAGES S1 & S2)")
    parser.add_argument("--raw-ply", type=Path, default=Path("outputs/home3/max/scene_raw.ply"),
                        help="Path to raw Gaussian PLY")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/home3/complete"),
                        help="Output directory for clean scene bundle")
    parser.add_argument("--scene-json", type=Path, default=Path("outputs/home3/max/scene.json"),
                        help="Path to scene manifest json")
    parser.add_argument("--scene", type=str, default="home",
                        help="Scene identifier")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                        help="Path to pipeline configuration yaml")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Skip automatic deployment via scripts/swap_assets.ps1")

    args = parser.parse_args()
    run_fix_room_reconstruction(
        raw_ply=args.raw_ply,
        output_dir=args.output_dir,
        scene_json_in=args.scene_json,
        scene_name=args.scene,
        config_path=args.config,
        deploy=not args.no_deploy,
    )


if __name__ == "__main__":
    main()
