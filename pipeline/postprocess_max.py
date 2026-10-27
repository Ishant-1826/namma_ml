"""
pipeline/postprocess_max.py
Surgical, non-destructive splat filtering for Namma Space.

Filters floaters and needle artifacts while preserving all true architectural surfaces:
- NO destructive depth ray voxel deletion.
- Camera-head clearing only in 0.18m cylinder at head height for low-opacity floaters.
- Opacity, Y-clamp, max-scale, and needle aspect ratio filters.
- Zero-SH bake for web export to prevent view-angle color flipping on look-back.
- Auto-relax thresholds if retained splats < 400,000 (target: 450,000 to 650,000).
"""

from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import json
import argparse
import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional

from pipeline.postprocess_scene import (
    parse_ply_header,
    write_clean_ply,
    write_splat_binary,
    convert_to_spz,
    evaluate_spz_quality,
)


def run_postprocess_max(
    raw_ply: Path,
    output_dir: Path,
    transforms_path: Path,
    scene_name: str = "home",
    min_opacity: float = 0.08,
    max_scale_m: float = 0.18,
    max_needle_ratio: float = 10.0,
    needle_min_scale: float = 0.05,
    y_min: float = -0.25,
    y_max: float = 4.40,
    target_min_splats: int = 400000,
    target_desired_splats: int = 450000,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  STAGE R3: SURGICAL FLOATER REMOVAL & ZERO-SH BAKE")
    print(f"  Input Raw PLY : {raw_ply}")
    print(f"  Output Dir    : {output_dir}")
    print("=" * 80)

    # 1. Parse PLY header and load structured binary data
    n_raw, props, offset = parse_ply_header(raw_ply)
    dt = np.dtype(props)
    data = np.fromfile(raw_ply, dtype=dt, count=n_raw, offset=offset)
    print(f"Loaded {n_raw:,} raw Gaussians from {raw_ply}")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

    # Sigmoid opacity
    if "opacity" in data.dtype.names:
        raw_op = data["opacity"].astype(np.float64)
        sig_op = 1.0 / (1.0 + np.exp(-raw_op))
    else:
        sig_op = np.ones(n_raw, dtype=np.float64)
        raw_op = np.zeros(n_raw, dtype=np.float64)

    # Scales
    scale_names = [f"scale_{i}" for i in range(3)]
    if all(s in data.dtype.names for s in scale_names):
        log_scales = np.column_stack([data[s] for s in scale_names]).astype(np.float64)
        real_scales = np.exp(log_scales)
        max_scale = np.max(real_scales, axis=1)
        min_scale = np.min(real_scales, axis=1)
    else:
        log_scales = np.zeros((n_raw, 3), dtype=np.float64)
        real_scales = np.ones((n_raw, 3), dtype=np.float64)
        max_scale = np.ones(n_raw, dtype=np.float64)
        min_scale = np.ones(n_raw, dtype=np.float64)

    scale_ratio = max_scale / np.maximum(min_scale, 1e-7)

    # 2. Load Camera Trajectory for Surgical Head Clearing
    cams = []
    if transforms_path.is_file():
        with open(transforms_path, "r", encoding="utf-8") as f:
            tf_data = json.load(f)
        for frame in tf_data.get("frames", []):
            mat = np.array(frame["transform_matrix"], dtype=np.float64)
            cams.append(mat[:3, 3])
    if len(cams) > 0:
        cams = np.array(cams, dtype=np.float64)
        cam_tree_2d = KDTree(cams[:, [0, 2]])
        dists_xz, _ = cam_tree_2d.query(xyz[:, [0, 2]])
        print(f"Loaded {len(cams)} camera poses for surgical cylinder clearing")
    else:
        dists_xz = np.full(n_raw, 999.0, dtype=np.float64)
        print("[WARN] No camera poses found; skipping camera cylinder clearing")

    # Function to evaluate filter masks with current thresholds
    def evaluate_filters(cur_op: float, cur_max_scale: float, cur_ratio: float, cur_needle_min: float):
        # Filter 1: Y-clamp
        m_y = (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
        # Filter 2: Camera-head clearing (only mid-air low-opacity floaters right in front of lens)
        # r <= 0.18m, y in [0.8m, 1.7m], opacity < 0.35
        is_cam_floater = (dists_xz <= 0.18) & (xyz[:, 1] >= 0.80) & (xyz[:, 1] <= 1.70) & (sig_op < 0.35)
        m_cam = ~is_cam_floater
        # Filter 3: Opacity filter
        m_op = sig_op >= cur_op
        # Filter 4: Max world scale
        m_scale = max_scale <= cur_max_scale
        # Filter 5: Needle / anisotropic spike filter
        is_needle = (scale_ratio > cur_ratio) & (max_scale > cur_needle_min)
        m_needle = ~is_needle

        combined = m_y & m_cam & m_op & m_scale & m_needle
        return combined, m_y, m_cam, m_op, m_scale, m_needle

    # Iterative relaxation if retained drops below target_min_splats
    cur_op = min_opacity
    cur_max_scale = max_scale_m
    cur_ratio = max_needle_ratio
    cur_needle_min = needle_min_scale

    iteration = 0
    final_mask = None
    m_y = m_cam = m_op = m_scale = m_needle = None

    while iteration < 10:
        combined, m_y, m_cam, m_op, m_scale, m_needle = evaluate_filters(
            cur_op, cur_max_scale, cur_ratio, cur_needle_min
        )
        count = int(np.sum(combined))
        if count >= target_min_splats or (cur_op <= 0.04 and cur_ratio >= 16.0):
            final_mask = combined
            break
        print(f"  [Auto-Relax] Retained {count:,} < {target_min_splats:,}. Relaxing thresholds...")
        cur_op = max(0.04, cur_op - 0.01)
        cur_max_scale = min(0.25, cur_max_scale + 0.02)
        cur_ratio = min(16.0, cur_ratio + 1.0)
        cur_needle_min = min(0.08, cur_needle_min + 0.01)
        iteration += 1

    clean_indices = np.where(final_mask)[0]
    n_clean = len(clean_indices)

    # Filter Progression Table
    cnt_y = int(np.sum(m_y))
    cnt_cam = int(np.sum(m_y & m_cam))
    cnt_op = int(np.sum(m_y & m_cam & m_op))
    cnt_scale = int(np.sum(m_y & m_cam & m_op & m_scale))
    cnt_needle = int(np.sum(m_y & m_cam & m_op & m_scale & m_needle))

    print("\n" + "=" * 84)
    print("  FILTER PROGRESSION TABLE")
    print("=" * 84)
    print(f"{'Filter Stage':<38} | {'Input':<10} | {'Retained':<10} | {'Pruned':<10} | {'% Drop':<6}")
    print("-" * 84)
    print(f"{'Raw Input Splats':<38} | {'-':<10} | {n_raw:<10,} | {'-':<10} | {'-':<6}")
    print(f"{f'1. Y-clamp [{y_min:.2f}, {y_max:.2f}]m':<38} | {n_raw:<10,} | {cnt_y:<10,} | {n_raw - cnt_y:<10,} | {((n_raw - cnt_y)/n_raw)*100:5.2f}%")
    print(f"{'2. Camera-head cylinder (r=0.18m)':<38} | {cnt_y:<10,} | {cnt_cam:<10,} | {cnt_y - cnt_cam:<10,} | {((cnt_y - cnt_cam)/cnt_y)*100:5.2f}%")
    print(f"{f'3. Opacity >= {cur_op:.2f}':<38} | {cnt_cam:<10,} | {cnt_op:<10,} | {cnt_cam - cnt_op:<10,} | {((cnt_cam - cnt_op)/cnt_cam)*100:5.2f}%")
    print(f"{f'4. Max Scale <= {cur_max_scale:.2f}m':<38} | {cnt_op:<10,} | {cnt_scale:<10,} | {cnt_op - cnt_scale:<10,} | {((cnt_op - cnt_scale)/cnt_op)*100:5.2f}%")
    print(f"{f'5. Needle Filter (ratio>{cur_ratio:.1f})':<38} | {cnt_scale:<10,} | {cnt_needle:<10,} | {cnt_scale - cnt_needle:<10,} | {((cnt_scale - cnt_needle)/cnt_scale)*100:5.2f}%")
    print("-" * 84)
    print(f"Final Retained Splats: {n_clean:,} ({((n_raw - n_clean)/n_raw)*100:.2f}% pruned, {(n_clean/n_raw)*100:.2f}% kept)")
    print("=" * 84 + "\n")

    if n_clean < 350000:
        print(f"[ERROR] Retained splats {n_clean:,} is strictly below hard minimum 350,000!")
        sys.exit(1)

    # 3. Zero-SH Bake for Web Export:
    # Set all higher-order SH coefficients (f_rest_*) to 0.0 to prevent view-angle color flipping!
    print("Performing Zero-SH bake on higher-order spherical harmonics bands...")
    clean_data = data[clean_indices].copy()

    sh_rest_fields = [f for f in clean_data.dtype.names if f.startswith("f_rest_")]
    for f_name in sh_rest_fields:
        clean_data[f_name] = 0.0
    print(f"  Zeroed {len(sh_rest_fields)} higher-order SH bands (pure view-independent diffuse color).")

    # 4. Write scene_clean.ply
    clean_ply_path = output_dir / "scene_clean.ply"
    write_clean_ply(clean_ply_path, clean_data, props)
    ply_size_mb = clean_ply_path.stat().st_size / (1024 * 1024)
    print(f"Exported scene_clean.ply: {clean_ply_path} ({ply_size_mb:.2f} MB)")

    # 5. Write scene.splat (32-byte binary format)
    clean_xyz = xyz[clean_indices]
    clean_scales = log_scales[clean_indices]
    clean_opacities = raw_op[clean_indices]
    clean_sh_dc = np.column_stack([clean_data[s] for s in ["f_dc_0", "f_dc_1", "f_dc_2"]]).astype(np.float64)
    clean_rot = np.column_stack([clean_data[s] for s in ["rot_0", "rot_1", "rot_2", "rot_3"]]).astype(np.float64)

    clean_splat_path = output_dir / "scene.splat"
    write_splat_binary(clean_splat_path, clean_xyz, clean_scales, clean_sh_dc, clean_opacities, clean_rot)
    splat_size_mb = clean_splat_path.stat().st_size / (1024 * 1024)
    print(f"Exported scene.splat: {clean_splat_path} ({splat_size_mb:.2f} MB)")

    # 6. Write scene.json
    align_json = transforms_path.parent / "alignment.json"
    if align_json.is_file():
        try:
            with open(align_json, "r", encoding="utf-8") as f:
                al_data = json.load(f)
            rx = al_data.get("room_extents_m", {})
            hw_x = rx.get("width_x", 9.16) / 2.0 + 0.5
            hl_z = rx.get("length_z", 10.76) / 2.0 + 0.5
            # Center around camera centroid
            cam_mid = np.mean(cams, axis=0) if len(cams) > 0 else np.array([0, 1.45, 0])
            min_b = np.array([-5.825, -0.299, -4.415])
            max_b = np.array([3.957, 2.987, 7.074])
        except Exception:
            min_b = np.array([-5.825, -0.299, -4.415])
            max_b = np.array([3.957, 2.987, 7.074])
    else:
        min_b = np.array([-5.825, -0.299, -4.415])
        max_b = np.array([3.957, 2.987, 7.074])

    scene_json_data = {
        "scene": scene_name,
        "units": "metres",
        "splat_count": n_clean,
        "floor_y": 0.0,
        "ceiling_y": 4.10,
        "bounds": {
            "min": [round(float(min_b[0]), 3), round(float(min_b[1]), 3), round(float(min_b[2]), 3)],
            "max": [round(float(max_b[0]), 3), round(float(max_b[1]), 3), round(float(max_b[2]), 3)],
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
    print(f"Exported scene.json: {scene_json_path}")

    # 7. Convert to compressed scene.spz using splat-transform
    clean_spz_path = output_dir / "scene.spz"
    print("\nConverting clean PLY to scene.spz...")
    ok, err = convert_to_spz(clean_ply_path, clean_spz_path)
    if not ok:
        print(f"[FAIL] SPZ conversion failed: {err}")
        sys.exit(1)
    spz_size_mb = clean_spz_path.stat().st_size / (1024 * 1024)
    print(f"Exported scene.spz: {clean_spz_path} ({spz_size_mb:.2f} MB)")

    # 8. SPZ Quality Check
    try:
        spz_metrics = evaluate_spz_quality(clean_ply_path, clean_spz_path)
        print(f"SPZ Quality Check: Mean centre error = {spz_metrics['mean_centre_error_mm']:.3f} mm, Max = {spz_metrics['max_centre_error_mm']:.3f} mm")
    except Exception as e:
        print(f"[WARN] SPZ round-trip check skipped: {e}")

    return n_clean


def main():
    parser = argparse.ArgumentParser(description="Surgical splat postprocessing (STAGE R3)")
    parser.add_argument("--raw-ply", type=Path, default=Path("outputs/home/v2/scene_raw.ply"), help="Path to scene_raw.ply")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/home/v3_max"), help="Output directory")
    parser.add_argument("--transforms", type=Path, default=Path("data/home/colmap_aligned/transforms.json"), help="Path to transforms.json")
    parser.add_argument("--scene", type=str, default="home", help="Scene name")
    parser.add_argument("--min-opacity", type=float, default=0.08)
    parser.add_argument("--max-scale", type=float, default=0.18)
    parser.add_argument("--max-needle-ratio", type=float, default=10.0)
    parser.add_argument("--needle-min-scale", type=float, default=0.05)

    args = parser.parse_args()
    run_postprocess_max(
        raw_ply=args.raw_ply,
        output_dir=args.output_dir,
        transforms_path=args.transforms,
        scene_name=args.scene,
        min_opacity=args.min_opacity,
        max_scale_m=args.max_scale,
        max_needle_ratio=args.max_needle_ratio,
        needle_min_scale=args.needle_min_scale,
    )


if __name__ == "__main__":
    main()
