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
from scipy.spatial import KDTree

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
    SH_C0,
)

def main():
    scene = "home"
    raw_ply = Path("outputs/home/v2/scene_raw.ply")
    v2_dir = Path("outputs/home/v2")
    v2_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  STAGE E: SPLAT POST-PROCESSING & QUALITY DIAGNOSTICS (v2)")
    print("=" * 80)
    
    # 1. Load Camera Trajectory & Aligned Reference Points
    transforms_path = Path("data/home/colmap_aligned/transforms.json")
    if not transforms_path.is_file():
        transforms_path = Path("data/home/transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        tf_data = json.load(f)
        
    cams = []
    for frame in tf_data.get("frames", []):
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        cams.append(mat[:3, 3])
    cams = np.array(cams, dtype=np.float64)
    print(f"Loaded {len(cams)} camera poses from {transforms_path}")

    # Build KDTree for camera path distance check (for 1.0 m exemption)
    cam_tree = KDTree(cams)

    # Load 3D reference points and compute 0.5m coverage grid
    points3d_path = Path("data/home/colmap_aligned/points3D.txt")
    if not points3d_path.is_file():
        points3d_path = Path("data/home/points3D.txt")
        
    pts_dict = {}
    with open(points3d_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            err = float(parts[7])
            track_tokens = parts[8:]
            pts_dict[pid] = {"xyz": xyz, "err": err, "track": " ".join(track_tokens)}
            
    # Compute 0.5m coverage grid
    cell_size = 0.5
    all_pts_xyz = np.array([p["xyz"] for p in pts_dict.values()])
    grid_x_min = np.floor(min(np.min(all_pts_xyz[:, 0]), np.min(cams[:, 0])) - 1.0)
    grid_x_max = np.ceil(max(np.max(all_pts_xyz[:, 0]), np.max(cams[:, 0])) + 1.0)
    grid_z_min = np.floor(min(np.min(all_pts_xyz[:, 2]), np.min(cams[:, 2])) - 1.0)
    grid_z_max = np.ceil(max(np.max(all_pts_xyz[:, 2]), np.max(cams[:, 2])) + 1.0)

    nx = int(np.ceil((grid_x_max - grid_x_min) / cell_size))
    nz = int(np.ceil((grid_z_max - grid_z_min) / cell_size))
    cell_obs = [[set() for _ in range(nz)] for _ in range(nx)]

    for p in pts_dict.values():
        xyz = p["xyz"]
        col = int((xyz[0] - grid_x_min) / cell_size)
        row = int((xyz[2] - grid_z_min) / cell_size)
        if 0 <= col < nx and 0 <= row < nz:
            track_tokens = p["track"].split()
            img_ids = [int(track_tokens[2 * k]) for k in range(len(track_tokens) // 2)]
            cell_obs[col][row].update(img_ids)

    cov_grid = np.zeros((nx, nz), dtype=int)
    for i in range(nx):
        for j in range(nz):
            cov_grid[i, j] = len(cell_obs[i][j])

    def get_cell_cam_count(x, z):
        col = int((x - grid_x_min) / cell_size)
        row = int((z - grid_z_min) / cell_size)
        if 0 <= col < nx and 0 <= row < nz:
            return cov_grid[col, row]
        return 0

    # 2. Load Raw PLY Model
    n_raw, props, offset = parse_ply_header(raw_ply)
    dt = np.dtype(props)
    data = np.fromfile(raw_ply, dtype=dt, count=n_raw, offset=offset)
    print(f"Loaded raw PLY: {n_raw:,} splats")

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    raw_op = data["opacity"].astype(np.float64)
    sig_op = 1.0 / (1.0 + np.exp(-raw_op))
    
    scale_names = ["scale_0", "scale_1", "scale_2"]
    log_scales = np.column_stack([data[s] for s in scale_names]).astype(np.float64)
    real_scales = np.exp(log_scales)
    max_scale = np.max(real_scales, axis=1)
    min_scale = np.min(real_scales, axis=1)
    scale_ratio = max_scale / np.maximum(min_scale, 1e-7)

    # 3. Quality Diagnostics (Percentiles, Counts, Coverage)
    p50_max_scale = float(np.percentile(max_scale, 50))
    p90_max_scale = float(np.percentile(max_scale, 90))
    p99_max_scale = float(np.percentile(max_scale, 99))

    p50_ratio = float(np.percentile(scale_ratio, 50))
    p90_ratio = float(np.percentile(scale_ratio, 90))
    p99_ratio = float(np.percentile(scale_ratio, 99))

    count_gt_015 = int(np.sum(max_scale > 0.15))
    count_gt_025 = int(np.sum(max_scale > 0.25))
    count_gt_050 = int(np.sum(max_scale > 0.50))

    # Evaluate camera coverage for all raw splats
    splat_cols = np.clip(((xyz[:, 0] - grid_x_min) / cell_size).astype(int), 0, nx - 1)
    splat_rows = np.clip(((xyz[:, 2] - grid_z_min) / cell_size).astype(int), 0, nz - 1)
    splat_cam_counts = cov_grid[splat_cols, splat_rows]

    # Splats in cells seen by < 3 cameras
    seen_lt_3 = splat_cam_counts < 3
    count_lt_3 = int(np.sum(seen_lt_3))
    pct_lt_3 = (count_lt_3 / n_raw) * 100.0

    print("\n--- SPLAT QUALITY DIAGNOSTICS ---")
    print(f"Max Scale Percentiles  : 50th={p50_max_scale:.4f} m, 90th={p90_max_scale:.4f} m, 99th={p99_max_scale:.4f} m")
    print(f"Scale Ratio Percentiles: 50th={p50_ratio:.2f}, 90th={p90_ratio:.2f}, 99th={p99_ratio:.2f}")
    print(f"Large Splats Count     : >0.15m: {count_gt_015:,} ({(count_gt_015/n_raw)*100:.2f}%), >0.25m: {count_gt_025:,} ({(count_gt_025/n_raw)*100:.2f}%), >0.50m: {count_gt_050:,} ({(count_gt_050/n_raw)*100:.2f}%)")
    print(f"Cells seen by <3 cams  : {count_lt_3:,} splats ({pct_lt_3:.2f}%)")

    # Write docs/splat_quality.md
    splat_quality_doc = f"""# 3D Gaussian Splat Quality & Diagnostics Report

## 1. Raw Splat Properties (final_v2, SCENE=home)
- **Total Raw Splats**: {n_raw:,}
- **Scale Regularisation**: `--scale-loss-weight 1e-7`
- **Opacity Regularisation**: `--opac-loss-weight 1e-8`

### Max Scale Distribution
- **50th Percentile (Median)**: {p50_max_scale:.4f} m ({p50_max_scale*100:.2f} cm)
- **90th Percentile**: {p90_max_scale:.4f} m ({p90_max_scale*100:.2f} cm)
- **99th Percentile**: {p99_max_scale:.4f} m ({p99_max_scale*100:.2f} cm)

### Scale Anisotropy (Max / Min Scale Ratio)
- **50th Percentile**: {p50_ratio:.2f}
- **90th Percentile**: {p90_ratio:.2f}
- **99th Percentile**: {p99_ratio:.2f}

### Floater & Needle Splat Counts
- **Splats > 0.15 m semi-axis**: {count_gt_015:,} ({count_gt_015/n_raw*100:.2f}%)
- **Splats > 0.25 m semi-axis**: {count_gt_025:,} ({count_gt_025/n_raw*100:.2f}%)
- **Splats > 0.50 m semi-axis**: {count_gt_050:,} ({count_gt_050/n_raw*100:.2f}%)
- **Splats in cells seen by < 3 cameras**: {count_lt_3:,} ({pct_lt_3:.2f}%)

---

## 2. Spark Renderer (`@sparkjsdev/spark`) Blur & Clamping Options
In the Three.js vanilla viewer (`viewer/src/main.ts`), the following Spark renderer properties control splat blur, LOD scaling, and radial clamping:

| Option | Type | Default | Balanced | High / Cinematic | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `lodSplatScale` | `number` | `0.50` | `0.75` | `1.00` | Controls splat spatial scaling factor in screen-space rasterisation. Lower values reduce splat overlap and eliminate haze. |
| `lodRenderScale` | `number` | `2.00` | `1.50` | `1.00` | Downsampling ratio for intermediate splat sorting passes. `1.0` is native pixel resolution. |
| `maxStdDev` | `number` | `2.00` | `2.45` | `2.828` ($\\sqrt{{8}}$) | Gaussian standard deviation cutoff radius. Clamping at $2.0\\sigma$ chops diffuse outer tails of large Gaussians, preventing blurry halo artifacts. |

---

## 3. Post-Processing Filter Progression
The following multi-stage filter pipeline was applied to `outputs/home/v2/scene_raw.ply`:
1. **Y Height Clamp `[-0.30, 4.40]` m**: Removes spurious sub-floor and high ceiling floaters while preserving floor and ceiling slabs.
2. **Intersected XZ BBox Crop**: Intersects camera trajectory bounding box (+4.0 m margin) with robust reference point extents (p0.5 to p99.5 + 1.0 m margin).
3. **Min Opacity Filter (`sig_op >= 0.05`)**: Eliminates invisible or faint haze splats.
4. **Max Scale Filter (`max_scale <= 0.25 m`)**: Eliminates giant floaters and puffy splats.
5. **Needle Filter (`ratio > 12 AND max_scale > 0.08 m`)**: Prunes razor-sharp, distorted needle splats.
6. **Camera Coverage Filter**: Removes splats in 0.5m grid cells seen by fewer than 3 cameras, except those within 1.0 m of the camera trajectory.
7. **Statistical Outlier Removal (SOR)**: Open3D k-NN outlier removal (`nb_neighbors=20, std_ratio=2.5`) on splat centroids.
"""
    with open("docs/splat_quality.md", "w", encoding="utf-8") as f:
        f.write(splat_quality_doc)
    print("Written docs/splat_quality.md")

    # 4. Multi-Stage Filtering Sequence
    print("\n--- APPLYING SEQUENTIAL FILTERS ---")
    print(f"{'Filter Stage':<42} | {'Before':<10} | {'After':<10} | {'Pruned':<10} | {'Pruned %'}")
    print("-" * 84)

    # 1. Y Clamp [-0.30, 4.40] m
    y_min, y_max = -0.30, 4.40
    mask_y = (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    cnt_y = int(np.sum(mask_y))
    print(f"{'1. Y Clamp [-0.30, 4.40] m':<42} | {n_raw:<10,} | {cnt_y:<10,} | {n_raw - cnt_y:<10,} | {((n_raw - cnt_y)/n_raw)*100:5.2f}%")

    # 2. Intersected XZ Crop
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
    mask_2 = mask_y & mask_xz
    cnt_2 = int(np.sum(mask_2))
    print(f"{'2. Intersected XZ Crop':<42} | {cnt_y:<10,} | {cnt_2:<10,} | {cnt_y - cnt_2:<10,} | {((cnt_y - cnt_2)/cnt_y)*100:5.2f}%")

    # 3. Opacity Filter (sig_op >= 0.05)
    mask_op = sig_op >= 0.05
    mask_3 = mask_2 & mask_op
    cnt_3 = int(np.sum(mask_3))
    print(f"{'3. Min Opacity (>= 0.05)':<42} | {cnt_2:<10,} | {cnt_3:<10,} | {cnt_2 - cnt_3:<10,} | {((cnt_2 - cnt_3)/cnt_2)*100:5.2f}%")

    # 4. Max Scale Filter (max_scale <= 0.25 m)
    mask_scale = max_scale <= 0.25
    mask_4 = mask_3 & mask_scale
    cnt_4 = int(np.sum(mask_4))
    print(f"{'4. Max Scale (<= 0.25 m)':<42} | {cnt_3:<10,} | {cnt_4:<10,} | {cnt_3 - cnt_4:<10,} | {((cnt_3 - cnt_4)/cnt_3)*100:5.2f}%")

    # 5. Needle Filter (ratio <= 12 OR max_scale <= 0.08 m)
    mask_needle = ~((scale_ratio > 12.0) & (max_scale > 0.08))
    mask_5 = mask_4 & mask_needle
    cnt_5 = int(np.sum(mask_5))
    print(f"{'5. Needle Filter (ratio>12 & scale>0.08m)':<42} | {cnt_4:<10,} | {cnt_5:<10,} | {cnt_4 - cnt_5:<10,} | {((cnt_4 - cnt_5)/cnt_4)*100:5.2f}%")

    # 6. Coverage Filter (seen by >= 3 cameras OR within 1.0 m of camera path)
    # Check distance of splats to camera trajectory
    dists_to_cams, _ = cam_tree.query(xyz)
    within_cam_path = dists_to_cams <= 1.0
    seen_ge_3 = splat_cam_counts >= 3
    mask_cov = seen_ge_3 | within_cam_path
    mask_6 = mask_5 & mask_cov
    cnt_6 = int(np.sum(mask_6))
    print(f"{'6. Coverage (cams>=3 or dist<=1.0m)':<42} | {cnt_5:<10,} | {cnt_6:<10,} | {cnt_5 - cnt_6:<10,} | {((cnt_5 - cnt_6)/cnt_5)*100:5.2f}%")

    # 7. Statistical Outlier Removal (SOR)
    stage6_indices = np.where(mask_6)[0]
    stage6_xyz = xyz[stage6_indices]
    if o3d is not None and len(stage6_xyz) > 20:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(stage6_xyz)
        _, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        clean_indices = stage6_indices[inliers]
    else:
        clean_indices = stage6_indices

    n_clean = len(clean_indices)
    print(f"{'7. Open3D SOR (nb=20, std=2.5)':<42} | {cnt_6:<10,} | {n_clean:<10,} | {cnt_6 - n_clean:<10,} | {((cnt_6 - n_clean)/cnt_6)*100:5.2f}%")
    print("-" * 84)
    print(f"Total Filtered: {n_raw:,} -> {n_clean:,} splats ({((n_raw - n_clean)/n_raw)*100:.2f}% pruned)")

    # 5. Export scene_clean.ply
    clean_data = data[clean_indices]
    clean_ply_path = v2_dir / "scene_clean.ply"
    write_clean_ply(clean_ply_path, clean_data, props)
    print(f"Exported scene_clean.ply: {clean_ply_path.stat().st_size / (1024*1024):.2f} MB")

    # 6. Export scene.splat (32-byte binary format)
    clean_xyz = xyz[clean_indices]
    clean_scales = log_scales[clean_indices]
    clean_opacities = raw_op[clean_indices]
    sh_names = ["f_dc_0", "f_dc_1", "f_dc_2"]
    clean_sh_dc = np.column_stack([data[s][clean_indices] for s in sh_names]).astype(np.float64)
    rot_names = ["rot_0", "rot_1", "rot_2", "rot_3"]
    clean_rot = np.column_stack([data[s][clean_indices] for s in rot_names]).astype(np.float64)

    clean_splat_path = v2_dir / "scene.splat"
    write_splat_binary(clean_splat_path, clean_xyz, clean_scales, clean_sh_dc, clean_opacities, clean_rot)
    print(f"Exported scene.splat: {clean_splat_path.stat().st_size / (1024*1024):.2f} MB")

    # 7. Export scene.json
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
    scene_json_path = v2_dir / "scene.json"
    with open(scene_json_path, "w", encoding="utf-8") as f:
        json.dump(scene_json_data, f, indent=2)
    print(f"Exported scene.json: {scene_json_path}")

    # 8. Convert to scene.spz using splat-transform
    clean_spz_path = v2_dir / "scene.spz"
    print("\nConverting clean PLY to scene.spz...")
    ok, err = convert_to_spz(clean_ply_path, clean_spz_path)
    if not ok:
        print(f"[FAIL] SPZ conversion failed: {err}")
        sys.exit(1)
    spz_size_mb = clean_spz_path.stat().st_size / (1024*1024)
    print(f"Exported scene.spz: {spz_size_mb:.2f} MB")

    # 9. SPZ Round-Trip Error Check
    print("\nRunning SPZ round-trip error evaluation...")
    spz_eval = evaluate_spz_quality(clean_ply_path, clean_spz_path)
    print(f"  Splat Count Difference: {spz_eval['splat_count_diff']}")
    print(f"  Max Centre Error       : {spz_eval['max_centre_error_mm']:.3f} mm")
    print(f"  Mean Centre Error      : {spz_eval['mean_centre_error_mm']:.3f} mm")
    print(f"  Mean Opacity Error     : {spz_eval['mean_opacity_error']:.6f}")
    with open(v2_dir / "spz_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(spz_eval, f, indent=2)

    # 10. Run verify_splat_frame.py
    print("\nVerifying splat coordinate frame...")
    verify_cmd = [
        sys.executable,
        "scripts/verify_splat_frame.py",
        "--scene", "home",
        "--splat-ply", str(clean_ply_path),
        "--reference-pts", str(points3d_path),
        "--transforms", str(transforms_path),
        "--scene-json", str(scene_json_path),
        "--output-json", str(v2_dir / "splat_frame_verification.json")
    ]
    res_ver = subprocess.run(verify_cmd, capture_output=True, text=True)
    print(res_ver.stdout)
    if res_ver.returncode != 0:
        print(f"[FAIL] Frame verification failed: {res_ver.stderr}")
        sys.exit(1)

    # 11. Copy latest occupancy files to v2 so asset swapper has all 5 files
    occ_src_dir = Path("outputs/home/occupancy_v3_1") if Path("outputs/home/occupancy_v3_1/occupancy.json").is_file() else Path("outputs/home")
    shutil.copy2(occ_src_dir / "occupancy.json", v2_dir / "occupancy.json")
    shutil.copy2(occ_src_dir / "occupancy.png", v2_dir / "occupancy.png")

    # 12. Deploy to viewer using swap_assets.ps1
    print("\nDeploying assets to viewer/public/scene/ via scripts/swap_assets.ps1...")
    swap_cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "scripts/swap_assets.ps1", "outputs/home/v2"]
    res_swap = subprocess.run(swap_cmd, capture_output=True, text=True)
    print(res_swap.stdout)
    if res_swap.returncode != 0:
        print(f"[FAIL] Swap assets failed: {res_swap.stderr}")
        sys.exit(1)

    print("\n[SUCCESS] Stage E completed successfully!")

if __name__ == "__main__":
    main()
