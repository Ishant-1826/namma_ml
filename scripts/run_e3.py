#!/usr/bin/env python3
"""Execute Experiment E3.
Add video 2 as a SECOND camera (one camera per video folder: --ImageReader.single_camera_per_folder 1),
keeping the settings from the best of E1/E2 (E1). Reconstruct both together.
"""

import os
import sys
import shutil
import sqlite3
from pathlib import Path
import numpy as np

# Ensure root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from scripts.sfm_experiment_runner import run_cmd, parse_model

exp_dir = Path("colmap_runs/E3")
exp_dir.mkdir(parents=True, exist_ok=True)

# Create structured images directory for one camera per video folder
images_root = exp_dir / "images"
v1_dir = images_root / "video1"
v2_dir = images_root / "video2"
v1_dir.mkdir(parents=True, exist_ok=True)
v2_dir.mkdir(parents=True, exist_ok=True)

src_images_dir = Path("data/home/images")
all_frames = sorted(src_images_dir.glob("frame_*.jpg"))

for f in all_frames:
    frame_num = int(f.stem.split("_")[1])
    target = (v1_dir if frame_num <= 222 else v2_dir) / f.name
    if not target.exists():
        try:
            os.link(f, target)
        except Exception:
            shutil.copy2(f, target)

v1_count = len(list(v1_dir.glob("*.jpg")))
v2_count = len(list(v2_dir.glob("*.jpg")))
print(f"E3 image directory structured: video1={v1_count} frames, video2={v2_count} frames.")

db_path = exp_dir / "database.db"
sparse_dir = exp_dir / "sparse"
sparse_dir.mkdir(parents=True, exist_ok=True)

# 1. Feature extraction with --ImageReader.single_camera_per_folder 1
feat_cmd = [
    "colmap", "feature_extractor",
    "--database_path", str(db_path),
    "--image_path", str(images_root),
    "--ImageReader.camera_model", "SIMPLE_RADIAL",
    "--ImageReader.single_camera_per_folder", "1",
    "--ImageReader.default_focal_length_factor", "0.75",
    "--ImageReader.camera_params", "1440,960,540,0",
    "--SiftExtraction.use_gpu", "1"
]
run_cmd(feat_cmd, "E3: Feature Extraction (single_camera_per_folder 1)")

# Verify 2 cameras created in database
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute("SELECT count(*) FROM cameras")
num_cams = cur.fetchone()[0]
cur.execute("SELECT camera_id, model, width, height, params FROM cameras")
cam_rows = cur.fetchall()
conn.close()
print(f"Cameras created in DB: {num_cams}")
for r in cam_rows:
    print(f"  Camera {r[0]}: model={r[1]}, size={r[2]}x{r[3]}")

# 2. Sequential Matching (from best of E1: overlap 15, loop detection 0, guided matching 1)
match_cmd = [
    "colmap", "sequential_matcher",
    "--database_path", str(db_path),
    "--SiftMatching.use_gpu", "1",
    "--SequentialMatching.overlap", "15",
    "--SequentialMatching.loop_detection", "0",
    "--SiftMatching.guided_matching", "1"
]
run_cmd(match_cmd, "E3: Sequential Matching")

# 3. Mapper (settings from E1: no refinement of principal point or extra params)
mapper_cmd = [
    "colmap", "mapper",
    "--database_path", str(db_path),
    "--image_path", str(images_root),
    "--output_path", str(sparse_dir),
    "--Mapper.ba_refine_focal_length", "1",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0"
]
run_cmd(mapper_cmd, "E3: Mapper")

# 4. Analyze results
sub_models = [d for d in sparse_dir.iterdir() if d.is_dir() and (d / "cameras.bin").exists() or (d / "cameras.txt").exists()]
print(f"\nNumber of models created: {len(sub_models)}")

if not sub_models:
    print("NO MODELS CREATED!")
    sys.exit(1)

model_stats = []
for m in sub_models:
    try:
        stats = parse_model(m, total_v1_frames=v1_count, total_v2_frames=v2_count)
        model_stats.append((m, stats))
    except Exception as e:
        print(f"Error parsing model {m}: {e}")

model_stats.sort(key=lambda x: x[1]["registered_total"], reverse=True)
best_model, best_stats = model_stats[0]

print(f"\n=== Best Model: {best_model.name} ===")
print(f"Total models: {len(sub_models)}")
reg_v1 = best_stats['v1_registered']
reg_v2 = best_stats['v2_registered']
reg_total = best_stats['registered_total']
total_frames = v1_count + v2_count
reg_v1_pct = (reg_v1 / v1_count) * 100
reg_v2_pct = (reg_v2 / v2_count) * 100
reg_total_pct = (reg_total / total_frames) * 100

print(f"Registered frames (Video 1): {reg_v1} / {v1_count} ({reg_v1_pct:.2f}%)")
print(f"Registered frames (Video 2): {reg_v2} / {v2_count} ({reg_v2_pct:.2f}%)")
print(f"Registered frames (Total): {reg_total} / {total_frames} ({reg_total_pct:.2f}%)")
print(f"Mean reprojection error: {best_stats['mean_reproj_error']:.4f} px")
print(f"Intrinsics: fx={best_stats['fx']:.2f}, fy={best_stats['fy']:.2f}, k1={best_stats['k1']:.4f}")
print(f"Step lengths: median={best_stats['median_step']:.4f}, largest={best_stats['largest_step']:.4f}, ratio={best_stats['step_ratio']:.2f}x")
print(f"Start-to-end gap: {best_stats['start_end_gap']:.4f} model units")

crit_reg = reg_total_pct >= 90.0
crit_reproj = best_stats['mean_reproj_error'] < 1.0
crit_cam = (abs(best_stats['fx'] - best_stats['fy']) / best_stats['fy'] <= 0.05) and (abs(best_stats['k1']) < 0.3)
crit_step = best_stats['step_ratio'] <= 5.0

passed = crit_reg and crit_reproj and crit_cam and crit_step
print(f"\nPass Criteria Check:")
print(f"  1. >=90% frames in ONE model: {'PASS' if crit_reg else 'FAIL'} ({reg_total_pct:.1f}%)")
print(f"  2. Reprojection error < 1.0 px: {'PASS' if crit_reproj else 'FAIL'} ({best_stats['mean_reproj_error']:.4f} px)")
print(f"  3. fx ~ fy within 5%, |k1| < 0.3: {'PASS' if crit_cam else 'FAIL'} (fx={best_stats['fx']:.1f}, fy={best_stats['fy']:.1f}, k1={best_stats['k1']:.4f})")
print(f"  4. Max step <= 5x median step: {'PASS' if crit_step else 'FAIL'} ({best_stats['step_ratio']:.2f}x)")
print(f"OVERALL E3: {'PASS' if passed else 'FAIL'}")
