#!/usr/bin/env python3
"""Execute Experiment E1."""

import os
import sys
import shutil
from pathlib import Path
import numpy as np

# Ensure root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from scripts.sfm_experiment_runner import run_cmd, parse_model

exp_dir = Path("colmap_runs/E1")
exp_dir.mkdir(parents=True, exist_ok=True)

images_dir = Path("data/home/images")
all_frames = sorted(images_dir.glob("frame_*.jpg"))
v1_frames = [f.name for f in all_frames if int(f.stem.split("_")[1]) <= 222]

image_list_path = exp_dir / "image_list.txt"
with open(image_list_path, "w", encoding="utf-8") as f:
    for name in v1_frames:
        f.write(f"{name}\n")

print(f"E1 image list written: {len(v1_frames)} frames (Video 1 alone).")

db_path = exp_dir / "database.db"
sparse_dir = exp_dir / "sparse"
sparse_dir.mkdir(parents=True, exist_ok=True)

# 1. Feature extraction
feat_cmd = [
    "colmap", "feature_extractor",
    "--database_path", str(db_path),
    "--image_path", str(images_dir),
    "--image_list_path", str(image_list_path),
    "--ImageReader.camera_model", "SIMPLE_RADIAL",
    "--ImageReader.single_camera", "1",
    "--ImageReader.default_focal_length_factor", "0.75",
    "--ImageReader.camera_params", "1440,960,540,0",
    "--SiftExtraction.use_gpu", "1"
]
run_cmd(feat_cmd, "E1: Feature Extraction")

# 2. Sequential Matching
match_cmd = [
    "colmap", "sequential_matcher",
    "--database_path", str(db_path),
    "--SiftMatching.use_gpu", "1",
    "--SequentialMatching.overlap", "15",
    "--SequentialMatching.loop_detection", "0",
    "--SiftMatching.guided_matching", "1"
]
run_cmd(match_cmd, "E1: Sequential Matching")

# 3. Mapper
mapper_cmd = [
    "colmap", "mapper",
    "--database_path", str(db_path),
    "--image_path", str(images_dir),
    "--image_list_path", str(image_list_path),
    "--output_path", str(sparse_dir),
    "--Mapper.ba_refine_focal_length", "1",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0"
]
run_cmd(mapper_cmd, "E1: Mapper")

# 4. Analyze results
sub_models = [d for d in sparse_dir.iterdir() if d.is_dir() and (d / "cameras.bin").exists() or (d / "cameras.txt").exists()]
print(f"\nNumber of models created: {len(sub_models)}")

if not sub_models:
    print("NO MODELS CREATED!")
    sys.exit(1)

# Pick largest model
model_stats = []
for m in sub_models:
    try:
        stats = parse_model(m, total_v1_frames=len(v1_frames), total_v2_frames=0)
        model_stats.append((m, stats))
    except Exception as e:
        print(f"Error parsing model {m}: {e}")

model_stats.sort(key=lambda x: x[1]["registered_total"], reverse=True)
best_model, best_stats = model_stats[0]

print(f"\n=== Best Model: {best_model.name} ===")
print(f"Total models: {len(sub_models)}")
reg_v1 = best_stats['v1_registered']
reg_v1_pct = (reg_v1 / len(v1_frames)) * 100
print(f"Registered frames (Video 1): {reg_v1} / {len(v1_frames)} ({reg_v1_pct:.2f}%)")
print(f"Mean reprojection error: {best_stats['mean_reproj_error']:.4f} px")
print(f"Intrinsics: fx={best_stats['fx']:.2f}, fy={best_stats['fy']:.2f}, k1={best_stats['k1']:.4f}")
print(f"Step lengths: median={best_stats['median_step']:.4f}, largest={best_stats['largest_step']:.4f}, ratio={best_stats['step_ratio']:.2f}x")
print(f"Start-to-end gap: {best_stats['start_end_gap']:.4f} model units")

# Check pass criteria:
# - At least 90% of frames of the video(s) used are in ONE model
# - Mean reprojection error under 1.0 px
# - fx and fy within 5% of each other, |k1| < 0.3
# - No frame-to-frame step larger than 5x the median step
crit_reg = reg_v1_pct >= 90.0
crit_reproj = best_stats['mean_reproj_error'] < 1.0
crit_cam = (abs(best_stats['fx'] - best_stats['fy']) / best_stats['fy'] <= 0.05) and (abs(best_stats['k1']) < 0.3)
crit_step = best_stats['step_ratio'] <= 5.0

passed = crit_reg and crit_reproj and crit_cam and crit_step
print(f"\nPass Criteria Check:")
print(f"  1. >=90% frames in ONE model: {'PASS' if crit_reg else 'FAIL'} ({reg_v1_pct:.1f}%)")
print(f"  2. Reprojection error < 1.0 px: {'PASS' if crit_reproj else 'FAIL'} ({best_stats['mean_reproj_error']:.4f} px)")
print(f"  3. fx ~ fy within 5%, |k1| < 0.3: {'PASS' if crit_cam else 'FAIL'} (fx={best_stats['fx']:.1f}, fy={best_stats['fy']:.1f}, k1={best_stats['k1']:.4f})")
print(f"  4. Max step <= 5x median step: {'PASS' if crit_step else 'FAIL'} ({best_stats['step_ratio']:.2f}x)")
print(f"OVERALL E1: {'PASS' if passed else 'FAIL'}")
