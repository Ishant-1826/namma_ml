#!/usr/bin/env python3
"""Execute Experiment E5:
Re-extract frames at 4 fps from the sharpest segments and rerun the best settings (E3).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Ensure root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.extract_frames import compute_blur_score, filter_blur_sliding_window, is_near_duplicate, resize_longest_edge
from scripts.sfm_experiment_runner import run_cmd, parse_model

exp_dir = Path("colmap_runs/E5")
exp_dir.mkdir(parents=True, exist_ok=True)

images_root = exp_dir / "images"
v1_dir = images_root / "video1"
v2_dir = images_root / "video2"
v1_dir.mkdir(parents=True, exist_ok=True)
v2_dir.mkdir(parents=True, exist_ok=True)

temp_dir = exp_dir / ".extract_tmp"
temp_dir.mkdir(parents=True, exist_ok=True)

v1_video = Path("data/home/raw/20261021_195555.mp4")
v2_video = Path("data/home/raw/20261021_195930.mp4")

ffmpeg_bin = shutil.which("ffmpeg") or str(_REPO_ROOT / "venv/Scripts/ffmpeg.exe")

def extract_sharp_frames(video_path, out_dir, target_count=225):
    print(f"\n--- Extracting {video_path.name} at 4 fps ---")
    v_tmp = temp_dir / video_path.stem
    v_tmp.mkdir(parents=True, exist_ok=True)
    out_pattern = str(v_tmp / "raw_%06d.jpg")
    
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vf", "fps=4.0",
        "-q:v", "2",
        out_pattern
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    raw_files = sorted(v_tmp.glob("raw_*.jpg"))
    print(f"Extracted {len(raw_files)} raw candidate frames at 4 fps.")
    
    # Compute sharpness
    blur_scores = []
    for rf in raw_files:
        img = cv2.imread(str(rf))
        blur_scores.append(compute_blur_score(img) if img is not None else 0.0)
        
    # Sliding window N=2 to pick sharpest in each segment, dropping bottom 20%
    kept_indices, threshold, dropped = filter_blur_sliding_window(
        blur_scores, window_size=2, percentile_threshold=20.0
    )
    print(f"Sliding window selected {len(kept_indices)} sharp frames (threshold={threshold:.1f}, dropped {dropped}).")
    
    # Near-duplicate filter
    final_indices = []
    last_img = None
    for idx in kept_indices:
        img = cv2.imread(str(raw_files[idx]))
        if last_img is not None and is_near_duplicate(last_img, img, ssim_threshold=0.985):
            continue
        final_indices.append(idx)
        last_img = img
        
    print(f"After near-duplicate filtering: {len(final_indices)} frames.")
    
    # Subsample to target_count if needed
    if len(final_indices) > target_count:
        sub_idx = np.round(np.linspace(0, len(final_indices) - 1, target_count)).astype(int)
        final_indices = [final_indices[i] for i in sub_idx]
        print(f"Uniformly subsampled to {len(final_indices)} frames.")
        
    # Copy/save to out_dir
    saved_paths = []
    for count, idx in enumerate(final_indices, start=1):
        src = raw_files[idx]
        dst = out_dir / f"frame_{count:06d}.jpg"
        img = cv2.imread(str(src))
        img = resize_longest_edge(img, 1920)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(dst)
        
    return len(saved_paths)

v1_num = extract_sharp_frames(v1_video, v1_dir, target_count=225)
v2_num = extract_sharp_frames(v2_video, v2_dir, target_count=225)
total_frames = v1_num + v2_num
print(f"\nTotal sharp frames prepared: {total_frames} (Video 1: {v1_num}, Video 2: {v2_num})")

# Clean temp directory
shutil.rmtree(temp_dir, ignore_errors=True)

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
run_cmd(feat_cmd, "E5: Feature Extraction")

# 2. Sequential Matching (overlap 15, guided matching)
match_cmd = [
    "colmap", "sequential_matcher",
    "--database_path", str(db_path),
    "--SiftMatching.use_gpu", "1",
    "--SequentialMatching.overlap", "15",
    "--SequentialMatching.loop_detection", "0",
    "--SiftMatching.guided_matching", "1"
]
run_cmd(match_cmd, "E5: Sequential Matching")

# 3. Mapper (best settings from E3: ba_refine_extra_params 0)
mapper_cmd = [
    "colmap", "mapper",
    "--database_path", str(db_path),
    "--image_path", str(images_root),
    "--output_path", str(sparse_dir),
    "--Mapper.ba_refine_focal_length", "1",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0"
]
run_cmd(mapper_cmd, "E5: Mapper")

# 4. Analyze results
sub_models = [d for d in sparse_dir.iterdir() if d.is_dir() and (d / "cameras.bin").exists() or (d / "cameras.txt").exists()]
print(f"\nNumber of models created: {len(sub_models)}")

if not sub_models:
    print("NO MODELS CREATED!")
    sys.exit(1)

model_stats = []
for m in sub_models:
    try:
        stats = parse_model(m, total_v1_frames=v1_num, total_v2_frames=v2_num)
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
reg_v1_pct = (reg_v1 / v1_num) * 100
reg_v2_pct = (reg_v2 / v2_num) * 100
reg_total_pct = (reg_total / total_frames) * 100

print(f"Registered frames (Video 1): {reg_v1} / {v1_num} ({reg_v1_pct:.2f}%)")
print(f"Registered frames (Video 2): {reg_v2} / {v2_num} ({reg_v2_pct:.2f}%)")
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
print(f"OVERALL E5: {'PASS' if passed else 'FAIL'}")
