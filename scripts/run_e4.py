#!/usr/bin/env python3
"""Execute Experiment E4: Run global_mapper (GLOMAP via pycolmap.global_mapping) on the best database."""

import os
import sys
import shutil
from pathlib import Path

# Ensure root
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import pycolmap
from scripts.sfm_experiment_runner import parse_model

exp_dir = Path("colmap_runs/E4")
exp_dir.mkdir(parents=True, exist_ok=True)

# Best database is E3 (contains both videos, 2 cameras)
best_db = Path("colmap_runs/E3/database.db")
best_images = Path("colmap_runs/E3/images")

# Copy DB to E4
e4_db = exp_dir / "database.db"
shutil.copy2(best_db, e4_db)

sparse_dir = exp_dir / "sparse"
sparse_dir.mkdir(parents=True, exist_ok=True)

print(f"Running global_mapper (GLOMAP) on {e4_db}...")
try:
    options = pycolmap.GlobalPipelineOptions()
    reconstructions = pycolmap.global_mapping(
        database_path=e4_db,
        image_path=best_images,
        output_path=sparse_dir,
        options=options
    )
    print(f"Global mapping completed. Number of reconstructions: {len(reconstructions)}")
    for idx, recon in reconstructions.items():
        print(f"  Recon {idx}: {len(recon.images)} images, {len(recon.points3D)} points, error={recon.compute_mean_reprojection_error():.4f}px")
        # Save recon to sparse/<idx>
        out_sub = sparse_dir / str(idx)
        out_sub.mkdir(parents=True, exist_ok=True)
        recon.write(out_sub)
        recon.write_text(out_sub)

    # Analyze with sfm_experiment_runner
    stats = parse_model(sparse_dir / "0", total_v1_frames=222, total_v2_frames=228)
    print("\n=== E4 Model 0 Stats ===")
    v1_count = 222
    v2_count = 228
    reg_v1 = stats['v1_registered']
    reg_v2 = stats['v2_registered']
    reg_total = stats['registered_total']
    total_frames = v1_count + v2_count
    reg_v1_pct = (reg_v1 / v1_count) * 100
    reg_v2_pct = (reg_v2 / v2_count) * 100
    reg_total_pct = (reg_total / total_frames) * 100

    print(f"Registered frames (Video 1): {reg_v1} / {v1_count} ({reg_v1_pct:.2f}%)")
    print(f"Registered frames (Video 2): {reg_v2} / {v2_count} ({reg_v2_pct:.2f}%)")
    print(f"Registered frames (Total): {reg_total} / {total_frames} ({reg_total_pct:.2f}%)")
    print(f"Mean reprojection error: {stats['mean_reproj_error']:.4f} px")
    print(f"Intrinsics: fx={stats['fx']:.2f}, fy={stats['fy']:.2f}, k1={stats['k1']:.4f}")
    print(f"Step lengths: median={stats['median_step']:.4f}, largest={stats['largest_step']:.4f}, ratio={stats['step_ratio']:.2f}x")
    print(f"Start-to-end gap: {stats['start_end_gap']:.4f} model units")

    crit_reg = reg_total_pct >= 90.0
    crit_reproj = stats['mean_reproj_error'] < 1.0
    crit_cam = (abs(stats['fx'] - stats['fy']) / stats['fy'] <= 0.05) and (abs(stats['k1']) < 0.3)
    crit_step = stats['step_ratio'] <= 5.0

    passed = crit_reg and crit_reproj and crit_cam and crit_step
    print(f"\nPass Criteria Check:")
    print(f"  1. >=90% frames in ONE model: {'PASS' if crit_reg else 'FAIL'} ({reg_total_pct:.1f}%)")
    print(f"  2. Reprojection error < 1.0 px: {'PASS' if crit_reproj else 'FAIL'} ({stats['mean_reproj_error']:.4f} px)")
    print(f"  3. fx ~ fy within 5%, |k1| < 0.3: {'PASS' if crit_cam else 'FAIL'} (fx={stats['fx']:.1f}, fy={stats['fy']:.1f}, k1={stats['k1']:.4f})")
    print(f"  4. Max step <= 5x median step: {'PASS' if crit_step else 'FAIL'} ({stats['step_ratio']:.2f}x)")
    print(f"OVERALL E4: {'PASS' if passed else 'FAIL'}")

except Exception as e:
    print(f"Global mapping failed with error: {e}")
    import traceback
    traceback.print_exc()
