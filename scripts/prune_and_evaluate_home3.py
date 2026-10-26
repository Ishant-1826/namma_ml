import os
import sys
import json
import shutil
import subprocess
import numpy as np
import pycolmap
from collections import defaultdict

def main():
    print("=" * 80)
    print("  STEP 2: Pruning new content in home3 (sparse_ext -> sparse_pruned)")
    print("=" * 80)

    sparse_0_path = "data/home3/colmap/sparse/0"
    sparse_ext_path = "data/home3/colmap/sparse_ext"
    sparse_pruned_path = "data/home3/colmap/sparse_pruned"
    sparse_pruned_ba_path = "data/home3/colmap/sparse_pruned_ba"

    rec_orig = pycolmap.Reconstruction(sparse_0_path)
    rec = pycolmap.Reconstruction(sparse_ext_path)

    orig_image_names = set(img.name.replace("\\", "/") for img in rec_orig.images.values())
    orig_point_ids = set(rec_orig.points3D.keys())

    print(f"Original baseline images : {len(orig_image_names)}")
    print(f"Original baseline points : {len(orig_point_ids)}")
    print(f"sparse_ext initial images: {rec.num_reg_images()}")
    print(f"sparse_ext initial points: {rec.num_points3D()}")

    # 1. Prune 3D points: new points only with reproj error > 2.0 or track < 3
    new_points_deleted = 0
    pids_to_del = []
    for pid, pt in rec.points3D.items():
        if pid not in orig_point_ids:
            if pt.error > 2.0 or len(pt.track.elements) < 3:
                pids_to_del.append(pid)

    for pid in pids_to_del:
        rec.delete_point3D(pid)
        new_points_deleted += 1

    print(f"Pruned {new_points_deleted} new 3D points (error > 2.0 px or track < 3).")
    print(f"Remaining 3D points: {rec.num_points3D()}")

    # 2. Prune new images: new images with mean error > 1.5 px or < 100 observations
    source_names = ['sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']
    imgs_to_delete = []

    for img in rec.images.values():
        name = img.name.replace("\\", "/")
        if name in orig_image_names:
            continue
        
        cam = rec.cameras[img.camera_id]
        rigid = img.cam_from_world()
        
        errs = []
        for p2d in img.points2D:
            if p2d.has_point3D():
                pid = p2d.point3D_id
                p3d = rec.points3D[pid]
                p_cam = rigid * p3d.xyz
                p_img = cam.img_from_cam(p_cam)
                if p_img is not None:
                    err = np.linalg.norm(p_img - p2d.xy)
                    errs.append(err)

        num_obs = len(errs)
        mean_err = float(np.mean(errs)) if errs else 999.0

        if mean_err > 1.5 or num_obs < 100:
            imgs_to_delete.append(img.name)

    print(f"Pruning {len(imgs_to_delete)} new images (mean error > 1.5 px or obs < 100)...")

    # Write delete list
    os.makedirs(sparse_pruned_path, exist_ok=True)
    del_txt = "data/home3/colmap/delete_images.txt"
    with open(del_txt, "w") as f:
        for name in imgs_to_delete:
            f.write(f"{name}\n")

    # Use colmap image_deleter or deregister in rec
    # First save intermediate point-filtered model
    temp_pt_filtered = "data/home3/colmap/sparse_pt_filtered"
    os.makedirs(temp_pt_filtered, exist_ok=True)
    rec.write(temp_pt_filtered)

    # Run colmap image_deleter
    cmd = [
        "colmap", "image_deleter",
        "--input_path", temp_pt_filtered,
        "--output_path", sparse_pruned_path,
        "--image_names_path", del_txt
    ]
    print("Running colmap image_deleter...")
    subprocess.run(cmd, check=True)

    if os.path.exists(temp_pt_filtered):
        shutil.rmtree(temp_pt_filtered, ignore_errors=True)

    rec_pruned = pycolmap.Reconstruction(sparse_pruned_path)
    print(f"sparse_pruned: {rec_pruned.num_reg_images()} images, {rec_pruned.num_points3D()} points.")

    # -----------------------------------------------------------------------
    # STEP 3: Bundle adjustment on sparse_pruned
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  STEP 3: Bundle Adjustment (sparse_pruned -> sparse_pruned_ba)")
    print("=" * 80)

    os.makedirs(sparse_pruned_ba_path, exist_ok=True)

    # Configure BA in pycolmap to strictly fix the 446 original image poses and cameras 1 & 2
    ba_config = pycolmap.BundleAdjustmentConfig()
    for img in rec_pruned.images.values():
        name = img.name.replace("\\", "/")
        if name in orig_image_names:
            ba_config.set_constant_rig_from_world_pose(img.frame_id)

    # Fix camera 1 and 2 intrinsics
    for cid in [1, 2]:
        if cid in rec_pruned.cameras:
            ba_config.set_constant_cam_intrinsics(cid)

    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_rig_from_world = True

    print("Running bundle adjustment with fixed original poses...")
    pycolmap.bundle_adjustment(rec_pruned, ba_config, ba_options)

    rec_pruned.write(sparse_pruned_ba_path)
    print(f"sparse_pruned_ba written: {rec_pruned.num_reg_images()} images, {rec_pruned.num_points3D()} points.")

    # Run model_analyzer
    subprocess.run(["colmap", "model_analyzer", "--path", sparse_pruned_ba_path], check=True)

    # -----------------------------------------------------------------------
    # STEP 4: Evaluate Stage B criteria on sparse_pruned_ba
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  STEP 4: Evaluating Stage B Criteria on sparse_pruned_ba")
    print("=" * 80)

    rec_ba = pycolmap.Reconstruction(sparse_pruned_ba_path)

    # Alignment scale
    with open("data/home/alignment.json") as f:
        align = json.load(f)
    metric_scale = align.get("s", 1.0)

    # Check shift of 446 originals
    orig_shifts = []
    matched_orig = 0
    for img in rec_orig.images.values():
        name = img.name.replace("\\", "/")
        img_ba = rec_ba.find_image_with_name(name)
        if img_ba and img_ba.has_pose:
            matched_orig += 1
            c_orig = img.projection_center()
            c_ba = img_ba.projection_center()
            shift_m = float(np.linalg.norm(c_orig - c_ba)) * metric_scale
            orig_shifts.append(shift_m)

    median_shift_m = float(np.median(orig_shifts)) if orig_shifts else 999.0
    max_shift_m = float(np.max(orig_shifts)) if orig_shifts else 999.0

    # Model-wide mean reprojection error
    model_mean_reproj = rec_ba.compute_mean_reprojection_error()

    # Per-source stats
    total_expected = {
        'originals': 450,
        'sweep_1': 43,
        'sweep_2': 41,
        'sweep_3': 38,
        'sweep_4': 52,
        'stills': 11
    }

    source_reg = defaultdict(int)
    source_errors = defaultdict(list)
    source_points = defaultdict(set)

    for img in rec_ba.images.values():
        if not img.has_pose:
            continue
        name = img.name.replace("\\", "/")
        src = 'originals' if name in orig_image_names or name.startswith("video") else None
        if not src:
            for s in ['sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']:
                if s in name:
                    src = s
                    break
        if not src:
            src = 'other'

        source_reg[src] += 1
        cam = rec_ba.cameras[img.camera_id]
        rigid = img.cam_from_world()

        for p2d in img.points2D:
            if p2d.has_point3D():
                pid = p2d.point3D_id
                source_points[src].add(pid)
                p3d = rec_ba.points3D[pid]
                p_cam = rigid * p3d.xyz
                p_img = cam.img_from_cam(p_cam)
                if p_img is not None:
                    err = np.linalg.norm(p_img - p2d.xy)
                    source_errors[src].append(err)

    print("\nPER-SOURCE EVALUATION TABLE (sparse_pruned_ba):")
    print("-" * 80)
    print(f"{'Source':12s} | {'Reg / Tot':12s} | {'Reg %':7s} | {'3D Points':10s} | {'Mean Reproj':12s} | {'Source Pass?'}")
    print("-" * 80)

    passing_new_sources = []
    for s in ['originals', 'sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']:
        reg_cnt = source_reg[s]
        tot_cnt = total_expected.get(s, 0)
        pct = (reg_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
        pts_cnt = len(source_points[s])
        mean_err = float(np.mean(source_errors[s])) if source_errors[s] else 0.0

        if s == 'originals':
            src_pass = (reg_cnt >= 446 and mean_err < 1.5)
            pass_str = "PASS" if src_pass else "FAIL"
        else:
            src_pass = (pct >= 60.0 and mean_err < 1.5)
            pass_str = "PASS (>=60%, <1.5px)" if src_pass else "FAIL (<60% or >=1.5px)"
            if src_pass:
                passing_new_sources.append(s)

        print(f"{s:12s} | {reg_cnt:3d} / {tot_cnt:3d}    | {pct:6.2f}% | {pts_cnt:10d} | {mean_err:8.4f} px | {pass_str}")

    print("-" * 80)
    print(f"Model-Wide Mean Reprojection Error: {model_mean_reproj:.4f} px (Target: < 1.1 px)")
    print(f"Originals Median Shift: {median_shift_m:.6f} m (Target: < 0.05 m)")
    print(f"Originals Max Shift   : {max_shift_m:.6f} m (Target: < 0.25 m)")
    print(f"Passing New Sources (>=60% reg & <1.5px reproj): {len(passing_new_sources)} {passing_new_sources}")

    # Decision rule
    keep_home3 = (
        matched_orig == 446 and
        median_shift_m < 0.05 and
        max_shift_m < 0.25 and
        model_mean_reproj < 1.1 and
        len(passing_new_sources) >= 2
    )

    decision = "home3" if keep_home3 else "home"
    print("\n" + "=" * 80)
    print(f"FINAL DECISION RULE RESULT: SCENE = {decision}")
    if not keep_home3:
        reasons = []
        if model_mean_reproj >= 1.1:
            reasons.append(f"Model-wide reproj {model_mean_reproj:.4f}px >= 1.1px")
        if len(passing_new_sources) < 2:
            reasons.append(f"Only {len(passing_new_sources)} new source(s) passed (need >= 2 with >=60% reg and <1.5px err)")
        print(f"Fallback Reason: {'; '.join(reasons)}")
    print("=" * 80)

    # Save report
    report = {
        "pruned_points_count": new_points_deleted,
        "pruned_images_count": len(imgs_to_delete),
        "pruned_ba_images": rec_ba.num_reg_images(),
        "pruned_ba_points": rec_ba.num_points3D(),
        "model_mean_reprojection_error": model_mean_reproj,
        "originals_median_shift_m": median_shift_m,
        "originals_max_shift_m": max_shift_m,
        "passing_new_sources": passing_new_sources,
        "decision": decision
    }

    with open("data/home3/reports/prune_ba_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
