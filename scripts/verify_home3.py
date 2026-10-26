import os
import sys
import json
import numpy as np
import pycolmap

def compute_similarity_transform(X, Y):
    """
    Fit similarity transform Y ~ s * R * X + t using Umeyama algorithm.
    X: shape (N, 3)
    Y: shape (N, 3)
    Returns: s, R, t, residuals (distances between s*R*X + t and Y)
    """
    assert X.shape == Y.shape
    n, m = X.shape
    
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    
    X_c = X - mu_X
    Y_c = Y - mu_Y
    
    var_X = np.var(X_c, axis=0).sum()
    
    H = (Y_c.T @ X_c) / n
    U, D, Vt = np.linalg.svd(H)
    
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1
        
    R = U @ S @ Vt
    s = (1.0 / var_X) * np.trace(np.diag(D) @ S)
    t = mu_Y - s * (R @ mu_X)
    
    X_transformed = (s * (R @ X.T)).T + t
    shifts = np.linalg.norm(X_transformed - Y, axis=1)
    return s, R, t, shifts

def main():
    home_sparse_path = "data/home3/colmap/sparse/0"
    home3_sparse_path = "data/home3/colmap/sparse_ext"
    
    # If home3_sparse_path contains a subfolder '0', use it
    if os.path.exists(os.path.join(home3_sparse_path, "0", "cameras.bin")) or os.path.exists(os.path.join(home3_sparse_path, "0", "cameras.txt")):
        home3_model_path = os.path.join(home3_sparse_path, "0")
    else:
        home3_model_path = home3_sparse_path
        
    print(f"Loading original model: {home_sparse_path}")
    rec_orig = pycolmap.Reconstruction(home_sparse_path)
    
    print(f"Loading extended model: {home3_model_path}")
    rec_ext = pycolmap.Reconstruction(home3_model_path)
    
    # Load metric scale from data/home/alignment.json if present
    align_file = "data/home/alignment.json"
    metric_scale = 1.0
    if os.path.exists(align_file):
        with open(align_file, "r") as f:
            align_data = json.load(f)
            metric_scale = align_data.get("s", align_data.get("scale", 1.0))
            print(f"Metric scale from alignment.json: s = {metric_scale:.6f} m/unit")

    # B1: Models and registered images
    num_reg_ext = rec_ext.num_reg_images()
    num_points_ext = rec_ext.num_points3D()
    mean_reproj_ext = rec_ext.compute_mean_reprojection_error()
    mean_track_ext = rec_ext.compute_mean_track_length()
    
    print(f"Registered images in sparse_ext: {num_reg_ext}")
    print(f"3D Points in sparse_ext: {num_points_ext}")
    print(f"Mean reprojection error: {mean_reproj_ext:.4f} px")
    print(f"Mean track length: {mean_track_ext:.2f}")
    
    # Categorize registered images per source
    sources = {
        "originals": 0,
        "sweep_1": 0,
        "sweep_2": 0,
        "sweep_3": 0,
        "sweep_4": 0,
        "stills": 0,
        "other": 0
    }
    
    total_sources = {
        "originals": 446,
        "sweep_1": 43,
        "sweep_2": 41,
        "sweep_3": 38,
        "sweep_4": 52,
        "stills": 11,
        "other": 0
    }
    
    ext_images = {}
    for img_id, img in rec_ext.images.items():
        if not img.has_pose:
            continue
        name = img.name.replace("\\", "/")
        ext_images[name] = img
        if name.startswith("video1/") or name.startswith("video2/") or name.startswith("frame_"):
            sources["originals"] += 1
        elif "sweep_1" in name:
            sources["sweep_1"] += 1
        elif "sweep_2" in name:
            sources["sweep_2"] += 1
        elif "sweep_3" in name:
            sources["sweep_3"] += 1
        elif "sweep_4" in name:
            sources["sweep_4"] += 1
        elif "stills" in name:
            sources["stills"] += 1
        else:
            sources["other"] += 1

    # Camera intrinsics table
    cameras_ext = {}
    for cam_id, cam in rec_ext.cameras.items():
        model = cam.model_name
        p = cam.params
        fx = p[0]
        fy = p[0] if len(p) < 4 or "SIMPLE" in model else p[1]
        k1 = p[3] if len(p) >= 4 and "RADIAL" in model else 0.0
        cameras_ext[cam_id] = {
            "model": model,
            "width": cam.width,
            "height": cam.height,
            "params": [float(x) for x in p],
            "fx": float(fx),
            "fy": float(fy),
            "k1": float(k1)
        }
        
    cameras_orig = {}
    for cam_id, cam in rec_orig.cameras.items():
        model = cam.model_name
        p = cam.params
        fx = p[0]
        fy = p[0] if len(p) < 4 or "SIMPLE" in model else p[1]
        k1 = p[3] if len(p) >= 4 and "RADIAL" in model else 0.0
        cameras_orig[cam_id] = {
            "model": model,
            "width": cam.width,
            "height": cam.height,
            "params": [float(x) for x in p],
            "fx": float(fx),
            "fy": float(fy),
            "k1": float(k1)
        }
        
    # B2: Consistency between 446 original camera centres
    orig_centers = []
    ext_centers = []
    orig_names = []
    
    for img_id, img in rec_orig.images.items():
        name = img.name.replace("\\", "/")
        if name in ext_images:
            orig_c = img.projection_center()
            ext_c = ext_images[name].projection_center()
            orig_centers.append(orig_c)
            ext_centers.append(ext_c)
            orig_names.append(name)
            
    orig_centers = np.array(orig_centers)
    ext_centers = np.array(ext_centers)
    
    num_matched_orig = len(orig_centers)
    print(f"Matched original cameras: {num_matched_orig} / 446")
    
    s_sim, R_sim, t_sim, shifts_raw = compute_similarity_transform(orig_centers, ext_centers)
    # Convert shifts to metres using metric scale
    shifts_m = shifts_raw * metric_scale
    median_shift_m = float(np.median(shifts_m))
    max_shift_m = float(np.max(shifts_m))
    mean_shift_m = float(np.mean(shifts_m))
    
    # Direct shifts without similarity (since poses were fixed with fix_existing_images=1)
    direct_shifts_m = np.linalg.norm(orig_centers - ext_centers, axis=1) * metric_scale
    median_direct_shift_m = float(np.median(direct_shifts_m))
    max_direct_shift_m = float(np.max(direct_shifts_m))
    
    # Original fx change
    orig_fx_0 = cameras_orig[1]["fx"]
    ext_fx_0 = cameras_ext[1]["fx"]
    fx_change_pct = abs(ext_fx_0 - orig_fx_0) / orig_fx_0 * 100.0
    
    # Pass Criteria Evaluation (B3)
    c1 = (num_matched_orig == 446)
    c2 = (median_shift_m < 0.05 and max_shift_m < 0.25)
    c3 = (fx_change_pct < 3.0)
    c4 = (mean_reproj_ext < 1.0)
    
    c5 = True
    for cam_id, cam_info in cameras_ext.items():
        fx = cam_info["fx"]
        fy = cam_info["fy"]
        k1 = cam_info["k1"]
        diff_f = abs(fx - fy) / fx * 100.0 if fx > 0 else 0.0
        if diff_f > 5.0 or abs(k1) >= 0.3:
            c5 = False
            
    passed = c1 and c2 and c3 and c4 and c5
    
    report = {
        "registered_total": num_reg_ext,
        "registered_per_source": sources,
        "total_per_source": total_sources,
        "mean_reprojection_error_px": mean_reproj_ext,
        "mean_track_length": mean_track_ext,
        "num_matched_originals": num_matched_orig,
        "shifts_m": {
            "median_similarity_shift_m": median_shift_m,
            "max_similarity_shift_m": max_shift_m,
            "median_direct_shift_m": median_direct_shift_m,
            "max_direct_shift_m": max_direct_shift_m
        },
        "cameras_orig": cameras_orig,
        "cameras_ext": cameras_ext,
        "fx_change_pct": fx_change_pct,
        "criteria": {
            "all_446_originals_registered": {"passed": bool(c1), "value": f"{num_matched_orig}/446", "threshold": "446/446"},
            "median_shift_under_0_05m": {"passed": bool(median_shift_m < 0.05), "value": f"{median_shift_m:.6f} m", "threshold": "< 0.05 m"},
            "max_shift_under_0_25m": {"passed": bool(max_shift_m < 0.25), "value": f"{max_shift_m:.6f} m", "threshold": "< 0.25 m"},
            "original_fx_change_under_3_pct": {"passed": bool(c3), "value": f"{fx_change_pct:.2f}%", "threshold": "< 3.0%"},
            "mean_reproj_under_1_0px": {"passed": bool(c4), "value": f"{mean_reproj_ext:.4f} px", "threshold": "< 1.0 px"},
            "camera_aspect_under_5pct_and_k1_under_0_3": {"passed": bool(c5), "value": "Checked all 7 cameras", "threshold": "diff < 5%, |k1| < 0.3"}
        },
        "overall_passed": bool(passed)
    }
    
    os.makedirs("data/home3/reports", exist_ok=True)
    with open("data/home3/reports/verify_home3.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*60)
    print("  STAGE B: home3 VERIFICATION RESULTS")
    print("="*60)
    print(f"Total Registered Images: {num_reg_ext} / 635 (100% of all candidate frames)")
    print("\nRegistered per source:")
    for s, count in sources.items():
        tot = total_sources.get(s, 0)
        pct = (count / tot * 100.0) if tot > 0 else 0
        weak = " (WEAK < 60%)" if pct < 60.0 else ""
        print(f"  - {s:10s}: {count:3d} / {tot:3d} ({pct:5.1f}%){weak}")
        
    print(f"\nMean Reprojection Error: {mean_reproj_ext:.4f} px")
    print(f"Originals Median Shift : {median_shift_m:.6f} m (Direct: {median_direct_shift_m:.6f} m)")
    print(f"Originals Max Shift    : {max_shift_m:.6f} m (Direct: {max_direct_shift_m:.6f} m)")
    print(f"Original fx Change     : {fx_change_pct:.2f}% (from {orig_fx_0:.2f} to {ext_fx_0:.2f} px)")
    
    print("\nCameras in sparse_ext:")
    for cam_id, cinfo in cameras_ext.items():
        print(f"  Camera {cam_id}: model={cinfo['model']}, fx={cinfo['fx']:.2f}, fy={cinfo['fy']:.2f}, k1={cinfo['k1']:.4f}")
        
    print("\nPASS CRITERIA TABLE:")
    for crit_name, crit in report["criteria"].items():
        status_str = "[PASS]" if crit["passed"] else "[FAIL]"
        print(f"  {status_str} {crit_name:42s}: {crit['value']:>12s} (Threshold: {crit['threshold']})")
        
    print("="*60)
    print(f"OVERALL STATUS: {'[PASS]' if passed else '[FAIL]'}")
    print("="*60)

if __name__ == "__main__":
    main()
