#!/usr/bin/env python3
"""scripts/check_sweep_parallax.py

Parallax check for sweep videos before SfM:
Reports:
  1. Median optical-flow magnitude (in pixels at full resolution).
  2. Share of frames where the view rotates a lot but the scene shifts little (pure-rotation suspects).
  3. Classification of which sweeps look rotation-only vs translation/parallax.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

def analyze_sweep(images_dir: Path, sweep_name: str):
    frames = sorted(images_dir.glob("frame_*.jpg"))
    if len(frames) < 2:
        return {"error": "Less than 2 frames"}

    flow_magnitudes = []
    rotation_angles_deg = []
    residual_parallax = []
    pure_rotation_flags = []

    # Process consecutive pairs
    for i in range(len(frames) - 1):
        img1 = cv2.imread(str(frames[i]))
        img2 = cv2.imread(str(frames[i + 1]))
        if img1 is None or img2 is None:
            continue

        h_full, w_full = img1.shape[:2]
        # Downscale for fast & robust optical flow
        scale = 0.5
        w_scaled = int(w_full * scale)
        h_scaled = int(h_full * scale)

        g1 = cv2.cvtColor(cv2.resize(img1, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(img2, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        u = flow[..., 0] / scale  # scale back to full-res px
        v = flow[..., 1] / scale
        mag = np.sqrt(u**2 + v**2)

        med_mag = float(np.median(mag))
        flow_magnitudes.append(med_mag)

        # Estimate camera motion using feature points for robust affine / homography
        pts1 = cv2.goodFeaturesToTrack(g1, maxCorners=500, qualityLevel=0.01, minDistance=10)
        if pts1 is not None and len(pts1) >= 20:
            pts2, status, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts1, None)
            good1 = pts1[status == 1] / scale
            good2 = pts2[status == 1] / scale

            if len(good1) >= 15:
                # Estimate affine partial 2D (rotation + scale + translation)
                M, inliers = cv2.estimateAffinePartial2D(good1, good2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None:
                    # M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
                    cos_val = M[0, 0]
                    sin_val = M[1, 0]
                    angle = np.degrees(np.arctan2(sin_val, cos_val))
                    rotation_angles_deg.append(abs(float(angle)))

                    # Compute residuals under homography / affine rotation
                    pred2 = cv2.transform(good1.reshape(-1, 1, 2), M).reshape(-1, 2)
                    res = np.linalg.norm(good2 - pred2, axis=1)
                    med_res = float(np.median(res))
                    residual_parallax.append(med_res)

                    # Pure-rotation suspect condition:
                    # The view rotates significantly (e.g. median flow > 8 px or rotation > 0.4 deg)
                    # BUT the residual parallax across scene depths is very small (med_res < 2.0 px
                    # or ratio of residual to total flow < 0.12)
                    rot_significant = (med_mag >= 6.0 or abs(angle) >= 0.35)
                    parallax_tiny = (med_res <= 2.5 or (med_res / (med_mag + 1e-4) < 0.12))
                    is_pure_rot = bool(rot_significant and parallax_tiny)
                    pure_rotation_flags.append(is_pure_rot)
                    continue

        pure_rotation_flags.append(False)

    total_pairs = len(flow_magnitudes)
    pure_rot_count = sum(pure_rotation_flags)
    pure_rot_share = (pure_rot_count / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    overall_med_flow = float(np.median(flow_magnitudes)) if flow_magnitudes else 0.0
    mean_rot_deg = float(np.mean(rotation_angles_deg)) if rotation_angles_deg else 0.0
    mean_residual = float(np.mean(residual_parallax)) if residual_parallax else 0.0

    # Verdict
    looks_rotation_only = pure_rot_share >= 60.0

    return {
        "sweep": sweep_name,
        "frames_count": len(frames),
        "total_pairs": total_pairs,
        "median_flow_px": round(overall_med_flow, 2),
        "mean_rotation_deg": round(mean_rot_deg, 2),
        "mean_residual_px": round(mean_residual, 2),
        "pure_rotation_frames": pure_rot_count,
        "pure_rotation_share_pct": round(pure_rot_share, 1),
        "looks_rotation_only": looks_rotation_only,
    }

def main():
    print("=" * 80)
    print("  PARALLAX & ROTATION CHECK BEFORE SFM (data/home2/images/)")
    print("=" * 80)
    print(f"{'Sweep':<12} {'Frames':<8} {'Median Flow (px)':<18} {'Mean Rot (deg)':<16} {'Pure-Rot Share (%)':<20} {'Classification'}")
    print("-" * 80)

    sweeps = ["sweep_1", "sweep_2", "sweep_3", "sweep_4"]
    results = []

    for sw in sweeps:
        sw_dir = REPO_ROOT / "data" / "home2" / "images" / sw
        if not sw_dir.is_dir():
            print(f"{sw:<12} Directory not found")
            continue
        res = analyze_sweep(sw_dir, sw)
        results.append(res)
        verdict = "[SUSPECT] Rotation-Only" if res["looks_rotation_only"] else "[OK] Sufficient Parallax"
        print(f"{sw:<12} {res['frames_count']:<8} {res['median_flow_px']:<18.2f} {res['mean_rotation_deg']:<16.2f} {res['pure_rotation_share_pct']:<20.1f} {verdict}")

    print("-" * 80)
    print("Summary of Sweep Classification:")
    for r in results:
        status = "ROTATION-ONLY" if r["looks_rotation_only"] else "PARALLAX (Translational motion present)"
        print(f"  * {r['sweep']}: {status} (Median Flow: {r['median_flow_px']} px, Pure-Rotation Share: {r['pure_rotation_share_pct']}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
