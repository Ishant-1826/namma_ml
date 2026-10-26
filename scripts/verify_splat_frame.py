#!/usr/bin/env python3
"""
scripts/verify_splat_frame.py  --  Verify Exported 3D Gaussian Splat Coordinate Frame.

Validates that the exported splat adheres to the metric Y-up aligned room coordinate system:
1. Percentile Extents (Filtered: opacity >= 0.2, max scale <= 0.5 m):
   - Reference points are cropped to the same x/z/y bounding box as the splat before comparison.
   - Compares X and Z 2nd-98th percentile corners and overhang (overhang <= 1.0 m, underhang <= 0.3 m).
   - Compares Y lower bound (floor p2 near y=0 within 0.30 m) and ceiling clearance (p98 <= 4.40 m).
   - Explains pass/fail with exact numbers.
2. Rigid Alignment Fit (Umeyama / Trimmed ICP):
   - Subsamples 50k splat centres vs reference sparse points (points3D.txt).
   - Tests unmirrored vs X-mirrored vs Z-mirrored.
   - FAILS unless unmirrored is best with |yaw| <= 5 deg, |t| <= 0.25 m, scale in [0.9, 1.1].
   - Explains pass/fail with exact numbers.
3. Camera Inclusion & Elevation:
   - Tests 100% of camera centres against scene.json bounds in X and Z and against Y in [floor_y, ceiling_y].
   - Reuses align_scene.py logic to split camera passes (Pass 1 eye vs Pass 2 chest).
   - Fails loudly if either pass has 0 cameras; reports camera count and mean height per pass.
4. Floor Plane & Camera Plane Alignment:
   - Filtered splat floor peak within +/- 0.10 m of y = 0.
   - Splat floor band-RANSAC tilt <= 3.0 deg.
   - Camera trajectory plane tilt <= 5.0 deg.

Outputs:
  Console report + outputs/<scene>/splat_frame_verification.json (or custom --output-json)
  Exits 0 on PASS, non-zero on FAIL.

Usage:
  python scripts/verify_splat_frame.py --scene home
  python scripts/verify_splat_frame.py --scene home --splat-ply outputs/home/scene_clean.ply
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

try:
    from pipeline.utils.point_cloud import o3d
except ImportError:
    try:
        import open3d as o3d
    except ImportError:
        o3d = None


def parse_points3d_txt(path: Path) -> np.ndarray:
    """Parses XYZ points from COLMAP points3D.txt."""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float64)


def load_ply_all(ply_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reader for Gaussian Splat PLY properties:
    Returns (xyz, sigmoid_opacity, max_real_scale).
    """
    with open(ply_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("latin1").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        is_binary = False
        is_little_endian = True
        num_vertices = 0
        properties = []
        in_vertex = False

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                if "binary_little_endian" in parts[1]:
                    is_binary = True
                    is_little_endian = True
                elif "binary_big_endian" in parts[1]:
                    is_binary = True
                    is_little_endian = False
                elif "ascii" in parts[1]:
                    is_binary = False
            elif parts[0] == "element" and parts[1] == "vertex":
                in_vertex = True
                num_vertices = int(parts[2])
            elif parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            elif parts[0] == "property" and in_vertex:
                properties.append((parts[2], parts[1]))

        if num_vertices == 0:
            raise ValueError(f"PLY file {ply_path} contains 0 vertices.")

        type_map = {
            "float": "<f4" if is_little_endian else ">f4",
            "float32": "<f4" if is_little_endian else ">f4",
            "double": "<f8" if is_little_endian else ">f8",
            "float64": "<f8" if is_little_endian else ">f8",
            "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
            "int": "<i4" if is_little_endian else ">i4",
            "int32": "<i4" if is_little_endian else ">i4",
            "uint": "<u4" if is_little_endian else ">u4",
            "uint32": "<u4" if is_little_endian else ">u4",
            "short": "<i2" if is_little_endian else ">i2",
            "int16": "<i2" if is_little_endian else ">i2",
            "ushort": "<u2" if is_little_endian else ">u2",
            "uint16": "<u2" if is_little_endian else ">u2",
        }

        dtype_fields = [(name, type_map.get(ptype, "<f4")) for name, ptype in properties]
        dt = np.dtype(dtype_fields)
        data = np.fromfile(f, dtype=dt, count=num_vertices)

        xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

        if "opacity" in data.dtype.names:
            raw_op = data["opacity"].astype(np.float64)
            sig_op = 1.0 / (1.0 + np.exp(-raw_op))
        else:
            sig_op = np.ones(len(xyz), dtype=np.float64)

        scale_names = [f"scale_{i}" for i in range(3)]
        if all(s in data.dtype.names for s in scale_names):
            log_scales = np.column_stack([data[s] for s in scale_names]).astype(np.float64)
            max_scale = np.max(np.exp(log_scales), axis=1)
        else:
            max_scale = np.zeros(len(xyz), dtype=np.float64)

        return xyz, sig_op, max_scale


def load_camera_centres_split(
    transforms_path: Path,
    scene: str = "home",
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads camera centres, splitting into Pass 1 and Pass 2 using the exact logic align_scene.py uses.
    Fails loudly if any pass has 0 cameras.
    """
    with open(transforms_path, "r", encoding="utf-8") as f:
        tf = json.load(f)

    p1, p2 = [], []
    pass_split = None

    # Look for frames_report.json to discover pass split
    search_dirs = [transforms_path.parent]
    if data_dir:
        search_dirs.append(data_dir)
    if scene and not scene.startswith("synth") and not scene.startswith("test"):
        search_dirs.append(Path("data") / scene)

    for s_dir in search_dirs:
        fr_path = s_dir / "frames_report.json"
        if fr_path.is_file():
            try:
                with open(fr_path, "r", encoding="utf-8") as fr_f:
                    vids = json.load(fr_f).get("videos", [])
                    if vids:
                        pass_split = vids[0].get("kept_after_duplicate", vids[0].get("raw_extracted", None))
                        if pass_split is not None:
                            break
            except Exception:
                pass

    total_cams = len(tf.get("frames", []))
    if pass_split is None:
        pass_split = min(222, total_cams // 2) if total_cams > 0 else 222

    for frame in tf.get("frames", []):
        fpath = frame["file_path"]
        fpath_lower = fpath.lower()
        fname = Path(fpath).name
        fnum = int("".join(filter(str.isdigit, fname))) if any(c.isdigit() for c in fname) else 0

        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        c = mat[:3, 3]

        # align_scene.py logic:
        # if "video2" in name or ("video1" not in name and f_num > pass_split):
        if "video2" in fpath_lower or ("video1" not in fpath_lower and fnum > pass_split):
            p2.append(c)
        else:
            p1.append(c)

    if len(p1) == 0:
        raise ValueError(
            f"[FATAL] Pass 1 has 0 cameras out of {total_cams} total frames! "
            f"Pass splitting logic failed (pass_split={pass_split})."
        )
    if len(p2) == 0:
        raise ValueError(
            f"[FATAL] Pass 2 has 0 cameras out of {total_cams} total frames! "
            f"Pass splitting logic failed (pass_split={pass_split})."
        )

    p1_arr = np.array(p1, dtype=np.float64)
    p2_arr = np.array(p2, dtype=np.float64)
    all_arr = np.vstack([p1_arr, p2_arr])
    return all_arr, p1_arr, p2_arr


def compute_umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    trim_ratio: float = 0.80,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """Fits similarity transform mapping src -> dst via trimmed Umeyama."""
    tree = KDTree(dst)
    dists, idxs = tree.query(src)
    raw_mean_dist = float(np.mean(dists))

    n_trim = max(10, int(len(dists) * trim_ratio))
    sort_idx = np.argsort(dists)[:n_trim]
    src_m = src[sort_idx]
    dst_m = dst[idxs[sort_idx]]

    mu_src = np.mean(src_m, axis=0)
    mu_dst = np.mean(dst_m, axis=0)
    src_c = src_m - mu_src
    dst_c = dst_m - mu_dst

    cov = (dst_c.T @ src_c) / len(src_m)
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt

    var_src = np.mean(np.sum(src_c**2, axis=1))
    s = float(np.trace(np.diag(S) @ D) / (var_src + 1e-12))
    t = mu_dst - s * (R @ mu_src)

    transformed = s * (src_m @ R.T) + t
    rmse = float(np.sqrt(np.mean(np.sum((transformed - dst_m)**2, axis=1))))
    yaw_deg = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    return s, R, t, yaw_deg, rmse, raw_mean_dist


def fit_svd_plane(pts: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """Fits best plane to 3D points via SVD. Returns (normal, d, tilt_vs_y, height_origin)."""
    mu = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - mu)
    normal = Vt[-1]
    if normal[1] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, mu))
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    h_origin = -d / normal[1]
    return normal, d, tilt_deg, h_origin


def fit_floor_band_ransac(
    pts: np.ndarray,
    y_center: float = 0.00,
    band: float = 0.10,
    dist_thresh: float = 0.03,
) -> Tuple[np.ndarray, float, float, int, int]:
    """RANSAC plane fit within elevation band [y_center - band, y_center + band]."""
    mask = (pts[:, 1] >= y_center - band) & (pts[:, 1] <= y_center + band)
    band_pts = pts[mask]
    if len(band_pts) < 50:
        raise ValueError(f"Too few points ({len(band_pts)}) in floor band [{y_center - band:.2f}, {y_center + band:.2f}]")

    if o3d is None:
        raise RuntimeError("Open3D is required for floor band RANSAC fitting.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(band_pts)
    plane, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=2000)
    a, b, c, d = plane
    normal = np.array([a, b, c], dtype=np.float64)
    if b < 0:
        normal = -normal
        d = -d
    normal /= np.linalg.norm(normal)
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[1], -1.0, 1.0))))
    h_origin = float(-d / normal[1])
    return normal, h_origin, tilt_deg, len(inliers), len(band_pts)


def verify_splat_frame(
    scene: str,
    splat_ply: Path,
    reference_pts_path: Path,
    transforms_path: Path,
    scene_json_path: Optional[Path] = None,
    corner_tolerance_m: float = 0.40,
    span_tolerance_pct: float = 15.0,
    yaw_tolerance_deg: float = 5.0,
    trans_tolerance_m: float = 0.25,
    scale_range: Tuple[float, float] = (0.90, 1.10),
    floor_peak_tol_m: float = 0.10,
    floor_tilt_tol_deg: float = 3.0,
    cam_tilt_tol_deg: float = 5.0,
    max_ceiling_bound_m: float = 4.40,
    y_p2_diff_tol_m: float = 0.30,
    output_json_path: Optional[Path] = None,
) -> bool:
    print("=" * 82)
    print("  VERIFY 3D GAUSSIAN SPLAT COORDINATE FRAME (PHASE 6)")
    print(f"  Scene          : {scene}")
    print(f"  Exported Splat : {splat_ply}")
    print(f"  Reference Pts  : {reference_pts_path}")
    print(f"  Transforms     : {transforms_path}")
    print("=" * 82)

    failure_reasons: List[str] = []

    if not splat_ply.is_file():
        print(f"\n[FAIL] Exported splat file not found: {splat_ply}", file=sys.stderr)
        return False
    if not reference_pts_path.is_file():
        print(f"\n[FAIL] Reference point cloud not found: {reference_pts_path}", file=sys.stderr)
        return False
    if not transforms_path.is_file():
        print(f"\n[FAIL] Transforms JSON not found: {transforms_path}", file=sys.stderr)
        return False

    # Load scene.json if available
    scene_data: Optional[Dict[str, Any]] = None
    search_json_paths = []
    if scene_json_path:
        search_json_paths.append(scene_json_path)
    search_json_paths.extend([
        splat_ply.parent / "scene.json",
        Path("outputs") / scene / "scene.json",
        Path("viewer") / "public" / "scene" / "scene.json",
    ])
    for sjp in search_json_paths:
        if sjp.is_file():
            try:
                with open(sjp, "r", encoding="utf-8") as f_sj:
                    scene_data = json.load(f_sj)
                    print(f"  Loaded scene.json manifest from: {sjp}")
                    break
            except Exception as e:
                print(f"  [!] Notice reading {sjp}: {e}")

    print("\n[Step 1/4] Loading point sets & filtering splats...")
    splat_raw_xyz, splat_op, splat_sc = load_ply_all(splat_ply)
    ref_pts_raw = parse_points3d_txt(reference_pts_path)
    all_cams, cams_p1, cams_p2 = load_camera_centres_split(transforms_path, scene=scene, data_dir=reference_pts_path.parent)

    # Opacity & scale filter: opacity >= 0.2, max scale <= 0.5 m
    mask_filt = (splat_op >= 0.20) & (splat_sc <= 0.50)
    splat_filt_xyz = splat_raw_xyz[mask_filt]

    mean_h_p1 = float(np.mean(cams_p1[:, 1]))
    mean_h_p2 = float(np.mean(cams_p2[:, 1]))
    ratio_p1_p2 = mean_h_p1 / mean_h_p2 if mean_h_p2 > 1e-4 else 1.0

    print(f"  Raw Splat Gaussians    : {len(splat_raw_xyz):,}")
    print(f"  Filtered Splats (>=0.2): {len(splat_filt_xyz):,} ({len(splat_filt_xyz)/len(splat_raw_xyz)*100:.1f}%)")
    print(f"  Reference points loaded: {len(ref_pts_raw):,}")
    print(f"  Camera poses loaded    : {len(all_cams):,} total")
    print(f"    - Pass 1 (Eye level)   : {len(cams_p1)} cameras, mean height = {mean_h_p1:.3f} m")
    print(f"    - Pass 2 (Chest level) : {len(cams_p2)} cameras, mean height = {mean_h_p2:.3f} m")
    print(f"    - Height ratio (P1/P2) : {ratio_p1_p2:.2f}x")

    if len(splat_filt_xyz) < 500:
        print(f"\n[FAIL] Too few high-confidence splats ({len(splat_filt_xyz)} < 500).", file=sys.stderr)
        return False

    # Determine reference box for cropping reference points (Requirement 2a)
    if scene_data and "bounds" in scene_data:
        box_min = np.array(scene_data["bounds"]["min"], dtype=np.float64)
        box_max = np.array(scene_data["bounds"]["max"], dtype=np.float64)
        floor_y_scene = float(scene_data.get("floor_y", 0.0))
        ceiling_y_scene = float(scene_data.get("ceiling_y", 4.10))
    else:
        box_min = np.min(splat_raw_xyz, axis=0)
        box_max = np.max(splat_raw_xyz, axis=0)
        floor_y_scene = 0.0
        ceiling_y_scene = 4.10

    # -----------------------------------------------------------------------
    # Step 2: Percentile Extents Check (Cropped Reference Points, Requirement 2a)
    # -----------------------------------------------------------------------
    print("\n[Step 2/4] Checking percentile extents (Requirement 2a: Reference points cropped to splat box)...")
    ref_crop_mask = (
        (ref_pts_raw[:, 0] >= box_min[0]) & (ref_pts_raw[:, 0] <= box_max[0]) &
        (ref_pts_raw[:, 1] >= box_min[1]) & (ref_pts_raw[:, 1] <= box_max[1]) &
        (ref_pts_raw[:, 2] >= box_min[2]) & (ref_pts_raw[:, 2] <= box_max[2])
    )
    ref_pts = ref_pts_raw[ref_crop_mask]
    if len(ref_pts) < 100:
        print(f"  [!] Warning: Only {len(ref_pts)} ref points inside splat box; using uncropped ref points.")
        ref_pts = ref_pts_raw
    else:
        print(f"  Cropped reference points: {len(ref_pts_raw):,} -> {len(ref_pts):,} inside splat box [{box_min[0]:.2f}..{box_max[0]:.2f}, {box_min[2]:.2f}..{box_max[2]:.2f}] m")

    p2_ref = np.percentile(ref_pts, 2, axis=0)
    p98_ref = np.percentile(ref_pts, 98, axis=0)
    span_ref = p98_ref - p2_ref

    p2_raw = np.percentile(splat_raw_xyz, 2, axis=0)
    p98_raw = np.percentile(splat_raw_xyz, 98, axis=0)
    span_raw = p98_raw - p2_raw

    p2_filt = np.percentile(splat_filt_xyz, 2, axis=0)
    p98_filt = np.percentile(splat_filt_xyz, 98, axis=0)
    span_filt = p98_filt - p2_filt

    print(f"  {'Axis':<4} | {'Ref [p2, p98]':<20} | {'Splat Filt [p2, p98]':<22} | {'Old Rule (Span/Corner)':<24} | {'Coverage Rule (a & b)'}")
    print("  " + "-" * 118)

    old_extents_passed = True
    new_extents_passed = True
    percentile_report = {}

    for i, ax in enumerate(["X", "Y", "Z"]):
        ref_s = f"[{p2_ref[i]:+.2f}, {p98_ref[i]:+.2f}]"
        filt_s = f"[{p2_filt[i]:+.2f}, {p98_filt[i]:+.2f}]"

        d_p2 = float(abs(p2_filt[i] - p2_ref[i]))
        d_p98 = float(abs(p98_filt[i] - p98_ref[i]))
        s_pct = float(abs(span_filt[i] - span_ref[i]) / span_ref[i] * 100) if span_ref[i] > 1e-4 else 0.0

        # Old Span-Equality & Corner Check
        old_axis_pass = True
        if ax in ["X", "Z"]:
            if d_p2 > corner_tolerance_m or d_p98 > corner_tolerance_m or s_pct > span_tolerance_pct:
                old_axis_pass = False
        elif ax == "Y":
            if p98_filt[1] > max_ceiling_bound_m or d_p2 > y_p2_diff_tol_m:
                old_axis_pass = False
        if not old_axis_pass:
            old_extents_passed = False
        old_tag = f"span={s_pct:4.1f}% [{ 'PASS' if old_axis_pass else 'FAIL' }]"

        # Coverage Rule
        if ax in ["X", "Z"]:
            # (a) Ref box lies inside filtered-splat box within corner tolerance:
            underhang_min = float(p2_filt[i] - p2_ref[i])
            underhang_max = float(p98_ref[i] - p98_filt[i])
            inside_ok = (underhang_min <= corner_tolerance_m) and (underhang_max <= corner_tolerance_m)

            # (b) Filtered-splat box exceeds reference box by at most 1.0 m:
            overhang_min = float(p2_ref[i] - p2_filt[i])
            overhang_max = float(p98_filt[i] - p98_ref[i])
            exceed_ok = (overhang_min <= 1.00) and (overhang_max <= 1.00)
            span_ok = (s_pct <= span_tolerance_pct)

            new_axis_pass = inside_ok and exceed_ok and span_ok
            if not new_axis_pass:
                new_extents_passed = False
                if not inside_ok:
                    failure_reasons.append(
                        f"{ax} box does not enclose reference box within {corner_tolerance_m:.2f}m tolerance "
                        f"(underhang_min={underhang_min:+.2f}m, underhang_max={underhang_max:+.2f}m)"
                    )
                if not exceed_ok:
                    failure_reasons.append(
                        f"{ax} splat box exceeds reference box by > 1.00m "
                        f"(overhang_min={overhang_min:+.2f}m, overhang_max={overhang_max:+.2f}m)"
                    )
                if not span_ok:
                    failure_reasons.append(f"{ax} span difference {s_pct:.1f}% > {span_tolerance_pct:.1f}% tolerance")
            new_tag = f"overhang=[{overhang_min:+.2f}, {overhang_max:+.2f}]m [{ 'PASS' if new_axis_pass else 'FAIL' }]"
        else:
            # Axis Y: Floor near y = 0.00 m (lowest dense layer) and ceiling clearance
            floor_near_zero = (abs(p2_filt[1]) <= y_p2_diff_tol_m)
            ceiling_ok = (p98_filt[1] <= max_ceiling_bound_m)
            new_axis_pass = floor_near_zero and ceiling_ok
            underhang_min = float(p2_filt[1])
            underhang_max = float(max_ceiling_bound_m - p98_filt[1])
            overhang_min = float(-p2_filt[1])
            overhang_max = float(p98_filt[1])
            inside_ok = floor_near_zero
            exceed_ok = ceiling_ok
            if not new_axis_pass:
                new_extents_passed = False
                if not floor_near_zero:
                    failure_reasons.append(f"Y lowest dense layer p2={p2_filt[1]:+.2f}m not near floor y=0 (tol: +/-{y_p2_diff_tol_m:.2f}m)")
                if not ceiling_ok:
                    failure_reasons.append(f"Y_max {p98_filt[1]:.2f}m > {max_ceiling_bound_m:.2f}m (above ceiling clearance)")
            new_tag = f"floor_p2={p2_filt[1]:+.2f}m, ceil={p98_filt[1]:.2f}m [{ 'PASS' if new_axis_pass else 'FAIL' }]"

        print(f"  {ax:<4} | {ref_s:<20} | {filt_s:<22} | {old_tag:<24} | {new_tag}")

        percentile_report[ax] = {
            "ref_p2": round(float(p2_ref[i]), 3),
            "ref_p98": round(float(p98_ref[i]), 3),
            "ref_span": round(float(span_ref[i]), 3),
            "filt_p2": round(float(p2_filt[i]), 3),
            "filt_p98": round(float(p98_filt[i]), 3),
            "filt_span": round(float(span_filt[i]), 3),
            "diff_p2_m": round(d_p2, 3),
            "diff_p98_m": round(d_p98, 3),
            "diff_span_pct": round(s_pct, 2),
            "old_rule_pass": bool(old_axis_pass),
            "coverage_rule": {
                "underhang_min_m": round(underhang_min, 3),
                "underhang_max_m": round(underhang_max, 3),
                "overhang_min_m": round(overhang_min, 3),
                "overhang_max_m": round(overhang_max, 3),
                "inside_tol_pass": bool(inside_ok),
                "exceed_tol_pass": bool(exceed_ok),
                "pass": bool(new_axis_pass),
            },
        }

    extents_passed = new_extents_passed

    # -----------------------------------------------------------------------
    # Step 3: Rigid Alignment Fit (Umeyama / Trimmed ICP)
    # -----------------------------------------------------------------------
    print("\n[Step 3/4] Estimating rigid similarity transform vs reference points (50k subsample)...")
    np.random.seed(42)
    sample_size = min(50000, len(splat_raw_xyz))
    splat_sample = splat_raw_xyz[np.random.choice(len(splat_raw_xyz), sample_size, replace=False)]

    s_un, R_un, t_un, yaw_un, rmse_un, raw_d_un = compute_umeyama_alignment(splat_sample, ref_pts)
    t_norm_un = float(np.linalg.norm(t_un))

    splat_mx = splat_sample.copy()
    splat_mx[:, 0] = -splat_mx[:, 0]
    s_mx, R_mx, t_mx, yaw_mx, rmse_mx, raw_d_mx = compute_umeyama_alignment(splat_mx, ref_pts)

    splat_mz = splat_sample.copy()
    splat_mz[:, 2] = -splat_mz[:, 2]
    s_mz, R_mz, t_mz, yaw_mz, rmse_mz, raw_d_mz = compute_umeyama_alignment(splat_mz, ref_pts)

    print(f"  Unmirrored fit : yaw = {yaw_un:+5.2f} deg, |t| = {t_norm_un:.4f} m, scale = {s_un:.4f}, RMSE = {rmse_un:.4f} m")
    print(f"  X-mirrored fit : yaw = {yaw_mx:+5.2f} deg, |t| = {np.linalg.norm(t_mx):.4f} m, scale = {s_mx:.4f}, RMSE = {rmse_mx:.4f} m")
    print(f"  Z-mirrored fit : yaw = {yaw_mz:+5.2f} deg, |t| = {np.linalg.norm(t_mz):.4f} m, scale = {s_mz:.4f}, RMSE = {rmse_mz:.4f} m")

    unmirrored_is_best = (rmse_un < rmse_mx) and (rmse_un < rmse_mz)
    yaw_pass = abs(yaw_un) <= yaw_tolerance_deg
    trans_pass = t_norm_un <= trans_tolerance_m
    scale_pass = (scale_range[0] <= s_un <= scale_range[1])
    alignment_passed = unmirrored_is_best and yaw_pass and trans_pass and scale_pass

    if not unmirrored_is_best:
        failure_reasons.append(
            f"Unmirrored fit RMSE ({rmse_un:.4f}m) not strictly better than mirrored fits (X: {rmse_mx:.4f}m, Z: {rmse_mz:.4f}m)"
        )
    if not yaw_pass:
        failure_reasons.append(f"Yaw |{yaw_un:.2f}°| > {yaw_tolerance_deg:.1f}° tolerance")
    if not trans_pass:
        failure_reasons.append(f"Translation |t| {t_norm_un:.4f}m > {trans_tolerance_m:.2f}m tolerance")
    if not scale_pass:
        failure_reasons.append(f"Scale {s_un:.4f} outside [{scale_range[0]}, {scale_range[1]}] tolerance")

    print(f"  Alignment fit verdict: {'[PASS]' if alignment_passed else '[FAIL]'}")

    # -----------------------------------------------------------------------
    # Step 4: Camera Inclusion (Requirement 2b), Floor Peak & Tilt Checks
    # -----------------------------------------------------------------------
    print("\n[Step 4/4] Checking camera inclusion, floor plane, and camera plane tilt...")

    # Requirement 2b: Test each camera center against scene.json bounds in x and z
    # and against y in [floor_y, ceiling_y]. Do NOT test against opacity-filtered splats bbox.
    cam_x_min, cam_x_max = float(box_min[0]), float(box_max[0])
    cam_z_min, cam_z_max = float(box_min[2]), float(box_max[2])
    cam_y_min, cam_y_max = float(floor_y_scene), float(ceiling_y_scene)

    inside_x = (all_cams[:, 0] >= cam_x_min) & (all_cams[:, 0] <= cam_x_max)
    inside_z = (all_cams[:, 2] >= cam_z_min) & (all_cams[:, 2] <= cam_z_max)
    inside_y = (all_cams[:, 1] >= cam_y_min) & (all_cams[:, 1] <= cam_y_max)
    inside_all = inside_x & inside_y & inside_z
    pct_cams_inside = float(np.mean(inside_all) * 100)
    cams_inc_pass = (pct_cams_inside == 100.0)

    print(f"  Camera inclusion (scene.json bounds X/Z and Y in [{cam_y_min:.2f}, {cam_y_max:.2f}] m):")
    print(f"    - In X [{cam_x_min:+.2f}, {cam_x_max:+.2f}] m : {np.sum(inside_x)}/{len(all_cams)} ({np.mean(inside_x)*100:.1f}%)")
    print(f"    - In Z [{cam_z_min:+.2f}, {cam_z_max:+.2f}] m : {np.sum(inside_z)}/{len(all_cams)} ({np.mean(inside_z)*100:.1f}%)")
    print(f"    - In Y [{cam_y_min:+.2f}, {cam_y_max:+.2f}] m : {np.sum(inside_y)}/{len(all_cams)} ({np.mean(inside_y)*100:.1f}%)")
    print(f"    - Total enclosed: {np.sum(inside_all)}/{len(all_cams)} ({pct_cams_inside:.1f}%) {'[PASS]' if cams_inc_pass else '[FAIL]'}")
    if not cams_inc_pass:
        failure_reasons.append(
            f"Camera centres enclosed in scene bounds is {pct_cams_inside:.1f}% ({np.sum(inside_all)}/{len(all_cams)}) < 100.0% "
            f"(X: {np.sum(inside_x)}/{len(all_cams)}, Y: {np.sum(inside_y)}/{len(all_cams)}, Z: {np.sum(inside_z)}/{len(all_cams)})"
        )

    # Splat floor band RANSAC & Floor Peak check
    try:
        norm_fl, h_fl, tilt_fl, inl_fl, tot_fl = fit_floor_band_ransac(splat_filt_xyz, y_center=0.00, band=0.10)
        splat_tilt_pass = tilt_fl <= floor_tilt_tol_deg

        floor_band_mask = np.abs(splat_filt_xyz[:, 1] - h_fl) < 0.05
        floor_band_pts = splat_filt_xyz[floor_band_mask]
        counts, edges = np.histogram(floor_band_pts[:, 1], bins=np.arange(-0.10, 0.12, 0.02))
        centres = 0.5 * (edges[:-1] + edges[1:])
        floor_peak_idx = int(np.argmax(counts))
        floor_peak_y = float(centres[floor_peak_idx])
        floor_peak_pass = abs(floor_peak_y) <= floor_peak_tol_m and abs(h_fl) <= floor_peak_tol_m
        print(f"  Filtered splat floor peak: {floor_peak_y:+.3f} m (allowed |y| <= {floor_peak_tol_m:.2f} m) {'[PASS]' if floor_peak_pass else '[FAIL]'}")
        if not floor_peak_pass:
            failure_reasons.append(f"Floor peak |{floor_peak_y:+.3f}m| > {floor_peak_tol_m:.2f}m")

        print(f"  Splat floor plane normal : [{norm_fl[0]:+.4f}, {norm_fl[1]:+.4f}, {norm_fl[2]:+.4f}]")
        print(f"  Splat floor plane height : {h_fl:+.4f} m (inliers: {inl_fl:,}/{tot_fl:,})")
        print(f"  Splat floor plane tilt   : {tilt_fl:.2f} deg (allowed <= {floor_tilt_tol_deg:.1f} deg) {'[PASS]' if splat_tilt_pass else '[FAIL]'}")
        if not splat_tilt_pass:
            failure_reasons.append(f"Splat floor tilt {tilt_fl:.2f}° > {floor_tilt_tol_deg:.1f}°")
    except Exception as e:
        norm_fl = np.array([0.0, 1.0, 0.0])
        h_fl = float(np.percentile(splat_filt_xyz[:, 1], 2))
        tilt_fl = 90.0
        inl_fl = 0
        tot_fl = len(splat_filt_xyz)
        floor_peak_y = h_fl
        floor_peak_pass = False
        splat_tilt_pass = False
        print(f"  Filtered splat floor detection: FAILED ({e}, lowest dense layer at y={h_fl:+.3f} m) [FAIL]")
        failure_reasons.append(f"Floor plane detection near y=0 failed: {e} (lowest dense layer at y={h_fl:+.3f} m)")

    # Camera plane tilts
    norm_p1, _, tilt_p1, _ = fit_svd_plane(cams_p1) if len(cams_p1) else (np.zeros(3), 0, 0, 0)
    norm_p2, _, tilt_p2, _ = fit_svd_plane(cams_p2) if len(cams_p2) else (np.zeros(3), 0, 0, 0)
    cam_tilt_pass = (tilt_p1 <= cam_tilt_tol_deg) and (tilt_p2 <= cam_tilt_tol_deg)
    print(f"  Pass 1 camera plane tilt : {tilt_p1:.2f} deg (allowed <= {cam_tilt_tol_deg:.1f} deg) {'[PASS]' if tilt_p1 <= cam_tilt_tol_deg else '[FAIL]'}")
    print(f"  Pass 2 camera plane tilt : {tilt_p2:.2f} deg (allowed <= {cam_tilt_tol_deg:.1f} deg) {'[PASS]' if tilt_p2 <= cam_tilt_tol_deg else '[FAIL]'}")
    if not cam_tilt_pass:
        failure_reasons.append(f"Camera plane tilt (P1: {tilt_p1:.2f}°, P2: {tilt_p2:.2f}°) > {cam_tilt_tol_deg:.1f}°")

    mean_cam_height_above_floor = float(np.mean(all_cams[:, 1]) - h_fl)
    below_neg_03_count = int(np.sum(splat_raw_xyz[:, 1] < -0.30))
    below_neg_03_pct = float(below_neg_03_count / len(splat_raw_xyz) * 100)
    print(f"  Mean camera height       : {mean_cam_height_above_floor:.3f} m above splat floor")
    print(f"  Gaussians below y=-0.3m  : {below_neg_03_count:,} ({below_neg_03_pct:.2f}%)")

    overall_pass = extents_passed and alignment_passed and cams_inc_pass and floor_peak_pass and splat_tilt_pass and cam_tilt_pass

    print("\n" + "=" * 82)
    if overall_pass:
        print("  VERIFICATION RESULT: [PASS] SPLAT COORDINATE FRAME IS VALID")
    else:
        print("  VERIFICATION RESULT: [FAIL] COORDINATE FRAME CRITERIA VIOLATED")
    print("=" * 82)

    if failure_reasons:
        print("  Failures:")
        for r in failure_reasons:
            print(f"    - {r}")

    res_data = {
        "scene": scene,
        "status": "PASS" if overall_pass else "FAIL",
        "splat_file": str(splat_ply.resolve()),
        "raw_gaussian_count": len(splat_raw_xyz),
        "filtered_gaussian_count": len(splat_filt_xyz),
        "camera_count": len(all_cams),
        "camera_passes": {
            "pass_1_count": len(cams_p1),
            "pass_1_mean_height_m": round(mean_h_p1, 3),
            "pass_2_count": len(cams_p2),
            "pass_2_mean_height_m": round(mean_h_p2, 3),
            "height_ratio": round(ratio_p1_p2, 2),
        },
        "camera_inclusion_bounds": {
            "x_bounds": [round(cam_x_min, 3), round(cam_x_max, 3)],
            "y_bounds": [round(cam_y_min, 3), round(cam_y_max, 3)],
            "z_bounds": [round(cam_z_min, 3), round(cam_z_max, 3)],
            "cameras_inside_count": int(np.sum(inside_all)),
            "cameras_inside_pct": round(pct_cams_inside, 2),
        },
        "checks": {
            "extents_within_tolerance": extents_passed,
            "alignment_unmirrored_best": unmirrored_is_best,
            "alignment_yaw_pass": yaw_pass,
            "alignment_trans_pass": trans_pass,
            "alignment_scale_pass": scale_pass,
            "cameras_100pct_inside": cams_inc_pass,
            "floor_peak_pass": floor_peak_pass,
            "splat_floor_tilt_pass": splat_tilt_pass,
            "camera_plane_tilt_pass": cam_tilt_pass,
        },
        "percentile_extents": percentile_report,
        "alignment_fit": {
            "unmirrored": {
                "scale": round(s_un, 4),
                "translation_norm_m": round(t_norm_un, 4),
                "yaw_deg": round(yaw_un, 2),
                "rmse_m": round(rmse_un, 4),
            },
            "x_mirrored": {
                "scale": round(s_mx, 4),
                "translation_norm_m": round(float(np.linalg.norm(t_mx)), 4),
                "yaw_deg": round(yaw_mx, 2),
                "rmse_m": round(rmse_mx, 4),
            },
            "z_mirrored": {
                "scale": round(s_mz, 4),
                "translation_norm_m": round(float(np.linalg.norm(t_mz)), 4),
                "yaw_deg": round(yaw_mz, 2),
                "rmse_m": round(rmse_mz, 4),
            },
        },
        "floor": {
            "floor_peak_y_m": round(floor_peak_y, 4),
            "splat_plane_normal": [round(float(v), 4) for v in norm_fl],
            "splat_plane_height_m": round(h_fl, 4),
            "splat_tilt_deg": round(tilt_fl, 2),
            "splat_inliers": inl_fl,
            "pass1_cam_plane_tilt_deg": round(tilt_p1, 2),
            "pass2_cam_plane_tilt_deg": round(tilt_p2, 2),
            "mean_camera_height_above_floor_m": round(mean_cam_height_above_floor, 3),
            "gaussians_below_neg_0p3m": below_neg_03_count,
            "gaussians_below_neg_0p3m_pct": round(below_neg_03_pct, 2),
        },
        "failure_reasons": failure_reasons,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(res_data, f, indent=2)
        print(f"\n  [+] Saved verification report: {output_json_path}")

    return overall_pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify coordinate frame of exported 3D Gaussian Splat.")
    ap.add_argument("--scene", default="home", help="Scene name")
    ap.add_argument("--splat-ply", default=None, help="Path to exported splat PLY")
    ap.add_argument("--reference-pts", default=None, help="Path to reference points3D.txt")
    ap.add_argument("--transforms", default=None, help="Path to transforms.json")
    ap.add_argument("--scene-json", default=None, help="Path to scene.json")
    ap.add_argument("--corner-tolerance", type=float, default=0.40, help="Corner tolerance in metres")
    ap.add_argument("--span-tolerance", type=float, default=15.0, help="Span tolerance percentage")
    ap.add_argument("--yaw-tolerance", type=float, default=5.0, help="Yaw tolerance in degrees")
    ap.add_argument("--trans-tolerance", type=float, default=0.25, help="Translation tolerance in metres")
    ap.add_argument("--floor-tilt-tolerance", type=float, default=3.0, help="Floor plane tilt tolerance in degrees")
    ap.add_argument("--cam-tilt-tolerance", type=float, default=5.0, help="Camera plane tilt tolerance in degrees")
    ap.add_argument("--output-json", default=None, help="Output JSON verification path")

    args = ap.parse_args()
    scene = args.scene
    data_dir = Path("data") / scene
    outputs_dir = Path("outputs") / scene

    splat_ply = Path(args.splat_ply) if args.splat_ply else (outputs_dir / "scene_raw.ply")
    ref_path = Path(args.reference_pts) if args.reference_pts else (data_dir / "colmap_aligned" / "points3D.txt")
    transforms_path = Path(args.transforms) if args.transforms else (data_dir / "colmap_aligned" / "transforms.json")
    output_json = Path(args.output_json) if args.output_json else (outputs_dir / "splat_frame_verification.json")
    scene_json = Path(args.scene_json) if args.scene_json else None

    success = verify_splat_frame(
        scene=scene,
        splat_ply=splat_ply,
        reference_pts_path=ref_path,
        transforms_path=transforms_path,
        scene_json_path=scene_json,
        corner_tolerance_m=args.corner_tolerance,
        span_tolerance_pct=args.span_tolerance,
        yaw_tolerance_deg=args.yaw_tolerance,
        trans_tolerance_m=args.trans_tolerance,
        floor_tilt_tol_deg=args.floor_tilt_tolerance,
        cam_tilt_tol_deg=args.cam_tilt_tolerance,
        output_json_path=output_json,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
