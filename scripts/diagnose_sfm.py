#!/usr/bin/env python3
"""
scripts/diagnose_sfm.py -- Full SfM diagnostic per task spec.

Writes docs/sfm_diagnosis.md covering items (a) through (g).

Usage:
    python scripts/diagnose_sfm.py --scene home
"""

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# COLMAP text-format parsers
# ---------------------------------------------------------------------------

def parse_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cameras[int(p[0])] = {
                "model": p[1], "width": int(p[2]), "height": int(p[3]),
                "params": [float(x) for x in p[4:]],
            }
    return cameras


def parse_images_txt(path):
    """COLMAP images.txt: 2 lines per image."""
    images = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        img_id = int(p[0])
        qw, qx, qy, qz = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        tx, ty, tz = float(p[5]), float(p[6]), float(p[7])
        cam_id = int(p[8])
        name = p[9]
        images[img_id] = {
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": tx, "ty": ty, "tz": tz,
            "camera_id": cam_id, "name": name,
        }
        i += 2
    return images


def parse_points3d_txt(path):
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            points[int(p[0])] = {
                "xyz": np.array([float(p[1]), float(p[2]), float(p[3])]),
                "error": float(p[7]),
            }
    return points


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ])


def camera_centre(img):
    R = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
    return -(R.T @ np.array([img["tx"], img["ty"], img["tz"]]))


def camera_yaw_deg(img):
    """Yaw (rotation around Y) of camera pose, in degrees."""
    R = quat_to_rot(img["qw"], img["qx"], img["qy"], img["qz"])
    # The forward direction in world space is the 3rd row of R^T (z-column of R)
    # Yaw = atan2(-forward_x, forward_z) or atan2 of the (0,2)/(2,2) elements
    fwd = R.T @ np.array([0., 0., 1.])
    return math.degrees(math.atan2(fwd[0], fwd[2]))


def frame_number(name):
    return int(name.replace("frame_", "").replace(".jpg", "").replace(".png", ""))


def signed_area_2d(pts):
    """Signed area via shoelace formula. Positive = CCW."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return list(points)
    # Find point with max distance from line start->end
    start, end = np.array(points[0]), np.array(points[-1])
    line = end - start
    line_len = np.linalg.norm(line)
    max_dist, max_idx = 0.0, 0
    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        if line_len < 1e-10:
            d = np.linalg.norm(pt - start)
        else:
            # 2D scalar cross product: (line × (start - pt))
            v = start - pt
            d = abs(line[0]*v[1] - line[1]*v[0]) / line_len
        if d > max_dist:
            max_dist, max_idx = d, i
    if max_dist > epsilon:
        left  = rdp_simplify(points[:max_idx+1], epsilon)
        right = rdp_simplify(points[max_idx:],    epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def turn_sequence(simplified):
    """Compute L/R turn sequence from simplified path vertices."""
    turns = []
    for i in range(1, len(simplified) - 1):
        a = np.array(simplified[i])   - np.array(simplified[i-1])
        b = np.array(simplified[i+1]) - np.array(simplified[i])
        cross = a[0]*b[1] - a[1]*b[0]
        turns.append("L" if cross > 0 else "R")
    return ",".join(turns)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def count_long_range_matches(db_path, pass1_names, pass2_names, gap=30):
    """
    Count verified pairs where |frame_i - frame_j| > gap (within each pass)
    and all cross-pass pairs. Returns (within_p1, within_p2, cross) dicts of
    {(name_i, name_j): num_matches}.
    """
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # Build image_id -> name, frame_num mapping
    cur.execute("SELECT image_id, name FROM images")
    id2name = {r[0]: r[1] for r in cur.fetchall()}
    name2id = {v: k for k, v in id2name.items()}

    def frame(name):
        try:
            return frame_number(name)
        except Exception:
            return -1

    p1_set = set(pass1_names)
    p2_set = set(pass2_names)

    cur.execute("SELECT pair_id, rows FROM two_view_geometries")
    rows = cur.fetchall()

    within_p1, within_p2, cross = {}, {}, {}
    for pair_id, n_matches in rows:
        if n_matches == 0:
            continue
        # Decode pair_id
        id1 = pair_id >> 32
        id2 = pair_id & 0xFFFFFFFF
        n1 = id2name.get(id1, "")
        n2 = id2name.get(id2, "")
        f1, f2 = frame(n1), frame(n2)
        in_p1_1 = n1 in p1_set
        in_p1_2 = n2 in p1_set
        in_p2_1 = n1 in p2_set
        in_p2_2 = n2 in p2_set
        # Cross-pass
        if (in_p1_1 and in_p2_2) or (in_p2_1 and in_p1_2):
            cross[(n1, n2)] = n_matches
        elif in_p1_1 and in_p1_2:
            if abs(f1 - f2) > gap:
                within_p1[(n1, n2)] = n_matches
        elif in_p2_1 and in_p2_2:
            if abs(f1 - f2) > gap:
                within_p2[(n1, n2)] = n_matches
    conn.close()
    return within_p1, within_p2, cross


# ---------------------------------------------------------------------------
# Floor-plane detection via RANSAC
# ---------------------------------------------------------------------------

def ransac_plane(pts, n_iter=500, thresh=0.05):
    """Fit a plane to 3D points by RANSAC. Returns (normal, d, inlier_frac)."""
    best_count, best_n, best_d = 0, np.array([0., 1., 0.]), 0.0
    rng = np.random.default_rng(42)
    n = len(pts)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(nrm) < 1e-10:
            continue
        nrm /= np.linalg.norm(nrm)
        d = -float(nrm @ p0)
        dists = np.abs(pts @ nrm + d)
        count = int((dists < thresh).sum())
        if count > best_count:
            best_count, best_n, best_d = count, nrm, d
    inlier_frac = best_count / n if n > 0 else 0.0
    # Orient normal upward
    if best_n[1] < 0:
        best_n, best_d = -best_n, -best_d
    return best_n, best_d, inlier_frac


# ---------------------------------------------------------------------------
# Similarity transform (rotation+scale+translation, 2D)
# ---------------------------------------------------------------------------

def fit_similarity_2d(src, dst):
    """
    Fit similarity transform (rotation, uniform scale, translation) from src->dst.
    Both are (N,2) arrays. Returns (R2, s, t, residual_rms).
    No reflection allowed.
    """
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    # Scale
    ss = float((src_c**2).sum())
    if ss < 1e-12:
        return np.eye(2), 1.0, mu_d - mu_s, float('inf')
    # Covariance
    K = dst_c.T @ src_c / ss
    U, S_sv, Vt = np.linalg.svd(K)
    # Enforce proper rotation (no reflection)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])
    R2 = U @ D @ Vt
    s = float(np.trace(np.diag(S_sv) @ D))
    t = mu_d - s * (R2 @ mu_s)
    # Residual
    pred = (s * (R2 @ src_c.T)).T + mu_d
    rms = float(np.sqrt(((pred - dst)**2).sum(axis=1).mean()))
    return R2, s, t, rms


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Diagnose SfM reconstruction quality.")
    ap.add_argument("--scene", default="home")
    ap.add_argument("--pass-split", type=int, default=None,
                    help="Frame number boundary between pass 1 and pass 2 (inclusive = pass 1)")
    args = ap.parse_args()

    scene     = args.scene
    data_dir  = Path("data") / scene
    sparse_dir = data_dir / "colmap" / "sparse" / "0"
    db_path   = data_dir / "colmap" / "database.db"
    aligned_dir = data_dir / "colmap" / "aligned"

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print("Loading sparse model (TXT)...")
    cam_path  = sparse_dir / "cameras.txt"
    img_path  = sparse_dir / "images.txt"
    pts_path  = sparse_dir / "points3D.txt"
    aln_img_path = aligned_dir / "images.txt"
    aln_pts_path = aligned_dir / "points3D.txt"

    cameras = parse_cameras_txt(cam_path)
    images  = parse_images_txt(img_path)
    points  = parse_points3d_txt(pts_path)
    print(f"  Raw model: {len(cameras)} cameras, {len(images)} images, {len(points)} points")

    has_aligned = aln_img_path.exists() and aln_pts_path.exists()
    if has_aligned:
        images_a = parse_images_txt(aln_img_path)
        points_a = parse_points3d_txt(aln_pts_path)
        print(f"  Aligned model: {len(images_a)} images, {len(points_a)} points")
    else:
        images_a, points_a = images, points
        print("  No aligned model found -- using raw model for alignment checks.")

    # ---- pass split --------------------------------------------------------
    frames_rpt_path = data_dir / "frames_report.json"
    pass_split = args.pass_split
    pass1_frames_raw = 0
    if frames_rpt_path.exists():
        with open(frames_rpt_path) as f:
            fr = json.load(f)
        vids = fr.get("videos", [])
        if len(vids) >= 2:
            kept_v0 = vids[0].get("kept_after_duplicate", 0)
            total_kept = sum(v.get("kept_after_duplicate", 0) for v in vids)
            final_count = fr.get("summary", {}).get("final_frame_count", total_kept)
            pass1_frames_raw = vids[0].get("raw_extracted", 0)
            if total_kept > final_count and total_kept > 0:
                ps = int(round(kept_v0 / total_kept * final_count))
            else:
                ps = int(kept_v0)
            if pass_split is None:
                pass_split = ps
            print(f"  Pass split from frames_report: {pass_split}")
            total_frames = fr.get("summary", {}).get("final_frame_count", 0)
            n_pass1_total = ps
            n_pass2_total = total_frames - ps
            print(f"  Total frames: {total_frames}  Pass1: {n_pass1_total}  Pass2: {n_pass2_total}")

    if pass_split is None:
        pass_split = len(images) // 2
        print(f"  No frames_report -- using midpoint pass_split={pass_split}")

    # Sort registered images by frame number
    registered_by_frame = sorted(images.keys(), key=lambda i: frame_number(images[i]["name"]))
    reg_names = {images[i]["name"] for i in images}
    all_frame_nums = [frame_number(images[i]["name"]) for i in images]

    p1_reg = [i for i in images if frame_number(images[i]["name"]) <= pass_split]
    p2_reg = [i for i in images if frame_number(images[i]["name"]) >  pass_split]
    p1_names = {images[i]["name"] for i in p1_reg}
    p2_names = {images[i]["name"] for i in p2_reg}

    # Count how many images exist on disk per pass
    images_dir = data_dir / "images"
    all_img_files = sorted(images_dir.glob("frame_*.jpg"))
    total_all = len(all_img_files)
    p1_total = sum(1 for f in all_img_files if frame_number(f.name) <= pass_split)
    p2_total = total_all - p1_total

    print(f"\n  Registered: Pass1={len(p1_reg)}/{p1_total} ({100*len(p1_reg)/max(p1_total,1):.1f}%)")
    print(f"              Pass2={len(p2_reg)}/{p2_total} ({100*len(p2_reg)/max(p2_total,1):.1f}%)")

    # ---- camera centres (raw model) ----------------------------------------
    centres_raw = {}
    for iid in images:
        C = camera_centre(images[iid])
        if np.linalg.norm(C) < 1000:
            centres_raw[iid] = C

    # ---- camera centres (aligned model) ------------------------------------
    centres_aln = {}
    for iid in images_a:
        C = camera_centre(images_a[iid])
        if np.linalg.norm(C) < 1000:
            centres_aln[iid] = C

    # =========================================================================
    # (a) Models / registration
    # =========================================================================
    print("\n[a] Model count...")
    sparse_base = data_dir / "colmap" / "sparse"
    model_dirs = sorted([d for d in sparse_base.iterdir() if d.is_dir()])
    model_info = []
    for md in model_dirs:
        ip = md / "images.txt"
        if ip.exists():
            imgs_m = parse_images_txt(ip)
            model_info.append((md.name, len(imgs_m)))
            print(f"  Model {md.name}: {len(imgs_m)} registered images")
    n_models = len(model_info)
    total_reg = sum(n for _, n in model_info)
    both_in_one = (n_models == 1) or (model_info[0][1] == len(images) if model_info else False)

    # =========================================================================
    # (b) Cameras / intrinsics
    # =========================================================================
    print("\n[b] Camera intrinsics...")
    cam_info_lines = []
    for cid, cam in sorted(cameras.items()):
        w, h = cam["width"], cam["height"]
        params = cam["params"]
        fx = params[0] if params else "?"
        line = (f"Camera {cid}: {cam['model']} {w}x{h} | "
                f"f={params[0]:.2f}px" if params else f"Camera {cid}: {cam['model']} {w}x{h}")
        if len(params) > 1:
            line += f" cx={params[1]:.1f} cy={params[2]:.1f}"
        if len(params) > 3:
            line += f" k1={params[3]:.6f}"
        if len(params) > 4:
            line += f" k2={params[4]:.6f}"
        cam_info_lines.append(line)
        print(f"  {line}")
    print(f"  Number of distinct cameras: {len(cameras)}")
    sharing_ok = len(cameras) == 1

    # =========================================================================
    # (c) Step lengths and yaw jumps (aligned model)
    # =========================================================================
    print("\n[c] Step lengths and yaw jumps (aligned model)...")
    p1_sorted = sorted(p1_reg, key=lambda i: frame_number(images[i]["name"]))
    p2_sorted = sorted(p2_reg, key=lambda i: frame_number(images[i]["name"]))

    def compute_steps(iids, centres, imgs):
        steps = []
        for k in range(1, len(iids)):
            a, b = iids[k-1], iids[k]
            if a in centres and b in centres:
                dist = float(np.linalg.norm(centres[b] - centres[a]))
                ya = camera_yaw_deg(imgs[a])
                yb = camera_yaw_deg(imgs[b])
                dyaw = abs(((yb - ya) + 180) % 360 - 180)
                steps.append({
                    "from": images[a]["name"], "to": images[b]["name"],
                    "dist_m": dist, "dyaw_deg": dyaw,
                })
        return steps

    steps_p1 = compute_steps(p1_sorted, centres_aln, images_a if has_aligned else images)
    steps_p2 = compute_steps(p2_sorted, centres_aln, images_a if has_aligned else images)

    big_jumps = sorted(steps_p1 + steps_p2, key=lambda s: -s["dist_m"])[:10]
    yaw_jumps_p1 = [s for s in steps_p1 if s["dyaw_deg"] > 40]
    yaw_jumps_p2 = [s for s in steps_p2 if s["dyaw_deg"] > 40]

    print(f"  Top 10 distance jumps:")
    for s in big_jumps:
        print(f"    {s['from']} -> {s['to']}: {s['dist_m']:.3f} m  Δyaw={s['dyaw_deg']:.1f}°")
    print(f"  Yaw jumps >40° in Pass1: {len(yaw_jumps_p1)}, Pass2: {len(yaw_jumps_p2)}")
    for s in yaw_jumps_p1 + yaw_jumps_p2:
        print(f"    {s['from']} -> {s['to']}: Δyaw={s['dyaw_deg']:.1f}°")

    # =========================================================================
    # (d) Long-range matches from database
    # =========================================================================
    print("\n[d] Long-range matches...")
    lr_p1, lr_p2, lr_cross = {}, {}, {}
    db_ok = False
    if db_path.exists():
        try:
            lr_p1, lr_p2, lr_cross = count_long_range_matches(db_path, p1_names, p2_names, gap=30)
            db_ok = True
            print(f"  Within-pass1 long-range pairs (|Δframe|>30): {len(lr_p1)}")
            print(f"  Within-pass2 long-range pairs (|Δframe|>30): {len(lr_p2)}")
            print(f"  Cross-pass pairs: {len(lr_cross)}")
            # Suspicious: within-pass pairs with few inliers
            sus_p1 = [(k, v) for k, v in lr_p1.items() if v < 20]
            sus_p2 = [(k, v) for k, v in lr_p2.items() if v < 20]
            print(f"  Suspicious (low-inlier) within-pass pairs: P1={len(sus_p1)} P2={len(sus_p2)}")
        except Exception as e:
            print(f"  DB read error: {e}")

    # =========================================================================
    # (e) Handedness / signed area / turn sequence
    # =========================================================================
    print("\n[e] Handedness (top-view XZ plane)...")

    def path_xz(iids, centres, imgs, aln_centres):
        """Return list of (x, z) tuples, sorted by frame number."""
        sorted_i = sorted(iids, key=lambda i: frame_number(imgs[i]["name"]))
        pts = []
        for i in sorted_i:
            if i in aln_centres:
                c = aln_centres[i]
                pts.append((c[0], c[2]))
        return pts

    p1_path_xz = path_xz(p1_reg, centres_aln, images if not has_aligned else images_a, centres_aln)
    p2_path_xz = path_xz(p2_reg, centres_aln, images if not has_aligned else images_a, centres_aln)

    # Use aligned centres keyed by img_id, but image names from raw images
    # Top-down view looking along -Y: X is East (right), -Z is North (up)
    def path_xz_v2(iids, raw_imgs, aln_ctrs):
        sorted_i = sorted(iids, key=lambda i: frame_number(raw_imgs[i]["name"]))
        return [(aln_ctrs[i][0], -aln_ctrs[i][2]) for i in sorted_i if i in aln_ctrs]

    p1_path_xz = path_xz_v2(p1_reg, images, centres_aln)
    p2_path_xz = path_xz_v2(p2_reg, images, centres_aln)

    # Estimate scale (median step distance) to convert RDP epsilon from metres
    all_steps_m = [s["dist_m"] for s in steps_p1 + steps_p2]
    median_step = float(np.median(all_steps_m)) if all_steps_m else 1.0

    # RDP in aligned (metric) coordinates; epsilon = 0.5 m
    rdp_eps = 0.5
    p1_simp = rdp_simplify(p1_path_xz, rdp_eps)
    p2_simp = rdp_simplify(p2_path_xz, rdp_eps)

    area_p1 = signed_area_2d(p1_path_xz) if p1_path_xz else 0.0
    area_p2 = signed_area_2d(p2_path_xz) if p2_path_xz else 0.0

    turns_p1 = turn_sequence(p1_simp) if len(p1_simp) >= 3 else "N/A"
    turns_p2 = turn_sequence(p2_simp) if len(p2_simp) >= 3 else "N/A"

    expected_turns = "L,L,L"
    plan_path = Path("docs/floorplan.json")
    if plan_path.exists():
        try:
            with open(plan_path) as f:
                fp_meta = json.load(f)
                expected_turns = fp_meta.get("turns", expected_turns)
        except Exception:
            pass

    print(f"  Pass 1 signed area (top-down X, -Z): {area_p1:.4f}  {'CCW' if area_p1 > 0 else 'CW'}")
    print(f"  Pass 2 signed area (top-down X, -Z): {area_p2:.4f}  {'CCW' if area_p2 > 0 else 'CW'}")
    print(f"  Pass 1 RDP simplified pts: {len(p1_simp)}  turns: {turns_p1}")
    print(f"  Pass 2 RDP simplified pts: {len(p2_simp)}  turns: {turns_p2}")
    print(f"  Expected: {expected_turns}")

    # Start-end gap (aligned coords)
    def start_end_gap(iids, raw_imgs, aln_ctrs):
        sorted_i = sorted(iids, key=lambda i: frame_number(raw_imgs[i]["name"]))
        valid = [i for i in sorted_i if i in aln_ctrs]
        if len(valid) < 2:
            return float('nan')
        return float(np.linalg.norm(aln_ctrs[valid[0]] - aln_ctrs[valid[-1]]))

    gap_p1 = start_end_gap(p1_reg, images, centres_aln)
    gap_p2 = start_end_gap(p2_reg, images, centres_aln)
    gap_both = start_end_gap(list(p1_reg) + list(p2_reg), images, centres_aln)
    print(f"\n  Start-end gap: Pass1={gap_p1:.3f} m  Pass2={gap_p2:.3f} m  Combined={gap_both:.3f} m")

    # Pass 1/2 median distance
    p1_ctrs_arr = np.array([centres_aln[i] for i in p1_reg if i in centres_aln])
    p2_ctrs_arr = np.array([centres_aln[i] for i in p2_reg if i in centres_aln])
    if len(p1_ctrs_arr) > 0 and len(p2_ctrs_arr) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(p1_ctrs_arr)
        dists_cross, _ = tree.query(p2_ctrs_arr, k=1)
        median_cross_dist = float(np.median(dists_cross))
        print(f"  Pass1<->Pass2 median nearest-neighbour distance: {median_cross_dist:.3f} m")
    else:
        median_cross_dist = float('nan')
        print("  Could not compute cross-pass distance (empty pass)")

    # =========================================================================
    # (f) Alignment quality
    # =========================================================================
    print("\n[f] Alignment quality...")

    # Camera up vectors in world space
    cam_ups_raw = []
    for iid in images:
        R = quat_to_rot(images[iid]["qw"], images[iid]["qx"], images[iid]["qy"], images[iid]["qz"])
        cam_ups_raw.append(R.T @ np.array([0., -1., 0.]))  # COLMAP: -Y is up
    avg_up_raw = np.mean(cam_ups_raw, axis=0)
    avg_up_raw /= np.linalg.norm(avg_up_raw)
    print(f"  Mean camera-up (raw model): [{avg_up_raw[0]:+.4f} {avg_up_raw[1]:+.4f} {avg_up_raw[2]:+.4f}]")

    if has_aligned:
        cam_ups_aln = []
        for iid in images_a:
            R = quat_to_rot(images_a[iid]["qw"], images_a[iid]["qx"],
                            images_a[iid]["qy"], images_a[iid]["qz"])
            cam_ups_aln.append(R.T @ np.array([0., -1., 0.]))
        avg_up_aln = np.mean(cam_ups_aln, axis=0)
        avg_up_aln /= np.linalg.norm(avg_up_aln)
        angle_to_Y = math.degrees(math.acos(min(1.0, abs(avg_up_aln[1]))))
        print(f"  Mean camera-up (aligned): [{avg_up_aln[0]:+.4f} {avg_up_aln[1]:+.4f} {avg_up_aln[2]:+.4f}]")
        print(f"  Angle of camera-up to world +Y: {angle_to_Y:.2f}° (pass if <= 8°)")

        # Floor-plane candidates
        pts_aln = np.array([p["xyz"] for p in points_a.values()])
        lo = np.percentile(pts_aln, 1, axis=0)
        hi = np.percentile(pts_aln, 99, axis=0)
        pts_in = pts_aln[np.all((pts_aln >= lo) & (pts_aln <= hi), axis=1)]

        # Candidate 1: all points near floor (lowest 20%)
        y_thresh = np.percentile(pts_in[:, 1], 20)
        floor_pts = pts_in[pts_in[:, 1] < y_thresh]
        floor_normal, floor_d, floor_inlier_frac = ransac_plane(floor_pts)
        angle_floor_to_up = math.degrees(math.acos(min(1.0, abs(float(floor_normal @ avg_up_aln)))))
        print(f"\n  Floor plane candidate (lowest 20% of pts):")
        print(f"    Normal: [{floor_normal[0]:+.4f} {floor_normal[1]:+.4f} {floor_normal[2]:+.4f}]")
        print(f"    Inlier fraction: {floor_inlier_frac:.3f}")
        print(f"    Angle to camera-up: {angle_floor_to_up:.2f}° (reject if > 8°)")

        # Per-pass camera heights
        p1_aln_ids = [i for i in images_a if frame_number(images[i]["name"] if i in images
                      else images_a[i]["name"]) <= pass_split] if has_aligned else p1_reg
        p2_aln_ids = [i for i in images_a if frame_number(images[i]["name"] if i in images
                      else images_a[i]["name"]) > pass_split] if has_aligned else p2_reg

        # Re-identify pass ids using images_a name-based split
        p1a_ids = [i for i in images_a if frame_number(images_a[i]["name"]) <= pass_split]
        p2a_ids = [i for i in images_a if frame_number(images_a[i]["name"]) > pass_split]

        h1 = [camera_centre(images_a[i])[1] for i in p1a_ids if i in centres_aln or True]
        h1 = [camera_centre(images_a[i])[1] for i in p1a_ids]
        h2 = [camera_centre(images_a[i])[1] for i in p2a_ids]
        med_h1 = float(np.median(h1)) if h1 else float('nan')
        med_h2 = float(np.median(h2)) if h2 else float('nan')
        ratio = med_h1 / med_h2 if med_h2 > 0 else float('nan')
        print(f"\n  Pass 1 median camera height: {med_h1:.3f} m  (expected 1.4-1.8 m)")
        print(f"  Pass 2 median camera height: {med_h2:.3f} m  (expected 1.0-1.4 m)")
        print(f"  Height ratio P1/P2: {ratio:.3f}  (expected ~1.3)")

        # Load align_report if present
        align_rpt_path = data_dir / "align_report.json"
        align_rpt = {}
        if align_rpt_path.exists():
            with open(align_rpt_path) as f:
                align_rpt = json.load(f)
            print(f"\n  align_report.json scale: {align_rpt.get('scale_m_per_unit', '?')} m/unit")
            print(f"  align_report method: {align_rpt.get('scale_method', '?')}")

    else:
        angle_to_Y = float('nan')
        floor_normal = np.array([0., 1., 0.])
        angle_floor_to_up = float('nan')
        med_h1 = med_h2 = ratio = float('nan')
        align_rpt = {}

    # =========================================================================
    # (g) Plan-fit residual (aligned model vs sketch)
    # =========================================================================
    print("\n[g] Plan-fit residual (combined pass cameras vs sketch path)...")

    with open("docs/floorplan.json") as f:
        floorplan = json.load(f)
    sketch_path = np.array(floorplan["path"])  # (N, 2) x/y in sketch units

    # Combined trajectory (all registered frames, sorted)
    all_sorted = sorted(list(p1_reg) + list(p2_reg),
                        key=lambda i: frame_number(images[i]["name"]))
    traj_xz = [(centres_aln[i][0], centres_aln[i][2]) for i in all_sorted if i in centres_aln]

    plan_fit_rms = float('nan')
    mirrored = False
    if len(traj_xz) >= 4:
        # Sample sketch path at uniform intervals matching trajectory density
        # Use a subset of trajectory points (every 5th) for fit
        traj_sub = np.array(traj_xz[::5])
        # The sketch is a closed loop -- sample at corresponding t values
        sketch_len = len(sketch_path)
        t_sub = np.linspace(0, sketch_len - 1, len(traj_sub))
        sketch_sub = np.array([sketch_path[int(t)] for t in t_sub])

        R2, s_fit, t_fit, rms_fit = fit_similarity_2d(traj_sub, sketch_sub)
        path_len_sketch = float(np.sum(np.linalg.norm(np.diff(sketch_path, axis=0), axis=1)))
        residual_pct = 100.0 * rms_fit / path_len_sketch if path_len_sketch > 0 else float('nan')
        det_R2 = float(np.linalg.det(R2))
        mirrored = det_R2 < 0

        plan_fit_rms = rms_fit
        print(f"  Similarity fit: s={s_fit:.4f}  det(R)={det_R2:.4f}  rms={rms_fit:.4f} sketch_units")
        print(f"  Path length (sketch): {path_len_sketch:.4f} sketch_units")
        print(f"  Plan-fit residual: {residual_pct:.1f}%  (pass < 15%)")
        print(f"  Mirror detected: {mirrored}  (must be False to pass)")
    else:
        print("  Not enough trajectory points for plan-fit.")
        residual_pct = float('nan')

    # Reprojection error from points3D.txt (direct, always accurate)
    reproj_error = float('nan')
    try:
        errs = [p["error"] for p in points.values() if p["error"] > 0]
        if errs:
            reproj_error = float(np.mean(errs))
            print(f"\n  Reprojection error (points3D.txt): mean={reproj_error:.4f} px  "
                  f"median={float(np.median(errs)):.4f} px")

    except Exception as e:
        print(f"  Could not compute reprojection error: {e}")

    # =========================================================================
    # Determine conclusion (g)
    # =========================================================================
    bad_sfm = (len(model_info) > 1) or (total_reg < 0.95 * total_all) or (reproj_error > 1.0)
    bad_alignment = (not math.isnan(angle_to_Y) and angle_to_Y > 8.0) or \
                    (not math.isnan(ratio) and (ratio < 1.1 or ratio > 1.6))
    bad_plot_only = (not bad_sfm) and (not bad_alignment) and (mirrored or (not math.isnan(residual_pct) and residual_pct > 15))

    conclusion_reason = []
    if len(model_info) > 1:
        conclusion_reason.append(f"(i) Bad SfM: {len(model_info)} sub-models instead of 1")
    if total_reg < 0.95 * total_all:
        conclusion_reason.append(f"(i) Bad SfM: {100*total_reg/total_all:.1f}% registered < 95% threshold")
    if reproj_error > 1.0:
        conclusion_reason.append(f"(i) Bad SfM: reprojection error {reproj_error:.3f} > 1 px")
    if not math.isnan(angle_to_Y) and angle_to_Y > 8.0:
        conclusion_reason.append(f"(ii) Bad alignment: camera-up angle to Y = {angle_to_Y:.2f}° > 8°")
    if not math.isnan(ratio) and (ratio < 1.1 or ratio > 1.6):
        conclusion_reason.append(f"(ii) Bad alignment/scale: height ratio P1/P2 = {ratio:.3f}, expected ~1.3")
    if mirrored:
        conclusion_reason.append("(iii) Plot bug: trajectory is mirrored (det(R)<0 in plan fit)")
    if not conclusion_reason:
        conclusion_reason.append("(iii) Plot-only issue or no clear cause found in this model")

    # =========================================================================
    # Write docs/sfm_diagnosis.md
    # =========================================================================
    print("\nWriting docs/sfm_diagnosis.md...")
    Path("docs").mkdir(exist_ok=True)

    lines = []
    lines += [
        "# SfM Diagnosis — Namma Space / home",
        "",
        f"> Generated: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        f"> Model: `data/home/colmap/sparse/0/`  (+ aligned at `data/home/colmap/aligned/`)",
        "",
        "---",
        "",
        "## (a) Models and Registration",
        "",
        f"**Number of sub-models in `sparse/`:** {n_models}",
        "",
    ]
    for name, n in model_info:
        lines.append(f"- Model `{name}`: **{n} images**")
    lines += [
        "",
        f"**Total registered:** {total_reg} / {total_all} "
        f"= **{100*total_reg/total_all:.1f}%** (pass criterion: ≥95% in ONE model)",
        "",
        f"| Pass | Registered | Total on disk | % |",
        f"|------|-----------|--------------|---|",
        f"| Pass 1 (eye-level, frames 1–{pass_split}) | {len(p1_reg)} | {p1_total} | {100*len(p1_reg)/max(p1_total,1):.1f}% |",
        f"| Pass 2 (chest-level, frames {pass_split+1}–{total_all}) | {len(p2_reg)} | {p2_total} | {100*len(p2_reg)/max(p2_total,1):.1f}% |",
        "",
        f"> **{'FAIL' if n_models > 1 or total_reg < 0.95*total_all else 'PASS'}**: ",
        f"> Both passes in ONE model: {'NO — split across ' + str(n_models) + ' sub-models' if n_models > 1 else 'YES'}",
        "",
        "---",
        "",
        "## (b) Camera Intrinsics",
        "",
        f"**Number of distinct camera entries:** {len(cameras)}",
        "",
    ]
    for line in cam_info_lines:
        lines.append(f"- `{line}`")
    lines += [
        "",
        f"**Shared single camera:** {'YES' if sharing_ok else 'NO — separate cameras per video (check if correct)'}",
        "",
        "> COLMAP SIMPLE_RADIAL: params = [f, cx, cy, k1]. Both videos are 1920×1080 30fps,",
        "> same phone, same zoom — sharing one camera model is appropriate here.",
        "",
        "---",
        "",
        "## (c) Step Lengths and Yaw Jumps (Aligned Model)",
        "",
        "### 10 Largest Frame-to-Frame Jumps",
        "",
        "| From | To | Distance (m) | Δyaw (°) |",
        "|------|-----|-------------|---------|",
    ]
    for s in big_jumps:
        lines.append(f"| `{s['from']}` | `{s['to']}` | {s['dist_m']:.3f} | {s['dyaw_deg']:.1f} |")
    lines += [
        "",
        "### Yaw Jumps > 40° in One Step",
        "",
        f"Pass 1: **{len(yaw_jumps_p1)}** events | Pass 2: **{len(yaw_jumps_p2)}** events",
        "",
    ]
    if yaw_jumps_p1 + yaw_jumps_p2:
        lines.append("| Pass | From | To | Δyaw (°) |")
        lines.append("|------|------|-----|---------|")
        for s in yaw_jumps_p1:
            lines.append(f"| P1 | `{s['from']}` | `{s['to']}` | {s['dyaw_deg']:.1f} |")
        for s in yaw_jumps_p2:
            lines.append(f"| P2 | `{s['from']}` | `{s['to']}` | {s['dyaw_deg']:.1f} |")
    else:
        lines.append("*No yaw jumps > 40° detected.*")
    lines += [
        "",
        "---",
        "",
        "## (d) Long-Range Matches (|Δframe| > 30)",
        "",
    ]
    if db_ok:
        lines += [
            f"| Category | Pairs | Notes |",
            f"|----------|-------|-------|",
            f"| Within Pass 1 (|Δ|>30) | {len(lr_p1)} | Loop closures or false aliases |",
            f"| Within Pass 2 (|Δ|>30) | {len(lr_p2)} | Loop closures or false aliases |",
            f"| Cross-pass | {len(lr_cross)} | Geometric links between passes |",
            "",
        ]
        sus_p1 = [(k, v) for k, v in lr_p1.items() if v < 20]
        sus_p2 = [(k, v) for k, v in lr_p2.items() if v < 20]
        lines.append(f"**Suspicious (< 20 inliers) within-pass long-range pairs:** "
                     f"P1={len(sus_p1)}, P2={len(sus_p2)}")
        if sus_p1[:5]:
            lines.append("")
            lines.append("Top suspicious Pass 1 pairs (low inlier count):")
            for (a, b), n in sorted(sus_p1, key=lambda x: x[1])[:5]:
                lines.append(f"  - `{a}` ↔ `{b}`: {n} inliers")
        if sus_p2[:5]:
            lines.append("")
            lines.append("Top suspicious Pass 2 pairs (low inlier count):")
            for (a, b), n in sorted(sus_p2, key=lambda x: x[1])[:5]:
                lines.append(f"  - `{a}` ↔ `{b}`: {n} inliers")
        lines += [
            "",
            "> **Note:** Many low-inlier long-range pairs within a pass indicate",
            "> perceptual aliasing (identical seat rows matching wrong locations).",
        ]
    else:
        lines.append("*Database not found or unreadable.*")
    lines += [
        "",
        "---",
        "",
        "## (e) Handedness and Turn Sequence",
        "",
        f"| Pass | Points | Signed area (XZ) | Winding | RDP pts | Turn sequence |",
        f"|------|--------|-----------------|---------|---------|--------------|",
        f"| Pass 1 | {len(p1_path_xz)} | {area_p1:.3f} | {'CCW ✓' if area_p1 > 0 else 'CW ✗'} | {len(p1_simp)} | `{turns_p1}` |",
        f"| Pass 2 | {len(p2_path_xz)} | {area_p2:.3f} | {'CCW ✓' if area_p2 > 0 else 'CW ✗'} | {len(p2_simp)} | `{turns_p2}` |",
        f"| **Expected** | — | > 0 | CCW | — | `L,R,L,L,R,L,L,R,L` |",
        "",
        f"**Start-end gap:**",
        f"- Pass 1: {gap_p1:.3f} m",
        f"- Pass 2: {gap_p2:.3f} m",
        f"- Combined first-to-last: {gap_both:.3f} m  *(pass criterion: < 0.5 m)*",
        "",
        f"**Cross-pass median nearest-neighbour distance:** {median_cross_dist:.3f} m  *(pass criterion: ≤ 0.4 m)*",
        "",
        "---",
        "",
        "## (f) Alignment Quality",
        "",
    ]
    if has_aligned:
        lines += [
            f"**Mean camera-up vector (aligned model):** "
            f"[{avg_up_aln[0]:+.4f}, {avg_up_aln[1]:+.4f}, {avg_up_aln[2]:+.4f}]",
            f"**Angle to world +Y:** {angle_to_Y:.2f}°  *(pass criterion: ≤ 8°)*",
            "",
            "### Floor Plane Candidates",
            "",
            "| Candidate | Normal | Inlier frac | Angle to cam-up | Accept? |",
            "|-----------|--------|-------------|----------------|---------|",
            f"| Lowest 20% of cloud | [{floor_normal[0]:+.4f},{floor_normal[1]:+.4f},{floor_normal[2]:+.4f}] | {floor_inlier_frac:.3f} | {angle_floor_to_up:.2f}° | {'YES' if angle_floor_to_up <= 8 else 'NO'} |",
            "",
            "### Per-Pass Camera Heights",
            "",
            f"| Pass | Median height (m) | Expected range (m) | Status |",
            f"|------|------------------|-------------------|--------|",
            f"| Pass 1 (eye-level) | {med_h1:.3f} | 1.4 – 1.8 | {'✓' if 1.4 <= med_h1 <= 1.8 else '✗ OUT OF RANGE'} |",
            f"| Pass 2 (chest-level) | {med_h2:.3f} | 1.0 – 1.4 | {'✓' if 1.0 <= med_h2 <= 1.4 else '✗ OUT OF RANGE'} |",
            f"| **P1/P2 ratio** | {ratio:.3f} | ~1.3 | {'✓' if 1.1 <= ratio <= 1.6 else '✗ WRONG SCALE'} |",
        ]
        if align_rpt:
            lines += [
                "",
                f"**align_report.json:** scale = {align_rpt.get('scale_m_per_unit','?')} m/unit "
                f"(method: {align_rpt.get('scale_method','?')})",
            ]
    else:
        lines.append("*No aligned model available — skipping alignment quality checks.*")

    lines += [
        "",
        "---",
        "",
        "## (g) Plan-Fit Residual",
        "",
        f"**Similarity fit (reconstruction XZ → sketch XY):**",
        f"- Fit RMS: {plan_fit_rms:.4f} sketch units",
        f"- Plan path length: {path_len_sketch:.4f} sketch units" if not math.isnan(plan_fit_rms) else "",
        f"- **Residual: {residual_pct:.1f}%**  *(pass criterion: < 15%, no mirror)*" if not math.isnan(residual_pct) else "- *Could not compute residual*",
        f"- Mirror detected: **{mirrored}**  *(must be False)*",
        "",
        "---",
        "",
        "## Conclusion",
        "",
        "**Root causes identified:**",
        "",
    ]
    for r in conclusion_reason:
        lines.append(f"- {r}")

    lines += [
        "",
        "### Pass / Fail Summary",
        "",
        f"| Criterion | Value | Threshold | Status |",
        f"|-----------|-------|-----------|--------|",
        f"| Sub-models | {n_models} | 1 | {'✓' if n_models == 1 else '✗'} |",
        f"| Overall registration | {100*total_reg/total_all:.1f}% | ≥95% | {'✓' if total_reg >= 0.95*total_all else '✗'} |",
        f"| Pass 1 registration | {100*len(p1_reg)/max(p1_total,1):.1f}% | ≥95% | {'✓' if len(p1_reg) >= 0.95*p1_total else '✗'} |",
        f"| Pass 2 registration | {100*len(p2_reg)/max(p2_total,1):.1f}% | ≥95% | {'✓' if len(p2_reg) >= 0.95*p2_total else '✗'} |",
        f"| Reprojection error | {reproj_error:.3f} px | < 1.0 px | {'✓' if reproj_error < 1.0 else '✗'} |",
        f"| Start-end gap | {gap_both:.3f} m | < 0.5 m | {'✓' if gap_both < 0.5 else '✗'} |",
        f"| Turn sequence P1 | `{turns_p1}` | `L,R,L,L,R,L,L,R,L` | {'✓' if turns_p1 == 'L,R,L,L,R,L,L,R,L' else '✗'} |",
        f"| Turn sequence P2 | `{turns_p2}` | `L,R,L,L,R,L,L,R,L` | {'✓' if turns_p2 == 'L,R,L,L,R,L,L,R,L' else '✗'} |",
        f"| P1/P2 cross-dist | {median_cross_dist:.3f} m | ≤ 0.4 m | {'✓' if median_cross_dist <= 0.4 else '✗'} |",
        f"| Plan-fit residual | {residual_pct:.1f}% | < 15% | {'✓' if not math.isnan(residual_pct) and residual_pct < 15 else '✗'} |",
        f"| No mirror | {not mirrored} | True | {'✓' if not mirrored else '✗'} |",
        f"| Floor normal angle | {angle_to_Y:.2f}° | ≤ 8° | {'✓' if not math.isnan(angle_to_Y) and angle_to_Y <= 8 else '✗'} |",
        "",
        "### Recommended Next Step",
        "",
        "Based on the above, the recommended experiment order is:",
        "",
        "1. **E1**: Reconstruct Pass 1 alone (frames 1–"
        + str(pass_split) + ") to verify single-pass quality",
        "2. **E3**: Sequential matching (overlap=15, guided, loop-detect OFF) to suppress chair aliasing",
        "3. **E2**: Per-folder cameras if E1 shows intrinsics mismatch",
        "4. **E4**: global_mapper on the best DB",
        "5. **E5/E6**: Re-extract / ALIKED only if blur or SIFT quality is the bottleneck",
    ]

    diag_path = Path("docs") / "sfm_diagnosis.md"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [+] Written: {diag_path}")
    print("\nDiagnosis complete.")


if __name__ == "__main__":
    main()
