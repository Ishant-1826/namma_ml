#!/usr/bin/env python3
"""
scripts/make_topview.py  --  Render trustworthy 2D top-down view and plan overlay.

Outputs:
  * outputs/<scene>/topview.png          -- 2D point density + multi-pass trajectories
  * outputs/<scene>/topview_overlay.png  -- with fitted sketch floor plan overlay
  * outputs/<scene>/topview_legs.jpg     -- 10-keyframe contact sheet verifying legs L1..L10

Usage:
  python scripts/make_topview.py --scene home
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = {
                "model": parts[1],
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(p) for p in parts[4:]],
            }
    return cameras


def parse_images_txt(path: Path) -> dict:
    images = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]
        images[img_id] = {
            "qw": qw, "qx": qy, "qy": qy, "qz": qz,  # Note: rotmat will use Shepperd/explicit
            "qvec": np.array([qw, qx, qy, qz]),
            "tvec": np.array([tx, ty, tz]),
            "camera_id": cam_id,
            "name": name,
        }
        i += 2
    return images


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def camera_centre(img: dict) -> np.ndarray:
    qw, qx, qy, qz = img["qvec"]
    R = quat_to_rot(qw, qx, qy, qz)
    return -(R.T @ img["tvec"])


def parse_points3d_txt(path: Path) -> np.ndarray:
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts) if pts else np.zeros((0, 3))


def frame_number(name: str) -> int:
    digits = "".join(filter(str.isdigit, name))
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# Similarity fit (sketch -> reconstruction)
# ---------------------------------------------------------------------------

def fit_similarity_2d(sketch_path: np.ndarray, traj_xz: np.ndarray):
    """
    Fit similarity transform (scale, rotation, translation) mapping sketch -> traj.
    traj_xz: (N, 2) in (X, -Z) where X=East, -Z=North.
    sketch_path: (M, 2) [x, y] where x=East, y=North.
    Enforces det(R) = +1 (proper rotation, no reflection).
    """
    # Resample sketch along polyline to match traj density
    path_diffs = np.diff(sketch_path, axis=0)
    seg_lens = np.linalg.norm(path_diffs, axis=1)
    cum_len = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = cum_len[-1]

    N = len(traj_xz)
    dists = np.linspace(0, total_len, N)
    sketch_resampled = []
    for d in dists:
        idx = np.searchsorted(cum_len, d) - 1
        idx = max(0, min(idx, len(seg_lens) - 1))
        ratio = (d - cum_len[idx]) / seg_lens[idx] if seg_lens[idx] > 1e-6 else 0.0
        pt = sketch_path[idx] + ratio * path_diffs[idx]
        sketch_resampled.append(pt)
    sketch_resampled = np.array(sketch_resampled)

    mu_s = sketch_resampled.mean(0)
    mu_t = traj_xz.mean(0)
    s_c = sketch_resampled - mu_s
    t_c = traj_xz - mu_t

    ss = (s_c**2).sum()
    K = t_c.T @ s_c / ss
    U, S, Vt = np.linalg.svd(K)
    D = np.diag([1.0, np.linalg.det(U @ Vt)])
    R_sim = U @ D @ Vt
    scale_sim = float(np.trace(np.diag(S) @ D))
    t_sim = mu_t - scale_sim * (R_sim @ mu_s)

    fitted_sketch_resampled = (scale_sim * (R_sim @ sketch_resampled.T)).T + t_sim
    fitted_full_sketch = (scale_sim * (R_sim @ sketch_path.T)).T + t_sim

    res = np.linalg.norm(traj_xz - fitted_sketch_resampled, axis=1)
    rms_m = float(np.sqrt(np.mean(res**2)))
    traj_len = float(np.sum(np.linalg.norm(np.diff(traj_xz, axis=0), axis=1)))
    res_pct = 100.0 * rms_m / traj_len if traj_len > 0 else 0.0

    return scale_sim, R_sim, t_sim, fitted_full_sketch, rms_m, res_pct


# ---------------------------------------------------------------------------
# Main rendering
# ---------------------------------------------------------------------------

def render_topview(
    scene: str,
    aligned_dir: Path,
    output_dir: Path,
    floorplan_path: Optional[Path] = None,
    pass_split: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dynamic report loading
    align_rpt_path = Path("data") / scene / "align_report.json"
    align_data = {}
    if align_rpt_path.exists():
        with open(align_rpt_path) as f:
            align_data = json.load(f)

    frames_rpt_path = Path("data") / scene / "frames_report.json"
    if pass_split is None:
        pass_split = 201
        if frames_rpt_path.exists():
            with open(frames_rpt_path) as f:
                fr_data = json.load(f)
            vids = fr_data.get("videos", [])
            if len(vids) >= 2:
                kept_v0 = vids[0].get("kept_after_duplicate") or vids[0].get("kept_frames") or 0
                tot = sum(v.get("kept_after_duplicate", 0) for v in vids)
                fin = fr_data.get("summary", {}).get("final_frame_count", tot)
                if tot > 0:
                    pass_split = int(round(kept_v0 / tot * fin))

    p1_h = align_data.get("pass_1_height_m", 1.58)
    p2_h = align_data.get("pass_2_height_m", 1.08)
    p1_type = "Eye-level" if p1_h >= p2_h else "Chest-level"
    p2_type = "Chest-level" if p1_h >= p2_h else "Eye-level"
    p1_desc = f"Pass 1 ({p1_type} ~{p1_h:.2f}m)"
    p2_desc = f"Pass 2 ({p2_type} ~{p2_h:.2f}m)"

    imgs = parse_images_txt(aligned_dir / "images.txt")
    pts = parse_points3d_txt(aligned_dir / "points3D.txt")
    print(f"Loaded aligned model: {len(imgs)} cameras, {len(pts)} 3D points (pass_split={pass_split})")

    # 1. Coordinate frame definition (strictly right-handed Y-up, viewing along -Y):
    # World frame:
    #   +Y is Up (normal to floor, floor at y = 0)
    #   +X is East (right)
    #   -Z is North (top of screen)
    #   +Z is South (bottom of screen)
    # Looking down along -Y:
    #   Horizontal axis (screen X) = +X
    #   Vertical axis (screen Y) = -Z (so North points UP, South points DOWN)

    # 2. Filter 3D points for background density:
    # Height between 0.3 m and 1.8 m above the floor (captures chairs, desks, walls, doors)
    mask = (pts[:, 1] >= 0.30) & (pts[:, 1] <= 1.80)
    pts_bg = pts[mask]
    print(f"Background points (0.3m <= y <= 1.8m): {len(pts_bg)} / {len(pts)}")

    # Camera trajectories
    p1_ids = sorted([i for i in imgs if frame_number(imgs[i]["name"]) <= pass_split], key=lambda i: frame_number(imgs[i]["name"]))
    p2_ids = sorted([i for i in imgs if frame_number(imgs[i]["name"]) > pass_split], key=lambda i: frame_number(imgs[i]["name"]))

    p1_c = np.array([camera_centre(imgs[i]) for i in p1_ids])
    p2_c = np.array([camera_centre(imgs[i]) for i in p2_ids])

    # Convert to 2D top-down coordinates: (x, -z)
    p1_xz = np.column_stack([p1_c[:, 0], -p1_c[:, 2]]) if len(p1_c) > 0 else np.zeros((0, 2))
    p2_xz = np.column_stack([p2_c[:, 0], -p2_c[:, 2]]) if len(p2_c) > 0 else np.zeros((0, 2))
    bg_xz = np.column_stack([pts_bg[:, 0], -pts_bg[:, 2]]) if len(pts_bg) > 0 else np.zeros((0, 2))

    # Determine bounding box
    all_xz = np.vstack([p1_xz, p2_xz, bg_xz])
    x_min, x_max = np.percentile(all_xz[:, 0], 0.5) - 0.5, np.percentile(all_xz[:, 0], 99.5) + 0.5
    y_min, y_max = np.percentile(all_xz[:, 1], 0.5) - 0.5, np.percentile(all_xz[:, 1], 99.5) + 0.5

    # 3. 2D Histogram Density (5 cm cells = 0.05 m)
    bin_size = 0.05
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)
    H, xedges, yedges = np.histogram2d(bg_xz[:, 0], bg_xz[:, 1], bins=[x_bins, y_bins])
    H = np.log1p(H)  # log scale for density

    # 4. Render two versions: Clean Topview and With Plan Overlay
    for with_overlay in [False, True]:
        fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
        ax.set_facecolor("#111318")
        fig.patch.set_facecolor("#111318")

        # Background point density (5cm cells, log scale, flipped along y for proper image orientation)
        ax.imshow(
            H.T,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            origin="lower",
            cmap="mako" if "mako" in plt.colormaps() else "Blues",
            alpha=0.65,
            interpolation="nearest",
            aspect="equal",
        )

        # 1m grid
        x_grid = np.arange(math.floor(x_min), math.ceil(x_max) + 1.0, 1.0)
        y_grid = np.arange(math.floor(y_min), math.ceil(y_max) + 1.0, 1.0)
        for gx in x_grid:
            ax.axvline(gx, color="#333842", linestyle=":", linewidth=0.6, alpha=0.6)
        for gy in y_grid:
            ax.axhline(gy, color="#333842", linestyle=":", linewidth=0.6, alpha=0.6)

        # Plot Pass 1 trajectory (Chest level ~1.12m)
        if len(p1_xz) > 1:
            points_p1 = p1_xz.reshape(-1, 1, 2)
            segments_p1 = np.concatenate([points_p1[:-1], points_p1[1:]], axis=1)
            norm_p1 = plt.Normalize(0, len(p1_xz))
            lc1 = matplotlib.collections.LineCollection(segments_p1, cmap="cool", norm=norm_p1, linewidth=2.5, alpha=0.9, zorder=4)
            lc1.set_array(np.arange(len(p1_xz)))
            ax.add_collection(lc1)

            # Direction arrows every ~5 seconds (~20 frames)
            for k in range(10, len(p1_xz) - 1, 20):
                dx = p1_xz[k+1, 0] - p1_xz[k-1, 0]
                dy = p1_xz[k+1, 1] - p1_xz[k-1, 1]
                L = math.hypot(dx, dy)
                if L > 0.05:
                    ax.annotate(
                        "",
                        xy=(p1_xz[k, 0] + dx/L*0.15, p1_xz[k, 1] + dy/L*0.15),
                        xytext=(p1_xz[k, 0], p1_xz[k, 1]),
                        arrowprops=dict(arrowstyle="->", color="#00ffcc", lw=1.8),
                        zorder=5,
                    )

        # Plot Pass 2 trajectory (Eye level ~1.54m)
        if len(p2_xz) > 1:
            points_p2 = p2_xz.reshape(-1, 1, 2)
            segments_p2 = np.concatenate([points_p2[:-1], points_p2[1:]], axis=1)
            norm_p2 = plt.Normalize(0, len(p2_xz))
            lc2 = matplotlib.collections.LineCollection(segments_p2, cmap="autumn", norm=norm_p2, linewidth=2.0, alpha=0.85, zorder=3)
            lc2.set_array(np.arange(len(p2_xz)))
            ax.add_collection(lc2)

            for k in range(10, len(p2_xz) - 1, 20):
                dx = p2_xz[k+1, 0] - p2_xz[k-1, 0]
                dy = p2_xz[k+1, 1] - p2_xz[k-1, 1]
                L = math.hypot(dx, dy)
                if L > 0.05:
                    ax.annotate(
                        "",
                        xy=(p2_xz[k, 0] + dx/L*0.15, p2_xz[k, 1] + dy/L*0.15),
                        xytext=(p2_xz[k, 0], p2_xz[k, 1]),
                        arrowprops=dict(arrowstyle="->", color="#ffaa00", lw=1.6),
                        zorder=5,
                    )

        # Start and End markers
        if len(p1_xz) > 0:
            ax.scatter(p1_xz[0, 0], p1_xz[0, 1], c="#00ff66", s=140, marker="o", edgecolors="white", linewidths=1.5, zorder=6, label="Start (Door)")
            ax.scatter(p1_xz[-1, 0], p1_xz[-1, 1], c="#ff0044", s=140, marker="X", edgecolors="white", linewidths=1.5, zorder=6, label="End (Door)")
            gap_p1 = np.linalg.norm(p1_xz[0] - p1_xz[-1])
            ax.plot([p1_xz[0, 0], p1_xz[-1, 0]], [p1_xz[0, 1], p1_xz[-1, 1]], "w--", linewidth=1.0, alpha=0.7, zorder=5)
            ax.text((p1_xz[0, 0] + p1_xz[-1, 0])/2 + 0.15, (p1_xz[0, 1] + p1_xz[-1, 1])/2, f"Gap: {gap_p1:.2f} m", color="white", fontsize=9, fontweight="bold", zorder=7)

        # Plan overlay if requested
        res_info = ""
        if with_overlay and floorplan_path and floorplan_path.exists():
            with open(floorplan_path) as f:
                fp = json.load(f)
            sketch_path = np.array(fp["path"])
            s_sim, R_sim, t_sim, fitted_plan, rms_m, res_pct = fit_similarity_2d(sketch_path, p1_xz)
            ax.plot(fitted_plan[:, 0], fitted_plan[:, 1], color="#ffff00", linestyle="--", linewidth=2.4, alpha=0.9, zorder=6, label="Plan (sketch, approximate)")

            # Fit landmarks as well
            landmarks = fp.get("landmarks", {})
            for name, box in landmarks.items():
                box_arr = np.array([
                    [box[0][0], box[0][1]],
                    [box[1][0], box[0][1]],
                    [box[1][0], box[1][1]],
                    [box[0][0], box[1][1]],
                    [box[0][0], box[0][1]],
                ])
                fitted_box = (s_sim * (R_sim @ box_arr.T)).T + t_sim
                ax.plot(fitted_box[:, 0], fitted_box[:, 1], color="#ffd700", linestyle=":", linewidth=1.2, alpha=0.6, zorder=5)
                # Label box
                center = fitted_box[:4].mean(axis=0)
                ax.text(center[0], center[1], name.replace("_", " "), color="#ffd700", fontsize=8, ha="center", va="center", alpha=0.8, zorder=6)

            res_info = f" | Plan residual: {rms_m:.2f} m ({res_pct:.1f}% path length)"

        # North Arrow (pointing towards +Y in plot, which is -Z in 3D scene)
        arrow_x = x_min + 0.8
        arrow_y = y_max - 1.0
        ax.annotate(
            "N",
            xy=(arrow_x, arrow_y + 0.6),
            xytext=(arrow_x, arrow_y),
            arrowprops=dict(facecolor="white", edgecolor="white", width=2.5, headwidth=8),
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=7
        )

        # Scale bar (2 metres)
        sb_len = 2.0
        sb_x = x_max - 2.8
        sb_y = y_min + 0.6
        ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="white", linewidth=4, zorder=7)
        ax.plot([sb_x, sb_x], [sb_y - 0.1, sb_y + 0.1], color="white", linewidth=2, zorder=7)
        ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - 0.1, sb_y + 0.1], color="white", linewidth=2, zorder=7)
        ax.text(sb_x + sb_len / 2.0, sb_y + 0.2, f"{sb_len:.0f} m", color="white", fontsize=10, ha="center", fontweight="bold", zorder=7)

        # Title & Subtitle
        title = "Namma Space -- Metric Orthographic Top-Down View"
        if with_overlay:
            title += " (with Floor-Plan Overlay)"
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

        subtitle = f"Scene: {scene} | {p1_desc} | {p2_desc}{res_info}"
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, color="#b0b8c4", fontsize=10, ha="center")

        # Custom Legend
        legend_elements = [
            Line2D([0], [0], color="#00ffff", lw=2.5, label=p1_desc),
            Line2D([0], [0], color="#ff6600", lw=2.0, label=p2_desc),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ff66", markersize=9, label="Start (Door)"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="#ff0044", markersize=9, label="End (Door)"),
        ]
        if with_overlay:
            legend_elements.append(Line2D([0], [0], color="#ffff00", lw=2.4, linestyle="--", label="Plan (sketch, approximate)"))

        leg = ax.legend(handles=legend_elements, loc="upper right", facecolor="#1a1d24", edgecolor="#404654", labelcolor="white", fontsize=9)
        leg.set_zorder(8)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X (metres, East →)", color="white", fontsize=10)
        ax.set_ylabel("Z (metres, North ↑)", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#404654")

        out_name = "topview_overlay.png" if with_overlay else "topview.png"
        out_path = output_dir / out_name
        plt.tight_layout()
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=200)
        plt.close(fig)
        print(f"  [+] Saved {out_path}")


# ---------------------------------------------------------------------------
# Keyframe Verification Sheet (topview_legs.jpg)
# ---------------------------------------------------------------------------

def generate_legs_contact_sheet(
    images_dir: Path,
    output_path: Path,
    aligned_dir: Path,
    floorplan_path: Path,
    pass_split: Optional[int] = None,
) -> None:
    """Generate topview_legs.jpg: keyframes labelled L1..LN with descriptions."""
    imgs = parse_images_txt(aligned_dir / "images.txt")

    if pass_split is None:
        pass_split = 201
        frames_rpt_path = images_dir.parent / "frames_report.json"
        if frames_rpt_path.exists():
            with open(frames_rpt_path) as f:
                fr_data = json.load(f)
            vids = fr_data.get("videos", [])
            if len(vids) >= 2:
                kept_v0 = vids[0].get("kept_after_duplicate") or vids[0].get("kept_frames") or 0
                tot = sum(v.get("kept_after_duplicate", 0) for v in vids)
                fin = fr_data.get("summary", {}).get("final_frame_count", tot)
                if tot > 0:
                    pass_split = int(round(kept_v0 / tot * fin))

    p1_ids = sorted([i for i in imgs if frame_number(imgs[i]["name"]) <= pass_split], key=lambda i: frame_number(imgs[i]["name"]))
    traj = np.array([[camera_centre(imgs[i])[0], -camera_centre(imgs[i])[2]] for i in p1_ids])
    names = [imgs[i]["name"] for i in p1_ids]

    dists = np.insert(np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1)), 0, 0.0)
    total_d = dists[-1]

    # Load floorplan to determine legs
    n_legs = 4
    leg_descriptions = []
    sketch_lens = [2.5, 3.5, 2.1, 3.5]
    if floorplan_path.exists():
        with open(floorplan_path) as f:
            fp = json.load(f)
        turns = fp.get("turns", "")
        if turns == "L,L,L":
            n_legs = 4
            sketch_lens = [2.5, 3.5, 2.1, 3.5]
            leg_descriptions = [
                "L1: Start at Door -> Head North along East side past East chair",
                "L2: Turn Left -> Head West along North wall past table & chairs",
                "L3: Turn Left -> Head South along West wall past chairs towards table",
                "L4: Turn Left -> Head East along South wall returning to Door",
            ]
        else:
            n_legs = 10
            sketch_lens = [1.9, 2.4, 0.95, 1.6, 1.2, 3.0, 0.95, 6.5, 0.7, 0.5]
            leg_descriptions = [
                "L1: Door -> North along right wall (seats on left)",
                "L2: Turn Left -> West across cross-aisle",
                "L3: Turn Right -> North into middle seating section",
                "L4: Turn Left -> West past central tables",
                "L5: Turn Left -> South along middle aisle",
                "L6: Turn Right -> West towards far left wall",
                "L7: Turn Left -> South along far left seating row",
                "L8: Turn Left -> East long leg (stage right, seats left)",
                "L9: Turn Right -> South heading back toward door",
                "L10: Turn Left -> East returning to start at door",
            ]
    cum_sketch = np.insert(np.cumsum(sketch_lens), 0, 0.0)
    sketch_tot = cum_sketch[-1]

    # Select representative keyframes for each leg
    keyframe_indices = []
    for k in range(n_legs):
        mid_frac = (cum_sketch[k] + cum_sketch[k+1]) / 2.0 / sketch_tot
        target_d = mid_frac * total_d
        idx = int(np.argmin(np.abs(dists - target_d)))
        keyframe_indices.append(idx)

    keyframe_indices[0] = min(10, len(names) - 1)
    keyframe_indices[-1] = len(names) - 1

    grid_cols = 2
    grid_rows = (n_legs + 1) // 2
    cell_w, cell_h = 640, 360
    header_h = 80
    sheet_w = cell_w * grid_cols
    sheet_h = cell_h * grid_rows + header_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (20, 24, 30)

    # Header
    cv2.putText(sheet, f"Namma Space -- {n_legs}-Leg Trajectory Visual Ground-Truth Verification", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(sheet, f"One representative keyframe per straight leg (L1 to L{n_legs}), confirming physical features in room", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 200), 1, cv2.LINE_AA)

    for k in range(n_legs):
        row = k // grid_cols
        col = k % grid_cols
        x_start = col * cell_w
        y_start = header_h + row * cell_h

        fname = names[keyframe_indices[k]]
        img_path = images_dir / fname
        if img_path.exists():
            thumb = cv2.imread(str(img_path))
            thumb = cv2.resize(thumb, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        else:
            thumb = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

        # Draw banner at bottom of each thumbnail
        cv2.rectangle(thumb, (0, cell_h - 55), (cell_w, cell_h), (10, 10, 15), -1)
        cv2.putText(thumb, f"{leg_descriptions[k]}", (12, cell_h - 30), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(thumb, f"Keyframe: {fname} (cum dist = {dists[keyframe_indices[k]]:.1f}m)", (12, cell_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

        # Draw badge in top-left
        cv2.rectangle(thumb, (0, 0), (90, 36), (0, 120, 255), -1)
        cv2.putText(thumb, f"LEG {k+1}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Border
        cv2.rectangle(thumb, (0, 0), (cell_w - 1, cell_h - 1), (60, 70, 80), 1)

        sheet[y_start:y_start + cell_h, x_start:x_start + cell_w] = thumb

    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [+] Saved {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4: Generate 2D top-down floor-plan and overlay.")
    ap.add_argument("--scene", default="home", help="Scene name")
    args = ap.parse_args()

    scene = args.scene
    aligned_dir = Path("data") / scene / "colmap" / "aligned"
    images_dir = Path("data") / scene / "images"
    output_dir = Path("outputs") / scene
    floorplan_path = Path("docs") / "floorplan.json"

    if not aligned_dir.exists():
        print(f"ERROR: aligned directory {aligned_dir} does not exist.")
        sys.exit(1)

    print("=" * 64)
    print("  NAMMA SPACE -- STAGE 4: TOP VIEW GENERATOR")
    print("=" * 64)
    print(f"  Scene      : {scene}")
    print(f"  Aligned dir: {aligned_dir}")
    print(f"  Output dir : {output_dir}")
    print("-" * 64)

    render_topview(
        scene=scene,
        aligned_dir=aligned_dir,
        output_dir=output_dir,
        floorplan_path=floorplan_path,
        pass_split=None,
    )

    generate_legs_contact_sheet(
        images_dir=images_dir,
        output_path=output_dir / "topview_legs.jpg",
        aligned_dir=aligned_dir,
        floorplan_path=floorplan_path,
        pass_split=None,
    )

    print("\nTop view generation complete.")


if __name__ == "__main__":
    main()
