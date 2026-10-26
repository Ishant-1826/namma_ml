#!/usr/bin/env python3
"""
scripts/check_alignment.py  --  Phase 4 Alignment Verification & Diagnostic Plotter.

Generates diagnostic PNGs in outputs/<scene>/:
  1. top.png: True orthographic top-down view (x = X, vertical = Z inverted so Z points down).
     Log-scale point density (0.3m to 1.8m above floor, 5 cm cells), camera paths color-coded
     by time with direction arrows every ~5s, Start (green) and End (red), 1 m grid, scale bar,
     and start-to-end gap.
  2. side_xy.png: Orthographic X vs Y elevation showing floor at y=0, cameras, and estimated ceiling.
  3. side_zy.png: Orthographic Z vs Y elevation showing floor at y=0, cameras, and estimated ceiling.
  4. alignment_check.png: Comprehensive 4-panel diagnostic dashboard.

All plots include headers with detected UP vector, floor-plane angle to UP,
camera heights per pass (mean and std), camera-loop footprint, robust point extents,
ceiling height with source label, and turn sequence.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
import matplotlib.patches as patches
import numpy as np
import yaml

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.align_scene import parse_cameras_txt, parse_images_txt, parse_points3d_txt, camera_centre, frame_number


# ---------------------------------------------------------------------------
# Trajectory Helpers: RDP, Smoothing, and Turn Sequence
# ---------------------------------------------------------------------------

def rdp_simplify(points: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    """Ramer-Douglas-Peucker 2D polyline simplification."""
    if len(points) < 3:
        return points
    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        line_unit = line_vec / line_len
        vecs = points - start
        proj = np.dot(vecs, line_unit)
        proj_clamped = np.clip(proj, 0, line_len)
        dists = np.linalg.norm(vecs - np.outer(proj_clamped, line_unit), axis=1)

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def compute_turn_sequence_and_corners(
    path_xz: np.ndarray,
    smooth_window: int = 5,
    rdp_epsilon: float = 1.0,
    min_leg_length: float = 1.0,
) -> Tuple[List[str], int, np.ndarray]:
    """
    Compute turn sequence ('L'/'R') and corner count:
    1. Smooth the camera path with a 5-frame moving average.
    2. Simplify via Ramer-Douglas-Peucker at 1.0 m.
    3. Merge legs shorter than 1.0 m.
    4. Compute L/R sequence and corner count.

    path_xz: (N, 2) [X, Z]. Converted to [X, -Z] where North is up.
    """
    if len(path_xz) < 3:
        return [], 0, path_xz

    # Convert to top-down coordinates where North (-Z) is UP:
    path_topdown = np.column_stack([path_xz[:, 0], -path_xz[:, 1]])

    # 1. Smooth camera path (moving average, 5 frames)
    if len(path_topdown) >= smooth_window:
        pad_w = smooth_window // 2
        padded = np.pad(path_topdown, ((pad_w, pad_w), (0, 0)), mode="edge")
        kernel = np.ones(smooth_window) / float(smooth_window)
        smoothed = np.column_stack([
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ])
    else:
        smoothed = path_topdown.copy()

    # 2. Run Ramer-Douglas-Peucker at 1.0 m
    simplified = rdp_simplify(smoothed, epsilon=rdp_epsilon)

    # 3. Merge legs shorter than 1.0 m
    pts = list(simplified)
    changed = True
    while changed and len(pts) > 2:
        changed = False
        for i in range(len(pts) - 1):
            leg_len = float(np.linalg.norm(pts[i + 1] - pts[i]))
            if leg_len < min_leg_length:
                if i < len(pts) - 2:
                    pts.pop(i + 1)
                else:
                    pts.pop(i)
                changed = True
                break

    merged_poly = np.array(pts)

    # 4. Compute turn sequence and corner count
    turns = []
    for i in range(1, len(merged_poly) - 1):
        v1 = merged_poly[i] - merged_poly[i - 1]
        v2 = merged_poly[i + 1] - merged_poly[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) > 1e-4:
            turns.append("L" if cross > 0 else "R")

    corner_count = len(turns)
    return turns, corner_count, merged_poly


def compute_turn_sequence(path_xz: np.ndarray, epsilon: float = 1.0) -> List[str]:
    """Backward-compatible turn sequence helper."""
    turns, _, _ = compute_turn_sequence_and_corners(path_xz, smooth_window=5, rdp_epsilon=epsilon, min_leg_length=1.0)
    return turns


def compute_signed_area(path_xz: np.ndarray) -> float:
    """Compute signed area of trajectory in top-down coordinates where North (-Z) is UP."""
    if len(path_xz) < 3:
        return 0.0
    x = path_xz[:, 0]
    y = -path_xz[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def get_source_name(name: str) -> str:
    """Extract source folder name from image path."""
    norm_name = name.replace("\\", "/")
    parts = norm_name.split("/")
    if len(parts) > 1:
        return parts[0]
    nl = name.lower()
    if "video1" in nl:
        return "video1"
    if "video2" in nl:
        return "video2"
    return "source"


SOURCE_PALETTE = {
    "20261021_195555": ("#1f77b4", "o", "Pass 1 Eye (1.45m)"),
    "video1": ("#1f77b4", "o", "Pass 1 Eye (1.45m)"),
    "20261021_195930": ("#e377c2", "s", "Pass 2 Chest"),
    "video2": ("#e377c2", "s", "Pass 2 Chest"),
    "sweep_1": ("#ff7f0e", "^", "Sweep 1"),
    "sweep_2": ("#2ca02c", "v", "Sweep 2"),
    "sweep_3": ("#d62728", "<", "Sweep 3"),
    "sweep_4": ("#9467bd", ">", "Sweep 4"),
    "stills": ("#8c564b", "*", "Stills"),
}


# ---------------------------------------------------------------------------
# Point Filtering and Ceiling Estimation
# ---------------------------------------------------------------------------

def filter_robust_points(points: Dict[int, dict]) -> np.ndarray:
    """Filter points with track length >= 4 and reprojection error <= 1.0 px."""
    filtered = []
    for p in points.values():
        track_str = p.get("track", "")
        track_tokens = track_str.split()
        track_len = len(track_tokens) // 2
        err = float(p.get("error", 0.0))
        if track_len >= 4 and err <= 1.0:
            filtered.append(p["xyz"])
    if len(filtered) < 50:
        return np.array([p["xyz"] for p in points.values()])
    return np.array(filtered)


def determine_ceiling_height(
    pts_xyz: np.ndarray,
    cam_centers: np.ndarray,
    meas_data: dict,
    cfg: dict,
    is_synthetic: bool = False,
) -> Tuple[float, str]:
    """
    Determine ceiling height following competition SOP:
    1. If measurements.json has ceiling_height_m, use it and label it "measured".
    2. Otherwise estimate it ONLY from points inside the camera-path footprint plus 1 m
       and above (max camera height + 0.3 m): use the 95th percentile of their heights
       if there are at least 500 such points, labeled "estimated".
    3. If not, print 'ceiling not reconstructed' and use configs/default.yaml ceiling_default_m.
    4. Never report a ceiling below max camera height + 0.5 m.
    """
    max_cam_h = float(np.max(cam_centers[:, 1])) if len(cam_centers) > 0 else 1.45
    min_ceiling_h = max_cam_h + 0.5

    meas_ceiling = meas_data.get("ceiling_height_m") if not is_synthetic else None
    if meas_ceiling is not None and float(meas_ceiling) > 0:
        c_val = float(meas_ceiling)
        c_val = max(c_val, min_ceiling_h)
        return c_val, "measured"

    # Footprint + 1m
    cam_x_min, cam_x_max = float(np.min(cam_centers[:, 0])), float(np.max(cam_centers[:, 0]))
    cam_z_min, cam_z_max = float(np.min(cam_centers[:, 2])), float(np.max(cam_centers[:, 2]))

    x_min_b = cam_x_min - 1.0
    x_max_b = cam_x_max + 1.0
    z_min_b = cam_z_min - 1.0
    z_max_b = cam_z_max + 1.0
    y_min_b = max_cam_h + 0.3

    mask = (
        (pts_xyz[:, 0] >= x_min_b) & (pts_xyz[:, 0] <= x_max_b) &
        (pts_xyz[:, 2] >= z_min_b) & (pts_xyz[:, 2] <= z_max_b) &
        (pts_xyz[:, 1] >= y_min_b)
    )
    ceil_candidate_pts = pts_xyz[mask]

    if len(ceil_candidate_pts) >= 500:
        c_val = float(np.percentile(ceil_candidate_pts[:, 1], 95))
        c_val = max(c_val, min_ceiling_h)
        return c_val, "estimated"
    else:
        print("ceiling not reconstructed")
        default_ceiling = float(
            cfg.get("align", {}).get("ceiling_default_m")
            or cfg.get("alignment", {}).get("ceiling_default_m")
            or cfg.get("alignment", {}).get("ceiling_height_m")
            or 3.0
        )
        c_val = max(default_ceiling, min_ceiling_h)
        return c_val, "default"


# ---------------------------------------------------------------------------
# Scale Bar Helper
# ---------------------------------------------------------------------------

def add_scale_bar(ax, length_m: float = 1.0, label: str = "1 m") -> None:
    """Draw a physical metric scale bar on the lower-left of the axes."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = abs(xlim[1] - xlim[0])
    dy = abs(ylim[1] - ylim[0])

    # Position in lower-left with margin
    x0 = min(xlim) + 0.08 * dx
    y0 = min(ylim) + 0.08 * dy if ylim[1] > ylim[0] else max(ylim) - 0.08 * dy

    x1 = x0 + length_m
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=4.0, zorder=100)
    ax.plot([x0, x0], [y0 - 0.02 * dy, y0 + 0.02 * dy], color="black", linewidth=2.0, zorder=100)
    ax.plot([x1, x1], [y0 - 0.02 * dy, y0 + 0.02 * dy], color="black", linewidth=2.0, zorder=100)
    ax.text((x0 + x1) / 2.0, y0 + 0.03 * dy if ylim[1] > ylim[0] else y0 - 0.03 * dy,
            label, color="black", fontsize=9, fontweight="bold", ha="center", va="bottom" if ylim[1] > ylim[0] else "top", zorder=101)


# ---------------------------------------------------------------------------
# Plot Top View (top.png generator)
# ---------------------------------------------------------------------------

def plot_top_view(
    cam_centers: np.ndarray,
    pts_top: np.ndarray,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    header_line1: str,
    header_line2: str,
    start_end_gap: float,
    source_cameras: Optional[Dict[str, np.ndarray]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Render top.png view with truly un-mirrored inverted Z-axis (+Z points down)."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # 2D point density histogram (5 cm = 0.05 m cells)
    step_cell = 0.05
    bx = np.arange(x_min, x_max + step_cell, step_cell)
    bz = np.arange(z_min, z_max + step_cell, step_cell)

    H_dense, xedges, zedges = np.histogram2d(pts_top[:, 0], pts_top[:, 2], bins=[bx, bz])
    log_dense = np.log1p(H_dense)

    # Display density with origin="lower" mapping to data coordinates
    ax.imshow(
        log_dense.T,
        extent=[bx[0], bx[-1], bz[0], bz[-1]],
        origin="lower",
        cmap="Blues",
        aspect="equal",
        alpha=0.85,
    )

    # Plot camera centres coloured by source
    if source_cameras and len(source_cameras) > 0:
        for s_name, s_pts in sorted(source_cameras.items()):
            if len(s_pts) == 0:
                continue
            color, marker, label = SOURCE_PALETTE.get(s_name, ("#7f7f7f", "o", s_name))
            is_stills = "still" in s_name.lower()
            sz = 60 if is_stills else 14
            ax.scatter(
                s_pts[:, 0],
                s_pts[:, 2],
                color=color,
                marker=marker,
                s=sz,
                edgecolor="black" if is_stills else "none",
                linewidth=0.8 if is_stills else 0,
                zorder=7 if is_stills else 5,
                label=f"{label} ({len(s_pts)})",
            )
            if not is_stills and len(s_pts) > 1:
                ax.plot(s_pts[:, 0], s_pts[:, 2], color=color, alpha=0.45, linewidth=1.2, zorder=4)
    else:
        t_vals = np.linspace(0.0, 1.0, len(cam_centers))
        ax.scatter(cam_centers[:, 0], cam_centers[:, 2], c=t_vals, cmap="coolwarm", s=12, zorder=5, label="Cameras (Time)")
        ax.plot(cam_centers[:, 0], cam_centers[:, 2], color="darkblue", alpha=0.4, linewidth=1.2, zorder=4)

    # Direction arrows every ~5 seconds (~15 frames at 3 fps)
    arrow_step = max(15, len(cam_centers) // 25)
    for idx in range(0, len(cam_centers) - 1, arrow_step):
        p0 = cam_centers[idx]
        p1 = cam_centers[idx + 1]
        dp = p1 - p0
        norm_dp = np.linalg.norm([dp[0], dp[2]])
        if norm_dp > 1e-4:
            ax.annotate(
                "",
                xy=(p1[0], p1[2]),
                xytext=(p0[0], p0[2]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8, mutation_scale=12),
                zorder=6,
            )

    start_pt = cam_centers[0]
    end_pt = cam_centers[-1]
    ax.scatter([start_pt[0]], [start_pt[2]], color="lime", edgecolor="black", s=140, marker="o", zorder=10, label=f"Start ({start_pt[0]:.2f}, {start_pt[2]:.2f})")
    ax.scatter([end_pt[0]], [end_pt[2]], color="red", edgecolor="black", s=140, marker="s", zorder=10, label=f"End ({end_pt[0]:.2f}, {end_pt[2]:.2f})")
    ax.plot([start_pt[0], end_pt[0]], [start_pt[2], end_pt[2]], "k--", lw=1.5, zorder=9, label=f"Gap: {start_end_gap:.2f} m")

    # Set axis limits explicitly: bottom=z_max, top=z_min so +Z points DOWN the image
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=z_max, top=z_min)

    # Assert ylim[0] > ylim[1] (Z truly points down the image)
    ylim = ax.get_ylim()
    assert ylim[0] > ylim[1], f"Expected inverted Z-axis with ylim[0] > ylim[1], got {ylim}"

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    add_scale_bar(ax, length_m=1.0, label="1 m")

    ax.set_xlabel("X (metres)  [← West (-X), East (+X) →]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Z (metres)  [↑ North (-Z), South (+Z) ↓]", fontsize=11, fontweight="bold")
    ax.set_title(f"TOP-DOWN VIEW (top.png)\n{header_line1}\n{header_line2}", fontsize=9, pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    return fig, ax


def plot_coverage_map(
    points: dict,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    header_line1: str,
    source_cameras: Optional[Dict[str, np.ndarray]] = None,
    cell_size: float = 0.5,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate coverage.png: number of distinct cameras observing points in each 0.5m cell."""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    bx = np.arange(x_min, x_max + cell_size, cell_size)
    bz = np.arange(z_min, z_max + cell_size, cell_size)
    nx = len(bx) - 1
    nz = len(bz) - 1

    cell_obs = [[set() for _ in range(nz)] for _ in range(nx)]

    for p in points.values():
        xyz = p["xyz"]
        col = int((xyz[0] - x_min) / cell_size)
        row = int((xyz[2] - z_min) / cell_size)
        if 0 <= col < nx and 0 <= row < nz:
            track_tokens = p.get("track", "").split()
            img_ids = [int(track_tokens[2 * k]) for k in range(len(track_tokens) // 2)]
            cell_obs[col][row].update(img_ids)

    cov_grid = np.zeros((nx, nz), dtype=int)
    for i in range(nx):
        for j in range(nz):
            cov_grid[i, j] = len(cell_obs[i][j])

    masked_cov = np.ma.masked_where(cov_grid == 0, cov_grid)

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="#1a1c23")

    im = ax.imshow(
        masked_cov.T,
        extent=[bx[0], bx[-1], bz[0], bz[-1]],
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distinct Observing Cameras (0.5m cell)", fontsize=10, fontweight="bold")

    # Overlay camera positions with subtle markers
    if source_cameras:
        for s_name, s_pts in source_cameras.items():
            if len(s_pts) > 0:
                color, _, _ = SOURCE_PALETTE.get(s_name, ("white", "o", s_name))
                ax.scatter(s_pts[:, 0], s_pts[:, 2], color=color, edgecolor="black", s=10, alpha=0.7, zorder=6)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=z_max, top=z_min)

    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(True, linestyle=":", color="gray", alpha=0.6)
    add_scale_bar(ax, length_m=1.0, label="1 m")

    ax.set_xlabel("X (metres)  [← West (-X), East (+X) →]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Z (metres)  [↑ North (-Z), South (+Z) ↓]", fontsize=11, fontweight="bold")
    ax.set_title(f"CAMERA COVERAGE MAP (coverage.png)\n{header_line1}\nDistinct cameras observing points per 0.5m cell", fontsize=9, pad=12)

    return fig, ax


# ---------------------------------------------------------------------------
# Main Check Alignment Function
# ---------------------------------------------------------------------------

def check_alignment(
    scene: str,
    config_path: str = "configs/default.yaml",
    data_dir: Optional[str] = None,
    outputs_dir: Optional[str] = None,
    aligned_dir: Optional[str] = None,
) -> Dict[str, Any]:
    repo_root = Path.cwd()
    data_dir_path = Path(data_dir) if data_dir else (repo_root / "data" / scene)
    outputs_dir_path = Path(outputs_dir) if outputs_dir else (repo_root / "outputs" / scene)
    outputs_dir_path.mkdir(parents=True, exist_ok=True)

    if aligned_dir:
        aligned_model_dir = Path(aligned_dir)
    elif (data_dir_path / "colmap_aligned").is_dir():
        aligned_model_dir = data_dir_path / "colmap_aligned"
    elif (data_dir_path / "colmap" / "aligned").is_dir():
        aligned_model_dir = data_dir_path / "colmap" / "aligned"
    elif (data_dir_path / "cameras.txt").is_file():
        aligned_model_dir = data_dir_path
    else:
        raise RuntimeError(f"Aligned model directory not found at {data_dir_path / 'colmap_aligned'}. Run pipeline/align_scene.py first.")
    aligned_dir = aligned_model_dir

    # Load alignment.json
    align_json_path = data_dir_path / "alignment.json"
    align_meta = {}
    if align_json_path.is_file():
        try:
            with open(align_json_path, "r", encoding="utf-8") as f:
                align_meta = json.load(f)
        except Exception:
            pass

    # Load config YAML
    cfg = {}
    cfg_file = Path(config_path)
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            pass

    # Load measurements.json
    meas_path = data_dir_path / "measurements.json"
    meas_data = {}
    is_synthetic = scene.startswith("synth") or not meas_path.is_file()
    if meas_path.is_file():
        try:
            with open(meas_path, "r", encoding="utf-8") as mf:
                meas_data = json.load(mf)
        except Exception:
            pass

    # Load cameras, images, points
    txt_cameras = aligned_dir / "cameras.txt"
    txt_images = aligned_dir / "images.txt"
    txt_points = aligned_dir / "points3D.txt"

    cameras = parse_cameras_txt(txt_cameras)
    images = parse_images_txt(txt_images)
    points = parse_points3d_txt(txt_points)

    sorted_imgs = sorted(images.values(), key=lambda im: frame_number(im["name"]))
    cam_centers = np.array([camera_centre(im) for im in sorted_imgs])

    # Filter robust points: track length >= 4 and reprojection error <= 1.0 px
    robust_pts = filter_robust_points(points)

    # 1st-99th percentile bounds
    x_p1, x_p99 = float(np.percentile(robust_pts[:, 0], 1)), float(np.percentile(robust_pts[:, 0], 99))
    y_p1, y_p99 = float(np.percentile(robust_pts[:, 1], 1)), float(np.percentile(robust_pts[:, 1], 99))
    z_p1, z_p99 = float(np.percentile(robust_pts[:, 2], 1)), float(np.percentile(robust_pts[:, 2], 99))

    robust_x_span = x_p99 - x_p1
    robust_z_span = z_p99 - z_p1

    # Camera-loop footprint
    cam_x_min, cam_x_max = float(np.min(cam_centers[:, 0])), float(np.max(cam_centers[:, 0]))
    cam_z_min, cam_z_max = float(np.min(cam_centers[:, 2])), float(np.max(cam_centers[:, 2]))
    cam_footprint_x = cam_x_max - cam_x_min
    cam_footprint_z = cam_z_max - cam_z_min

    # Passes split
    pass_split = 222
    frames_rep_path = data_dir_path / "frames_report.json"
    if frames_rep_path.is_file():
        try:
            with open(frames_rep_path, "r", encoding="utf-8") as fr_f:
                vids = json.load(fr_f).get("videos", [])
                if vids:
                    pass_split = vids[0].get("kept_after_duplicate", vids[0].get("raw_extracted", 222))
        except Exception:
            pass

    p1_indices = []
    p2_indices = []
    for idx, (im, c) in enumerate(zip(sorted_imgs, cam_centers)):
        name = im["name"].lower()
        f_num = frame_number(name)
        if "video2" in name or ("video1" not in name and f_num > pass_split):
            p2_indices.append(idx)
        else:
            p1_indices.append(idx)

    p1_cams = cam_centers[p1_indices] if p1_indices else cam_centers
    p2_cams = cam_centers[p2_indices] if p2_indices else None

    p1_h_mean = float(np.mean(p1_cams[:, 1]))
    p1_h_std = float(np.std(p1_cams[:, 1]))
    p2_h_mean = float(np.mean(p2_cams[:, 1])) if p2_cams is not None and len(p2_cams) > 0 else None
    p2_h_std = float(np.std(p2_cams[:, 1])) if p2_cams is not None and len(p2_cams) > 0 else None

    # Ceiling determination
    ceiling_est, ceiling_label = determine_ceiling_height(
        pts_xyz=robust_pts,
        cam_centers=cam_centers,
        meas_data=meas_data,
        cfg=cfg,
        is_synthetic=is_synthetic,
    )

    # Gap between start and end
    start_pt = cam_centers[0]
    end_pt = cam_centers[-1]
    start_end_gap = float(np.linalg.norm(end_pt - start_pt))

    # Turn sequence: smooth 5 frames, RDP 1.0m, merge legs < 1m
    cam_xz = cam_centers[:, [0, 2]]
    turn_sequence, corner_count, simplified_path = compute_turn_sequence_and_corners(
        path_xz=cam_xz,
        smooth_window=5,
        rdp_epsilon=1.0,
        min_leg_length=1.0,
    )
    signed_area = compute_signed_area(cam_xz)

    # Print turn sequence and corner count to console
    print(f"\nTurn Sequence: {turn_sequence} | Corner Count: {corner_count}")
    print(f"Camera-Loop Footprint (X x Z): {cam_footprint_x:.2f} m x {cam_footprint_z:.2f} m")
    print(f"Robust Point Extents  (X x Z): {robust_x_span:.2f} m x {robust_z_span:.2f} m")
    print(f"Ceiling Height               : {ceiling_est:.2f} m ({ceiling_label})")

    # Measurements comparison
    if not is_synthetic and meas_data:
        meas_list = meas_data.get("measurements", [])
        ref_dim = None
        for m in meas_list:
            m_name = m.get("name", "")
            if "square" in m_name or "side" in m_name or "footprint" in m_name:
                ref_dim = float(m.get("metres", 0.0))
                break
        if ref_dim and ref_dim > 0:
            diff_cam_x = ((cam_footprint_x - ref_dim) / ref_dim) * 100.0
            diff_cam_z = ((cam_footprint_z - ref_dim) / ref_dim) * 100.0
            diff_ext_x = ((robust_x_span - ref_dim) / ref_dim) * 100.0
            diff_ext_z = ((robust_z_span - ref_dim) / ref_dim) * 100.0
            print(f"Comparison vs measurements.json reference ({ref_dim:.2f} m):")
            print(f"  Camera Footprint X: {cam_footprint_x:.2f} m ({diff_cam_x:+.1f}%) | Z: {cam_footprint_z:.2f} m ({diff_cam_z:+.1f}%)")
            print(f"  Robust Extents   X: {robust_x_span:.2f} m ({diff_ext_x:+.1f}%) | Z: {robust_z_span:.2f} m ({diff_ext_z:+.1f}%)")

        meas_c_val = meas_data.get("ceiling_height_m")
        if meas_c_val:
            meas_c = float(meas_c_val)
            c_diff_pct = ((ceiling_est - meas_c) / meas_c) * 100.0
            print(f"  Ceiling Height    : {ceiling_est:.2f} m vs Measured {meas_c:.2f} m ({c_diff_pct:+.1f}%)")

    # Header metadata string
    up_vec = align_meta.get("up_vector_detected", [0.0, 1.0, 0.0])
    up_str = f"[{up_vec[0]:+.2f}, {up_vec[1]:+.2f}, {up_vec[2]:+.2f}]"
    floor_angle = align_meta.get("floor_plane", {}).get("angle_to_up_deg", 0.0)

    header_line1 = (
        f"Detected UP: {up_str} | Floor Angle: {floor_angle:.1f}° | "
        f"Footprint: {cam_footprint_x:.2f}m x {cam_footprint_z:.2f}m | "
        f"Extents: {robust_x_span:.2f}m x {robust_z_span:.2f}m (Ceiling: {ceiling_est:.2f}m [{ceiling_label}])"
    )
    if p2_h_mean is not None:
        header_line2 = (
            f"Pass 1 (Eye): {p1_h_mean:.2f}m ± {p1_h_std:.2f}m | Pass 2 (Chest): {p2_h_mean:.2f}m ± {p2_h_std:.2f}m | "
            f"Gap: {start_end_gap:.2f}m | Turns: {turn_sequence} ({corner_count} corners)"
        )
    else:
        header_line2 = (
            f"Camera Height: {p1_h_mean:.2f}m ± {p1_h_std:.2f}m | "
            f"Gap: {start_end_gap:.2f}m | Turns: {turn_sequence} ({corner_count} corners)"
        )

    # -----------------------------------------------------------------------
    # 1. TOP.PNG
    # -----------------------------------------------------------------------
    wall_mask = (robust_pts[:, 1] >= 0.30) & (robust_pts[:, 1] <= 1.80)
    pts_top = robust_pts[wall_mask] if np.sum(wall_mask) > 50 else robust_pts

    x_min_top = x_p1 - 0.5
    x_max_top = x_p99 + 0.5
    z_min_top = z_p1 - 0.5
    z_max_top = z_p99 + 0.5

    # Separate cameras by source
    source_cameras: Dict[str, List[np.ndarray]] = {}
    for im, c in zip(sorted_imgs, cam_centers):
        s_name = get_source_name(im["name"])
        if s_name not in source_cameras:
            source_cameras[s_name] = []
        source_cameras[s_name].append(c)
    source_cams_np = {k: np.array(v) for k, v in source_cameras.items()}

    fig_top, ax_top = plot_top_view(
        cam_centers=cam_centers,
        pts_top=pts_top,
        x_min=x_min_top,
        x_max=x_max_top,
        z_min=z_min_top,
        z_max=z_max_top,
        header_line1=header_line1,
        header_line2=header_line2,
        start_end_gap=start_end_gap,
        source_cameras=source_cams_np,
    )
    top_png_path = outputs_dir_path / "top.png"
    fig_top.tight_layout()
    fig_top.savefig(str(top_png_path), dpi=150)
    plt.close(fig_top)

    # -----------------------------------------------------------------------
    # 1b. COVERAGE.PNG (0.5m grid of distinct observing cameras)
    # -----------------------------------------------------------------------
    fig_cov, ax_cov = plot_coverage_map(
        points=points,
        x_min=x_min_top,
        x_max=x_max_top,
        z_min=z_min_top,
        z_max=z_max_top,
        header_line1=header_line1,
        source_cameras=source_cams_np,
        cell_size=0.5,
    )
    cov_png_path = outputs_dir_path / "coverage.png"
    fig_cov.tight_layout()
    fig_cov.savefig(str(cov_png_path), dpi=150)
    plt.close(fig_cov)

    # -----------------------------------------------------------------------
    # 2. SIDE_XY.PNG
    # -----------------------------------------------------------------------
    fig_xy, ax_xy = plt.subplots(figsize=(10, 6), dpi=150)

    # Subsample robust points for scatter
    sub_step = max(1, len(robust_pts) // 8000)
    ax_xy.scatter(robust_pts[::sub_step, 0], robust_pts[::sub_step, 1], c="silver", s=2, alpha=0.4, label="Robust Points")

    # Floor line at y = 0
    margin_x = 0.05 * robust_x_span
    x_lim_low = x_p1 - margin_x
    x_lim_high = x_p99 + margin_x
    ax_xy.plot([x_lim_low, x_lim_high], [0.0, 0.0], color="green", linewidth=2.5, zorder=8, label="Floor (y = 0.0 m)")

    # Ceiling line
    ax_xy.plot([x_lim_low, x_lim_high], [ceiling_est, ceiling_est], color="purple", linestyle="--", linewidth=2.0, zorder=8, label=f"Ceiling ({ceiling_est:.2f} m [{ceiling_label}])")

    # Cameras
    if len(source_cams_np) > 2:
        for s_name, s_pts in sorted(source_cams_np.items()):
            color, marker, label = SOURCE_PALETTE.get(s_name, ("dodgerblue", "o", s_name))
            ax_xy.scatter(s_pts[:, 0], s_pts[:, 1], color=color, marker=marker, s=14, zorder=9, label=f"{label} ({len(s_pts)})")
    elif p2_cams is not None and len(p2_cams) > 0:
        ax_xy.scatter(p1_cams[:, 0], p1_cams[:, 1], color="dodgerblue", s=14, zorder=9, label=f"Pass 1 Eye ({p1_h_mean:.2f}m)")
        ax_xy.scatter(p2_cams[:, 0], p2_cams[:, 1], color="deeppink", s=14, zorder=9, label=f"Pass 2 Chest ({p2_h_mean:.2f}m)")
    else:
        ax_xy.scatter(cam_centers[:, 0], cam_centers[:, 1], color="dodgerblue", s=14, zorder=9, label=f"Cameras ({p1_h_mean:.2f}m)")

    y_floor_base = min(0.0, y_p1)
    y_ceil_top = max(ceiling_est, y_p99)
    y_span = y_ceil_top - y_floor_base
    margin_y = 0.05 * y_span

    ax_xy.set_xlim(x_lim_low, x_lim_high)
    ax_xy.set_ylim(y_floor_base - margin_y, y_ceil_top + margin_y)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.xaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
    ax_xy.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax_xy.grid(True, linestyle=":", color="gray", alpha=0.6)
    add_scale_bar(ax_xy, length_m=1.0, label="1 m")

    ax_xy.set_xlabel("X (metres)", fontsize=11, fontweight="bold")
    ax_xy.set_ylabel("Y (metres)  ↑ Up", fontsize=11, fontweight="bold")
    ax_xy.set_title(f"SIDE ELEVATION X-Y (side_xy.png)\n{header_line1}\n{header_line2}", fontsize=9, pad=12)
    ax_xy.legend(loc="upper right", framealpha=0.9, fontsize=8)

    side_xy_path = outputs_dir_path / "side_xy.png"
    fig_xy.tight_layout()
    fig_xy.savefig(str(side_xy_path), dpi=150)
    plt.close(fig_xy)

    # -----------------------------------------------------------------------
    # 3. SIDE_ZY.PNG
    # -----------------------------------------------------------------------
    fig_zy, ax_zy = plt.subplots(figsize=(10, 6), dpi=150)

    ax_zy.scatter(robust_pts[::sub_step, 2], robust_pts[::sub_step, 1], c="silver", s=2, alpha=0.4, label="Robust Points")

    margin_z = 0.05 * robust_z_span
    z_lim_low = z_p1 - margin_z
    z_lim_high = z_p99 + margin_z
    ax_zy.plot([z_lim_low, z_lim_high], [0.0, 0.0], color="green", linewidth=2.5, zorder=8, label="Floor (y = 0.0 m)")
    ax_zy.plot([z_lim_low, z_lim_high], [ceiling_est, ceiling_est], color="purple", linestyle="--", linewidth=2.0, zorder=8, label=f"Ceiling ({ceiling_est:.2f} m [{ceiling_label}])")

    if len(source_cams_np) > 2:
        for s_name, s_pts in sorted(source_cams_np.items()):
            color, marker, label = SOURCE_PALETTE.get(s_name, ("dodgerblue", "o", s_name))
            ax_zy.scatter(s_pts[:, 2], s_pts[:, 1], color=color, marker=marker, s=14, zorder=9, label=f"{label} ({len(s_pts)})")
    elif p2_cams is not None and len(p2_cams) > 0:
        ax_zy.scatter(p1_cams[:, 2], p1_cams[:, 1], color="dodgerblue", s=14, zorder=9, label=f"Pass 1 Eye ({p1_h_mean:.2f}m)")
        ax_zy.scatter(p2_cams[:, 2], p2_cams[:, 1], color="deeppink", s=14, zorder=9, label=f"Pass 2 Chest ({p2_h_mean:.2f}m)")
    else:
        ax_zy.scatter(cam_centers[:, 2], cam_centers[:, 1], color="dodgerblue", s=14, zorder=9, label=f"Cameras ({p1_h_mean:.2f}m)")

    ax_zy.set_xlim(z_lim_low, z_lim_high)
    ax_zy.set_ylim(y_floor_base - margin_y, y_ceil_top + margin_y)
    ax_zy.set_aspect("equal", adjustable="box")
    ax_zy.xaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
    ax_zy.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax_zy.grid(True, linestyle=":", color="gray", alpha=0.6)
    add_scale_bar(ax_zy, length_m=1.0, label="1 m")

    ax_zy.set_xlabel("Z (metres)", fontsize=11, fontweight="bold")
    ax_zy.set_ylabel("Y (metres)  ↑ Up", fontsize=11, fontweight="bold")
    ax_zy.set_title(f"SIDE ELEVATION Z-Y (side_zy.png)\n{header_line1}\n{header_line2}", fontsize=9, pad=12)
    ax_zy.legend(loc="upper right", framealpha=0.9, fontsize=8)

    side_zy_path = outputs_dir_path / "side_zy.png"
    fig_zy.tight_layout()
    fig_zy.savefig(str(side_zy_path), dpi=150)
    plt.close(fig_zy)

    # -----------------------------------------------------------------------
    # 4. ALIGNMENT_CHECK.PNG (4-panel summary dashboard)
    # -----------------------------------------------------------------------
    fig_dash, axs = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig_dash.suptitle(f"ALIGNMENT AUDIT DASHBOARD -- {scene}\n{header_line1}", fontsize=11, fontweight="bold")

    # Panel (0, 0): Top view plan
    ax0 = axs[0, 0]
    bx_d = np.arange(x_min_top, x_max_top + 0.05, 0.05)
    bz_d = np.arange(z_min_top, z_max_top + 0.05, 0.05)
    H_d, _, _ = np.histogram2d(pts_top[:, 0], pts_top[:, 2], bins=[bx_d, bz_d])
    log_d = np.log1p(H_d)
    ax0.imshow(log_d.T, extent=[bx_d[0], bx_d[-1], bz_d[0], bz_d[-1]], origin="lower", cmap="Blues", aspect="equal", alpha=0.85)
    t_vals = np.linspace(0.0, 1.0, len(cam_centers))
    ax0.scatter(cam_centers[:, 0], cam_centers[:, 2], c=t_vals, cmap="coolwarm", s=10, zorder=5)
    ax0.scatter([start_pt[0]], [start_pt[2]], color="lime", s=80, marker="o", label="Start", zorder=6)
    ax0.scatter([end_pt[0]], [end_pt[2]], color="red", s=80, marker="s", label="End", zorder=6)

    # Invert Z-axis
    ax0.set_xlim(x_min_top, x_max_top)
    ax0.set_ylim(bottom=z_max_top, top=z_min_top)
    assert ax0.get_ylim()[0] > ax0.get_ylim()[1], f"Dashboard ax0 y-axis must be inverted, got {ax0.get_ylim()}"

    ax0.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.set_xlabel("X (m)", fontsize=9)
    ax0.set_ylabel("Z (m) [↓ South, ↑ North]", fontsize=9)
    ax0.set_title("Plan (X-Z) [Inverted Z]", fontsize=10, fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8)

    # Panel (0, 1): Side X-Y
    ax1 = axs[0, 1]
    ax1.scatter(robust_pts[::sub_step, 0], robust_pts[::sub_step, 1], c="silver", s=2, alpha=0.4)
    ax1.plot([x_lim_low, x_lim_high], [0.0, 0.0], "g-", lw=2, label="Floor y=0")
    ax1.plot([x_lim_low, x_lim_high], [ceiling_est, ceiling_est], "m--", lw=1.5, label=f"Ceiling ({ceiling_est:.2f}m)")
    ax1.scatter(cam_centers[:, 0], cam_centers[:, 1], c=t_vals, cmap="coolwarm", s=10)
    ax1.set_xlim(x_lim_low, x_lim_high)
    ax1.set_ylim(y_floor_base - margin_y, y_ceil_top + margin_y)
    ax1.set_aspect("equal", adjustable="box")
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlabel("X (m)", fontsize=9)
    ax1.set_ylabel("Y (m) ↑ Up", fontsize=9)
    ax1.set_title("Elevation (X-Y)", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel (1, 0): Side Z-Y
    ax2 = axs[1, 0]
    ax2.scatter(robust_pts[::sub_step, 2], robust_pts[::sub_step, 1], c="silver", s=2, alpha=0.4)
    ax2.plot([z_lim_low, z_lim_high], [0.0, 0.0], "g-", lw=2, label="Floor y=0")
    ax2.plot([z_lim_low, z_lim_high], [ceiling_est, ceiling_est], "m--", lw=1.5, label=f"Ceiling ({ceiling_est:.2f}m)")
    ax2.scatter(cam_centers[:, 2], cam_centers[:, 1], c=t_vals, cmap="coolwarm", s=10)
    ax2.set_xlim(z_lim_low, z_lim_high)
    ax2.set_ylim(y_floor_base - margin_y, y_ceil_top + margin_y)
    ax2.set_aspect("equal", adjustable="box")
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_xlabel("Z (m)", fontsize=9)
    ax2.set_ylabel("Y (m) ↑ Up", fontsize=9)
    ax2.set_title("Elevation (Z-Y)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)

    # Panel (1, 1): Summary statistics
    ax3 = axs[1, 1]
    ax3.axis("off")
    stats_text = [
        "METRIC ALIGNMENT VERIFICATION REPORT",
        "=" * 50,
        f"Scene Name           : {scene}",
        f"Input Model Dir      : {aligned_dir}",
        f"Detected UP Vector   : {up_str}",
        f"Floor Angle to UP    : {floor_angle:.2f}°",
        f"Total Registered Imgs: {len(cam_centers)}",
        f"Robust 3D Points     : {len(robust_pts)} (track>=4, err<=1.0)",
        "-" * 50,
        "CAMERA HEIGHTS ABOVE FLOOR (y = 0.0):",
        f"  Pass 1 (Eye Level) : {p1_h_mean:.3f} m  (std: ±{p1_h_std:.3f} m)",
    ]
    if p2_h_mean is not None:
        stats_text.extend([
            f"  Pass 2 (Chest Level): {p2_h_mean:.3f} m  (std: ±{p2_h_std:.3f} m)",
            f"  Eye / Chest Ratio  : {p1_h_mean/p2_h_mean:.2f}x  (Expected: 1.1 - 1.5x)",
        ])
    stats_text.extend([
        "-" * 50,
        "TRAJECTORY & ROOM GEOMETRY:",
        f"  Start-to-End Gap   : {start_end_gap:.3f} m",
        f"  RDP Turn Sequence  : {turn_sequence} ({corner_count} corners)",
        f"  Signed Area        : {signed_area:+.4f} (Positive = CCW)",
        f"  Camera Footprint   : {cam_footprint_x:.2f} m x {cam_footprint_z:.2f} m",
        f"  Robust Extents     : {robust_x_span:.2f} m x {robust_z_span:.2f} m",
        f"  Ceiling Height     : {ceiling_est:.2f} m ({ceiling_label})",
    ])
    if not is_synthetic and meas_data:
        stats_text.append("-" * 50)
        stats_text.append("GROUND TRUTH COMPARISON:")
        ref_dim = None
        for m in meas_data.get("measurements", []):
            if "square" in m.get("name", "") or "side" in m.get("name", ""):
                ref_dim = float(m.get("metres", 0.0))
                break
        if ref_dim:
            stats_text.append(f"  Ref Dimension      : {ref_dim:.2f} m")
            stats_text.append(f"  Footprint vs Ref   : X: {((cam_footprint_x - ref_dim)/ref_dim)*100:+.1f}%, Z: {((cam_footprint_z - ref_dim)/ref_dim)*100:+.1f}%")
            stats_text.append(f"  Extents vs Ref     : X: {((robust_x_span - ref_dim)/ref_dim)*100:+.1f}%, Z: {((robust_z_span - ref_dim)/ref_dim)*100:+.1f}%")
        if meas_data.get("ceiling_height_m"):
            c_val = float(meas_data["ceiling_height_m"])
            stats_text.append(f"  Ceiling vs Ground  : {ceiling_est:.2f}m vs {c_val:.2f}m ({((ceiling_est - c_val)/c_val)*100:+.1f}%)")

    stats_text.append("=" * 50)
    ax3.text(0.05, 0.95, "\n".join(stats_text), fontfamily="monospace", fontsize=8.5, va="top", transform=ax3.transAxes)

    dash_path = outputs_dir_path / "alignment_check.png"
    fig_dash.tight_layout()
    fig_dash.savefig(str(dash_path), dpi=150)
    plt.close(fig_dash)

    print(f"\n[+] Generated 4 verification PNGs in {outputs_dir_path}:")
    print(f"    - {top_png_path.name}")
    print(f"    - {side_xy_path.name}")
    print(f"    - {side_zy_path.name}")
    print(f"    - {dash_path.name}")

    report = {
        "scene": scene,
        "top_png": str(top_png_path),
        "side_xy_png": str(side_xy_path),
        "side_zy_png": str(side_zy_path),
        "dashboard_png": str(dash_path),
        "start_end_gap_m": round(start_end_gap, 3),
        "turn_sequence": turn_sequence,
        "corner_count": corner_count,
        "signed_area": round(signed_area, 4),
        "pass_1_height_m": round(p1_h_mean, 3),
        "pass_2_height_m": round(p2_h_mean, 3) if p2_h_mean is not None else None,
        "camera_footprint_m": {"width_x": round(cam_footprint_x, 2), "length_z": round(cam_footprint_z, 2)},
        "robust_extents_m": {"width_x": round(robust_x_span, 2), "length_z": round(robust_z_span, 2)},
        "ceiling_height_m": round(ceiling_est, 2),
        "ceiling_source": ceiling_label,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and verify metric alignment.")
    parser.add_argument("--scene", required=True, help="Scene name (under data/ and outputs/)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", default=None, help="Data directory override (default: data/<scene>)")
    parser.add_argument("--outputs-dir", default=None, help="Outputs directory override (default: outputs/<scene>)")
    args = parser.parse_args()

    try:
        check_alignment(
            scene=args.scene,
            config_path=args.config,
            data_dir=args.data_dir,
            outputs_dir=args.outputs_dir,
        )
    except Exception as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
