#!/usr/bin/env python3
"""Extract sharp, high-quality frames from video captures for COLMAP SfM & Gaussian Splatting.

Stage 1 of the Namma Space pipeline.

Input:
  Videos (and optional stills in raw/stills/) in data/<scene>/raw/ (or data/<scene>/).

Output:
  - data/<scene>/images/ (frame_000001.jpg ...)
  - data/<scene>/frames_report.json
  - data/<scene>/contact_sheet.jpg (8x6 thumbnails grid)

Features:
- ffmpeg extraction at configurable fps (default 3), high-quality JPEG.
- Proper handling of smartphone rotation metadata (ffmpeg auto-rotation).
- Downscaled Laplacian variance blur filter.
- Keeps the sharpest frame in each sliding window of N=3 and drops frames below
  a per-video percentile threshold (bottom 15%).
- Near-duplicate filter: drops frames with tiny displacement vs previous kept frame (camera paused).
- Resizes longest edge to max_dimension (default 1920) preserving aspect ratio.
- Merges supplementary stills at the end.
- Caps at max_frames (default 450) via uniform subsampling; warns if < 120 remain.
- Generates 8x6 thumbnail contact sheet for visual coverage inspection.
- Writes comprehensive frames_report.json.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Ensure repo root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Refresh Windows PATH from registry so ffmpeg/ffprobe is discovered
if sys.platform == "win32":
    try:
        import winreg

        def _get_reg_path(key, subkey):
            try:
                with winreg.OpenKey(key, subkey) as k:
                    val, _ = winreg.QueryValueEx(k, "Path")
                    return val
            except Exception:
                return ""

        _m = _get_reg_path(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        )
        _u = _get_reg_path(winreg.HKEY_CURRENT_USER, r"Environment")
        os.environ["PATH"] = f"{sys.prefix}/Scripts;{_m};{_u};{os.environ.get('PATH', '')}"
    except Exception:
        pass

from pipeline.utils.config import load_config


def compute_blur_score(
    image: np.ndarray, downscale_max_dim: int = 640
) -> float:
    """Compute blur score using variance of Laplacian on a downscaled copy.

    Args:
        image: BGR or grayscale image array.
        downscale_max_dim: Maximum dimension for the downscaled copy.

    Returns:
        Variance of Laplacian (higher means sharper).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    largest = max(h, w)
    if largest > downscale_max_dim:
        scale = downscale_max_dim / float(largest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_blur_sliding_window(
    blur_scores: List[float],
    window_size: int = 3,
    percentile_threshold: float = 15.0,
) -> Tuple[List[int], float, int]:
    """Select the sharpest frame in each window and filter bottom percentile.

    Args:
        blur_scores: List of blur scores for each candidate frame in order.
        window_size: Number of consecutive frames per window (default: 3).
        percentile_threshold: Per-video percentile threshold (default: 15.0%).

    Returns:
        (kept_indices, threshold_value, dropped_count)
    """
    if not blur_scores:
        return [], 0.0, 0

    scores_arr = np.array(blur_scores, dtype=np.float64)
    threshold = float(np.percentile(scores_arr, percentile_threshold))

    kept_indices: List[int] = []
    dropped_count = 0
    total_frames = len(blur_scores)

    # Group candidate frames into non-overlapping windows of window_size
    for start_idx in range(0, total_frames, window_size):
        end_idx = min(start_idx + window_size, total_frames)
        window = list(range(start_idx, end_idx))

        # Select the sharpest frame in this window
        best_idx = max(window, key=lambda idx: blur_scores[idx])

        # Drop if even the sharpest frame falls below the per-video bottom percentile
        if blur_scores[best_idx] >= threshold:
            kept_indices.append(best_idx)
        else:
            dropped_count += 1

    return kept_indices, threshold, dropped_count


def is_near_duplicate(
    img1: np.ndarray,
    img2: np.ndarray,
    downscale_dim: Tuple[int, int] = (160, 90),
    ssim_threshold: float = 0.985,
) -> bool:
    """Check if img2 is a near-duplicate of img1 (camera was paused/stationary).

    Args:
        img1: Previous kept image (BGR).
        img2: Current candidate image (BGR).
        downscale_dim: Dimension (width, height) for comparison.
        ssim_threshold: SSIM threshold above which frames are near-duplicates.

    Returns:
        True if frames are near-duplicates, False otherwise.
    """
    # Downscale and convert to grayscale for fast comparison
    g1 = cv2.cvtColor(
        cv2.resize(img1, downscale_dim, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    g2 = cv2.cvtColor(
        cv2.resize(img2, downscale_dim, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )

    similarity = ssim(g1, g2)
    return bool(similarity >= ssim_threshold)


def resize_longest_edge(
    image: np.ndarray, max_dim: Optional[int] = 1920
) -> np.ndarray:
    """Resize image preserving aspect ratio if longest edge > max_dim."""
    if not max_dim:
        return image
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image
    scale = max_dim / float(largest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_video_with_ffmpeg(
    video_path: Path, temp_dir: Path, fps: float
) -> List[Path]:
    """Extract frames from a video at target fps using ffmpeg.

    ffmpeg automatically honors container rotation metadata (e.g. smartphone displaymatrix)
    so output frames are always oriented correctly.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg executable not found in PATH.")

    video_temp = temp_dir / video_path.stem
    video_temp.mkdir(parents=True, exist_ok=True)

    out_pattern = str(video_temp / "raw_%06d.jpg")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        out_pattern,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract frames from {video_path.name}:\n{res.stderr}"
        )

    extracted_files = sorted(video_temp.glob("raw_*.jpg"))
    return extracted_files


def build_contact_sheet(
    image_paths: List[Path],
    output_path: Path,
    cols: int = 8,
    rows: int = 6,
    cell_w: int = 240,
    cell_h: int = 135,
) -> None:
    """Generate an 8x6 thumbnail contact sheet for eyeballing visual coverage."""
    if not image_paths:
        return

    num_slots = cols * rows  # 48 thumbnails
    total_imgs = len(image_paths)

    # Pick evenly spaced indices
    selected_indices = np.round(
        np.linspace(0, total_imgs - 1, min(num_slots, total_imgs))
    ).astype(int)

    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    contact_sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for slot_idx, img_idx in enumerate(selected_indices):
        r = slot_idx // cols
        c = slot_idx % cols
        if r >= rows:
            break

        img = cv2.imread(str(image_paths[img_idx]))
        if img is None:
            continue

        thumb = cv2.resize(img, (cell_w - 4, cell_h - 4), interpolation=cv2.INTER_AREA)

        y0 = r * cell_h + 2
        y1 = y0 + thumb.shape[0]
        x0 = c * cell_w + 2
        x1 = x0 + thumb.shape[1]

        contact_sheet[y0:y1, x0:x1] = thumb

        # Draw frame index label
        label = f"#{img_idx + 1}"
        cv2.putText(
            contact_sheet,
            label,
            (x0 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            contact_sheet,
            label,
            (x0 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def discover_inputs(scene_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Discover input video files and optional stills under scene directory."""
    raw_dir = scene_dir / "raw"
    search_dir = raw_dir if raw_dir.is_dir() else scene_dir

    video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    videos = [
        f for f in search_dir.iterdir()
        if f.is_file() and f.suffix.lower() in video_exts
    ]
    videos = sorted(videos, key=lambda p: p.name)

    # Stills
    stills_dirs = [raw_dir / "stills", scene_dir / "stills"]
    stills: List[Path] = []
    for s_dir in stills_dirs:
        if s_dir.is_dir():
            still_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            stills.extend([
                f for f in s_dir.iterdir()
                if f.is_file() and f.suffix.lower() in still_exts
            ])

    stills = sorted(stills, key=lambda p: p.name)
    return videos, stills


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract sharp frames from smartphone captures for Namma Space pipeline."
    )
    parser.add_argument("--scene", required=True, help="Scene name (subfolder of data/)")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to pipeline configuration YAML (default: configs/default.yaml)",
    )
    parser.add_argument("--fps", type=float, default=None, help="Extraction FPS (default: 3)")
    parser.add_argument("--quality", type=int, default=None, help="Output JPEG quality (default: 95)")
    parser.add_argument(
        "--max-dimension", type=int, default=None, help="Resize longest edge (default: 1920)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum final frame cap (default: 450)"
    )
    parser.add_argument(
        "--min-frames-warn", type=int, default=None, help="Minimum frame warning threshold (default: 120)"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[FAIL] Configuration load error: {e}", file=sys.stderr)
        return 1

    ext_cfg = config.get("extract_frames", {})
    fps = float(args.fps if args.fps is not None else ext_cfg.get("fps", 3.0))
    quality = int(args.quality if args.quality is not None else ext_cfg.get("quality", 95))
    max_dim = int(args.max_dimension if args.max_dimension is not None else ext_cfg.get("max_dimension", 1920))
    max_frames = int(args.max_frames if args.max_frames is not None else ext_cfg.get("max_frames", 450))
    min_frames_warn = int(
        args.min_frames_warn if args.min_frames_warn is not None else ext_cfg.get("min_frames_warn", 120)
    )
    window_size = int(ext_cfg.get("window_size", 3))
    blur_percentile = float(ext_cfg.get("blur_percentile", 15.0))
    duplicate_threshold = float(ext_cfg.get("duplicate_threshold", 0.985))
    contact_cols = int(ext_cfg.get("contact_sheet_cols", 8))
    contact_rows = int(ext_cfg.get("contact_sheet_rows", 6))

    scene_dir = Path("data") / args.scene
    out_images_dir = scene_dir / "images"
    report_path = scene_dir / "frames_report.json"
    contact_path = scene_dir / "contact_sheet.jpg"
    temp_dir = scene_dir / ".extract_tmp"

    if not scene_dir.is_dir():
        print(
            f"[FAIL] Scene directory not found: '{scene_dir}'.\n"
            f"Action: Create '{scene_dir}' and place video file(s) in '{scene_dir / 'raw'}'.",
            file=sys.stderr,
        )
        return 1

    video_files, still_files = discover_inputs(scene_dir)
    if not video_files and not still_files:
        print(
            f"[FAIL] No video files (.mp4, .mov) or stills found in '{scene_dir}'.\n"
            f"Action: Place videos in '{scene_dir / 'raw'}' or '{scene_dir}'. See docs/SOP.md.",
            file=sys.stderr,
        )
        return 1

    print("=" * 70)
    print("  NAMMA SPACE -- STAGE 1: FRAME EXTRACTION & FILTERING")
    print("=" * 70)
    print(f"  Scene             : {args.scene}")
    print(f"  Input directory   : {scene_dir}")
    print(f"  Output directory  : {out_images_dir}")
    print(f"  Target extraction : {fps} fps")
    print(f"  JPEG Quality      : {quality}")
    print(f"  Max dimension     : {max_dim} px")
    print(f"  Blur filter       : window N={window_size}, per-video bottom {blur_percentile}%")
    print(f"  Duplicate filter  : SSIM >= {duplicate_threshold}")
    print(f"  Videos found ({len(video_files)}):")
    for v in video_files:
        print(f"    - {v.name} ({v.stat().st_size / (1024*1024):.1f} MB)")
    if still_files:
        print(f"  Stills found ({len(still_files)}): in {still_files[0].parent}")
    print("-" * 70)

    # Clean / prepare output and temp directories
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing images in output images dir
    for old_img in out_images_dir.glob("frame_*.jpg"):
        try:
            old_img.unlink()
        except Exception:
            pass

    total_raw_extracted = 0
    total_dropped_blur = 0
    total_dropped_duplicate = 0
    candidate_image_paths: List[Path] = []
    video_reports: List[Dict[str, Any]] = []

    last_kept_img: Optional[np.ndarray] = None

    for v_idx, video_path in enumerate(video_files, start=1):
        print(f"\n[Video {v_idx}/{len(video_files)}] Processing: {video_path.name}")

        # 1. ffmpeg extraction
        raw_frames = extract_video_with_ffmpeg(video_path, temp_dir, fps=fps)
        total_raw_extracted += len(raw_frames)
        print(f"  -> Extracted {len(raw_frames)} raw candidate frames at {fps} fps.")

        if not raw_frames:
            continue

        # 2. Compute blur scores on downscaled copies
        blur_scores: List[float] = []
        for rf in tqdm(raw_frames, desc="  -> Computing sharpness", unit="frame"):
            img = cv2.imread(str(rf))
            if img is None:
                blur_scores.append(0.0)
            else:
                blur_scores.append(compute_blur_score(img))

        # 3. Sliding window selection (N=3) + per-video bottom 15% threshold
        kept_indices, p_thresh, dropped_blur = filter_blur_sliding_window(
            blur_scores,
            window_size=window_size,
            percentile_threshold=blur_percentile,
        )
        total_dropped_blur += (len(raw_frames) - len(kept_indices))
        print(
            f"  -> Window N={window_size} + 15th percentile filter: {len(kept_indices)} kept, "
            f"{len(raw_frames) - len(kept_indices)} dropped (threshold: {p_thresh:.1f})."
        )

        # 4. Near-duplicate filter (stationary camera detection)
        v_kept_paths: List[Path] = []
        v_dropped_dup = 0

        for k_idx in kept_indices:
            candidate_path = raw_frames[k_idx]
            curr_img = cv2.imread(str(candidate_path))
            if curr_img is None:
                continue

            if last_kept_img is not None and is_near_duplicate(
                last_kept_img, curr_img, ssim_threshold=duplicate_threshold
            ):
                v_dropped_dup += 1
                continue

            last_kept_img = curr_img
            v_kept_paths.append(candidate_path)

        total_dropped_duplicate += v_dropped_dup
        candidate_image_paths.extend(v_kept_paths)
        print(f"  -> Near-duplicate filter: dropped {v_dropped_dup} stationary frames.")

        video_reports.append({
            "filename": video_path.name,
            "raw_extracted": len(raw_frames),
            "kept_after_blur": len(kept_indices),
            "kept_after_duplicate": len(v_kept_paths),
            "blur_threshold_p15": round(p_thresh, 1),
            "mean_blur_score": round(float(np.mean(blur_scores)), 1) if blur_scores else 0.0,
        })

    # 5. Merge supplementary stills
    stills_merged_count = 0
    if still_files:
        print(f"\n[Stills] Merging {len(still_files)} supplementary stills...")
        for sf in still_files:
            candidate_image_paths.append(sf)
            stills_merged_count += 1

    # 6. Cap at max_frames by uniform subsampling if needed
    total_before_cap = len(candidate_image_paths)
    if total_before_cap > max_frames:
        print(
            f"\n[Capping] Frame count ({total_before_cap}) exceeds max_frames ({max_frames}). "
            f"Applying uniform subsampling..."
        )
        subsample_indices = np.round(
            np.linspace(0, total_before_cap - 1, max_frames)
        ).astype(int)
        final_selected_paths = [candidate_image_paths[i] for i in subsample_indices]
    else:
        final_selected_paths = candidate_image_paths

    # 7. Write final resized images sequentially: frame_000001.jpg, ...
    print(f"\nWriting {len(final_selected_paths)} final images to {out_images_dir}...")
    final_saved_paths: List[Path] = []
    final_blur_scores: List[float] = []
    final_res = [0, 0]

    for f_idx, src_path in enumerate(
        tqdm(final_selected_paths, desc="  -> Writing images", unit="img"), start=1
    ):
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        # Resize longest edge to max_dimension
        processed = resize_longest_edge(img, max_dim=max_dim)
        h, w = processed.shape[:2]
        final_res = [w, h]

        out_name = f"frame_{f_idx:06d}.jpg"
        out_path = out_images_dir / out_name
        cv2.imwrite(
            str(out_path),
            processed,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )

        final_saved_paths.append(out_path)
        final_blur_scores.append(compute_blur_score(processed))

    # 8. Generate contact sheet (8x6 grid)
    print(f"Generating contact sheet ({contact_cols}x{contact_rows}) at {contact_path}...")
    build_contact_sheet(
        final_saved_paths,
        contact_path,
        cols=contact_cols,
        rows=contact_rows,
    )

    # 9. Clean up temporary extraction folder
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    # 10. Generate frames_report.json
    blur_stats = {
        "mean": round(float(np.mean(final_blur_scores)), 1) if final_blur_scores else 0.0,
        "std": round(float(np.std(final_blur_scores)), 1) if final_blur_scores else 0.0,
        "min": round(float(np.min(final_blur_scores)), 1) if final_blur_scores else 0.0,
        "max": round(float(np.max(final_blur_scores)), 1) if final_blur_scores else 0.0,
    }

    report = {
        "scene": args.scene,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "raw_extracted": total_raw_extracted,
            "kept_after_filtering": len(final_saved_paths),
            "dropped_blur": total_dropped_blur,
            "dropped_duplicate": total_dropped_duplicate,
            "stills_merged": stills_merged_count,
            "final_frame_count": len(final_saved_paths),
            "final_resolution": final_res,
        },
        "blur_statistics": blur_stats,
        "configuration": {
            "fps": fps,
            "quality": quality,
            "max_dimension": max_dim,
            "window_size": window_size,
            "blur_percentile": blur_percentile,
            "duplicate_threshold": duplicate_threshold,
            "max_frames": max_frames,
        },
        "videos": video_reports,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 11. Final Summary & Warnings
    print("\n" + "=" * 70)
    print("  EXTRACTION REPORT SUMMARY")
    print("=" * 70)
    print(f"  Raw frames extracted  : {total_raw_extracted}")
    print(f"  Dropped (blur filter) : {total_dropped_blur}")
    print(f"  Dropped (duplicates)  : {total_dropped_duplicate}")
    if stills_merged_count > 0:
        print(f"  Stills merged         : {stills_merged_count}")
    print(f"  Final frames kept     : {len(final_saved_paths)}")
    print(f"  Final resolution      : {final_res[0]}x{final_res[1]}")
    print(f"  Mean blur score       : {blur_stats['mean']}")
    print(f"  Images saved to       : {out_images_dir}")
    print(f"  Contact sheet         : {contact_path}")
    print(f"  Report saved to       : {report_path}")

    if len(final_saved_paths) < min_frames_warn:
        print(
            f"\n  [WARN] Only {len(final_saved_paths)} frames remain (< {min_frames_warn} recommended minimum).\n"
            f"         Reconstruction quality may suffer. Ensure video duration is at least 2 minutes."
        )

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
