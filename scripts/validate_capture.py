#!/usr/bin/env python3
"""Validate indoor smartphone video capture against the Namma Space SOP.

Checks:
- Container metadata via ffprobe: resolution, fps, duration, rotation, codec.
- Frame quality via OpenCV: blur score (variance of Laplacian), mean brightness,
  and brightness drift (flicker/AE hunting).
- Overall verdict: PASS, WARN, or FAIL with actionable SOP recommendations.

Usage:
  python scripts/validate_capture.py --video path/to/video.mp4
  python scripts/validate_capture.py --scene <scene_name>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Refresh Windows PATH from registry so ffprobe is found if installed recently
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
        os.environ["PATH"] = f"{_m};{_u};{os.environ.get('PATH', '')}"
    except Exception:
        pass


def find_ffprobe() -> Optional[str]:
    """Locate the ffprobe executable."""
    path = shutil.which("ffprobe")
    if path:
        return path

    # Check alongside ffmpeg if ffmpeg is in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).parent / (
            "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        )
        if candidate.is_file():
            return str(candidate)

    return None


def probe_video(video_path: Path, ffprobe_bin: Optional[str]) -> Dict[str, Any]:
    """Extract stream and format metadata using ffprobe JSON output."""
    if not ffprobe_bin:
        return {}

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration,codec_name,tags,side_data_list",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as e:
        print(f"[WARN] ffprobe failed: {e}", file=sys.stderr)

    return {}


def parse_metadata(
    video_path: Path, probe_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Parse resolution, fps, duration, rotation, and codec."""
    info: Dict[str, Any] = {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "duration_s": 0.0,
        "rotation": 0,
        "codec": "unknown",
    }

    streams = probe_data.get("streams", [])
    if streams:
        v = streams[0]
        info["width"] = int(v.get("width", 0))
        info["height"] = int(v.get("height", 0))
        info["codec"] = v.get("codec_name", "unknown")

        # Duration
        dur_str = v.get("duration") or probe_data.get("format", {}).get(
            "duration"
        )
        if dur_str:
            try:
                info["duration_s"] = float(dur_str)
            except ValueError:
                pass

        # FPS
        fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate")
        if fps_str and "/" in fps_str:
            num, den = fps_str.split("/")
            try:
                den_f = float(den)
                if den_f > 0:
                    info["fps"] = round(float(num) / den_f, 2)
            except ValueError:
                pass

        # Rotation
        tags = v.get("tags", {})
        if "rotate" in tags:
            try:
                info["rotation"] = int(tags["rotate"])
            except ValueError:
                pass
        for side in v.get("side_data_list", []):
            if "rotation" in side:
                try:
                    info["rotation"] = int(side["rotation"])
                except ValueError:
                    pass

    # Fallback to OpenCV if ffprobe did not get everything
    if info["width"] == 0 or info["duration_s"] == 0.0:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            if info["width"] == 0:
                info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if info["fps"] == 0.0:
                info["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
            if info["duration_s"] == 0.0:
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if info["fps"] > 0:
                    info["duration_s"] = round(frame_count / info["fps"], 2)
            cap.release()

    return info


def sample_frames_quality(
    video_path: Path, num_samples: int = 30
) -> Dict[str, Any]:
    """Sample frames uniformly and measure blur and brightness metrics."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Unable to read frames from video (total_frames={total_frames})"
        )

    sample_indices = np.linspace(
        0, total_frames - 1, min(num_samples, total_frames), dtype=int
    )

    blur_scores: List[float] = []
    brightness_vals: List[float] = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Variance of Laplacian for blur score
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_scores.append(lap_var)

        # Mean pixel intensity for brightness (0 to 255)
        mean_bright = float(np.mean(gray))
        brightness_vals.append(mean_bright)

    cap.release()

    if not blur_scores:
        raise RuntimeError("Failed to decode any sample frames from the video.")

    return {
        "samples_analyzed": len(blur_scores),
        "mean_blur": round(float(np.mean(blur_scores)), 1),
        "min_blur": round(float(np.min(blur_scores)), 1),
        "max_blur": round(float(np.max(blur_scores)), 1),
        "blur_threshold": 80.0,
        "blurry_frame_pct": round(
            float(np.sum(np.array(blur_scores) < 80.0))
            / len(blur_scores)
            * 100.0,
            1,
        ),
        "mean_brightness": round(float(np.mean(brightness_vals)), 1),
        "brightness_drift_std": round(float(np.std(brightness_vals)), 1),
        "min_brightness": round(float(np.min(brightness_vals)), 1),
        "max_brightness": round(float(np.max(brightness_vals)), 1),
    }


def evaluate_checks(
    meta: Dict[str, Any], quality: Dict[str, Any]
) -> Tuple[str, List[str], List[str]]:
    """Evaluate capture against SOP rules and return (verdict, issues, warnings)."""
    issues: List[str] = []
    warnings: List[str] = []

    width, height = meta["width"], meta["height"]
    short_edge = min(width, height) if width > 0 and height > 0 else 0
    long_edge = max(width, height) if width > 0 and height > 0 else 0

    # 1. Resolution Check
    if short_edge < 1080:
        issues.append(
            f"Resolution {width}x{height} is below 1080p minimum. "
            "SOP requires at least 1080p, 4K (3840x2160) strongly recommended."
        )
    elif long_edge < 3840 and short_edge < 2160:
        warnings.append(
            f"Resolution is {width}x{height} (1080p). "
            "Acceptable, but 4K (3840x2160) delivers superior splat reconstruction."
        )

    # 2. FPS Check
    fps = meta["fps"]
    if fps < 24.0:
        issues.append(
            f"Framerate is {fps} fps (< 24 fps). Insufficient frame rate for smooth SfM baseline."
        )
    elif fps < 29.0:
        warnings.append(
            f"Framerate is {fps} fps. Standard 30 fps or 60 fps is recommended."
        )

    # 3. Duration Check
    duration_s = meta["duration_s"]
    dur_min = duration_s / 60.0
    if duration_s > 720:  # > 12 min
        warnings.append(
            f"Duration is {dur_min:.1f} min (> 12 min limit). "
            "Large video generates thousands of frames, which drastically slows down COLMAP."
        )
    elif duration_s < 90:  # < 1.5 min
        warnings.append(
            f"Duration is {dur_min:.1f} min (< 1.5 min). "
            "May be too brief to cover a room + adjoining corridor in two elevation passes with loop closure."
        )

    # 4. Blur Score Check (Laplacian Variance)
    blurry_pct = quality["blurry_frame_pct"]
    mean_blur = quality["mean_blur"]
    if blurry_pct > 35.0 or mean_blur < 50.0:
        issues.append(
            f"High motion blur detected ({blurry_pct}% frames below threshold, mean blur {mean_blur}). "
            "Ensure shutter speed >= 1/120s and maintain ~0.5 m/s slow walking pace."
        )
    elif blurry_pct > 15.0:
        warnings.append(
            f"{blurry_pct}% of sampled frames showed marginal sharpness. Walk more steadily."
        )

    # 5. Brightness & Exposure Drift Check
    mean_bright = quality["mean_brightness"]
    bright_drift = quality["brightness_drift_std"]

    if mean_bright < 35.0:
        issues.append(
            f"Scene is severely underexposed (mean brightness {mean_bright}/255). Turn on more indoor lights."
        )
    elif mean_bright > 225.0:
        issues.append(
            f"Scene is severely overexposed/blown out (mean brightness {mean_bright}/255). Lower exposure/draw blinds."
        )

    if bright_drift > 28.0:
        warnings.append(
            f"Significant brightness drift detected (std deviation {bright_drift}). "
            "Indicates Auto-Exposure (AE) hunted during capture. Lock AE before recording."
        )

    # Final Verdict
    if issues:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return verdict, issues, warnings


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate smartphone video capture against Namma Space SOP criteria."
    )
    parser.add_argument(
        "--video", type=str, help="Direct path to video file (.mp4, .mov)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene name (looks under data/<scene>/ for video)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of frames to sample for quality audit (default: 30)",
    )
    args = parser.parse_args()

    # Resolve target video
    video_path: Optional[Path] = None
    if args.video:
        video_path = Path(args.video)
    elif args.scene:
        scene_dir = Path("data") / args.scene
        candidates = (
            list(scene_dir.glob("input.*"))
            + list(scene_dir.glob("*.mp4"))
            + list(scene_dir.glob("*.mov"))
        )
        if candidates:
            video_path = candidates[0]
        else:
            print(
                f"[ERROR] No video found in {scene_dir}. Place video at {scene_dir / 'input.mp4'}",
                file=sys.stderr,
            )
            return 1
    else:
        parser.print_help()
        print("\n[ERROR] Either --video <path> or --scene <name> must be provided.")
        return 1

    if not video_path.is_file():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        return 1

    print("=" * 68)
    print(f"  NAMMA SPACE -- CAPTURE VALIDATOR")
    print(f"  Target Video: {video_path}")
    print("=" * 68)

    # 1. ffprobe extraction
    ffprobe_bin = find_ffprobe()
    if not ffprobe_bin:
        print("  [INFO] ffprobe not found in PATH; falling back to OpenCV decoder.")

    probe_raw = probe_video(video_path, ffprobe_bin)
    meta = parse_metadata(video_path, probe_raw)

    print("\n[1] Container & Stream Metadata:")
    print(
        f"    - Resolution     : {meta['width']}x{meta['height']} "
        f"{'(Portrait/Rotated)' if meta['rotation'] in (90, 270) else ''}"
    )
    print(f"    - Framerate      : {meta['fps']} fps")
    print(
        f"    - Duration       : {format_duration(meta['duration_s'])} ({meta['duration_s']:.1f} s)"
    )
    print(f"    - Rotation       : {meta['rotation']} degrees")
    print(f"    - Codec          : {meta['codec']}")

    # 2. Quality sampling
    print(f"\n[2] Frame Quality Analysis ({args.samples} uniform samples):")
    try:
        quality = sample_frames_quality(video_path, num_samples=args.samples)
    except Exception as e:
        print(f"[FAIL] Frame decoding failed: {e}", file=sys.stderr)
        return 1

    print(
        f"    - Mean Blur Score: {quality['mean_blur']}  (min: {quality['min_blur']}, max: {quality['max_blur']})"
    )
    print(
        f"    - Blurry Frames  : {quality['blurry_frame_pct']}% of samples (threshold < {quality['blur_threshold']})"
    )
    print(f"    - Mean Brightness: {quality['mean_brightness']} / 255")
    print(
        f"    - Brightness Dev : +/-{quality['brightness_drift_std']} (Flicker / AE hunt metric)"
    )

    # 3. Verdict
    verdict, issues, warnings = evaluate_checks(meta, quality)

    print("\n" + "-" * 68)
    print(f"  VERDICT: [ {verdict} ]")
    print("-" * 68)

    if issues:
        print("\n  FAILURES (Action Required):")
        for iss in issues:
            print(f"   [X] {iss}")

    if warnings:
        print("\n  WARNINGS / ADVISORIES:")
        for warn in warnings:
            print(f"   [!] {warn}")

    if verdict == "PASS":
        print("\n  [+] All checks passed! Video adheres strictly to SOP guidelines.")
        print("      Ready for Stage 1 (pipeline/extract_frames.py).")
        return 0
    elif verdict == "WARN":
        print("\n  [~] Capture is usable, but review warnings above for best results.")
        return 0
    else:
        print("\n  [-] Capture failed critical SOP quality checks.")
        print("      Refer to docs/SOP.md and re-record before running SfM.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
