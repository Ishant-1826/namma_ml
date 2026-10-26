#!/usr/bin/env python3
"""Setup script for scene home2:
- Moves stills to data/home2/raw/stills/
- Copies original videos from data/home/raw/ to data/home2/raw/
- Renames the four new videos in time order to sweep_1.mp4 .. sweep_4.mp4
- Prints resolution, fps, duration, rotation metadata for every video
- Prints size and dimensions for every still
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Registry PATH resolution for Windows
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
        _m = _get_reg_path(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        _u = _get_reg_path(winreg.HKEY_CURRENT_USER, r"Environment")
        os.environ["PATH"] = f"{_m};{_u};{os.environ.get('PATH', '')}"
    except Exception:
        pass

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)

home_raw = REPO_ROOT / "data" / "home" / "raw"
home2_raw = REPO_ROOT / "data" / "home2" / "raw"
stills_dir = home2_raw / "stills"
stills_dir.mkdir(parents=True, exist_ok=True)

# 1. Move stills into data/home2/raw/stills/
still_candidates = set(home2_raw.glob("*.jpg.jpeg")) | set(home2_raw.glob("*.jpeg")) | set(home2_raw.glob("*.jpg"))
for s in sorted(still_candidates):
    if s.is_file() and s.parent == home2_raw and not s.name.startswith("sweep"):
        # Normalize name to .jpg
        clean_name = s.name.replace(".jpg.jpeg", ".jpg").replace(".jpeg", ".jpg")
        target = stills_dir / clean_name
        shutil.move(str(s), str(target))
        print(f"Moved still: {s.name} -> stills/{target.name}")

# 2. Rename the 4 sweep videos in time order
sweep_mapping = [
    ("20261024_222349.mp4", "sweep_1.mp4"),
    ("20261024_222501.mp4", "sweep_2.mp4"),
    ("20261024_222544.mp4", "sweep_3.mp4"),
    ("20261024_222642.mp4", "sweep_4.mp4"),
]

for old_name, new_name in sweep_mapping:
    old_p = home2_raw / old_name
    new_p = home2_raw / new_name
    if old_p.is_file():
        old_p.rename(new_p)
        print(f"Renamed sweep video: {old_name} -> {new_name}")
    elif new_p.is_file():
        print(f"Sweep video already renamed: {new_name}")

# 3. Copy the two original videos from data/home/raw/ to data/home2/raw/
# Eye-level pass first (20261021_195555.mp4), chest-level second (20261021_195930.mp4)
orig_videos = [
    ("20261021_195555.mp4", "20261021_195555.mp4", "Eye-level Pass 1"),
    ("20261021_195930.mp4", "20261021_195930.mp4", "Chest-level Pass 2"),
]

for src_name, dst_name, desc in orig_videos:
    src_p = home_raw / src_name
    dst_p = home2_raw / dst_name
    if not dst_p.is_file():
        print(f"Copying {desc}: {src_name} -> data/home2/raw/{dst_name} ...")
        shutil.copy2(str(src_p), str(dst_p))
        print(f"  Copied {dst_name} ({dst_p.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"Original video already exists in home2: {dst_name}")

# 4. Probe video metadata using ffmpeg and cv2
def get_video_metadata(video_path: Path):
    ffmpeg_bin = shutil.which("ffmpeg") or str(REPO_ROOT / "venv" / "Scripts" / "ffmpeg.exe")
    
    cmd = [ffmpeg_bin, "-i", str(video_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    
    # Parse duration
    import re
    dur_m = re.search(r"Duration:\s*(\d+):(\d+):([\d\.]+)", out)
    if dur_m:
        h, m, s = float(dur_m.group(1)), float(dur_m.group(2)), float(dur_m.group(3))
        duration_sec = h * 3600 + m * 60 + s
    else:
        duration_sec = 0.0
        
    # Parse resolution and fps
    stream_m = re.search(r"Stream.*Video:.*,\s*(\d+)x(\d+).*?,\s*([\d\.]+)\s*(?:fps|tbr)", out)
    if stream_m:
        w = int(stream_m.group(1))
        h = int(stream_m.group(2))
        fps = float(stream_m.group(3))
    else:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if duration_sec == 0:
            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_sec = fc / fps if fps > 0 else 0.0
        cap.release()
        
    # Parse rotation metadata
    rot_m = re.search(r"(?:displaymatrix: rotation of|rotate\s*:)\s*(-?\d+)", out, re.IGNORECASE)
    rotation = int(rot_m.group(1)) if rot_m else 0
    
    return {
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 2),
        "rotation": rotation,
        "size_bytes": video_path.stat().st_size,
    }

print("\n" + "=" * 80)
print("  VIDEO METADATA AUDIT (data/home2/raw/)")
print("=" * 80)
print(f"{'Filename':<25} {'Resolution':<12} {'FPS':<8} {'Duration (s)':<14} {'Rotation':<10} {'Size (MB)':<10}")
print("-" * 80)

videos = sorted([f for f in home2_raw.glob("*.mp4") if f.is_file()])
for v in videos:
    m = get_video_metadata(v)
    res_str = f"{m['width']}x{m['height']}"
    size_mb = m['size_bytes'] / (1024 * 1024)
    print(f"{v.name:<25} {res_str:<12} {m['fps']:<8.2f} {m['duration_sec']:<14.2f} {m['rotation']:<10} {size_mb:<10.1f}")

print("\n" + "=" * 80)
print("  STILLS METADATA AUDIT (data/home2/raw/stills/)")
print("=" * 80)
print(f"{'Filename':<28} {'Resolution':<14} {'Size (bytes)':<14} {'Size (MB)':<10}")
print("-" * 80)

stills = sorted(list(stills_dir.glob("*.jpg")) + list(stills_dir.glob("*.jpeg")) + list(stills_dir.glob("*.png")))
for s in stills:
    im = cv2.imread(str(s))
    h, w = im.shape[:2] if im is not None else (0, 0)
    size_b = s.stat().st_size
    size_mb = size_b / (1024 * 1024)
    print(f"{s.name:<28} {f'{w}x{h}':<14} {size_b:<14} {size_mb:<10.2f}")

print("=" * 80)
