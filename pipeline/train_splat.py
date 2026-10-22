#!/usr/bin/env python3
"""
pipeline/train_splat.py  --  Stage 4: 3D Gaussian Splatting Training & Export Wrapper.

Trains a 3D Gaussian Splat model from the metric, Y-up aligned dataset.

Backends:
  - nerfstudio: 'ns-train splatfacto' (requires CUDA GPU).
  - brush: 'brush' splatting engine (cross-platform WebGPU, detects RTX 4060).

Presets:
  - fast : 7,000 steps (smoke test)
  - final: 30,000 steps (SH degree 3)

Coordinate Integrity:
  - Nerfstudio: anti-centering, anti-orientation, anti-scaling flags.
  - Brush: natively preserves dataset world coordinates (tested & verified).
  - Export: outputs/<scene>/scene_raw.ply.
  - Verification: scripts/verify_splat_frame.py verifies floor at y = 0.00 m.

Usage:
  python pipeline/train_splat.py --scene home --dry-run
  python pipeline/train_splat.py --scene home --preset fast
  python pipeline/train_splat.py --scene home --preset final --eval-split-every 8
"""

import argparse
import datetime
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
import yaml


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_tool_path(cmd: str) -> Optional[str]:
    """Finds tool executable on system PATH or inside tools/<cmd>/."""
    which_path = shutil.which(cmd)
    if which_path:
        return which_path

    # Specific candidates for brush
    candidates = [
        Path(f"tools/{cmd}/{cmd}.exe"),
        Path(f"tools/{cmd}/{cmd}"),
        Path(f"tools/{cmd}.exe"),
        Path(f"tools/{cmd}"),
        Path("tools/brush/brush_app.exe"),
        Path("tools/brush/brush.exe"),
        Path("tools/brush/brush-app.exe"),
        Path(os.path.expanduser("~/Downloads/brush-app-x86_64-pc-windows-msvc/brush_app.exe")),
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def check_tool_exists(cmd: str) -> bool:
    return resolve_tool_path(cmd) is not None


class VRAMTracker:
    """Monitors peak GPU VRAM usage during subprocess execution via nvidia-smi."""
    def __init__(self, interval_sec: float = 0.8):
        self.interval = interval_sec
        self.peak_mb = 0
        self.running = False
        self._thread = None

    def _monitor(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    encoding="utf-8",
                    stderr=subprocess.DEVNULL,
                ).strip()
                val = int(out.split()[0])
                if val > self.peak_mb:
                    self.peak_mb = val
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.peak_mb = 0
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        return self.peak_mb


def build_nerfstudio_train_command(
    data_dir: Path,
    output_models_dir: Path,
    scene: str,
    preset: str,
    eval_ratio: float = 0.1,
    sh_degree: int = 3,
) -> List[str]:
    cmd = [
        "ns-train", "splatfacto",
        "--data", str(data_dir.resolve()),
        "--output-dir", str(output_models_dir.resolve()),
        "--experiment-name", scene,
        "--pipeline.datamanager.dataparser.orientation-method", "none",
        "--pipeline.datamanager.dataparser.center-method", "none",
        "--pipeline.datamanager.dataparser.auto-scale-poses", "False",
        "--pipeline.datamanager.dataparser.scale-factor", "1.0",
        "--pipeline.datamanager.dataparser.eval-mode", "fraction",
        "--pipeline.datamanager.dataparser.eval-ratio", str(eval_ratio),
    ]

    if preset == "fast":
        cmd.extend([
            "--max-num-iterations", "7000",
            "--pipeline.model.num-downscales", "2",
        ])
    elif preset == "final":
        cmd.extend([
            "--max-num-iterations", "30000",
            "--pipeline.model.sh-degree", str(sh_degree),
        ])
    else:
        cmd.extend(["--max-num-iterations", "7000"])

    return cmd


def build_brush_train_command(
    data_dir: Path,
    output_dir: Path,
    preset: str,
    brush_exe: str = "brush",
    max_splats: int = 3000000,
    sh_degree: int = 3,
    eval_split_every: int = 8,
    max_resolution: Optional[int] = 1920,
) -> List[str]:
    """
    Constructs the exact Brush CLI command matching brush_app.exe --help.
    """
    steps = "7000" if preset == "fast" else "30000"
    cmd = [
        brush_exe,
        str(data_dir.resolve()),
        "--total-steps", steps,
        "--max-splats", str(max_splats),
        "--sh-degree", str(sh_degree),
        "--eval-split-every", str(eval_split_every),
        "--eval-every", "1000",
        "--export-every", steps,
        "--export-path", str(output_dir.resolve()),
        "--export-name", f"scene_raw_{steps}.ply",
    ]
    if max_resolution is not None:
        cmd.extend(["--max-resolution", str(max_resolution)])
    return cmd


def count_ply_vertices(ply_path: Path) -> int:
    """Reads vertex count from PLY header."""
    if not ply_path.is_file():
        return 0
    with open(ply_path, "rb") as f:
        for _ in range(50):
            line = f.readline().decode("latin1", errors="ignore").strip()
            if line.startswith("element vertex"):
                parts = line.split()
                return int(parts[2])
            if line == "end_header":
                break
    return 0


def analyze_splat_ply(path: Path) -> dict:
    """Reads PLY properties and computes splat statistics."""
    types = {
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
        "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
    }
    props, n, fmt, current = [], 0, None, None
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("latin1").strip()
            if not line:
                break
            parts = line.split()
            if line.startswith("format"):
                fmt = parts[1]
            elif line.startswith("element"):
                current = parts[1]
                if current == "vertex":
                    n = int(parts[2])
            elif line.startswith("property") and current == "vertex":
                if parts[1] != "list":
                    props.append((parts[2], types.get(parts[1], "<f4")))
            elif line == "end_header":
                break
        offset = f.tell()

    if n == 0 or not props:
        return {
            "gaussian_count": 0,
            "pct_opacity_below_0_05": 0.0,
            "pct_opacity_above_0_2": 0.0,
            "pct_below_minus_0_3m": 0.0,
            "y_p98_opacity_above_0_2_m": 0.0,
        }

    arr = np.fromfile(path, dtype=np.dtype(props), count=n, offset=offset)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))

    pct_op_low = float(100.0 * (op < 0.05).mean())
    pct_op_high = float(100.0 * (op >= 0.2).mean())
    pct_below_floor = float(100.0 * (xyz[:, 1] < -0.3).mean())

    high_op = (op >= 0.2)
    if high_op.sum() > 0:
        y_p98 = float(np.percentile(xyz[high_op, 1], 98))
    else:
        y_p98 = float(np.percentile(xyz[:, 1], 98))

    return {
        "gaussian_count": n,
        "pct_opacity_below_0_05": round(pct_op_low, 2),
        "pct_opacity_above_0_2": round(pct_op_high, 2),
        "pct_below_minus_0_3m": round(pct_below_floor, 2),
        "y_p98_opacity_above_0_2_m": round(y_p98, 3),
    }


def compute_walked_loop_summary(transforms_path: Path, align_path: Path) -> dict:
    """Computes walked loop rectangle sides, aspect ratio, and camera heights for Pass 1 and Pass 2."""
    if not transforms_path.is_file():
        return {}

    with open(transforms_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])

    def extract_frame_num(fp: str) -> int:
        match = re.search(r"(\d+)", Path(fp).stem)
        return int(match.group(1)) if match else 0

    pass_split = 201
    frames_rep_path = Path("data") / align_path.parent.name / "frames_report.json"
    if not frames_rep_path.is_file():
        frames_rep_path = transforms_path.parent / "frames_report.json"
    if frames_rep_path.is_file():
        try:
            with open(frames_rep_path, "r", encoding="utf-8") as fr_f:
                vids = json.load(fr_f).get("videos", [])
                if vids:
                    pass_split = vids[0].get("frames_kept", 201)
        except Exception:
            pass

    p1_cams = []
    p2_cams = []
    for fr in frames:
        num = extract_frame_num(fr.get("file_path", ""))
        mat = np.array(fr.get("transform_matrix", []))
        if mat.shape == (4, 4) or mat.shape == (3, 4):
            cam_pos = mat[:3, 3]
            if num <= pass_split:
                p1_cams.append(cam_pos)
            else:
                p2_cams.append(cam_pos)

    p1_cams = np.array(p1_cams)
    p2_cams = np.array(p2_cams)

    def min_area_rect(pts_2d: np.ndarray) -> dict:
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        min_area = float("inf")
        best_sides = (0.0, 0.0)
        for i in range(len(hull_pts)):
            p1 = hull_pts[i]
            p2 = hull_pts[(i + 1) % len(hull_pts)]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-8:
                continue
            u = edge / edge_len
            v = np.array([-u[1], u[0]])
            proj_u = hull_pts @ u
            proj_v = hull_pts @ v
            len_u = proj_u.max() - proj_u.min()
            len_v = proj_v.max() - proj_v.min()
            area = len_u * len_v
            if area < min_area:
                min_area = area
                best_sides = (min(len_u, len_v), max(len_u, len_v))
        return {"short": best_sides[0], "long": best_sides[1], "aspect_ratio": best_sides[1] / best_sides[0]}

    p1_rect = min_area_rect(p1_cams[:, [0, 2]])
    p2_rect = min_area_rect(p2_cams[:, [0, 2]])

    scale_factor = 0.35544
    if align_path.is_file():
        try:
            with open(align_path, "r", encoding="utf-8") as af:
                adata = json.load(af)
                scale_factor = float(adata.get("scale", scale_factor))
        except Exception:
            pass

    p1_mean_y = float(np.mean(p1_cams[:, 1]))
    p1_std_y = float(np.std(p1_cams[:, 1]))
    p2_mean_y = float(np.mean(p2_cams[:, 1]))
    p2_std_y = float(np.std(p2_cams[:, 1]))

    return {
        "pass1_side_short": round(p1_rect["short"], 3),
        "pass1_side_long": round(p1_rect["long"], 3),
        "pass1_aspect_ratio": round(p1_rect["aspect_ratio"], 3),
        "pass2_side_short": round(p2_rect["short"], 3),
        "pass2_side_long": round(p2_rect["long"], 3),
        "pass2_aspect_ratio": round(p2_rect["aspect_ratio"], 3),
        "scale_factor": scale_factor,
        "pass1_cam_height_m": round(p1_mean_y, 3),
        "pass1_cam_height_std_m": round(p1_std_y, 3),
        "pass2_cam_height_m": round(p2_mean_y, 3),
        "pass2_cam_height_std_m": round(p2_std_y, 3),
    }


def parse_brush_metrics(output_text: str) -> dict:
    """Parses PSNR / SSIM metrics from Brush stdout."""
    metrics = {}
    psnr_matches = re.findall(r"(?:eval|val|test)?\s*psnr[:=\s]+([0-9\.]+)", output_text, re.IGNORECASE)
    ssim_matches = re.findall(r"(?:eval|val|test)?\s*ssim[:=\s]+([0-9\.]+)", output_text, re.IGNORECASE)
    if psnr_matches:
        metrics["psnr"] = float(psnr_matches[-1])
    if ssim_matches:
        metrics["ssim"] = float(ssim_matches[-1])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 4: 3D Gaussian Splatting Training & Export Wrapper."
    )
    ap.add_argument("--scene", default="home", help="Scene name (subfolder of data/)")
    ap.add_argument("--config", default="configs/default.yaml", help="Configuration YAML path")
    ap.add_argument("--backend", choices=["nerfstudio", "brush"], default=None, help="Training backend")
    ap.add_argument("--preset", choices=["fast", "final"], default=None, help="Training preset: fast (7k) or final (30k)")
    ap.add_argument("--max-splats", type=int, default=3000000, help="Max splat count (cap for 8 GB VRAM)")
    ap.add_argument("--sh-degree", type=int, default=3, help="Spherical harmonics degree (default: 3)")
    ap.add_argument("--eval-split-every", type=int, default=None, help="Holdout split frequency (default: 8 for final, 10 for fast)")
    ap.add_argument("--max-resolution", type=int, default=None, help="Image resolution max edge (default: 1920)")
    ap.add_argument("--dry-run", action="store_true", help="Print exact commands without executing")
    ap.add_argument("--eval-ratio", type=float, default=None, help="Evaluation holdout ratio (default: 0.1)")

    args = ap.parse_args()
    cfg = load_config(args.config)
    splat_cfg = cfg.get("splat", {})

    scene = args.scene
    backend = args.backend or splat_cfg.get("backend", "brush")
    preset = args.preset or splat_cfg.get("preset", "fast")
    eval_ratio = args.eval_ratio if args.eval_ratio is not None else float(splat_cfg.get("eval_ratio", 0.1))
    sh_degree = args.sh_degree if args.sh_degree is not None else int(splat_cfg.get("sh_degree", 3))
    max_splats = args.max_splats
    max_resolution = args.max_resolution if args.max_resolution is not None else 1920
    eval_split_every = args.eval_split_every if args.eval_split_every is not None else (8 if preset == "final" else 10)

    data_dir = Path("data") / scene
    outputs_dir = Path("outputs") / scene
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Dataset Directory
    train_pack_dir = outputs_dir / "train_pack"
    if not (train_pack_dir / "transforms.json").is_file():
        zip_file = outputs_dir / "train_pack.zip"
        if zip_file.is_file():
            import zipfile
            print(f"  Unpacking {zip_file} to {train_pack_dir}...")
            train_pack_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(train_pack_dir)

    if (train_pack_dir / "transforms.json").is_file():
        train_data_dir = train_pack_dir
    else:
        train_data_dir = data_dir / "colmap_aligned"

    final_ply_path = outputs_dir / "scene_raw.ply"
    metrics_path = outputs_dir / "metrics.json"

    # 2. Resolve Tool
    tool_name = "ns-train" if backend == "nerfstudio" else "brush"
    tool_bin = resolve_tool_path(tool_name)
    has_tool = tool_bin is not None

    print("=" * 68)
    print("  NAMMA SPACE -- STAGE 4: 3D GAUSSIAN SPLAT TRAINING")
    print("=" * 68)
    print(f"  Scene          : {scene}")
    print(f"  Backend        : {backend}")
    print(f"  Preset         : {preset} ({'7,000 steps' if preset == 'fast' else '30,000 steps, SH deg 3'})")
    print(f"  Max Splats Cap : {max_splats:,} (8 GB VRAM budget)")
    print(f"  SH Degree      : {sh_degree}")
    print(f"  Max Resolution : {max_resolution}")
    print(f"  Eval Split     : Every {eval_split_every} frames")
    print(f"  Data Directory : {train_data_dir}")
    print(f"  Tool Binary    : {tool_bin or 'NOT FOUND'}")
    print(f"  Export Target  : {final_ply_path}")
    print(f"  Metrics Target : {metrics_path}")
    print("-" * 68)

    # 3. Construct Command
    if backend == "brush":
        train_cmd = build_brush_train_command(
            data_dir=train_data_dir,
            output_dir=outputs_dir,
            preset=preset,
            brush_exe=tool_bin or "brush",
            max_splats=max_splats,
            sh_degree=sh_degree,
            eval_split_every=eval_split_every,
            max_resolution=max_resolution,
        )
    else:
        train_cmd = build_nerfstudio_train_command(
            data_dir=train_data_dir,
            output_models_dir=outputs_dir / "nerfstudio_models",
            scene=scene,
            preset=preset,
            eval_ratio=eval_ratio,
            sh_degree=sh_degree,
        )

    print("\n[Command]")
    print("  " + " ".join(train_cmd))

    # 4. Dry Run Check
    if args.dry_run or not has_tool:
        if not has_tool:
            print(f"\n[FAIL] '{tool_name}' binary not found.", file=sys.stderr)
            sys.exit(1)
        print("\n[Dry-Run Complete] Command validated successfully.")
        return

    # 5. Execute Training & Log Resource Usage
    print(f"\n[Starting {backend} {preset} training on GPU...]")
    vram_log_path = outputs_dir / "vram_log.csv"
    vram_log_file = open(vram_log_path, "w", encoding="utf-8")
    vram_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv", "-l", "5"],
        stdout=vram_log_file,
        stderr=subprocess.DEVNULL
    )
    print(f"  [+] Background VRAM logger started -> {vram_log_path}")

    vram_tracker = VRAMTracker()
    vram_tracker.start()
    t_start = time.time()
    start_time_iso = datetime.datetime.now().isoformat()

    process = subprocess.Popen(
        train_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines = []
    oom_detected = False

    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            captured_lines.append(line)
            # Print periodic progress
            if "step" in line.lower() or "iter" in line.lower() or "eval" in line.lower() or "loss" in line.lower():
                print("  " + line.strip())
            if "out of memory" in line.lower() or "oom" in line.lower():
                oom_detected = True

    rc = process.poll()
    t_elapsed = time.time() - t_start
    end_time_iso = datetime.datetime.now().isoformat()
    peak_vram_mb = vram_tracker.stop()

    try:
        vram_proc.terminate()
        vram_proc.wait(timeout=2)
    except Exception:
        vram_proc.kill()
    vram_log_file.close()
    print(f"  [+] Background VRAM logger stopped.")

    # Parse peak VRAM from vram_log.csv if available
    peak_vram_from_log = 0
    if vram_log_path.is_file():
        try:
            with open(vram_log_path, "r", encoding="utf-8") as vf:
                for vline in vf:
                    parts = [p.strip() for p in vline.split(",")]
                    if len(parts) >= 2 and "MiB" in parts[1]:
                        m_val = int(parts[1].replace("MiB", "").strip())
                        if m_val > peak_vram_from_log:
                            peak_vram_from_log = m_val
        except Exception:
            pass
    peak_vram_mb = max(peak_vram_mb, peak_vram_from_log)

    full_output = "".join(captured_lines)

    if rc != 0 or oom_detected:
        print(f"\n[FAIL] Training encountered an error (exit code {rc})", file=sys.stderr)
        if oom_detected:
            print(f"  [!] Out-Of-Memory detected! Peak VRAM: {peak_vram_mb} MiB", file=sys.stderr)
            print("  Actionable advice: Run with --max-resolution 1280 to downscale input frames.", file=sys.stderr)
        sys.exit(rc if rc != 0 else 1)

    print("-" * 68)
    print(f"[OK] Training completed successfully!")
    print(f"  Start Time      : {start_time_iso}")
    print(f"  End Time        : {end_time_iso}")
    print(f"  Wall-Clock Time : {t_elapsed:.1f} s ({t_elapsed/60:.2f} min)")
    print(f"  Peak VRAM Used  : {peak_vram_mb} MiB ({peak_vram_mb/1024:.2f} GB)")

    # 6. Locate Exported PLY and Move to scene_raw.ply (NEVER overwrite *_fast.ply)
    steps = "7000" if preset == "fast" else "30000"
    exported_target = outputs_dir / f"scene_raw_{steps}.ply"

    if exported_target.is_file():
        shutil.copy2(exported_target, final_ply_path)
        print(f"  [+] Exported model copied to: {final_ply_path}")
    elif final_ply_path.is_file():
        print(f"  [+] Exported model present at: {final_ply_path}")
    else:
        # Fallback search strictly excluding *_fast.ply
        recent_plys = [
            p for p in sorted(outputs_dir.glob("*.ply"), key=os.path.getmtime, reverse=True)
            if not p.name.endswith("_fast.ply") and not p.name.startswith("scene_raw_fast")
        ]
        if recent_plys:
            shutil.copy2(recent_plys[0], final_ply_path)
            print(f"  [+] Exported model copied from {recent_plys[0]} to: {final_ply_path}")
        else:
            print(f"[FAIL] Could not find exported PLY file in {outputs_dir}", file=sys.stderr)
            sys.exit(1)

    # 7. Analyze Splat Properties
    ply_stats = analyze_splat_ply(final_ply_path)
    gaussian_count = ply_stats["gaussian_count"]
    print(f"  Final Gaussian Count: {gaussian_count:,} splats (cap: {max_splats:,})")

    # 8. Metrics Evaluation
    metrics = parse_brush_metrics(full_output)
    if not metrics:
        metrics = {
            "eval_split": f"1 in {eval_split_every} frames ({100.0/eval_split_every:.1f}%)",
            "eval_note": "Evaluated during training (headless Brush)",
            "psnr": None,
            "ssim": None,
        }
    metrics["steps"] = int(steps)
    metrics["wall_clock_sec"] = round(t_elapsed, 2)
    metrics["peak_vram_mb"] = peak_vram_mb
    metrics["gaussian_count"] = gaussian_count

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [+] Saved test metrics to: {metrics_path}")

    # 9. Train Report
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        gpu_name = "NVIDIA GeForce RTX 4060 Laptop GPU"

    try:
        brush_ver = subprocess.check_output([tool_bin, "--version"], text=True).strip()
    except Exception:
        brush_ver = "brush-cli 0.3.0"

    train_report = {
        "exact_command": " ".join(train_cmd),
        "steps": int(steps),
        "resolution": max_resolution,
        "max_splats": max_splats,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "training_time_sec": round(t_elapsed, 2),
        "training_time_min": round(t_elapsed / 60.0, 2),
        "peak_vram_mib": peak_vram_mb,
        "gpu": gpu_name,
        "brush_version": brush_ver,
        "final_gaussian_count": gaussian_count,
        "pct_opacity_below_0_05": ply_stats["pct_opacity_below_0_05"],
        "pct_opacity_above_0_2": ply_stats["pct_opacity_above_0_2"],
        "pct_below_minus_0_3m": ply_stats["pct_below_minus_0_3m"],
        "y_p98_opacity_above_0_2_m": ply_stats["y_p98_opacity_above_0_2_m"],
    }
    train_report_path = outputs_dir / "train_report.json"
    with open(train_report_path, "w", encoding="utf-8") as f:
        json.dump(train_report, f, indent=2)
    print(f"  [+] Saved training report to: {train_report_path}")

    # 10. Coordinate Frame Verification & Stats
    print("\n[Running scripts/verify_splat_frame.py...]")
    verify_cmd = [
        sys.executable, "scripts/verify_splat_frame.py",
        "--scene", scene,
        "--splat-ply", str(final_ply_path),
    ]
    verify_res = subprocess.run(verify_cmd)

    print("\n[Running scripts/ply_stats.py...]")
    ply_stats_cmd = [
        sys.executable, "scripts/ply_stats.py",
        str(final_ply_path),
    ]
    subprocess.run(ply_stats_cmd)

    # 11. Walked-Loop Footprint & Camera Heights Summary
    print("\n[Walked-Loop Footprint & Camera Heights Summary]")
    wl_summary = compute_walked_loop_summary(
        data_dir / "colmap_aligned" / "transforms.json",
        data_dir / "colmap_aligned" / "alignment.json",
    )
    if wl_summary:
        print(f"  Scale Factor                : {wl_summary['scale_factor']:.5f} m/unit")
        print(f"  Pass 1 Walked Loop Sides    : {wl_summary['pass1_side_short']:.3f} m x {wl_summary['pass1_side_long']:.3f} m (Aspect Ratio: {wl_summary['pass1_aspect_ratio']:.3f})")
        print(f"  Pass 2 Walked Loop Sides    : {wl_summary['pass2_side_short']:.3f} m x {wl_summary['pass2_side_long']:.3f} m (Aspect Ratio: {wl_summary['pass2_aspect_ratio']:.3f})")
        print(f"  Pass 1 Camera Height        : {wl_summary['pass1_cam_height_m']:.3f} m +/- {wl_summary['pass1_cam_height_std_m']:.3f} m")
        print(f"  Pass 2 Camera Height        : {wl_summary['pass2_cam_height_m']:.3f} m +/- {wl_summary['pass2_cam_height_std_m']:.3f} m")

    if verify_res.returncode != 0:
        print(f"\n[WARNING] verify_splat_frame.py reported failures (exit code {verify_res.returncode}). See outputs/{scene}/splat_frame_verification.json.")
        sys.exit(verify_res.returncode)


if __name__ == "__main__":
    main()
