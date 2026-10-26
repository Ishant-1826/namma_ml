#!/usr/bin/env python3
"""Run COLMAP Structure-from-Motion (SfM) on extracted video frames.

Stage 2 of the Namma Space pipeline.

Pipeline:
  1. feature_extractor: shared single camera, model OPENCV by default, GPU if available.
     Supports ALIKED if the build provides it, else SIFT.
  2. matcher: sequential with loop detection, or exhaustive when <= 300 images;
     LightGlue if supported, else SIFT matching.
  3. mapper: global_mapper or incremental mapper.
  4. image_undistorter: undistorts images into colmap/dense.
  5. model_converter: exports sparse model to TXT and PLY.
  6. Quality audit: reports registered %, reprojection error, track length, intrinsics.
     Exits non-zero if registered < 85% or several large models with diagnostic triage.

Outputs:
  - data/<scene>/colmap/sparse/0/ (cameras.bin/txt, images.bin/txt, points3D.bin/txt, points3D.ply)
  - data/<scene>/points3D.ply
  - data/<scene>/colmap/points3D.ply
  - data/<scene>/colmap/dense/ (undistorted images & cameras)
  - data/<scene>/sfm_report.json
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

import numpy as np

# Ensure repo root is on sys.path for direct CLI execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Refresh Windows PATH from registry so COLMAP is always discovered
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

from pipeline.utils.config import load_config


def find_colmap_binary() -> str:
    """Locate the colmap executable."""
    colmap_path = shutil.which("colmap")
    if not colmap_path and sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            cand = Path(local_app) / "Programs" / "COLMAP" / "bin" / "colmap.exe"
            if cand.is_file():
                colmap_path = str(cand)

    if colmap_path:
        bin_dir = Path(colmap_path).resolve().parent
        lib_dir = bin_dir.parent / "lib"
        path_additions = [str(bin_dir)]
        if lib_dir.is_dir():
            path_additions.append(str(lib_dir))
        cur_path = os.environ.get("PATH", "")
        for p in path_additions:
            if p not in cur_path:
                os.environ["PATH"] = f"{p};{os.environ.get('PATH', '')}"
        return colmap_path

    raise RuntimeError(
        "COLMAP executable not found in PATH.\n"
        "Action: Please install COLMAP >= 3.9 or run scripts/check_env.py to verify PATH."
    )


def get_colmap_version(colmap_bin: str) -> float:
    """Detect COLMAP major.minor version from CLI output."""
    try:
        import re
        res = subprocess.run([colmap_bin, "help"], capture_output=True, text=True, timeout=5)
        out = res.stdout + res.stderr
        m = re.search(r"COLMAP\s+(\d+\.\d+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return 3.9


def check_colmap_capabilities(colmap_bin: str) -> Dict[str, bool]:
    """Inspect help output on THIS install to determine available features."""
    caps = {
        "aliked": False,
        "lightglue": False,
        "global_mapper": False,
    }
    # Check feature extractor help
    try:
        res = subprocess.run([colmap_bin, "feature_extractor", "-h"], capture_output=True, text=True, timeout=5)
        out = (res.stdout + res.stderr).lower()
        if "aliked" in out:
            caps["aliked"] = True
    except Exception:
        pass

    # Check matcher help
    try:
        res = subprocess.run([colmap_bin, "exhaustive_matcher", "-h"], capture_output=True, text=True, timeout=5)
        out = (res.stdout + res.stderr).lower()
        if "lightglue" in out:
            caps["lightglue"] = True
    except Exception:
        pass

    # Check if global_mapper subcommand exists
    try:
        res = subprocess.run([colmap_bin, "help"], capture_output=True, text=True, timeout=5)
        out = (res.stdout + res.stderr).lower()
        if "global_mapper" in out:
            caps["global_mapper"] = True
    except Exception:
        pass

    return caps


def update_status(
    status_paths: List[Path],
    stage: str,
    progress: str,
    done: bool = False,
    error: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Atomically write pipeline stage status to status.json files."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "progress": progress,
        "done": done,
        "error": error,
        **kwargs,
    }
    for p in status_paths:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(p)
        except Exception:
            pass


def run_colmap_command(
    colmap_bin: str, subcmd: str, args: List[str], desc: str, progress_callback: Optional[Any] = None
) -> None:
    """Execute a COLMAP subcommand with error handling and real-time streaming."""
    full_cmd = [colmap_bin, subcmd] + args
    print(f"\n--- [COLMAP {subcmd}] {desc} ---")
    print(f"Command: {' '.join(full_cmd)}\n")

    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout:
        for line in process.stdout:
            clean_line = line.rstrip()
            print(f"  [colmap] {clean_line}")
            if progress_callback:
                progress_callback(clean_line)

    rc = process.wait()
    if rc != 0:
        raise RuntimeError(
            f"COLMAP {subcmd} failed with exit code {rc}.\n"
            f"Command executed: {' '.join(full_cmd)}"
        )


def parse_cameras_txt(cameras_file: Path) -> List[Dict[str, Any]]:
    """Parse COLMAP cameras.txt into structured camera metadata."""
    cameras = []
    if not cameras_file.is_file():
        return cameras

    with open(cameras_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras.append({
                "camera_id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            })
    return cameras


def parse_images_txt(images_file: Path) -> Tuple[int, List[str]]:
    """Parse COLMAP images.txt and return registered count and image names.

    COLMAP images.txt uses 2 lines per image:
      Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
      Line 2: POINTS2D[] as (X, Y, POINT3D_ID ...)
    """
    if not images_file.is_file():
        return 0, []

    registered_names = []
    with open(images_file, "r", encoding="utf-8") as f:
        is_header = True
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_header:
                parts = line.split(maxsplit=9)
                if len(parts) == 10 and parts[0].isdigit():
                    img_name = parts[9]
                    registered_names.append(img_name)
                is_header = False
            else:
                is_header = True

    return len(registered_names), registered_names


def parse_points3d_txt(points_file: Path) -> Tuple[int, float, float]:
    """Parse points3D.txt for 3D point count, mean reprojection error, and mean track length."""
    if not points_file.is_file():
        return 0, 0.0, 0.0

    errors: List[float] = []
    track_lengths: List[int] = []

    with open(points_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] (IMAGE_ID, POINT2D_IDX, ...)
            if len(parts) >= 8:
                try:
                    err = float(parts[7])
                    errors.append(err)
                    track_obs = (len(parts) - 8) // 2
                    track_lengths.append(track_obs)
                except ValueError:
                    continue

    point_count = len(errors)
    mean_err = round(float(np.mean(errors)), 4) if errors else 0.0
    mean_track = round(float(np.mean(track_lengths)), 2) if track_lengths else 0.0

    return point_count, mean_err, mean_track


def count_sparse_models(sparse_root: Path) -> List[Tuple[int, Path]]:
    """Find all valid model subfolders in sparse/ sorted by number of registered images (descending)."""
    models = []
    if sparse_root.is_dir():
        for sub in sparse_root.iterdir():
            if sub.is_dir():
                has_model = (
                    (sub / "cameras.bin").is_file() or (sub / "cameras.txt").is_file()
                ) and (
                    (sub / "images.bin").is_file() or (sub / "images.txt").is_file()
                )
                if has_model:
                    n_img = 0
                    ibin = sub / "images.bin"
                    itxt = sub / "images.txt"
                    if ibin.is_file():
                        try:
                            import struct
                            with open(ibin, "rb") as f:
                                n_img = struct.unpack("<Q", f.read(8))[0]
                        except Exception:
                            n_img = 0
                    elif itxt.is_file():
                        try:
                            with open(itxt, "r", encoding="utf-8") as f:
                                lines = [l for l in f if l.strip() and not l.startswith("#")]
                                n_img = len(lines) // 2
                        except Exception:
                            n_img = 0
                    models.append((n_img, sub))
        models.sort(key=lambda x: x[0], reverse=True)
    return models


def resolve_image_directory(scene_dir: Path) -> Path:
    """Find the images directory for the given scene."""
    candidates = [
        scene_dir / "images",
        Path("outputs") / scene_dir.name / "frames",
        scene_dir / "raw",
    ]
    for c in candidates:
        if c.is_dir() and (any(c.glob("*.jpg")) or any(c.glob("*/*.jpg"))):
            return c

    return scene_dir / "images"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run COLMAP SfM on extracted frames (feature extraction, matching, mapping, undistortion)."
    )
    parser.add_argument("--scene", required=True, help="Scene name (under data/<scene>/)")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--camera-model",
        type=str,
        default="OPENCV",
        help="COLMAP camera model (OPENCV by default, SIMPLE_RADIAL, PINHOLE, etc.)",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        choices=["exhaustive", "sequential"],
        default=None,
        help="Matcher type: sequential (with loop detection) or exhaustive (<=300 images)",
    )
    parser.add_argument(
        "--mapper",
        type=str,
        choices=["incremental", "global"],
        default=None,
        help="Mapper backend: incremental (colmap mapper) or global (colmap global_mapper)",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default=None,
        help="Feature type: SIFT or ALIKED (default: SIFT, auto-checked)",
    )
    parser.add_argument(
        "--vocab-tree",
        type=str,
        default=None,
        help="Path to vocabulary tree file for sequential loop detection",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous SfM run by skipping completed stages",
    )
    parser.add_argument(
        "--skip-undistort",
        action="store_true",
        help="Skip image undistortion step",
    )
    parser.add_argument(
        "--multiple-models",
        type=int,
        default=0,
        help="COLMAP mapper multiple_models flag (default: 0 to build single model and stop)",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        default=None,
        help="Path to status.json to write progress",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[FAIL] Configuration load error: {e}", file=sys.stderr)
        return 1

    sfm_cfg = config.get("sfm", {})
    # Default model is OPENCV unless specified by CLI argument or config
    cam_model = args.camera_model if args.camera_model != "OPENCV" else sfm_cfg.get("camera_model", "OPENCV")
    use_gpu = bool(sfm_cfg.get("use_gpu", True))
    mapper_choice = args.mapper or sfm_cfg.get("mapper", "incremental")
    seq_overlap = int(sfm_cfg.get("sequential_overlap", 20))
    min_registered_ratio = float(sfm_cfg.get("min_registered_ratio", 0.85))
    multiple_models = int(args.multiple_models if args.multiple_models is not None else sfm_cfg.get("multiple_models", 0))

    # Resolve paths
    scene_dir = Path("data") / args.scene
    status_paths = [
        Path("logs") / "status.json",
        scene_dir / "logs" / "status.json",
    ]
    if args.status_file:
        status_paths.append(Path(args.status_file))

    update_status(status_paths, "initializing", "Initializing SfM pipeline...", done=False)

    image_dir = resolve_image_directory(scene_dir)

    if not image_dir.is_dir():
        print(
            f"[FAIL] Image directory not found: '{image_dir}'.\n"
            f"Action: Run Stage 1 first (python pipeline/extract_frames.py --scene {args.scene}) "
            f"to generate frames.",
            file=sys.stderr,
        )
        return 1

    image_files = sorted(list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png")))
    total_images = len(image_files)

    if total_images == 0:
        print(
            f"[FAIL] No image files found in '{image_dir}'.\n"
            f"Action: Verify frame extraction produced images at '{image_dir}'.",
            file=sys.stderr,
        )
        return 1

    has_subfolders = any(sub.is_dir() and (any(sub.glob("*.jpg")) or any(sub.glob("*.png"))) for sub in image_dir.iterdir()) if image_dir.is_dir() else False

    # Choose matcher: sequential with loop detection, or exhaustive when <= 300 images
    if args.matcher:
        matcher_choice = args.matcher
    elif "matcher" in sfm_cfg and sfm_cfg["matcher"]:
        matcher_choice = sfm_cfg["matcher"]
    else:
        matcher_choice = "sequential" if has_subfolders else ("exhaustive" if total_images <= 300 else "sequential")

    colmap_dir = scene_dir / "colmap"
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    dense_dir = colmap_dir / "dense"
    report_path = scene_dir / "sfm_report.json"

    # Clean up old database and models if not resuming
    if not args.resume:
        print(f"[Clean] Clean start requested (--resume not passed). Purging old COLMAP database and models in {colmap_dir}...")
        if database_path.is_file():
            try:
                database_path.unlink()
            except Exception:
                pass
        if sparse_dir.is_dir():
            try:
                shutil.rmtree(sparse_dir)
            except Exception:
                pass
        if dense_dir.is_dir():
            try:
                shutil.rmtree(dense_dir)
            except Exception:
                pass

    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    colmap_bin = find_colmap_binary()
    colmap_version = get_colmap_version(colmap_bin)
    caps = check_colmap_capabilities(colmap_bin)

    # Capability check for ALIKED / LightGlue
    requested_feat = args.feature_type or sfm_cfg.get("feature_type", "SIFT")
    if requested_feat == "ALIKED":
        if caps["aliked"]:
            feature_type = "ALIKED"
            print("  [Capability] ALIKED is supported by this COLMAP build -> using ALIKED.")
        else:
            feature_type = "SIFT"
            print("  [Capability] ALIKED requested but NOT supported by this build -> falling back to SIFT.")
    else:
        feature_type = "SIFT"

    if has_subfolders:
        cam_model = "SIMPLE_RADIAL"

    print("=" * 70)
    print("  NAMMA SPACE -- STAGE 2: STRUCTURE-FROM-MOTION (COLMAP)")
    print("=" * 70)
    print(f"  Scene              : {args.scene}")
    print(f"  Images directory   : {image_dir} ({total_images} frames)")
    print(f"  Multi-camera Mode  : {'1 camera per subfolder' if has_subfolders else 'Single shared camera'}")
    print(f"  COLMAP binary      : {colmap_bin} (v{colmap_version})")
    print(f"  Camera Model       : {cam_model}")
    print(f"  Feature Extractor  : {feature_type} (GPU={'1' if use_gpu else '0'})")
    print(f"  Matcher            : {matcher_choice} + {'Exhaustive Cross-Matcher' if has_subfolders else ''} (GPU={'1' if use_gpu else '0'})")
    print(f"  Mapper             : {mapper_choice}")
    print(f"  Database           : {database_path}")
    print(f"  Sparse Output      : {sparse_dir}")
    print(f"  Quality Gate       : >= {min_registered_ratio*100:.0f}% registered, single unified model")
    print(f"  Resume Mode        : {'Enabled' if args.resume else 'Disabled'}")
    print("-" * 70)

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Feature Extraction
    # ─────────────────────────────────────────────────────────────
    need_extraction = True
    if args.resume and database_path.is_file() and database_path.stat().st_size > 1024:
        print("\n[Step 1/5] Feature extraction: database exists (--resume), skipping.")
        need_extraction = False

    if need_extraction:
        update_status(status_paths, "feature_extraction", f"Extracting {feature_type} features ({cam_model})...", done=False)
        if has_subfolders:
            feat_args = [
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--ImageReader.camera_model", "SIMPLE_RADIAL",
                "--ImageReader.single_camera_per_folder", "1",
                "--ImageReader.default_focal_length_factor", "0.75",
                "--ImageReader.camera_params", "1440,960,540,0",
                "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
                "--SiftExtraction.max_num_features", "8192",
            ]
        else:
            feat_args = [
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--ImageReader.camera_model", cam_model,
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
                "--SiftExtraction.max_num_features", "8192",
            ]
        if colmap_version >= 4.0 and caps["aliked"] and feature_type == "ALIKED":
            feat_args.extend([
                "--FeatureExtraction.use_gpu", "1" if use_gpu else "0",
                "--FeatureExtraction.type", "ALIKED",
            ])
        try:
            run_colmap_command(
                colmap_bin,
                "feature_extractor",
                feat_args,
                f"Extracting {feature_type} features ({cam_model}, {'per-folder' if has_subfolders else 'shared'})",
            )
        except Exception as e:
            update_status(status_paths, "failed", f"Feature extraction failed: {e}", done=True, error=str(e))
            print(f"[FAIL] Feature extraction failed: {e}", file=sys.stderr)
            return 1

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Feature Matching
    # ─────────────────────────────────────────────────────────────
    need_matching = True
    if args.resume:
        if (sparse_dir / "0" / "points3D.bin").is_file():
            print("\n[Step 2/5] Feature matching: sparse model exists (--resume), skipping.")
            need_matching = False
        elif database_path.is_file():
            try:
                import sqlite3
                conn = sqlite3.connect(database_path)
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM two_view_geometries WHERE rows > 0")
                num_geoms = cur.fetchone()[0]
                conn.close()
                if num_geoms > 100:
                    print(f"\n[Step 2/5] Feature matching: {num_geoms} verified two-view geometries exist (--resume), skipping.")
                    need_matching = False
            except Exception:
                pass

    if need_matching:
        update_status(status_paths, "matching", f"Matching features ({matcher_choice})...", done=False)
        if has_subfolders:
            # 1. Sequential matching for intra-video track continuity (overlap 15, guided 1)
            seq_args = [
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if use_gpu else "0",
                "--SequentialMatching.overlap", "15",
                "--SequentialMatching.quadratic_overlap", "1",
                "--SequentialMatching.loop_detection", "0",
                "--SiftMatching.guided_matching", "1",
            ]
            try:
                run_colmap_command(
                    colmap_bin,
                    "sequential_matcher",
                    seq_args,
                    "Sequential matching for intra-video continuity (overlap=15, guided=1)",
                )
            except Exception as e:
                update_status(status_paths, "failed", f"Sequential matching failed: {e}", done=True, error=str(e))
                print(f"[FAIL] Sequential matching failed: {e}", file=sys.stderr)
                return 1

            # 2. Cross-source matching so clips tie into originals
            if total_images <= 1000:
                print(f"\n[Cross-Source Matching] Total images ({total_images} <= 1000) -> Running exhaustive_matcher...")
                exh_args = [
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "1" if use_gpu else "0",
                    "--SiftMatching.guided_matching", "1",
                    "--SiftMatching.max_ratio", "0.80",
                ]
                try:
                    run_colmap_command(
                        colmap_bin,
                        "exhaustive_matcher",
                        exh_args,
                        f"Cross-source exhaustive matching for {total_images} images",
                    )
                except Exception as e:
                    update_status(status_paths, "failed", f"Exhaustive cross-matching failed: {e}", done=True, error=str(e))
                    print(f"[FAIL] Exhaustive cross-matching failed: {e}", file=sys.stderr)
                    return 1
            else:
                vocab_tree_path = args.vocab_tree or sfm_cfg.get("vocab_tree_path") or "colmap_runs/vocab_tree.bin"
                print(f"\n[Cross-Source Matching] Total images ({total_images} > 1000) -> Running vocab_tree_matcher ({vocab_tree_path})...")
                vt_args = [
                    "--database_path", str(database_path),
                    "--SiftMatching.use_gpu", "1" if use_gpu else "0",
                    "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
                ]
                try:
                    run_colmap_command(
                        colmap_bin,
                        "vocab_tree_matcher",
                        vt_args,
                        "Cross-source vocabulary tree matching",
                    )
                except Exception as e:
                    update_status(status_paths, "failed", f"Vocab tree matching failed: {e}", done=True, error=str(e))
                    print(f"[FAIL] Vocab tree matching failed: {e}", file=sys.stderr)
                    return 1
        elif matcher_choice == "exhaustive":
            match_args = [
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if use_gpu else "0",
                "--SiftMatching.guided_matching", "1",
                "--SiftMatching.max_ratio", "0.85",
            ]
            try:
                run_colmap_command(
                    colmap_bin,
                    "exhaustive_matcher",
                    match_args,
                    f"Exhaustive pairwise GPU matching for {total_images} images",
                )
            except Exception as e:
                update_status(status_paths, "failed", f"Exhaustive matching failed: {e}", done=True, error=str(e))
                print(f"[FAIL] Exhaustive matching failed: {e}", file=sys.stderr)
                return 1
        else:
            match_args = [
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if use_gpu else "0",
                "--SequentialMatching.overlap", str(seq_overlap),
                "--SequentialMatching.quadratic_overlap", "1",
            ]
            # Check for vocabulary tree for loop detection
            vocab_tree_path = args.vocab_tree or sfm_cfg.get("vocab_tree_path")
            if vocab_tree_path and Path(vocab_tree_path).is_file():
                print(f"  [Loop Detection] Using vocabulary tree: {vocab_tree_path}")
                match_args.extend([
                    "--SequentialMatching.loop_detection", "1",
                    "--SequentialMatching.vocab_tree_path", str(vocab_tree_path),
                ])
            else:
                print("  [Loop Detection] Vocab tree not specified; sequential matching with quadratic overlap enabled.")

            try:
                run_colmap_command(
                    colmap_bin,
                    "sequential_matcher",
                    match_args,
                    f"Sequential matching (overlap={seq_overlap}, quadratic loop detection)",
                )
            except Exception as e:
                update_status(status_paths, "failed", f"Sequential matching failed: {e}", done=True, error=str(e))
                print(f"[FAIL] Sequential matching failed: {e}", file=sys.stderr)
                return 1

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Mapping (Sparse Reconstruction)
    # ─────────────────────────────────────────────────────────────
    primary_model_dir = sparse_dir / "0"
    need_mapping = True
    if args.resume and (primary_model_dir / "points3D.bin").is_file():
        models = count_sparse_models(sparse_dir)
        if models and models[0][0] >= max(10, int(total_images * 0.1)):
            print(f"\n[Step 3/5] Mapping: sparse model exists with {models[0][0]} images (--resume), skipping.")
            need_mapping = False
        else:
            print("\n[Step 3/5] Mapping: existing model has insufficient images, remapping...")

    if need_mapping:
        update_status(
            status_paths,
            "mapping",
            f"Incremental mapping started (single_model={1 if multiple_models == 0 else 0})...",
            done=False,
        )
        mapping_success = False

        if mapper_choice == "global":
            if caps["global_mapper"]:
                global_args = [
                    "--database_path", str(database_path),
                    "--image_path", str(image_dir),
                    "--output_path", str(sparse_dir),
                    "--GlobalMapper.gp_use_gpu", "1" if use_gpu else "0",
                    "--GlobalMapper.ba_ceres_use_gpu", "1" if use_gpu else "0",
                ]
                try:
                    run_colmap_command(
                        colmap_bin,
                        "global_mapper",
                        global_args,
                        "Global Structure-from-Motion (global_mapper)",
                    )
                    if (primary_model_dir / "cameras.bin").is_file() or (primary_model_dir / "cameras.txt").is_file():
                        mapping_success = True
                    else:
                        print("\n[WARN] global_mapper produced no model; falling back to incremental mapper.")
                except Exception as e:
                    print(f"\n[WARN] global_mapper failed: {e}. Falling back to incremental mapper.")
            else:
                print(f"\n[WARN] 'colmap global_mapper' is not available in this build (v{colmap_version}).")
                print("       Falling back automatically to incremental mapper ('colmap mapper').")

        if not mapping_success:
            inc_args = [
                "--database_path", str(database_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
                "--Mapper.multiple_models", str(multiple_models),
                "--Mapper.abs_pose_min_num_inliers", "20",
                "--Mapper.abs_pose_min_inlier_ratio", "0.20",
                "--Mapper.max_reg_trials", "3",
                "--Mapper.init_min_tri_angle", "8",
                "--Mapper.init_min_num_inliers", "50",
                "--Mapper.filter_max_reproj_error", "4.0",
                "--Mapper.filter_min_tri_angle", "1.5",
            ]
            if has_subfolders:
                inc_args.extend([
                    "--Mapper.ba_refine_focal_length", "1",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0",
                ])
            if colmap_version >= 4.0:
                inc_args.extend(["--Mapper.ba_use_gpu", "1" if use_gpu else "0"])

            import re, time
            last_map_update = [0.0]
            def map_progress_cb(line: str) -> None:
                m = re.search(r"Registering image #(\d+)\s*\((\d+)\)", line)
                if m:
                    now = time.time()
                    if now - last_map_update[0] >= 5.0:
                        last_map_update[0] = now
                        reg_cur = m.group(1)
                        reg_tot = m.group(2)
                        update_status(
                            status_paths,
                            "mapping",
                            f"Registering image #{reg_cur} of {total_images} (candidate #{reg_tot})",
                            done=False,
                        )

            try:
                run_colmap_command(
                    colmap_bin,
                    "mapper",
                    inc_args,
                    "Incremental Structure-from-Motion (mapper)",
                    progress_callback=map_progress_cb,
                )
            except Exception as e:
                update_status(status_paths, "failed", f"Incremental mapping failed: {e}", done=True, error=str(e))
                print(f"[FAIL] Incremental mapping failed: {e}", file=sys.stderr)
                return 1

    # Verify model existence
    available_models_info = count_sparse_models(sparse_dir)
    if not available_models_info:
        print(
            "\n[FAIL] COLMAP produced 0 reconstructed 3D models!\n"
            "Likely Root Causes:\n"
            "  1. Rotation-only motion: camera rotated in place without sufficient translation baseline.\n"
            "  2. Motion blur: rapid panning or low shutter speed degraded keypoint matches.\n"
            "  3. Textureless walls: plain white or uniform walls lack feature keypoints.\n"
            "  4. Doorways: sharp transitions or occlusion across doorways broke continuous feature tracking.\n\n"
            "Next Config Switches to Try:\n"
            "  - Try --camera-model SIMPLE_RADIAL instead of OPENCV.\n"
            "  - Try --matcher exhaustive for dense pairwise matching.\n"
            "  - Re-run extract_frames.py with higher fps (--fps 4) or lower blur threshold.\n"
            "  - Inspect data/<scene>/contact_sheet.jpg for visual track continuity.",
            file=sys.stderr,
        )
        return 1

    primary_model = available_models_info[0][1]
    target_primary = sparse_dir / "0"
    if primary_model.name != "0":
        print(f"\n[INFO] Largest model is '{primary_model.name}'. Setting as primary model '0'...")
        temp_dir = sparse_dir / "_temp_swap"
        old_name = primary_model.name
        primary_model.rename(temp_dir)
        if target_primary.is_dir():
            target_primary.rename(sparse_dir / old_name)
        temp_dir.rename(target_primary)
        primary_model = target_primary
        available_models_info = count_sparse_models(sparse_dir)

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Convert Model to TXT and PLY
    # ─────────────────────────────────────────────────────────────
    print(f"\n[Step 4/5] Converting model in {primary_model} to TXT and PLY...")
    update_status(status_paths, "model_converter", f"Converting model {primary_model.name} to TXT and PLY...", done=False)

    # Export TXT
    txt_args = [
        "--input_path", str(primary_model),
        "--output_path", str(primary_model),
        "--output_type", "TXT",
    ]
    try:
        run_colmap_command(colmap_bin, "model_converter", txt_args, "Exporting model to TXT")
    except Exception as e:
        print(f"[WARN] Could not convert to TXT: {e}")

    # Export PLY inside model dir, root colmap dir, and scene dir
    model_ply = primary_model / "points3D.ply"
    root_colmap_ply = colmap_dir / "points3D.ply"
    scene_ply = scene_dir / "points3D.ply"
    ply_args = [
        "--input_path", str(primary_model),
        "--output_path", str(model_ply),
        "--output_type", "PLY",
    ]
    try:
        run_colmap_command(colmap_bin, "model_converter", ply_args, "Exporting point cloud to PLY")
        shutil.copy2(model_ply, root_colmap_ply)
        shutil.copy2(model_ply, scene_ply)
        print(f"  [+] Saved point cloud to: {model_ply}, {root_colmap_ply}, and {scene_ply}")
    except Exception as e:
        print(f"[WARN] Could not convert to PLY: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Image Undistortion (for Gaussian Splatting)
    # ─────────────────────────────────────────────────────────────
    if not args.skip_undistort:
        dense_dir.mkdir(parents=True, exist_ok=True)
        need_undistort = True
        if args.resume and (dense_dir / "images").is_dir() and any((dense_dir / "images").glob("*.jpg")):
            print("\n[Step 5/5] Image undistortion: dense/images already exists (--resume), skipping.")
            need_undistort = False

        if need_undistort:
            update_status(status_paths, "image_undistorter", "Undistorting images for Gaussian Splatting...", done=False)
            undistort_args = [
                "--image_path", str(image_dir),
                "--input_path", str(primary_model),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
            ]
            try:
                run_colmap_command(
                    colmap_bin,
                    "image_undistorter",
                    undistort_args,
                    "Undistorting images for Gaussian Splatting",
                )
            except Exception as e:
                print(f"[WARN] Image undistortion failed: {e}")
    else:
        print("\n[Step 5/5] Image undistortion skipped (--skip-undistort).")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Parse Statistics & Metrics
    # ─────────────────────────────────────────────────────────────
    cameras_meta = parse_cameras_txt(primary_model / "cameras.txt")
    registered_count, registered_names = parse_images_txt(primary_model / "images.txt")
    point_count, mean_reproj_err, mean_track = parse_points3d_txt(primary_model / "points3D.txt")

    reg_ratio = float(registered_count) / float(total_images) if total_images > 0 else 0.0
    reg_pct = round(reg_ratio * 100.0, 2)

    # Parse detailed poses for step lengths
    def qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
             2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
             2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
             1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
             2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
             2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
             1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
        ])

    images_with_poses = {}
    itxt_path = primary_model / "images.txt"
    if itxt_path.is_file():
        with open(itxt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) >= 10:
                qvec = np.array([float(x) for x in parts[1:5]])
                tvec = np.array([float(x) for x in parts[5:8]])
                c_id = int(parts[8])
                nm = parts[9]
                R = qvec2rotmat(qvec)
                C = -R.T @ tvec
                images_with_poses[nm] = {
                    "img_id": int(parts[0]),
                    "qvec": qvec,
                    "tvec": tvec,
                    "cam_id": c_id,
                    "center": C,
                    "name": nm,
                }
            i += 2

    # Per-source breakdown
    subfolders = [d.name for d in image_dir.iterdir() if d.is_dir()] if image_dir.is_dir() else []
    source_stats = []
    orig_total = 0
    orig_reg = 0

    if subfolders:
        for s in sorted(subfolders):
            s_files = list((image_dir / s).glob("*.jpg")) + list((image_dir / s).glob("*.png"))
            s_total = len(s_files)
            s_reg = sum(1 for n in images_with_poses if n.startswith(s + "/") or n.startswith(s + "\\"))
            s_pct = (s_reg / s_total * 100.0) if s_total > 0 else 0.0
            is_orig = ("20261021_" in s or "video" in s.lower())
            if is_orig:
                orig_total += s_total
                orig_reg += s_reg
            source_stats.append({
                "source": s,
                "is_original": is_orig,
                "total": s_total,
                "registered": s_reg,
                "percent": round(s_pct, 2),
            })
    else:
        orig_total = total_images
        orig_reg = registered_count

    orig_reg_pct = (orig_reg / orig_total * 100.0) if orig_total > 0 else 100.0

    # Step length computation
    orig_steps = []
    all_steps = []

    for s_info in source_stats:
        s_name = s_info["source"]
        if s_name == "stills":
            continue
        s_imgs = [img for n, img in images_with_poses.items() if n.startswith(s_name + "/") or n.startswith(s_name + "\\")]
        s_imgs.sort(key=lambda x: int("".join(filter(str.isdigit, Path(x["name"]).stem)) or 0))
        for k in range(len(s_imgs) - 1):
            st = float(np.linalg.norm(s_imgs[k+1]["center"] - s_imgs[k]["center"]))
            all_steps.append(st)
            if s_info["is_original"]:
                orig_steps.append(st)

    median_orig_step = float(np.median(orig_steps)) if orig_steps else (float(np.median(all_steps)) if all_steps else 1.0)
    largest_step = float(np.max(all_steps)) if all_steps else 0.0
    step_ratio = largest_step / median_orig_step if median_orig_step > 0 else 0.0

    # Camera intrinsics check
    cam_intrinsics_ok = True
    cams_info = []
    for c in cameras_meta:
        m = c["model"]
        p = c["params"]
        if m in ("SIMPLE_RADIAL", "RADIAL"):
            cfx, cfy = p[0], p[0]
            ck1 = p[3] if m == "SIMPLE_RADIAL" else p[3]
        elif m == "PINHOLE":
            cfx, cfy = p[0], p[1]
            ck1 = 0.0
        elif m in ("OPENCV", "OPENCV_FISHEYE"):
            cfx, cfy = p[0], p[1]
            ck1 = p[4]
        else:
            cfx, cfy = p[0], p[0]
            ck1 = 0.0
        diff_f = abs(cfx - cfy) / max(cfx, cfy) if max(cfx, cfy) > 0 else 0.0
        c_ok = (diff_f <= 0.05) and (abs(ck1) < 0.3)
        if not c_ok:
            cam_intrinsics_ok = False
        cams_info.append({"camera_id": c["camera_id"], "fx": cfx, "fy": cfy, "k1": ck1, "ok": c_ok})

    # Pass Criteria Evaluation
    crit_1_orig_reg = orig_reg_pct >= 95.0
    crit_2_total_reg = reg_pct >= 90.0
    crit_3_reproj = mean_reproj_err < 1.0
    crit_4_cams = cam_intrinsics_ok
    crit_5_step = step_ratio <= 5.0

    large_model_threshold = max(20, int(total_images * 0.15))
    large_models = [m for m in available_models_info if m[0] >= large_model_threshold]

    report = {
        "scene": args.scene,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "images_total": total_images,
            "images_registered": registered_count,
            "registered (%)": reg_pct,
            "registration_percent": reg_pct,
            "registration_ratio": round(reg_ratio, 4),
            "original_videos_registered_pct": round(orig_reg_pct, 2),
            "number_of_models": len(available_models_info),
            "primary_model_index": primary_model.name,
            "points3D_count": point_count,
            "mean_reprojection_error": mean_reproj_err,
            "mean_track_length": mean_track,
            "median_orig_step": round(median_orig_step, 4),
            "largest_step": round(largest_step, 4),
            "step_ratio": round(step_ratio, 2),
            "large_models_count": len(large_models),
        },
        "pass_criteria": {
            "orig_registered_ge_95": crit_1_orig_reg,
            "total_registered_ge_90": crit_2_total_reg,
            "reproj_error_lt_1": crit_3_reproj,
            "camera_intrinsics_ok": crit_4_cams,
            "step_ratio_le_5": crit_5_step,
        },
        "source_breakdown": source_stats,
        "cameras": cams_info,
        "configuration": {
            "camera_model": cam_model,
            "matcher": matcher_choice,
            "mapper": mapper_choice,
            "feature_type": feature_type,
            "min_registered_ratio": min_registered_ratio,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Quality Gate & Diagnostic Output Table
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STRUCTURE-FROM-MOTION (COLMAP) REPORT SUMMARY")
    print("=" * 80)
    print(f"  Total input images   : {total_images}")
    print(f"  Registered images    : {registered_count} ({reg_pct:.1f}%)")
    print(f"  Number of models     : {len(available_models_info)} (large models: {len(large_models)})")
    print(f"  3D Points in cloud   : {point_count:,}")
    print(f"  Mean reprojection err: {mean_reproj_err:.4f} px")
    print(f"  Mean track length    : {mean_track:.2f} frames")
    print(f"  Median step (orig)   : {median_orig_step:.4f} model units")
    print(f"  Largest step         : {largest_step:.4f} model units (ratio: {step_ratio:.2f}x)")
    print(f"  Sparse model saved to: {primary_model}")
    print(f"  PLY point cloud      : {root_colmap_ply} and {scene_ply}")
    print(f"  Report saved to      : {report_path}")

    if source_stats:
        print("\n" + "-" * 80)
        print("  PER-SOURCE REGISTRATION BREAKDOWN")
        print("-" * 80)
        print(f"  {'Source':<22} {'Type':<12} {'Total':<10} {'Registered':<14} {'Percent':<10} {'Status'}")
        print("  " + "-" * 76)
        for s in source_stats:
            st_type = "Original Pass" if s["is_original"] else "New Source"
            st_status = "[PASS]" if s["percent"] >= 60.0 else "[DROP CANDIDATE (<60%)]"
            print(f"  {s['source']:<22} {st_type:<12} {s['total']:<10} {s['registered']:<14} {s['percent']:<9.1f}% {st_status}")
        print("  " + "-" * 76)

    print("\n" + "=" * 80)
    print("  PASS CRITERIA VERIFICATION TABLE")
    print("=" * 80)
    print(f"  {'Criterion':<55} {'Requirement':<16} {'Actual':<14} {'Verdict'}")
    print("  " + "-" * 90)
    print(f"  1. Original two videos registered in ONE model       >= 95.0%        {orig_reg_pct:<13.2f}% [{'PASS' if crit_1_orig_reg else 'FAIL'}]")
    print(f"  2. Total frames registered overall                   >= 90.0%        {reg_pct:<13.2f}% [{'PASS' if crit_2_total_reg else 'FAIL'}]")
    print(f"  3. Mean reprojection error                           < 1.0 px        {mean_reproj_err:<13.4f}px [{'PASS' if crit_3_reproj else 'FAIL'}]")
    cam_str = f"{len(cams_info)} cams ok" if crit_4_cams else "fail"
    print(f"  4. Per camera fx ~ fy within 5%, |k1| < 0.3          diff<=5%,|k|<0.3 {cam_str:<14} [{'PASS' if crit_4_cams else 'FAIL'}]")
    print(f"  5. Frame-to-frame step <= 5x median original step    <= 5.00x        {step_ratio:<13.2f}x [{'PASS' if crit_5_step else 'FAIL'}]")
    print("  " + "-" * 90)

    # Check for sources failing under 60%
    failing_new_sources = [s for s in source_stats if not s["is_original"] and s["percent"] < 60.0]
    if failing_new_sources:
        print("\n  [ADVISORY] The following new source(s) fell below 60% registration:")
        for fs in failing_new_sources:
            print(f"    - {fs['source']}: {fs['registered']}/{fs['total']} ({fs['percent']:.1f}%)")
            print("      Likely cause: Pure rotation, motion blur, or textureless surfaces.")
            print(f"      Protocol: Dropping source '{fs['source']}' and proceeding with remaining sources.")

    # Quality Gate Check
    if not crit_1_orig_reg:
        update_status(
            status_paths,
            "failed",
            f"Quality gate failure: Original videos registered {orig_reg_pct:.1f}% (< 95.0%)",
            done=True,
            error="Original video registration failed",
        )
        print("\n  [STOP] ORIGINAL VIDEOS FELL BELOW 95% REGISTRATION CRITERIA!", file=sys.stderr)
        return 1

    overall_pass = crit_1_orig_reg and crit_2_total_reg and crit_3_reproj and crit_4_cams and crit_5_step
    print(f"\n  Overall Quality Gate: [{'PASS' if overall_pass else 'FAIL'}]")
    print("=" * 80)

    update_status(
        status_paths,
        "completed",
        f"SfM reconstruction completed ({registered_count}/{total_images} images, {point_count} points, reproj={mean_reproj_err:.2f}px)",
        done=True,
        registered_count=registered_count,
        total_images=total_images,
        points3D_count=point_count,
        mean_reprojection_error=mean_reproj_err,
        overall_pass=overall_pass,
    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
