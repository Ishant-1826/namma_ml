#!/usr/bin/env python3
"""
scripts/pack_dataset.py  --  Pack training dataset into a minimal zip bundle.

Collects only what 3D Gaussian Splatting training requires:
  1. Aligned images (only registered frames).
  2. Aligned transforms.json (Nerfstudio format with relative image paths).
  3. Initial sparse point cloud (sparse_pc.ply converted from points3D.txt).

Output:
  outputs/<scene>/train_pack.zip

Usage:
  python scripts/pack_dataset.py --scene home
"""

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


def parse_points3d_txt(path: Path) -> list:
    """Parses COLMAP points3D.txt and returns list of (x, y, z, r, g, b)."""
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            pts.append((x, y, z, r, g, b))
    return pts


def write_ply(pts: list, output_path: Path) -> None:
    """Writes (x, y, z, r, g, b) points to binary little-endian PLY."""
    n = len(pts)
    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    ).encode("ascii")

    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.empty(n, dtype=dt)
    for i, p in enumerate(pts):
        arr[i] = (p[0], p[1], p[2], p[3], p[4], p[5])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def pack_dataset(
    scene: str,
    aligned_dir: Path,
    images_dir: Path,
    output_zip: Path,
) -> bool:
    print("=" * 64)
    print(f"  PACK TRAINING DATASET: '{scene}'")
    print(f"  Aligned Dir : {aligned_dir}")
    print(f"  Images Dir  : {images_dir}")
    print(f"  Output Zip  : {output_zip}")
    print("=" * 64)

    # 1. Verify source files
    txt_points = aligned_dir / "points3D.txt"
    src_transforms = aligned_dir / "transforms.json"
    if not src_transforms.is_file():
        src_transforms = aligned_dir.parent / "transforms.json"

    if not (txt_points.is_file() and src_transforms.is_file()):
        print(f"[FAIL] Missing aligned data files in {aligned_dir}", file=sys.stderr)
        print("  Actionable advice: Run pipeline/align_scene.py --scene <scene> first.", file=sys.stderr)
        return False

    with open(src_transforms, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Ensure transforms.json references "images/<name>" and "sparse_pc.ply"
    clean_frames = []
    registered_image_names = set()
    for fr in meta.get("frames", []):
        fp = fr["file_path"].replace("\\", "/")
        img_name = Path(fp).name
        clean_frames.append({
            "file_path": f"images/{img_name}",
            "transform_matrix": fr["transform_matrix"],
        })
        registered_image_names.add(img_name)

    pack_transforms = dict(meta)
    pack_transforms["frames"] = clean_frames
    pack_transforms["ply_file_path"] = "sparse_pc.ply"

    print(f"  Registered frames in dataset: {len(clean_frames)}")

    # 2. Convert points3D.txt to sparse_pc.ply
    print("  Converting points3D.txt to binary sparse_pc.ply...")
    pts = parse_points3d_txt(txt_points)
    tmp_ply = output_zip.parent / "sparse_pc.ply"
    write_ply(pts, tmp_ply)
    print(f"  Generated sparse_pc.ply ({len(pts)} points, {tmp_ply.stat().st_size / 1024:.1f} KB)")

    # 3. Create zip file
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Packing into {output_zip} ...")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # A. transforms.json
        transforms_str = json.dumps(pack_transforms, indent=2)
        zf.writestr("transforms.json", transforms_str)

        # B. sparse_pc.ply
        zf.write(tmp_ply, arcname="sparse_pc.ply")

        # C. images/
        img_count = 0
        missing_count = 0
        for img_name in sorted(registered_image_names):
            img_path = images_dir / img_name
            if img_path.is_file():
                zf.write(img_path, arcname=f"images/{img_name}")
                img_count += 1
            else:
                missing_count += 1

        print(f"  Added {img_count} images (missing: {missing_count})")

    # Clean up temporary ply
    if tmp_ply.is_file():
        tmp_ply.unlink()

    # Print final size
    zip_bytes = output_zip.stat().st_size
    zip_mb = zip_bytes / (1024 * 1024)
    print("-" * 64)
    print(f"  [+] Pack created successfully: {output_zip}")
    print(f"      Size: {zip_mb:.2f} MB ({zip_bytes:,} bytes)")
    print(f"      Contents: transforms.json, sparse_pc.ply, {img_count} images")
    print("=" * 64)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack training dataset into minimal train_pack.zip.")
    ap.add_argument("--scene", default="home", help="Scene name (subfolder of data/)")
    ap.add_argument("--aligned-dir", default=None, help="Path to aligned COLMAP directory")
    ap.add_argument("--images-dir", default=None, help="Path to images directory")
    ap.add_argument("--output-zip", default=None, help="Output zip file path")

    args = ap.parse_args()
    scene = args.scene
    data_dir = Path("data") / scene
    aligned_dir = Path(args.aligned_dir) if args.aligned_dir else (data_dir / "colmap_aligned")
    images_dir = Path(args.images_dir) if args.images_dir else (data_dir / "images")
    output_zip = Path(args.output_zip) if args.output_zip else (Path("outputs") / scene / "train_pack.zip")

    success = pack_dataset(
        scene=scene,
        aligned_dir=aligned_dir,
        images_dir=images_dir,
        output_zip=output_zip,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
