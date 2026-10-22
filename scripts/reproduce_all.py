#!/usr/bin/env python3
"""
scripts/reproduce_all.py -- One-command reproduction script for Namma Space Phase 3.

Runs the complete sequence:
  1. scripts/diagnose_sfm.py --scene <scene>
  2. pipeline/align_scene.py --scene <scene>
  3. scripts/make_topview.py --scene <scene>

Outputs:
  - docs/sfm_diagnosis.md
  - data/<scene>/colmap/aligned/ (cameras.txt, images.txt, points3D.txt)
  - data/<scene>/align_report.json
  - outputs/<scene>/topview.png
  - outputs/<scene>/topview_overlay.png
  - outputs/<scene>/topview_legs.jpg

Usage:
  python scripts/reproduce_all.py --scene home
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd: list, desc: str) -> None:
    print("\n" + "=" * 64)
    print(f"  RUNNING: {desc}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 64)
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {res.returncode}: {desc}")
        print("  Actionable advice: Check error message above and verify virtual environment.")
        sys.exit(res.returncode)
    print(f"[OK] Completed in {elapsed:.2f}s: {desc}")


def main():
    ap = argparse.ArgumentParser(description="Reproduce all Phase 3 diagnosis, alignment, and top-view steps.")
    ap.add_argument("--scene", default="home", help="Scene name (subfolder of data/)")
    args = ap.parse_args()

    scene = args.scene
    python_exe = sys.executable

    print("=" * 64)
    print(f"  NAMMA SPACE -- COMPLETE PHASE 3 REPRODUCTION PIPELINE")
    print(f"  Scene      : {scene}")
    print(f"  Python     : {python_exe}")
    print("=" * 64)

    # Step 1: Diagnosis
    run_step(
        [python_exe, "scripts/diagnose_sfm.py", "--scene", scene],
        "Step 1: SfM Model Diagnosis (docs/sfm_diagnosis.md)"
    )

    # Step 2: Alignment
    run_step(
        [python_exe, "pipeline/align_scene.py", "--scene", scene],
        "Step 2: Metric Scene Alignment (pipeline/align_scene.py)"
    )

    # Step 3: Alignment Verification
    run_step(
        [python_exe, "scripts/check_alignment.py", "--scene", scene],
        "Step 3: Alignment Verification (scripts/check_alignment.py)"
    )

    # Step 4: Top-View Generation
    run_step(
        [python_exe, "scripts/make_topview.py", "--scene", scene],
        "Step 4: Top-View Plot & Overlay Generation (scripts/make_topview.py)"
    )

    print("\n" + "=" * 64)
    print("  ALL STEPS SUCCESSFULLY REPRODUCED!")
    print("=" * 64)
    print("Key output files generated:")
    outputs = [
        Path("docs/sfm_diagnosis.md"),
        Path(f"data/{scene}/colmap_aligned/images.txt"),
        Path(f"data/{scene}/transforms.json"),
        Path(f"data/{scene}/alignment.json"),
        Path(f"outputs/{scene}/alignment_check.png"),
        Path(f"outputs/{scene}/alignment_side.png"),
        Path(f"outputs/{scene}/alignment_top.png"),
        Path(f"outputs/{scene}/topview.png"),
        Path(f"outputs/{scene}/topview_overlay.png"),
        Path(f"outputs/{scene}/topview_legs.jpg"),
    ]
    for p in outputs:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  [+] {p} ({size_kb:.1f} KB)")
        else:
            print(f"  [-] MISSING: {p}")


if __name__ == "__main__":
    main()
