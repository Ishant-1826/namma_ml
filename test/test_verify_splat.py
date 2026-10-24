#!/usr/bin/env python3
"""
test/test_verify_splat.py  --  Unit Tests for scripts/verify_splat_frame.py.

Tests coordinate frame verification on tiny synthetic Gaussian Splat PLYs:
1. Valid synthetic splat (floor at y=0, extents matching reference within 15%, upright) -> PASS.
2. Displaced floor (floor at y = 0.80 m) -> FAIL loudly with numbers.
3. Inverted / upside-down splat (y -> -y) -> FAIL loudly with numbers.
4. Extents mismatch (scaled X/Z by 1.4x, > 15% span deviation) -> FAIL loudly with numbers.

All tests operate inside a temporary directory and clean up on exit.
"""

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

# Ensure repo root is on sys.path
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.verify_splat_frame import verify_splat_frame


def write_synthetic_splat_ply(
    path: Path,
    pts: np.ndarray,
    opacity_val: float = 2.0,  # in logit space: sigmoid(2.0) ~ 0.88
    log_scale_val: float = -3.0,  # exp(-3.0) ~ 0.05 m
) -> None:
    """Writes synthetic 3D Gaussian Splat binary PLY."""
    n = len(pts)
    props = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]

    header = (
        f"ply\n"
        f"format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property float f_dc_0\n"
        f"property float f_dc_1\n"
        f"property float f_dc_2\n"
        f"property float opacity\n"
        f"property float scale_0\n"
        f"property float scale_1\n"
        f"property float scale_2\n"
        f"property float rot_0\n"
        f"property float rot_1\n"
        f"property float rot_2\n"
        f"property float rot_3\n"
        f"end_header\n"
    ).encode("ascii")

    dt = np.dtype(props)
    arr = np.zeros(n, dtype=dt)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["f_dc_0"] = 0.5
    arr["f_dc_1"] = 0.5
    arr["f_dc_2"] = 0.5
    arr["opacity"] = opacity_val
    arr["scale_0"] = log_scale_val
    arr["scale_1"] = log_scale_val
    arr["scale_2"] = log_scale_val
    arr["rot_0"] = 1.0  # unit quaternion (1, 0, 0, 0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def write_synthetic_points3d_txt(path: Path, pts: np.ndarray) -> None:
    """Writes COLMAP-style points3D.txt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (x, y, z) in enumerate(pts):
            f.write(f"{i + 1} {x:.6f} {y:.6f} {z:.6f} 128 128 128 0.5 1 0 2 0\n")


def write_synthetic_transforms_json(path: Path, cam_positions: np.ndarray) -> None:
    """Writes minimal Nerfstudio transforms.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, c in enumerate(cam_positions):
        # Look towards origin roughly
        c2w = np.eye(4)
        c2w[:3, 3] = c
        frames.append({
            "file_path": f"images/frame_{i+1:06d}.jpg",
            "transform_matrix": c2w.tolist(),
        })

    data = {
        "camera_model": "OPENCV",
        "fl_x": 1477.94,
        "fl_y": 1477.94,
        "cx": 960.0,
        "cy": 540.0,
        "w": 1920,
        "h": 1080,
        "frames": frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_synthetic_room_points(n_floor: int = 1200, n_walls: int = 1200) -> np.ndarray:
    """Generates synthetic 3D points for a 4m x 6m room with floor at y=0, walls up to y=2.5m."""
    rng = np.random.default_rng(42)
    # Floor: x in [-2, 2], z in [-3, 3], y ~ 0.0
    fx = rng.uniform(-2.0, 2.0, n_floor)
    fz = rng.uniform(-3.0, 3.0, n_floor)
    fy = rng.normal(0.0, 0.015, n_floor)
    floor_pts = np.column_stack([fx, fy, fz])

    # 4 Walls: y in [0.1, 2.5]
    n_per_wall = n_walls // 4
    # Wall 1: x = -2.0
    w1_x = np.full(n_per_wall, -2.0) + rng.normal(0, 0.01, n_per_wall)
    w1_z = rng.uniform(-3.0, 3.0, n_per_wall)
    w1_y = rng.uniform(0.1, 2.5, n_per_wall)
    w1 = np.column_stack([w1_x, w1_y, w1_z])

    # Wall 2: x = +2.0
    w2_x = np.full(n_per_wall, 2.0) + rng.normal(0, 0.01, n_per_wall)
    w2_z = rng.uniform(-3.0, 3.0, n_per_wall)
    w2_y = rng.uniform(0.1, 2.5, n_per_wall)
    w2 = np.column_stack([w2_x, w2_y, w2_z])

    # Wall 3: z = -3.0
    w3_x = rng.uniform(-2.0, 2.0, n_per_wall)
    w3_z = np.full(n_per_wall, -3.0) + rng.normal(0, 0.01, n_per_wall)
    w3_y = rng.uniform(0.1, 2.5, n_per_wall)
    w3 = np.column_stack([w3_x, w3_y, w3_z])

    # Wall 4: z = +3.0
    w4_x = rng.uniform(-2.0, 2.0, n_per_wall)
    w4_z = np.full(n_per_wall, 3.0) + rng.normal(0, 0.01, n_per_wall)
    w4_y = rng.uniform(0.1, 2.5, n_per_wall)
    w4 = np.column_stack([w4_x, w4_y, w4_z])

    return np.vstack([floor_pts, w1, w2, w3, w4])


class TestVerifySplatFrame(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="splat_test_")
        self.temp_path = Path(self.temp_dir)

        # Base synthetic scene
        self.pts = generate_synthetic_room_points()
        self.ref_txt = self.temp_path / "points3D.txt"
        write_synthetic_points3d_txt(self.ref_txt, self.pts)

        # Camera loop at y = 1.45 m
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        cam_x = 1.2 * np.cos(theta)
        cam_z = 2.0 * np.sin(theta)
        cam_y = np.full_like(cam_x, 1.45)
        cams = np.column_stack([cam_x, cam_y, cam_z])

        self.tf_json = self.temp_path / "transforms.json"
        write_synthetic_transforms_json(self.tf_json, cams)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_splat_passes(self):
        """A valid aligned splat matching reference points must PASS."""
        valid_ply = self.temp_path / "scene_valid.ply"
        # Add slight jitter to splat points
        splat_pts = self.pts + np.random.default_rng(123).normal(0, 0.01, self.pts.shape)
        write_synthetic_splat_ply(valid_ply, splat_pts)

        rep_path = self.temp_path / "rep_valid.json"
        res = verify_splat_frame(
            scene="synth",
            splat_ply=valid_ply,
            reference_pts_path=self.ref_txt,
            transforms_path=self.tf_json,
            span_tolerance_pct=15.0,
            output_json_path=rep_path,
        )
        self.assertTrue(res, "Valid synthetic splat should PASS verification")
        self.assertTrue(rep_path.is_file(), "Verification report JSON should be created")

        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "PASS")
        self.assertAlmostEqual(data["floor"]["floor_peak_y_m"], 0.0, delta=0.08)

    def test_displaced_floor_fails(self):
        """A splat with floor shifted up to y = 0.80 m must FAIL loudly."""
        displaced_ply = self.temp_path / "scene_displaced_floor.ply"
        pts_bad = self.pts.copy()
        pts_bad[:, 1] += 0.80  # Shift entire scene vertically
        write_synthetic_splat_ply(displaced_ply, pts_bad)

        rep_path = self.temp_path / "rep_displaced.json"
        res = verify_splat_frame(
            scene="synth",
            splat_ply=displaced_ply,
            reference_pts_path=self.ref_txt,
            transforms_path=self.tf_json,
            span_tolerance_pct=15.0,
            output_json_path=rep_path,
        )
        self.assertFalse(res, "Displaced floor must FAIL verification")
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "FAIL")
        # Check that floor failure reason is present
        reasons = " ".join(data.get("failure_reasons", []))
        self.assertTrue("Floor peak" in reasons or "Translation" in reasons or "Y" in reasons,
                        f"Expected floor failure in reasons: {reasons}")

    def test_upside_down_splat_fails(self):
        """An inverted (upside down) splat must FAIL loudly."""
        inverted_ply = self.temp_path / "scene_inverted.ply"
        pts_bad = self.pts.copy()
        pts_bad[:, 1] = -pts_bad[:, 1]  # Invert vertical axis
        write_synthetic_splat_ply(inverted_ply, pts_bad)

        rep_path = self.temp_path / "rep_inverted.json"
        res = verify_splat_frame(
            scene="synth",
            splat_ply=inverted_ply,
            reference_pts_path=self.ref_txt,
            transforms_path=self.tf_json,
            span_tolerance_pct=15.0,
            output_json_path=rep_path,
        )
        self.assertFalse(res, "Inverted splat must FAIL verification")
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "FAIL")

    def test_extents_mismatch_fails(self):
        """A splat with extents scaled by 1.45x (> 15% error) must FAIL loudly with numbers."""
        scaled_ply = self.temp_path / "scene_scaled.ply"
        pts_bad = self.pts.copy()
        pts_bad[:, 0] *= 1.45  # Expand X by 45%
        pts_bad[:, 2] *= 1.45  # Expand Z by 45%
        write_synthetic_splat_ply(scaled_ply, pts_bad)

        rep_path = self.temp_path / "rep_scaled.json"
        res = verify_splat_frame(
            scene="synth",
            splat_ply=scaled_ply,
            reference_pts_path=self.ref_txt,
            transforms_path=self.tf_json,
            span_tolerance_pct=15.0,
            output_json_path=rep_path,
        )
        self.assertFalse(res, "Scaled extents (>15%) must FAIL verification")
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "FAIL")
        reasons = " ".join(data.get("failure_reasons", []))
        self.assertTrue("exceeds" in reasons or "span" in reasons or "Scale" in reasons,
                        f"Expected extents failure message with numbers: {reasons}")


if __name__ == "__main__":
    unittest.main()
