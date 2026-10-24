#!/usr/bin/env python3
"""Unit tests for top-view generation using a synthetic counter-clockwise camera loop.

Tests:
1. UP vector estimation via SVD matches known ground-truth UP within 2 degrees
   under an arbitrary random 3D world rotation.
2. Plotted path has a positive signed area in image coordinates (counter-clockwise).
3. Entire top-view generation script runs headless and outputs valid PNG.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from scripts.view_sparse import (
    build_rotation_to_y_up,
    compute_signed_area,
    compute_turn_sequence,
    estimate_up_svd,
    parse_images,
    parse_points3d,
    qvec2rotmat,
)


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        return np.array([
            0.25 * S,
            (R[2, 1] - R[1, 2]) / S,
            (R[0, 2] - R[2, 0]) / S,
            (R[1, 0] - R[0, 1]) / S,
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([
            (R[2, 1] - R[1, 2]) / S,
            0.25 * S,
            (R[0, 1] + R[1, 0]) / S,
            (R[0, 2] + R[2, 0]) / S,
        ])
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([
            (R[0, 2] - R[2, 0]) / S,
            (R[0, 1] + R[1, 0]) / S,
            0.25 * S,
            (R[1, 2] + R[2, 1]) / S,
        ])
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        return np.array([
            (R[1, 0] - R[0, 1]) / S,
            (R[0, 2] + R[2, 0]) / S,
            (R[1, 2] + R[2, 1]) / S,
            0.25 * S,
        ])


def generate_synthetic_ccw_dataset(
    output_dir: Path,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic COLMAP model in output_dir.

    Creates:
    - Counter-clockwise closed rectangular camera loop in (X, Z) at height Y = 1.5:
      (0, 0) -> (4, 0) -> (4, -4) -> (0, -4) -> (0, 0)
    - Floor points at Y = 0 and wall points.
    - Arbitrary random rotation R_rand in SO(3) applied to the entire world frame.

    Returns:
      (known_up_vector, R_rand, canonical_cam_centers)
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Arbitrary random 3D rotation in SO(3) with det = +1
    H = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R_rand = Q
    assert abs(np.linalg.det(R_rand) - 1.0) < 1e-6

    # Ground-truth canonical UP is +Y
    canonical_up = np.array([0.0, 1.0, 0.0])
    known_up = R_rand @ canonical_up
    known_up /= np.linalg.norm(known_up)

    # 2. Counter-clockwise trajectory in canonical frame (eye level Y = 1.5)
    # Looking down from +Y:
    # (0, 0) -> (4, 0) [East, +X]
    # (4, 0) -> (4, -4) [North, -Z]  (Turn Left)
    # (4, -4) -> (0, -4) [West, -X]  (Turn Left)
    # (0, -4) -> (0, 0) [South, +Z]  (Turn Left)
    corners = [
        np.array([0.0, 1.5, 0.0]),
        np.array([4.0, 1.5, 0.0]),
        np.array([4.0, 1.5, -4.0]),
        np.array([0.0, 1.5, -4.0]),
    ]

    cam_centers = []
    cam_forwards = []
    points_per_edge = 8

    for c_i in range(len(corners)):
        p_start = corners[c_i]
        p_end = corners[(c_i + 1) % len(corners)]
        fwd = (p_end - p_start)
        fwd[1] = 0.0
        fwd /= np.linalg.norm(fwd)

        for step in range(points_per_edge):
            t = step / float(points_per_edge)
            pos = p_start + t * (p_end - p_start)
            cam_centers.append(pos)
            cam_forwards.append(fwd)

    cam_centers = np.array(cam_centers)
    n_cams = len(cam_centers)

    # 3. Create synthetic points: floor (Y = 0) and walls (Y in [0, 2.5])
    pts_list = []
    # Floor grid
    for fx in np.linspace(-1.0, 5.0, 20):
        for fz in np.linspace(-5.0, 1.0, 20):
            pts_list.append([fx, 0.0 + rng.normal(0, 0.01), fz])

    # Wall points
    for py in np.linspace(0.2, 2.4, 8):
        for wx in np.linspace(-0.5, 4.5, 15):
            pts_list.append([wx, py, 0.5])
            pts_list.append([wx, py, -4.5])
        for wz in np.linspace(-4.5, 0.5, 15):
            pts_list.append([-0.5, py, wz])
            pts_list.append([4.5, py, wz])

    pts_canonical = np.array(pts_list)

    # 4. Apply R_rand to world:
    # Points in rotated world: P' = R_rand @ P
    pts_rotated = (R_rand @ pts_canonical.T).T

    # Write cameras.txt
    with open(output_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 1920 1080 1400.0 1400.0 960.0 540.0\n")

    # Write images.txt
    with open(output_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx in range(n_cams):
            c_pos = cam_centers[idx]
            fwd = cam_forwards[idx]

            # In COLMAP camera frame:
            # - up is -Ycam [0, -1, 0]
            # - forward is +Zcam [0, 0, 1]
            # In canonical world:
            # - world up is +Yworld [0, 1, 0] -> -R_cw[:, 1] = [0, 1, 0] => R_cw[:, 1] = [0, -1, 0]
            # - world forward is fwd -> R_cw[:, 2] = fwd
            # - right is R_cw[:, 0] = cross(R_cw[:, 1], R_cw[:, 2])
            r_up = np.array([0.0, -1.0, 0.0])
            r_right = np.cross(r_up, fwd)
            r_right /= np.linalg.norm(r_right)
            R_cw_canonical = np.column_stack([r_right, r_up, fwd])

            # In rotated world:
            C_rot = R_rand @ c_pos
            R_cw_rot = R_rand @ R_cw_canonical
            R_wc_rot = R_cw_rot.T
            t_rot = -R_wc_rot @ C_rot

            qvec = rotmat2qvec(R_wc_rot)
            img_id = idx + 1
            name = f"frame_{img_id:04d}.jpg"

            f.write(f"{img_id} {qvec[0]:.9f} {qvec[1]:.9f} {qvec[2]:.9f} {qvec[3]:.9f} "
                    f"{t_rot[0]:.9f} {t_rot[1]:.9f} {t_rot[2]:.9f} 1 {name}\n")
            f.write("0.0 0.0 1\n")

    # Write points3D.txt
    with open(output_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p_id, pt in enumerate(pts_rotated, start=1):
            f.write(f"{p_id} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} 180 180 180 0.5 1 0 2 0 3 0\n")

    return known_up, R_rand, cam_centers


class TestTopView(unittest.TestCase):
    """Test suite for top-view generation using synthetic CCW trajectory."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="namma_test_topview_"))
        cls.model_dir = cls.temp_dir / "sparse" / "0"
        cls.known_up, cls.R_rand, cls.canonical_centers = generate_synthetic_ccw_dataset(
            cls.model_dir, seed=123
        )

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.is_dir():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_up_vector_estimation_within_two_degrees(self):
        """Estimate UP via SVD of camera up vectors and verify it matches known UP within 2.0 degrees."""
        poses = parse_images(self.model_dir)
        self.assertGreater(len(poses), 10, "Should load synthetic camera poses.")

        up_est = estimate_up_svd(poses)

        # Dot product with known UP
        dot = float(np.clip(np.dot(up_est, self.known_up), -1.0, 1.0))
        angle_err_deg = float(np.degrees(np.arccos(dot)))

        print(f"\n[Test] Known UP:    [{self.known_up[0]:.4f}, {self.known_up[1]:.4f}, {self.known_up[2]:.4f}]")
        print(f"[Test] Detected UP: [{up_est[0]:.4f}, {up_est[1]:.4f}, {up_est[2]:.4f}]")
        print(f"[Test] Angular error: {angle_err_deg:.4f} degrees")

        self.assertLess(
            angle_err_deg, 2.0,
            f"Detected UP angle error {angle_err_deg:.3f} deg exceeds 2.0 deg threshold!"
        )

    def test_signed_area_positive_for_ccw_loop(self):
        """Assert that the plotted path has positive signed area in image coordinates (CCW)."""
        poses = parse_images(self.model_dir)
        up_est = estimate_up_svd(poses)

        # Build proper rotation matrix to +Y
        R_align = build_rotation_to_y_up(up_est)
        det_R = float(np.linalg.det(R_align))
        self.assertAlmostEqual(det_R, 1.0, places=4, msg="Alignment matrix must have det = +1")

        # Transform camera centers
        cam_centers = np.array([R_align @ p["C"] for p in poses])
        cam_xz = cam_centers[:, [0, 2]]

        # Compute signed area
        signed_area = compute_signed_area(cam_xz)
        print(f"[Test] CCW loop signed area: {signed_area:.4f}")

        self.assertGreater(
            signed_area, 0.0,
            f"Counter-clockwise camera loop must have positive signed area, got {signed_area}"
        )

        # Verify turn sequence (RDP 0.5m) is predominantly left turns
        turns = compute_turn_sequence(cam_xz, epsilon=0.5)
        print(f"[Test] CCW loop turn sequence: {turns}")
        self.assertTrue(len(turns) >= 3, "Expected at least 3 corner turns for rectangle loop.")
        self.assertTrue(all(t == "L" for t in turns), f"Expected all Left turns for CCW loop, got {turns}")

    def test_script_runs_headless(self):
        """Execute view_sparse.py on synthetic model and verify headless execution and output."""
        output_png = self.temp_dir / "synth_top_view.png"
        script_path = _REPO_ROOT / "scripts" / "view_sparse.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--model-dir", str(self.model_dir),
            "--output", str(output_png),
            "--debug",
        ]

        # Execute headless in isolated process
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

        res = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            env=env,
        )

        if res.returncode != 0:
            print("STDERR:\n", res.stderr)
            print("STDOUT:\n", res.stdout)

        self.assertEqual(res.returncode, 0, "view_sparse.py should exit with return code 0.")
        self.assertTrue(output_png.is_file(), f"Output image {output_png} must exist.")
        self.assertGreater(output_png.stat().st_size, 10000, "Generated PNG must not be empty.")

        # Check stdout contents
        self.assertIn("Signed area sign: +", res.stdout)
        self.assertIn("Turn sequence", res.stdout)
        self.assertIn("Start-end gap", res.stdout)
        self.assertIn("Model count", res.stdout)


if __name__ == "__main__":
    unittest.main()
