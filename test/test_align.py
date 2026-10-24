#!/usr/bin/env python3
"""
test/test_align.py  --  Unit Tests for Phase 4 Scene Alignment.

Tests alignment on a SYNTHETIC room:
- Flat floor at y = 0
- Four vertical walls
- Camera loop at 1.45 m height, counter-clockwise from above
- Arbitrary random rotation (SO(3), det = +1), random translation, and random scale (0.3x to 3.0x)

Assertions:
1. Floor at y = 0 within 1 cm (0.01 m)
2. Recovered UP = +Y within 1 degree
3. Recovered camera height within 1% of 1.45 m
4. Determinant of rotation matrix = +1.0
5. Trajectory loop is counter-clockwise in top.png coordinates (positive signed area)
6. Real render test: Render top.png, transform cameras via ax.transData.transform,
   assert shoelace area in DISPLAY coordinates is positive (counter-clockwise on screen),
   assert first leg points UP the image, and assert ylim[0] > ylim[1].
All tests write to temporary directory, never to data/.
"""

import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Ensure repository root on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.align_scene import (
    align_scene,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
    camera_centre,
    frame_number,
    rotmat_to_quat,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
)
from scripts.check_alignment import (
    check_alignment,
    compute_signed_area,
    compute_turn_sequence_and_corners,
    plot_top_view,
)


def create_synthetic_room_dataset(
    output_dir: Path,
    seed: int = 1234,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Generate synthetic room COLMAP model with known ground truth:
    Floor at y=0, walls, cameras at 1.45m in counter-clockwise loop.
    Applies random scale s in [0.3, 3.0], random translation t, and random SO(3) rotation R.

    Returns:
      (ground_truth_cam_height, R_rand, t_rand, s_rand)
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Canonical geometry
    target_cam_h = 1.45  # metres

    # Floor points at y = 0 (dense grid in [-3, 3])
    floor_pts = []
    for x in np.linspace(-3.0, 3.0, 35):
        for z in np.linspace(-3.0, 3.0, 35):
            floor_pts.append([x, 0.0 + rng.normal(0, 0.002), z])

    # Four walls: x = +/-3, z = +/-3, y in [0.1, 2.8]
    wall_pts = []
    for y in np.linspace(0.1, 2.8, 12):
        for w in np.linspace(-3.0, 3.0, 25):
            wall_pts.append([w, y, -3.0])  # North wall
            wall_pts.append([w, y, 3.0])   # South wall
            wall_pts.append([-3.0, y, w])  # West wall
            wall_pts.append([3.0, y, w])   # East wall

    pts_canon = np.vstack([np.array(floor_pts), np.array(wall_pts)])

    # Camera loop at y = 1.45 m:
    # Closed CCW rectangular loop in (X, Z):
    # Looking down from +Y:
    # Corner 0: (0, 1.8) [South] -> Corner 1: (0, -1.8) [North, heading -Z, points UP top-view]
    # Corner 2: (-1.8, -1.8) [West, turns Left]
    # Corner 3: (-1.8, 1.8)  [South, turns Left]
    # Corner 4: (1.8, 1.8)   [East, turns Left]
    # Corner 5: (1.8, -1.8)  [North, turns Left]
    # Corner 6: (0, -1.8)
    corners = [
        np.array([0.0, target_cam_h, 1.8]),
        np.array([0.0, target_cam_h, -1.8]),
        np.array([-1.8, target_cam_h, -1.8]),
        np.array([-1.8, target_cam_h, 1.8]),
        np.array([1.8, target_cam_h, 1.8]),
        np.array([1.8, target_cam_h, -1.8]),
        np.array([0.0, target_cam_h, -1.8]),
    ]

    cam_centers = []
    cam_forwards = []
    steps_per_edge = 8

    for c_idx in range(len(corners) - 1):
        p0 = corners[c_idx]
        p1 = corners[c_idx + 1]
        fwd = p1 - p0
        fwd[1] = 0.0
        norm_fwd = np.linalg.norm(fwd)
        if norm_fwd < 1e-6:
            continue
        fwd /= norm_fwd
        for s_idx in range(steps_per_edge):
            t = s_idx / float(steps_per_edge)
            pos = p0 + t * (p1 - p0)
            cam_centers.append(pos)
            cam_forwards.append(fwd)

    cam_centers = np.array(cam_centers)
    n_cams = len(cam_centers)

    # 2. Random rotation in SO(3) with det = +1.0
    H = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R_rand = Q
    assert abs(np.linalg.det(R_rand) - 1.0) < 1e-6

    # Random scale in [0.3, 3.0]
    s_rand = float(rng.uniform(0.3, 3.0))

    # Random translation
    t_rand = rng.uniform(-5.0, 5.0, size=(3,))

    # 3. Transform geometry into random world frame:
    pts_world = s_rand * (pts_canon @ R_rand.T) + t_rand

    # Cameras in world frame
    cameras_dict = {
        1: {
            "model": "PINHOLE",
            "width": 1920,
            "height": 1080,
            "params": [1450.0, 1450.0, 960.0, 540.0],
        }
    }

    images_dict = {}
    for idx in range(n_cams):
        c_pos = cam_centers[idx]
        fwd = cam_forwards[idx]

        r_up = np.array([0.0, -1.0, 0.0])
        r_right = np.cross(r_up, fwd)
        r_right /= np.linalg.norm(r_right)
        R_cw_canon = np.column_stack([r_right, r_up, fwd])

        C_world = s_rand * (R_rand @ c_pos) + t_rand
        R_cw_world = R_rand @ R_cw_canon
        R_wc_world = R_cw_world.T
        t_wc_world = -R_wc_world @ C_world

        qw, qx, qy, qz = rotmat_to_quat(R_wc_world)
        img_id = idx + 1
        name = f"video1/frame_{img_id:06d}.jpg"

        images_dict[img_id] = {
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "tx": t_wc_world[0], "ty": t_wc_world[1], "tz": t_wc_world[2],
            "camera_id": 1,
            "name": name,
            "point2d_str": "0.0 0.0 1",
        }

    points_dict = {}
    for p_idx, pt in enumerate(pts_world, start=1):
        points_dict[p_idx] = {
            "xyz": pt,
            "rgb": (200, 200, 200),
            "error": 0.5,
            # Track length >= 4 for robust point filtering
            "track": "1 0 2 0 3 0 4 0",
        }

    write_cameras_txt(cameras_dict, output_dir / "cameras.txt")
    write_images_txt(images_dict, output_dir / "images.txt")
    write_points3d_txt(points_dict, output_dir / "points3D.txt")

    return target_cam_h, R_rand, t_rand, s_rand


class TestSceneAlignment(unittest.TestCase):
    """Test Phase 4 alignment on a synthetic room with random rigid transform."""

    @classmethod
    def setUpClass(cls):
        # Clean any leftover data/synth_scene before tests start
        stray_dir = _REPO_ROOT / "data" / "synth_scene"
        if stray_dir.is_dir():
            shutil.rmtree(stray_dir, ignore_errors=True)

        cls.temp_dir = Path(tempfile.mkdtemp(prefix="namma_test_align_"))
        cls.input_model_dir = cls.temp_dir / "synth_sparse"
        cls.data_dir = cls.temp_dir / "data" / "synth_scene"
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.aligned_dir = cls.data_dir / "colmap_aligned"
        cls.outputs_dir = cls.temp_dir / "outputs" / "synth_scene"
        cls.outputs_dir.mkdir(parents=True, exist_ok=True)

        cls.config_file = cls.temp_dir / "test_config.yaml"
        test_cfg = {
            "align": {
                "input_model": str(cls.input_model_dir),
                "eye_level_height_m": 1.45,
                "chest_level_height_m": 1.15,
                "max_floor_angle_deg": 8.0,
                "min_inlier_ratio": 0.015,
                "max_planes": 6,
                "dist_thresh": 0.08,
                "ceiling_default_m": 3.0,
            },
            "alignment": {
                "ceiling_height_m": 3.0,
            }
        }
        with open(cls.config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(test_cfg, f)

        # Generate synthetic dataset with random pose and scale
        cls.gt_cam_h, cls.R_rand, cls.t_rand, cls.s_rand = create_synthetic_room_dataset(
            cls.input_model_dir, seed=42
        )

        print(f"\n[Synthetic Test Setup]")
        print(f"  Random Scale       : {cls.s_rand:.4f}x (between 0.3x and 3.0x)")
        print(f"  Random Translation : [{cls.t_rand[0]:.2f}, {cls.t_rand[1]:.2f}, {cls.t_rand[2]:.2f}]")
        print(f"  Random Rotation det: {np.linalg.det(cls.R_rand):.6f}")

        # Run align_scene entirely inside temporary directories
        cls.alignment_result = align_scene(
            scene="synth_scene",
            config_path=str(cls.config_file),
            input_model=str(cls.input_model_dir),
            output_dir=str(cls.aligned_dir),
            data_dir=str(cls.data_dir),
        )

        # Load aligned model
        cls.aligned_cameras = parse_cameras_txt(cls.aligned_dir / "cameras.txt")
        cls.aligned_images = parse_images_txt(cls.aligned_dir / "images.txt")
        cls.aligned_points = parse_points3d_txt(cls.aligned_dir / "points3D.txt")

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.is_dir():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
        stray_dir = _REPO_ROOT / "data" / "synth_scene"
        if stray_dir.is_dir():
            shutil.rmtree(stray_dir, ignore_errors=True)

    def test_no_writes_to_repo_data(self):
        """Assertion: Tests must never write to data/."""
        stray_dir = _REPO_ROOT / "data" / "synth_scene"
        self.assertFalse(stray_dir.exists(), "Tests must write only to temporary directory, never to data/!")

    def test_floor_at_y0_within_one_centimeter(self):
        """Assertion 1: Floor plane must be at y = 0 within 1 cm (0.01 m)."""
        pts_xyz = np.array([p["xyz"] for p in self.aligned_points.values()])
        lowest_pts = pts_xyz[pts_xyz[:, 1] < 0.20]
        floor_y_median = float(np.median(lowest_pts[:, 1]))

        print(f"\n[Test 1] Floor y level: {floor_y_median:.5f} m (Error: {abs(floor_y_median):.5f} m)")
        self.assertLessEqual(
            abs(floor_y_median), 0.01,
            f"Floor level |y| = {abs(floor_y_median):.4f} m exceeds 1 cm (0.01 m) tolerance!"
        )

    def test_up_vector_equals_plus_y_within_one_degree(self):
        """Assertion 2: Detected UP vector must equal canonical +Y [0, 1, 0] within 1 degree."""
        R_total = np.array(self.alignment_result["R"])
        up_orig = np.array(self.alignment_result["up_vector_detected"])
        up_rotated = R_total @ up_orig
        up_rotated /= np.linalg.norm(up_rotated)

        dot = float(np.clip(np.dot(up_rotated, np.array([0.0, 1.0, 0.0])), -1.0, 1.0))
        angle_err_deg = float(np.degrees(np.arccos(dot)))

        print(f"\n[Test 2] Rotated UP: [{up_rotated[0]:.4f}, {up_rotated[1]:.4f}, {up_rotated[2]:.4f}]")
        print(f"         Angle to +Y: {angle_err_deg:.4f} degrees")
        self.assertLessEqual(
            angle_err_deg, 1.0,
            f"UP vector angle to +Y = {angle_err_deg:.3f}° exceeds 1.0° tolerance!"
        )

    def test_recovered_camera_height_within_one_percent(self):
        """Assertion 3: Recovered mean camera height must match ground truth 1.45 m within 1%."""
        cam_heights = [camera_centre(im)[1] for im in self.aligned_images.values()]
        mean_h = float(np.mean(cam_heights))
        rel_error = abs(mean_h - self.gt_cam_h) / self.gt_cam_h

        print(f"\n[Test 3] Recovered camera height: {mean_h:.4f} m (Target: {self.gt_cam_h:.2f} m, Error: {rel_error*100:.3f}%)")
        self.assertLessEqual(
            rel_error, 0.01,
            f"Camera height relative error {rel_error*100:.2f}% exceeds 1.0% tolerance!"
        )

    def test_rotation_determinant_equals_plus_one(self):
        """Assertion 4: Transform rotation matrix must be a proper rotation with det(R) = +1.0."""
        R = np.array(self.alignment_result["R"])
        det_R = float(np.linalg.det(R))

        print(f"\n[Test 4] Alignment rotation determinant: {det_R:.8f}")
        self.assertAlmostEqual(
            det_R, 1.0, places=4,
            msg=f"Rotation matrix is not a proper rotation! det(R) = {det_R}"
        )

    def test_counter_clockwise_loop_positive_signed_area(self):
        """Assertion 5: Trajectory loop must be counter-clockwise in top-down coordinates."""
        sorted_imgs = sorted(self.aligned_images.values(), key=lambda im: frame_number(im["name"]))
        cam_xz = np.array([camera_centre(im)[[0, 2]] for im in sorted_imgs])

        signed_area = compute_signed_area(cam_xz)
        print(f"\n[Test 5] Loop signed area: {signed_area:.4f}")

        self.assertGreater(
            signed_area, 0.0,
            f"Counter-clockwise trajectory must have positive signed area in top-down coordinates, got {signed_area}!"
        )

        turns, corner_count, _ = compute_turn_sequence_and_corners(cam_xz, smooth_window=5, rdp_epsilon=1.0, min_leg_length=1.0)
        print(f"         Turn sequence: {turns} (Corners: {corner_count})")
        self.assertTrue(all(t == "L" for t in turns), f"Expected all Left turns for CCW loop, got {turns}")

    def test_real_render_display_shoelace_area_and_first_leg_up(self):
        """
        Assertion 6 (Real Render Test):
        1. Render top.png for the synthetic COUNTER-CLOCKWISE loop.
        2. Assert ylim[0] > ylim[1] (vertical axis is inverted so Z points down the image).
        3. Transform camera positions with ax.transData.transform.
        4. Assert shoelace area of the loop in DISPLAY coordinates is positive (counter-clockwise on screen).
        5. Assert the first leg of the path points UP the image (display y increases).
        """
        # Run check_alignment to produce all verification outputs in temporary output directory
        rep = check_alignment(
            scene="synth_scene",
            config_path=str(self.config_file),
            data_dir=str(self.data_dir),
            outputs_dir=str(self.outputs_dir),
        )
        self.assertTrue(Path(rep["top_png"]).is_file(), "top.png must be created by check_alignment")

        # Now perform direct render inspection with ax.transData.transform
        sorted_imgs = sorted(self.aligned_images.values(), key=lambda im: frame_number(im["name"]))
        cam_centers = np.array([camera_centre(im) for im in sorted_imgs])
        pts_xyz = np.array([p["xyz"] for p in self.aligned_points.values()])

        fig, ax = plot_top_view(
            cam_centers=cam_centers,
            pts_top=pts_xyz,
            x_min=-4.0, x_max=4.0,
            z_min=-4.0, z_max=4.0,
            header_line1="Synthetic Render Test",
            header_line2="CCW Loop Verification",
            start_end_gap=0.0,
        )
        fig.canvas.draw()

        # Check that Z axis is inverted: bottom > top
        ylim = ax.get_ylim()
        print(f"\n[Test 6] Rendered top view ylim: (bottom={ylim[0]:.2f}, top={ylim[1]:.2f})")
        self.assertGreater(
            ylim[0], ylim[1],
            f"top.png vertical axis must have ylim[0] > ylim[1] so +Z points down the image, got {ylim}"
        )

        # Transform camera positions to DISPLAY coordinates
        cam_xz = cam_centers[:, [0, 2]]
        cam_disp = ax.transData.transform(cam_xz)

        # Compute shoelace area of loop in DISPLAY coordinates
        x_disp = cam_disp[:, 0]
        y_disp = cam_disp[:, 1]
        shoelace_disp = 0.5 * float(np.sum(x_disp * np.roll(y_disp, -1) - np.roll(x_disp, -1) * y_disp))

        print(f"         Shoelace area in DISPLAY coordinates: {shoelace_disp:+.2f}")
        self.assertGreater(
            shoelace_disp, 0.0,
            f"Shoelace area in display coordinates must be positive (counter-clockwise on screen), got {shoelace_disp}"
        )

        # First leg check: corner 0 -> corner 1 (first 8 steps)
        # In display coordinates, pointing UP the image means display y increases
        step_edge = 8
        leg1_start_y = cam_disp[0, 1]
        leg1_end_y = cam_disp[step_edge, 1]

        print(f"         First leg display Y: start={leg1_start_y:.2f}, end={leg1_end_y:.2f} (diff={leg1_end_y - leg1_start_y:+.2f})")
        self.assertGreater(
            leg1_end_y, leg1_start_y,
            f"First leg must point UP the image (display y must increase): start={leg1_start_y}, end={leg1_end_y}"
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
