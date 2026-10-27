#!/usr/bin/env python3
"""
test/test_occupancy.py  --  Unit tests for 3D voxel ray-carving occupancy grid.

Tests the core principles on a synthetic room:
1. A wall (dense occupied evidence) is BLOCKED.
2. A 0.9 m aisle between two chair blocks is WALKABLE.
3. A floater in free space seen by multiple cameras is REMOVED (overridden by free space).
4. An unobserved pocket is UNKNOWN and NOT WALKABLE.
5. A 0.35 m gap between two obstacle pillars (pillar 1 at x in [0.80, 1.00], pillar 2 at
   x in [1.35, 1.55]; gap 0.35 m centered at x = 1.175) with inflation 0.07 m and player
   radius 0.08 m (total 0.15 m each side = 0.30 m clearance) is PASSABLE (data == 0, walkable == 1).
"""

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from pipeline.build_occupancy import build_occupancy_grid


class TestOccupancyGrid(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_synthetic_room_occupancy(self):
        """
        Synthesizes a 5.0m x 5.0m room with:
        - Wall at x in [0.0, 0.4] m, z in [0.0, 5.0] m
        - Chair Block 1 at x in [1.2, 1.8] m, z in [1.5, 3.5] m
        - Chair Block 2 at x in [2.7, 3.3] m, z in [1.5, 3.5] m
          -> Aisle between chair blocks: x in (1.8, 2.7) m (width = 0.90 m)
        - Floater at (2.25, 0.80, 4.20) m in free space
        - Unobserved pocket at x in [4.2, 5.0] m, z in [4.2, 5.0] m
        - Obstacle Pillar 1 at x in [0.80, 1.00] m, z in [0.90, 1.10] m
        - Obstacle Pillar 2 at x in [1.35, 1.55] m, z in [0.90, 1.10] m
          -> Passable gap between pillars: x in (1.00, 1.35) m (width = 0.35 m, centered at x = 1.175 m)
        """
        cell_size = 0.10
        min_x, max_x = 0.0, 5.0
        min_z, max_z = 0.0, 5.0
        width = int(round((max_x - min_x) / cell_size))   # 50
        height = int(round((max_z - min_z) / cell_size))  # 50

        # Cameras walking along aisle from z=0.4 to z=4.6 at x=2.25 (cams 1..15)
        cams_list = []
        for z_val in np.linspace(0.40, 4.60, 15):
            # Alternating eye (1.45m) and chest (1.02m) heights
            h = 1.45 if len(cams_list) % 2 == 0 else 1.02
            cams_list.append(np.array([2.25, h, z_val]))

        # Branch cameras walking along x=1.175 visiting through the 0.35m gap (cams 16..20)
        for z_branch in [0.40, 0.70, 1.00, 1.30, 1.60]:
            h = 1.45 if len(cams_list) % 2 == 0 else 1.02
            cams_list.append(np.array([1.175, h, z_branch]))

        # Connector path along z=0.40 from aisle x=2.25 to branch x=1.175 (cams 21..31)
        for x_conn in np.linspace(1.20, 2.20, 11):
            h = 1.45 if len(cams_list) % 2 == 0 else 1.02
            cams_list.append(np.array([x_conn, h, 0.40]))

        cam_data = {}
        for idx, c in enumerate(cams_list, 1):
            cam_data[idx] = {"center": c, "name": f"frame_{idx:04d}.jpg"}

        # Generate SfM Points
        points_data = {}
        pt_id = 1

        # 1. Wall points (dense at x in [0.1, 0.3], z along full wall, y in [0.3, 1.3])
        # Aisle cameras see unoccluded wall; pillar region z in [0.7, 1.3] is occluded by pillars
        for z_w in np.linspace(0.1, 4.9, 49):
            if 0.7 <= z_w <= 1.3:
                continue
            for y_w in [0.4, 0.7, 1.0]:
                points_data[pt_id] = {
                    "xyz": np.array([0.20, y_w, z_w]),
                    "error": 0.5,
                    "track_len": 4,
                    "image_ids": [1, 2, 3, 4],
                }
                pt_id += 1

        # 2. Chair 1 points (x in [1.3, 1.7], z in [1.7, 3.3])
        for z_c in np.linspace(1.7, 3.3, 8):
            for y_c in [0.4, 0.7, 0.9]:
                points_data[pt_id] = {
                    "xyz": np.array([1.50, y_c, z_c]),
                    "error": 0.6,
                    "track_len": 3,
                    "image_ids": [2, 3, 4],
                }
                pt_id += 1

        # 3. Chair 2 points (x in [2.8, 3.2], z in [1.7, 3.3])
        for z_c in np.linspace(1.7, 3.3, 8):
            for y_c in [0.4, 0.7, 0.9]:
                points_data[pt_id] = {
                    "xyz": np.array([3.00, y_c, z_c]),
                    "error": 0.6,
                    "track_len": 3,
                    "image_ids": [2, 3, 4],
                }
                pt_id += 1

        # 4. Target points at both ends of aisle observed along the aisle
        for y_t in np.linspace(0.3, 1.3, 10):
            for x_t in np.linspace(1.9, 2.6, 5):
                points_data[pt_id] = {
                    "xyz": np.array([x_t, y_t, 4.80]),
                    "error": 0.4,
                    "track_len": 10,
                    "image_ids": list(range(1, 11)),
                }
                pt_id += 1
                points_data[pt_id] = {
                    "xyz": np.array([x_t, y_t, 0.10]),
                    "error": 0.4,
                    "track_len": 10,
                    "image_ids": list(range(6, 16)),
                }
                pt_id += 1

        # 5. Floater point at (2.25, 0.80, 4.20) in free space
        floater_pt_id = pt_id
        points_data[floater_pt_id] = {
            "xyz": np.array([2.25, 0.80, 4.20]),
            "error": 0.5,
            "track_len": 3,
            "image_ids": [10, 11, 12],
        }
        pt_id += 1

        # 6. Obstacle Pillar 1 points: x in [0.80, 1.00], z in [0.90, 1.10]
        for y_p in [0.4, 0.7, 1.0]:
            for x_p in [0.85, 0.95]:
                points_data[pt_id] = {
                    "xyz": np.array([x_p, y_p, 1.00]),
                    "error": 0.5,
                    "track_len": 3,
                    "image_ids": [16, 17, 18],
                }
                pt_id += 1

        # 7. Obstacle Pillar 2 points: x in [1.35, 1.55], z in [0.90, 1.10]
        for y_p in [0.4, 0.7, 1.0]:
            for x_p in [1.40, 1.50]:
                points_data[pt_id] = {
                    "xyz": np.array([x_p, y_p, 1.00]),
                    "error": 0.5,
                    "track_len": 3,
                    "image_ids": [18, 19, 20],
                }
                pt_id += 1

        # 8. Targets observed through gap along x=1.175 by branch cameras 16..20
        branch_cam_ids = [16, 17, 18, 19, 20]
        for z_tgt in [0.10, 0.40, 0.70, 1.30, 1.60, 2.00, 3.00, 4.80]:
            for y_t in np.linspace(0.25, 1.35, 12):
                points_data[pt_id] = {
                    "xyz": np.array([1.175, y_t, z_tgt]),
                    "error": 0.3,
                    "track_len": 5,
                    "image_ids": branch_cam_ids,
                }
                pt_id += 1

        # 9. Targets along connector path (z=0.40): observed by connector cameras
        conn_cam_ids = list(range(21, len(cams_list) + 1)) + [1, 16]
        for y_t in np.linspace(0.3, 1.3, 10):
            for x_t in np.linspace(0.20, 2.25, 10):
                points_data[pt_id] = {
                    "xyz": np.array([x_t, y_t, 0.40]),
                    "error": 0.3,
                    "track_len": len(conn_cam_ids),
                    "image_ids": conn_cam_ids,
                }
                pt_id += 1

        # Generate Clean Splats
        splat_xyz_list = []
        splat_op_list = []
        splat_scale_list = []

        # Wall splats
        for z_w in np.linspace(0.1, 4.9, 49):
            for y_w in np.linspace(0.25, 1.35, 10):
                splat_xyz_list.append([0.20, y_w, z_w])
                splat_op_list.append(0.85)
                splat_scale_list.append(0.08)

        # Chair 1 splats (x in [1.2, 1.8])
        for z_c in np.linspace(1.5, 3.5, 15):
            for y_c in np.linspace(0.3, 1.1, 8):
                splat_xyz_list.append([1.50, y_c, z_c])
                splat_op_list.append(0.80)
                splat_scale_list.append(0.08)

        # Chair 2 splats (x in [2.7, 3.3])
        for z_c in np.linspace(1.5, 3.5, 15):
            for y_c in np.linspace(0.3, 1.1, 8):
                splat_xyz_list.append([3.00, y_c, z_c])
                splat_op_list.append(0.80)
                splat_scale_list.append(0.08)

        # Floater splat in free space
        splat_xyz_list.append([2.25, 0.80, 4.20])
        splat_op_list.append(0.75)
        splat_scale_list.append(0.08)
        splat_xyz_list.append([2.25, 0.85, 4.20])
        splat_op_list.append(0.75)
        splat_scale_list.append(0.08)

        # Pillar 1 splats (x in [0.80, 1.00])
        for y_p in np.linspace(0.3, 1.1, 8):
            for x_p in [0.85, 0.95]:
                splat_xyz_list.append([x_p, y_p, 1.00])
                splat_op_list.append(0.85)
                splat_scale_list.append(0.08)

        # Pillar 2 splats (x in [1.35, 1.55])
        for y_p in np.linspace(0.3, 1.1, 8):
            for x_p in [1.40, 1.50]:
                splat_xyz_list.append([x_p, y_p, 1.00])
                splat_op_list.append(0.85)
                splat_scale_list.append(0.08)

        clean_xyz = np.array(splat_xyz_list, dtype=np.float64)
        clean_sig_op = np.array(splat_op_list, dtype=np.float64)
        clean_scale = np.array(splat_scale_list, dtype=np.float64)

        # Mock aligned_dir and existing_occ
        aligned_dir = self.test_dir / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)

        # Config with defaults (inflation 0.07 m and player radius 0.08 m)
        cfg = {
            "occupancy": {
                "slab_y_min_m": 0.20,
                "slab_y_max_m": 1.40,
                "voxel_size_m": 0.10,
                "occ_sfm_min_track": 3,
                "occ_sfm_max_error": 1.0,
                "occ_splat_min_opacity": 0.5,
                "occ_splat_max_scale": 0.15,
                "occ_min_support": 2,
                "free_min_cameras": 2,
                "free_ray_step_m": 0.05,
                "free_ray_stop_before_point_m": 0.15,
                "collapse_free_slab_ratio": 0.40,
                "remove_isolated_occupied": True,
                "inflation_m": 0.07,
                "player_radius_m": 0.08,
                "unknown_ok_radius_m": 0.30,
            }
        }

        # Create mock COLMAP text files in aligned_dir
        with open(aligned_dir / "cameras.txt", "w") as f:
            f.write("1 SIMPLE_RADIAL 1920 1080 1440 960 540 0.0\n")
        with open(aligned_dir / "images.txt", "w") as f:
            for idx, c in cam_data.items():
                f.write(f"{idx} 1.0 0.0 0.0 0.0 {-c['center'][0]} {-c['center'][1]} {-c['center'][2]} 1 {c['name']}\n\n")
        with open(aligned_dir / "points3D.txt", "w") as f:
            for pid, p in points_data.items():
                track_str = " ".join(f"{im} 1" for im in p["image_ids"])
                f.write(f"{pid} {p['xyz'][0]} {p['xyz'][1]} {p['xyz'][2]} 255 255 255 {p['error']} {track_str}\n")

        # Create synthetic existing occupancy.json for extents
        existing_occ_path = self.test_dir / "existing_occ.json"
        with open(existing_occ_path, "w") as f:
            json.dump({
                "origin": [min_x, min_z],
                "cell_size": cell_size,
                "width": width,
                "height": height,
                "data": [0] * (width * height),
                "walkable": [0] * (width * height),
            }, f)

        res = build_occupancy_grid(
            aligned_dir=aligned_dir,
            splat_ply_path=None,
            scene="synthetic_test",
            cfg=cfg,
            existing_occ_path=existing_occ_path,
            clean_xyz=clean_xyz,
            clean_sig_op=clean_sig_op,
            clean_scale=clean_scale,
        )

        data = np.array(res["data"]).reshape(height, width)
        walkable = np.array(res["walkable"]).reshape(height, width)
        state = np.array(res["state"]).reshape(height, width)

        # -------------------------------------------------------------
        # Assertion 1: The 0.9 m aisle between chair blocks is WALKABLE
        # -------------------------------------------------------------
        # Center of aisle is x = 2.25 m (col 22)
        aisle_col = int(round((2.25 - min_x) / cell_size))
        # Samples along z in [2.0, 3.0] m (rows 20 to 30)
        aisle_rows = [int(round((z - min_z) / cell_size)) for z in [2.0, 2.5, 3.0]]
        for r in aisle_rows:
            self.assertEqual(
                walkable[r, aisle_col], 1,
                f"Aisle cell at row={r}, col={aisle_col} (x=2.25, z={r*cell_size:.2f}) must be WALKABLE (1)",
            )
            self.assertEqual(
                data[r, aisle_col], 0,
                f"Aisle cell at row={r}, col={aisle_col} must be unblocked data=0",
            )

        # -------------------------------------------------------------
        # Assertion 2: The wall is BLOCKED and NOT walkable
        # -------------------------------------------------------------
        wall_col = int(round((0.20 - min_x) / cell_size))  # col 2
        for r in [20, 30, 40]:
            self.assertEqual(
                data[r, wall_col], 1,
                f"Wall cell at row={r}, col={wall_col} must be BLOCKED (data=1)",
            )
            self.assertEqual(
                walkable[r, wall_col], 0,
                f"Wall cell at row={r}, col={wall_col} must NOT be walkable (0)",
            )
            self.assertEqual(
                state[r, wall_col], 2,
                f"Wall cell at row={r}, col={wall_col} must be OCCUPIED state=2",
            )

        # -------------------------------------------------------------
        # Assertion 3: The floater is REMOVED by ray carving
        # -------------------------------------------------------------
        # Floater was at x=2.25, z=4.20. Cameras 1, 2, 3, 4 saw target at z=4.80 right through it.
        floater_col = int(round((2.25 - min_x) / cell_size))
        floater_row = int(round((4.20 - min_z) / cell_size))

        self.assertEqual(
            data[floater_row, floater_col], 0,
            f"Floater cell at row={floater_row}, col={floater_col} must be unblocked (data=0)",
        )
        self.assertEqual(
            walkable[floater_row, floater_col], 1,
            f"Floater cell at row={floater_row}, col={floater_col} must be WALKABLE (1) after removal",
        )
        self.assertEqual(
            state[floater_row, floater_col], 1,
            f"Floater cell at row={floater_row}, col={floater_col} must be KNOWN FREE (state=1)",
        )

        # -------------------------------------------------------------
        # Assertion 4: Unobserved pocket is UNKNOWN and NOT WALKABLE
        # -------------------------------------------------------------
        # Corner pocket at x in [4.5, 4.8], z in [4.5, 4.8]
        pocket_col = int(round((4.60 - min_x) / cell_size))
        pocket_row = int(round((4.60 - min_z) / cell_size))

        self.assertEqual(
            state[pocket_row, pocket_col], 0,
            f"Unobserved pocket cell at row={pocket_row}, col={pocket_col} must be UNKNOWN (state=0)",
        )
        self.assertEqual(
            data[pocket_row, pocket_col], 1,
            f"Unobserved pocket cell at row={pocket_row}, col={pocket_col} must be BLOCKED (data=1)",
        )
        self.assertEqual(
            walkable[pocket_row, pocket_col], 0,
            f"Unobserved pocket cell at row={pocket_row}, col={pocket_col} must NOT be walkable (0)",
        )

        # -------------------------------------------------------------
        # Assertion 5: 0.35 m gap between obstacle pillars IS PASSABLE
        # -------------------------------------------------------------
        # Pillar 1 at x in [0.80, 1.00], Pillar 2 at x in [1.35, 1.55]
        # Gap is 0.35 m centered at x = 1.175 m (col 11) at z = 1.00 m (row 10)
        gap_col = int(math.floor((1.175 - min_x) / cell_size))  # col 11
        gap_row = int(round((1.000 - min_z) / cell_size))       # row 10

        self.assertEqual(
            data[gap_row, gap_col], 0,
            f"Gap cell at row={gap_row}, col={gap_col} (x=1.175, z=1.00) must be UNBLOCKED (data=0)",
        )
        self.assertEqual(
            walkable[gap_row, gap_col], 1,
            f"Gap cell at row={gap_row}, col={gap_col} must be WALKABLE (1)",
        )
        self.assertEqual(
            state[gap_row, gap_col], 1,
            f"Gap cell at row={gap_row}, col={gap_col} must be KNOWN FREE (state=1)",
        )


if __name__ == "__main__":
    unittest.main()
