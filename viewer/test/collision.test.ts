import { describe, it, expect } from "vitest";
import * as fs from "fs";
import * as path from "path";
import {
  OccupancyGrid,
  worldToCell,
  isBlocked,
  isPointBlocked,
  isPointInUnknownCell,
  isCircleBlocked,
  resolveMovement,
  fixSpawnPosition,
  clampToBounds,
  getDistanceToFree,
} from "../src/collision";

/**
 * Creates a synthetic grid for deterministic collision testing.
 * Origin [0, 0], cellSize = 0.1 m.
 */
function createSyntheticGrid(
  width: number,
  height: number,
  occupiedCells: Array<[number, number]> = [],
  unwalkableCells: Array<[number, number]> = [],
  cellStates?: Uint8Array
): OccupancyGrid {
  const data = new Uint8Array(width * height);
  for (const [col, row] of occupiedCells) {
    if (col >= 0 && col < width && row >= 0 && row < height) {
      data[row * width + col] = 1;
    }
  }

  const walkable = new Uint8Array(width * height);
  // Default all cells to walkable (1)
  walkable.fill(1);
  for (const [col, row] of unwalkableCells) {
    if (col >= 0 && col < width && row >= 0 && row < height) {
      walkable[row * width + col] = 0;
    }
  }

  return {
    origin: [0, 0],
    cell_size: 0.1,
    width,
    height,
    data,
    walkable,
    state: cellStates,
  };
}

describe("Phase 8 Collision Detection & Sliding Maths", () => {
  const RADIUS = 0.08; // 0.08 m circle against already-inflated grid

  describe("1. World to Cell Mapping", () => {
    it("maps world coordinates to cell indices correctly", () => {
      const cell = worldToCell(-5.9, -4.7, 0.1, -5.85, -4.65);
      expect(cell.col).toBe(0);
      expect(cell.row).toBe(0);

      const cell2 = worldToCell(0, 0, 0.1, 0.55, 1.25);
      expect(cell2.col).toBe(5);
      expect(cell2.row).toBe(12);
    });
  });

  describe("2. Perpendicular Wall Collision", () => {
    it("stops a perpendicular move into a wall", () => {
      // 10x10 grid with a wall at column 5 (x from 0.5 to 0.6)
      const occupied: Array<[number, number]> = [];
      for (let r = 0; r < 10; r++) occupied.push([5, r]);
      const grid = createSyntheticGrid(10, 10, occupied);

      // Start at x = 0.35, z = 0.45 (centre of cell col=3, row=4)
      // Moving +X towards the wall by 0.3 m
      const res = resolveMovement(grid, 0.35, 0.45, 0.3, 0.0, RADIUS);

      // The wall is at x = 0.50. With radius 0.08, the player cannot cross x = 0.42.
      expect(res.x).toBeLessThanOrEqual(0.4201);
      expect(res.x).toBeGreaterThan(0.35); // made some forward progress
      expect(res.z).toBeCloseTo(0.45, 5); // Z did not change
      expect(res.hitX).toBe(true);
    });
  });

  describe("3. Wall Sliding (Diagonal Move)", () => {
    it("slides along a wall during a diagonal move, keeping the parallel component", () => {
      // Wall along X=5 (col 5) for all rows
      const occupied: Array<[number, number]> = [];
      for (let r = 0; r < 10; r++) occupied.push([5, r]);
      const grid = createSyntheticGrid(10, 10, occupied);

      // Start at x = 0.40, z = 0.20
      // Move diagonally: dx = 0.20 (into wall), dz = 0.30 (parallel to wall)
      const res = resolveMovement(grid, 0.40, 0.20, 0.20, 0.30, RADIUS);

      // X should be stopped near x = 0.42
      expect(res.x).toBeLessThanOrEqual(0.4201);
      // Z should have fully progressed from 0.20 to 0.50 (parallel move kept)
      expect(res.z).toBeCloseTo(0.50, 4);
      expect(res.hitX).toBe(true);
      expect(res.hitZ).toBe(false);
    });
  });

  describe("4. Inside Corner Collision", () => {
    it("stops both axes without jitter when moving diagonally into an inside corner", () => {
      // Inside corner formed by wall at col=5 and wall at row=5
      const occupied: Array<[number, number]> = [];
      for (let i = 0; i < 10; i++) {
        occupied.push([5, i]); // vertical wall
        occupied.push([i, 5]); // horizontal wall
      }
      const grid = createSyntheticGrid(10, 10, occupied);

      // Start at x = 0.40, z = 0.40 (close to the corner at (0.5, 0.5))
      // Move diagonally towards corner: dx = +0.20, dz = +0.20
      const res = resolveMovement(grid, 0.40, 0.40, 0.20, 0.20, RADIUS);

      // Both axes should hit and be blocked
      expect(res.x).toBeLessThanOrEqual(0.4201);
      expect(res.z).toBeLessThanOrEqual(0.4201);
      expect(res.hitX).toBe(true);
      expect(res.hitZ).toBe(true);
    });
  });

  describe("5. Passable vs Impassable Gaps", () => {
    it("passes through a 1.0 m (10 cells) gap", () => {
      // 20x20 grid with walls leaving a 10-cell gap (rows 5 to 14) at column 10
      const occupied: Array<[number, number]> = [];
      for (let r = 0; r < 20; r++) {
        if (r < 5 || r >= 15) {
          occupied.push([10, r]);
        }
      }
      const grid = createSyntheticGrid(20, 20, occupied);

      // Move from x = 0.8, z = 1.0 (in the middle of gap, row 10) to x = 1.3, z = 1.0
      const res = resolveMovement(grid, 0.8, 1.0, 0.5, 0.0, RADIUS);

      expect(res.x).toBeCloseTo(1.3, 4);
      expect(res.hitX).toBe(false);
    });

    it("blocks movement through a 0.3 m gap with a 0.25 m radius player", () => {
      // 10x10 grid with walls leaving only a 3-cell (0.30 m) gap at row 4, 5, 6
      const occupied: Array<[number, number]> = [];
      for (let r = 0; r < 10; r++) {
        if (r < 4 || r > 6) {
          occupied.push([5, r]);
        }
      }
      const grid = createSyntheticGrid(10, 10, occupied);

      // With player radius 0.25 m (diameter 0.50 m), a 0.30 m gap is impassable!
      const largeRadius = 0.25;
      const res = resolveMovement(grid, 0.25, 0.55, 0.5, 0.0, largeRadius);

      // Should be blocked from crossing the wall at column 5 (x=0.5)
      expect(res.x).toBeLessThanOrEqual(0.40);
      expect(res.hitX).toBe(true);
    });
  });

  describe("6. Fast Frame Anti-Tunneling", () => {
    it("stops a long move of 2.0 m in one step from tunneling through a 1-cell wall", () => {
      // 30x10 grid with a single 1-cell thick wall at col 10 (x = 1.0 to 1.1)
      const occupied: Array<[number, number]> = [];
      for (let r = 0; r < 10; r++) occupied.push([10, r]);
      const grid = createSyntheticGrid(30, 10, occupied);

      // Attempt to move 2.0 m across the wall in a single frame: from x = 0.5 to x = 2.5
      const res = resolveMovement(grid, 0.5, 0.5, 2.0, 0.0, RADIUS);

      // Must be stopped before reaching x = 1.0; cannot tunnel to x = 2.5!
      expect(res.x).toBeLessThanOrEqual(0.9201);
      expect(res.hitX).toBe(true);
    });
  });

  describe("7. Blocked Spawn Fix", () => {
    it("moves a blocked spawn to the nearest free cell centre", () => {
      // Grid with wall at col 5, row 5
      const grid = createSyntheticGrid(10, 10, [[5, 5]]);

      // Attempt spawn at centre of blocked cell (col 5, row 5 => x = 0.55, z = 0.55)
      const fix = fixSpawnPosition(grid, 0.55, 0.55, RADIUS);

      expect(fix.wasMoved).toBe(true);
      // Spawn should now be in an unblocked cell
      expect(isCircleBlocked(grid, fix.x, fix.z, RADIUS)).toBe(false);
      // Distance from original spawn is 2 cells (0.2 m) because cell col=6 centre (0.65) is only 0.05 m from wall (col 5), which is < radius 0.08 m
      const dist = Math.hypot(fix.x - 0.55, fix.z - 0.55);
      expect(dist).toBeCloseTo(0.2, 1);
    });
  });

  describe("8. Stuck Escape", () => {
    it("allows a player inside a blocked cell to walk out towards free space", () => {
      // Grid with blocked cell at col 5, row 5
      const grid = createSyntheticGrid(10, 10, [[5, 5]]);

      // Player starts inside the blocked cell at (0.55, 0.55)
      expect(isPointBlocked(grid, 0.55, 0.55)).toBe(true);

      // Walk left towards free cell at col 4 (dx = -0.20, dz = 0)
      const res = resolveMovement(grid, 0.55, 0.55, -0.20, 0.0, RADIUS);

      // Movement should be allowed towards free space
      expect(res.x).toBeLessThan(0.55);
      expect(res.x).toBeCloseTo(0.35, 2);
      // Player has exited the blocked cell into free space
      expect(isPointBlocked(grid, res.x, res.z)).toBe(false);
    });
  });

  describe("9. Walkable Limit", () => {
    it("blocks a move into an unwalkable free cell when limitToWalkable is true", () => {
      // Grid where col 5 is free in data (data=0), but unwalkable (walkable=0)
      const unwalkable: Array<[number, number]> = [];
      for (let r = 0; r < 10; r++) unwalkable.push([5, r]);
      const grid = createSyntheticGrid(10, 10, [], unwalkable);

      // Move into col 5 with limitToWalkable = true
      const res = resolveMovement(grid, 0.35, 0.45, 0.30, 0.0, RADIUS, true);

      // Should be blocked just like a wall
      expect(res.x).toBeLessThanOrEqual(0.4201);
      expect(res.hitX).toBe(true);

      // But with limitToWalkable = false, it passes through freely
      const resFree = resolveMovement(grid, 0.35, 0.45, 0.30, 0.0, RADIUS, false);
      expect(resFree.x).toBeCloseTo(0.65, 4);
      expect(resFree.hitX).toBe(false);
    });
  });

  describe("10. Scene Bounds Fallback", () => {
    it("clamps position to scene bounds", () => {
      const bounds = {
        min: [-5.0, 0.0, -4.0] as [number, number, number],
        max: [4.0, 3.0, 7.0] as [number, number, number],
      };

      const clamped = clampToBounds(-6.5, 8.2, bounds);
      expect(clamped.x).toBe(-5.0);
      expect(clamped.z).toBe(7.0);

      const inside = clampToBounds(1.2, 3.4, bounds);
      expect(inside.x).toBe(1.2);
      expect(inside.z).toBe(3.4);
    });
  });

  describe("11. Real Occupancy Grid Verification", () => {
    it("loads viewer/public/scene/occupancy.json, verifies keys and asserts spawn is free", () => {
      const occPath = path.resolve(__dirname, "../public/scene/occupancy.json");
      expect(fs.existsSync(occPath)).toBe(true);

      const raw = fs.readFileSync(occPath, "utf-8");
      const occData = JSON.parse(raw);

      // Assert expected keys
      const expectedKeys = [
        "origin",
        "cell_size",
        "width",
        "height",
        "data",
        "walkable",
      ];
      for (const k of expectedKeys) {
        expect(occData).toHaveProperty(k);
      }

      const grid: OccupancyGrid = {
        origin: occData.origin,
        cell_size: occData.cell_size,
        width: occData.width,
        height: occData.height,
        data: occData.data,
        walkable: occData.walkable,
      };

      // Load scene.json for spawn
      const sceneJsonPath = path.resolve(__dirname, "../public/scene/scene.json");
      const sceneData = JSON.parse(fs.readFileSync(sceneJsonPath, "utf-8"));
      const [spawnX, , spawnZ] = sceneData.spawn_position;

      // Fix spawn if needed and verify
      const fix = fixSpawnPosition(grid, spawnX, spawnZ, RADIUS, true);
      const isSpawnFree = !isCircleBlocked(grid, fix.x, fix.z, RADIUS, true);
      expect(isSpawnFree).toBe(true);

      // Check for consecutive recorded camera positions
      if (Array.isArray(sceneData.camera_positions) && sceneData.camera_positions.length > 1) {
        const cams = sceneData.camera_positions;
        let blockedSteps = 0;
        let totalSteps = 0;

        for (let i = 0; i < cams.length - 1; i++) {
          const p1 = cams[i];
          const p2 = cams[i + 1];
          const dist = Math.hypot(p2[0] - p1[0], p2[2] - p1[2]);
          const steps = Math.max(1, Math.ceil(dist / 0.05));

          for (let s = 0; s <= steps; s++) {
            const t = s / steps;
            const x = p1[0] * (1 - t) + p2[0] * t;
            const z = p1[2] * (1 - t) + p2[2] * t;
            totalSteps++;
            if (isPointBlocked(grid, x, z, false)) {
              blockedSteps++;
            }
          }
        }

        const blockedPct = (blockedSteps / totalSteps) * 100;
        expect(blockedPct).toBeLessThanOrEqual(5.0);
      } else {
        console.log(
          "[Test Skip] scene.json does not contain recorded camera_positions array; skipped camera trajectory interpolation test."
        );
      }
    });
  });

  describe("12. Explore Mode (Key U) Collision Behaviour", () => {
    // 6x5 grid:
    // col 0..1: Free cells (state = 1, data = 0)
    // col 2..3: Unknown cells (state = 0, data = 1)
    // col 4: Occupied cells (state = 2, data = 1)
    // col 5: Inflated cells (state = 3, data = 1)
    const width = 6;
    const height = 5;
    const state = new Uint8Array(width * height);
    const data = new Uint8Array(width * height);
    const walkable = new Uint8Array(width * height);

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        if (c <= 1) {
          state[idx] = 1; // free
          data[idx] = 0;
          walkable[idx] = 1;
        } else if (c <= 3) {
          state[idx] = 0; // unknown
          data[idx] = 1;
          walkable[idx] = 0;
        } else if (c === 4) {
          state[idx] = 2; // occupied
          data[idx] = 1;
          walkable[idx] = 0;
        } else {
          state[idx] = 3; // inflated
          data[idx] = 1;
          walkable[idx] = 0;
        }
      }
    }

    const grid: OccupancyGrid = {
      origin: [0, 0],
      cell_size: 0.1,
      width,
      height,
      data,
      walkable,
      state,
    };

    it("blocks unknown cells (state 0) when Explore Mode is OFF (default)", () => {
      // Cell (col=2, row=2) is unknown (state 0)
      expect(isBlocked(grid, 2, 2, false, false)).toBe(true);
      expect(isPointBlocked(grid, 0.25, 0.25, false, false)).toBe(true);
    });

    it("allows unknown cells (state 0) when Explore Mode is ON", () => {
      // Cell (col=2, row=2) is unknown (state 0)
      expect(isBlocked(grid, 2, 2, false, true)).toBe(false);
      expect(isPointBlocked(grid, 0.25, 0.25, false, true)).toBe(false);
    });

    it("strictly blocks occupied cells (state 2) in BOTH Explore OFF and Explore ON", () => {
      // Cell (col=4, row=2) is occupied (state 2)
      expect(isBlocked(grid, 4, 2, false, false)).toBe(true);
      expect(isBlocked(grid, 4, 2, false, true)).toBe(true);
      expect(isPointBlocked(grid, 0.45, 0.25, false, false)).toBe(true);
      expect(isPointBlocked(grid, 0.45, 0.25, false, true)).toBe(true);
    });

    it("strictly blocks inflated cells (state 3) in BOTH Explore OFF and Explore ON", () => {
      // Cell (col=5, row=2) is inflated (state 3)
      expect(isBlocked(grid, 5, 2, false, false)).toBe(true);
      expect(isBlocked(grid, 5, 2, false, true)).toBe(true);
      expect(isPointBlocked(grid, 0.55, 0.25, false, false)).toBe(true);
      expect(isPointBlocked(grid, 0.55, 0.25, false, true)).toBe(true);
    });

    it("allows free cells (state 1) in both modes", () => {
      // Cell (col=0, row=2) is free (state 1)
      expect(isBlocked(grid, 0, 2, false, false)).toBe(false);
      expect(isBlocked(grid, 0, 2, false, true)).toBe(false);
      expect(isPointBlocked(grid, 0.05, 0.25, false, false)).toBe(false);
      expect(isPointBlocked(grid, 0.05, 0.25, false, true)).toBe(false);
    });

    it("resolves movement into unknown cell: blocked when OFF, allowed when ON", () => {
      // Start at free cell x = 0.05, z = 0.25 (col 0, row 2)
      // Cell 2 (unknown) starts at x = 0.20. With radius 0.08, player cannot cross x = 0.12.
      // When OFF: player cannot cross into unknown cell
      const resOff = resolveMovement(grid, 0.05, 0.25, 0.20, 0.0, RADIUS, false, false);
      expect(resOff.hitX).toBe(true);
      expect(resOff.x).toBeLessThanOrEqual(0.1201);

      // When ON: player smoothly enters unknown cell towards x = 0.25
      const resOn = resolveMovement(grid, 0.05, 0.25, 0.20, 0.0, RADIUS, false, true);
      expect(resOn.x).toBeCloseTo(0.25, 2);
    });

    it("resolves movement into occupied cell: blocked in both modes", () => {
      // Start inside unknown cell at x = 0.25, z = 0.25 with Explore ON
      // Occupied cell begins at x = 0.40. With radius 0.08, player cannot cross x = 0.32.
      const resIntoOcc = resolveMovement(grid, 0.25, 0.25, 0.20, 0.0, RADIUS, false, true);
      expect(resIntoOcc.hitX).toBe(true);
      expect(resIntoOcc.x).toBeLessThanOrEqual(0.3201);
    });

    it("detects unknown cells accurately via isPointInUnknownCell", () => {
      // (0.05, 0.25) is free -> false
      expect(isPointInUnknownCell(grid, 0.05, 0.25)).toBe(false);
      // (0.25, 0.25) is unknown -> true
      expect(isPointInUnknownCell(grid, 0.25, 0.25)).toBe(true);
      // (0.45, 0.25) is occupied -> false
      expect(isPointInUnknownCell(grid, 0.45, 0.25)).toBe(false);
      // Outside grid -> false
      expect(isPointInUnknownCell(grid, -1.0, 0.25)).toBe(false);
    });
  });
});
