/**
 * collision.ts
 * Pure collision detection and movement resolution against 2D occupancy grids.
 * Framework-free, pure TypeScript: NO Three.js and NO DOM imports so it can be tested in Vitest.
 */

export interface OccupancyGrid {
  origin: [number, number]; // [min_x, min_z]
  cell_size: number;
  width: number;
  height: number;
  data: number[] | Uint8Array;
  walkable?: number[] | Uint8Array;
}

export interface MoveResult {
  x: number;
  z: number;
  hitX: boolean;
  hitZ: boolean;
}

export interface SceneBounds {
  min: [number, number, number];
  max: [number, number, number];
}

/**
 * Converts world coordinates (x, z) to grid cell indices (col, row).
 * col is along X (0 to width - 1).
 * row is along Z (0 to height - 1).
 */
export function worldToCell(
  originX: number,
  originZ: number,
  cellSize: number,
  x: number,
  z: number
): { col: number; row: number } {
  return {
    col: Math.floor((x - originX) / cellSize),
    row: Math.floor((z - originZ) / cellSize),
  };
}

/**
 * Returns true if the cell (col, row) is blocked (occupied or outside the grid).
 * If limitToWalkable is true and walkable array is provided, unwalkable cells (walkable === 0) are also blocked.
 */
export function isBlocked(
  grid: OccupancyGrid,
  col: number,
  row: number,
  limitToWalkable = false
): boolean {
  if (col < 0 || col >= grid.width || row < 0 || row >= grid.height) {
    return true; // outside grid is always blocked
  }
  const idx = row * grid.width + col;
  if (grid.data[idx] === 1) {
    return true; // occupied cell
  }
  if (limitToWalkable && grid.walkable && grid.walkable[idx] === 0) {
    return true; // unwalkable cell
  }
  return false;
}

/**
 * Returns true if the world point (x, z) falls in a blocked cell or outside the grid.
 */
export function isPointBlocked(
  grid: OccupancyGrid,
  x: number,
  z: number,
  limitToWalkable = false
): boolean {
  const col = Math.floor((x - grid.origin[0]) / grid.cell_size);
  const row = Math.floor((z - grid.origin[1]) / grid.cell_size);
  return isBlocked(grid, col, row, limitToWalkable);
}

/**
 * Circle-vs-grid collision test.
 * The circle is blocked if any cell its bounding square overlaps that is blocked
 * lies within the radius of the circle centre (nearest-point-on-cell test).
 */
export function isCircleBlocked(
  grid: OccupancyGrid,
  cx: number,
  cz: number,
  radius: number,
  limitToWalkable = false
): boolean {
  const originX = grid.origin[0];
  const originZ = grid.origin[1];
  const cellSize = grid.cell_size;
  const rSq = radius * radius;

  // Bounding box of the circle in grid cell indices
  const minCol = Math.floor((cx - radius - originX) / cellSize);
  const maxCol = Math.floor((cx + radius - originX) / cellSize);
  const minRow = Math.floor((cz - radius - originZ) / cellSize);
  const maxRow = Math.floor((cz + radius - originZ) / cellSize);

  for (let row = minRow; row <= maxRow; row++) {
    for (let col = minCol; col <= maxCol; col++) {
      if (isBlocked(grid, col, row, limitToWalkable)) {
        // Find nearest point on cell (col, row) to circle centre (cx, cz)
        const cellMinX = originX + col * cellSize;
        const cellMaxX = cellMinX + cellSize;
        const cellMinZ = originZ + row * cellSize;
        const cellMaxZ = cellMinZ + cellSize;

        const nearestX = Math.max(cellMinX, Math.min(cellMaxX, cx));
        const nearestZ = Math.max(cellMinZ, Math.min(cellMaxZ, cz));

        const dx = cx - nearestX;
        const dz = cz - nearestZ;
        if (dx * dx + dz * dz < rSq) {
          return true;
        }
      }
    }
  }

  return false;
}

/**
 * Computes the distance from world position (x, z) to the nearest free cell boundary.
 * Returns 0 if (x, z) is already in a free cell.
 */
export function getDistanceToFree(
  grid: OccupancyGrid,
  x: number,
  z: number,
  limitToWalkable = false,
  maxSearchRings = 25
): number {
  if (!isPointBlocked(grid, x, z, limitToWalkable)) {
    return 0;
  }

  const originX = grid.origin[0];
  const originZ = grid.origin[1];
  const cellSize = grid.cell_size;

  const centerCol = Math.floor((x - originX) / cellSize);
  const centerRow = Math.floor((z - originZ) / cellSize);

  let minDistSq = Infinity;
  let foundFree = false;

  for (let ring = 1; ring <= maxSearchRings; ring++) {
    for (let r = centerRow - ring; r <= centerRow + ring; r++) {
      for (let c = centerCol - ring; c <= centerCol + ring; c++) {
        // Only evaluate cells on the perimeter of the current ring
        if (
          r !== centerRow - ring &&
          r !== centerRow + ring &&
          c !== centerCol - ring &&
          c !== centerCol + ring
        ) {
          continue;
        }

        if (!isBlocked(grid, c, r, limitToWalkable)) {
          foundFree = true;
          const cellMinX = originX + c * cellSize;
          const cellMaxX = cellMinX + cellSize;
          const cellMinZ = originZ + r * cellSize;
          const cellMaxZ = cellMinZ + cellSize;

          const nearestX = Math.max(cellMinX, Math.min(cellMaxX, x));
          const nearestZ = Math.max(cellMinZ, Math.min(cellMaxZ, z));

          const dx = x - nearestX;
          const dz = z - nearestZ;
          const distSq = dx * dx + dz * dz;
          if (distSq < minDistSq) {
            minDistSq = distSq;
          }
        }
      }
    }

    if (foundFree) {
      break;
    }
  }

  return Math.sqrt(minDistSq);
}

/**
 * Checks if moving from (fromX, fromZ) to (toX, toZ) is allowed.
 * If the current position is already inside a blocked area, stuck escape allows
 * any move that does not increase distance to free space.
 */
export function isMoveAllowed(
  grid: OccupancyGrid,
  fromX: number,
  fromZ: number,
  toX: number,
  toZ: number,
  radius: number,
  limitToWalkable = false
): boolean {
  if (!isCircleBlocked(grid, toX, toZ, radius, limitToWalkable)) {
    return true;
  }

  // Stuck escape: if already blocked, allow moves that move towards or maintain distance to free space
  if (isCircleBlocked(grid, fromX, fromZ, radius, limitToWalkable)) {
    const fromDist = getDistanceToFree(grid, fromX, fromZ, limitToWalkable);
    const toDist = getDistanceToFree(grid, toX, toZ, limitToWalkable);
    return toDist <= fromDist + 1e-5;
  }

  return false;
}

/**
 * Resolves movement with wall sliding and sub-stepping.
 * Limits sub-steps to at most half a cell (0.05 m) so fast frames cannot tunnel through walls.
 * If a move is blocked, tries X only, then Z only, and keeps whichever succeeds.
 * Supports passing an optional outResult object to avoid per-frame heap allocations.
 */
export function resolveMovement(
  grid: OccupancyGrid,
  curX: number,
  curZ: number,
  dx: number,
  dz: number,
  radius: number,
  limitToWalkable = false,
  outResult?: MoveResult
): MoveResult {
  const result: MoveResult = outResult ?? { x: curX, z: curZ, hitX: false, hitZ: false };
  result.x = curX;
  result.z = curZ;
  result.hitX = false;
  result.hitZ = false;

  const totalDist = Math.hypot(dx, dz);
  if (totalDist < 1e-6) {
    return result;
  }

  const maxSubStep = grid.cell_size * 0.5; // 0.05 m per sub-step
  const numSteps = Math.max(1, Math.ceil(totalDist / maxSubStep));
  const stepDx = dx / numSteps;
  const stepDz = dz / numSteps;

  for (let s = 0; s < numSteps; s++) {
    const targetX = result.x + stepDx;
    const targetZ = result.z + stepDz;

    // 1. Try full diagonal move
    if (isMoveAllowed(grid, result.x, result.z, targetX, targetZ, radius, limitToWalkable)) {
      result.x = targetX;
      result.z = targetZ;
    } else {
      // 2. Full move blocked: Try sliding along X or Z
      const canX = stepDx !== 0 && isMoveAllowed(grid, result.x, result.z, targetX, result.z, radius, limitToWalkable);
      const canZ = stepDz !== 0 && isMoveAllowed(grid, result.x, result.z, result.x, targetZ, radius, limitToWalkable);

      if (canX && !canZ) {
        result.x = targetX;
        result.hitZ = true;
      } else if (!canX && canZ) {
        result.z = targetZ;
        result.hitX = true;
      } else if (canX && canZ) {
        // Both succeed independently (e.g. corner tip) - pick larger component
        if (Math.abs(stepDx) >= Math.abs(stepDz)) {
          result.x = targetX;
          result.hitZ = true;
        } else {
          result.z = targetZ;
          result.hitX = true;
        }
      } else {
        // Inside corner or flat perpendicular wall: stopped on both axes
        result.hitX = true;
        result.hitZ = true;
        break; // Stop further sub-steps
      }
    }
  }

  return result;
}

/**
 * Searches concentric rings outward for the nearest free cell centre if spawn is blocked.
 * Logs a warning if spawn was moved.
 */
export function fixSpawnPosition(
  grid: OccupancyGrid,
  spawnX: number,
  spawnZ: number,
  radius = 0.08,
  limitToWalkable = false
): { x: number; z: number; wasMoved: boolean } {
  if (!isCircleBlocked(grid, spawnX, spawnZ, radius, limitToWalkable)) {
    return { x: spawnX, z: spawnZ, wasMoved: false };
  }

  const originX = grid.origin[0];
  const originZ = grid.origin[1];
  const cellSize = grid.cell_size;

  const startCol = Math.floor((spawnX - originX) / cellSize);
  const startRow = Math.floor((spawnZ - originZ) / cellSize);

  const maxRings = Math.max(grid.width, grid.height);
  let bestX = spawnX;
  let bestZ = spawnZ;
  let bestDistSq = Infinity;
  let found = false;

  for (let ring = 1; ring <= maxRings; ring++) {
    for (let r = startRow - ring; r <= startRow + ring; r++) {
      for (let c = startCol - ring; c <= startCol + ring; c++) {
        if (
          r !== startRow - ring &&
          r !== startRow + ring &&
          c !== startCol - ring &&
          c !== startCol + ring
        ) {
          continue;
        }

        if (c >= 0 && c < grid.width && r >= 0 && r < grid.height) {
          const centerX = originX + (c + 0.5) * cellSize;
          const centerZ = originZ + (r + 0.5) * cellSize;

          if (!isCircleBlocked(grid, centerX, centerZ, radius, limitToWalkable)) {
            const dx = centerX - spawnX;
            const dz = centerZ - spawnZ;
            const distSq = dx * dx + dz * dz;
            if (distSq < bestDistSq) {
              bestDistSq = distSq;
              bestX = centerX;
              bestZ = centerZ;
              found = true;
            }
          }
        }
      }
    }

    if (found) {
      break;
    }
  }

  if (found) {
    console.warn(
      `[Collision] Blocked spawn at (${spawnX.toFixed(3)}, ${spawnZ.toFixed(3)}) adjusted to nearest free cell at (${bestX.toFixed(3)}, ${bestZ.toFixed(3)})`
    );
    return { x: bestX, z: bestZ, wasMoved: true };
  }

  return { x: spawnX, z: spawnZ, wasMoved: false };
}

/**
 * Fallback clamp to scene bounding box.
 */
export function clampToBounds(
  x: number,
  z: number,
  bounds?: SceneBounds
): { x: number; z: number } {
  if (!bounds) return { x, z };
  return {
    x: Math.max(bounds.min[0], Math.min(bounds.max[0], x)),
    z: Math.max(bounds.min[2], Math.min(bounds.max[2], z)),
  };
}
