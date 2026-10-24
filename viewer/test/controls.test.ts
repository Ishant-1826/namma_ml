import { describe, it, expect } from "vitest";
import {
  clampPitch,
  computeWishDirection,
  updateVelocity,
  yawToDirection,
  DEFAULT_PITCH_LIMIT_DEG,
  WALK_SPEED,
  SPRINT_SPEED,
  ACCELERATION,
  STOP_TIME_S,
  EYE_HEIGHT_M,
} from "../src/controls";

describe("First-Person Movement Maths", () => {
  describe("1. Diagonal Normalisation", () => {
    it("computes magnitude 1.0 for cardinal forward movement", () => {
      const wishDir = { x: 0, y: 0, z: 0 };
      const isMoving = computeWishDirection(0, 1, 0, wishDir);
      expect(isMoving).toBe(true);

      const length = Math.hypot(wishDir.x, wishDir.z);
      expect(length).toBeCloseTo(1.0, 5);
      // At yaw 0, forward is along -Z
      expect(wishDir.x).toBeCloseTo(0.0, 5);
      expect(wishDir.z).toBeCloseTo(-1.0, 5);
    });

    it("normalises diagonal movement (W+D) so length is 1.0, not sqrt(2)", () => {
      const wishDir = { x: 0, y: 0, z: 0 };
      const isMoving = computeWishDirection(1, 1, 0, wishDir);
      expect(isMoving).toBe(true);

      const length = Math.hypot(wishDir.x, wishDir.z);
      // Must be 1.0, ensuring diagonal movement is NOT faster than cardinal
      expect(length).toBeCloseTo(1.0, 5);
      expect(wishDir.x).toBeCloseTo(1 / Math.SQRT2, 5);
      expect(wishDir.z).toBeCloseTo(-1 / Math.SQRT2, 5);
    });

    it("normalises all 4 diagonal quadrants to unit length", () => {
      const wishDir = { x: 0, y: 0, z: 0 };
      const diagonals = [
        [1, 1],   // Forward-Right
        [-1, 1],  // Forward-Left
        [1, -1],  // Backward-Right
        [-1, -1], // Backward-Left
      ];

      for (const [x, z] of diagonals) {
        computeWishDirection(x, z, 0.45, wishDir);
        const length = Math.hypot(wishDir.x, wishDir.z);
        expect(length).toBeCloseTo(1.0, 5);
      }
    });

    it("returns false and zero vector when no input keys are pressed", () => {
      const wishDir = { x: 99, y: 99, z: 99 };
      const isMoving = computeWishDirection(0, 0, 0, wishDir);
      expect(isMoving).toBe(false);
      expect(wishDir.x).toBe(0);
      expect(wishDir.y).toBe(0);
      expect(wishDir.z).toBe(0);
    });

    it("ensures steady-state velocity for diagonal movement matches cardinal walk speed", () => {
      const velCardinal = { x: 0, y: 0, z: 0 };
      const velDiagonal = { x: 0, y: 0, z: 0 };
      const wishCardinal = { x: 0, y: 0, z: -1 };
      const wishDiagonal = { x: 1 / Math.SQRT2, y: 0, z: -1 / Math.SQRT2 };

      // Step 20 frames at 60fps (0.33s) with acceleration
      const dt = 0.0166;
      for (let i = 0; i < 20; i++) {
        updateVelocity(velCardinal, wishCardinal, true, WALK_SPEED, dt);
        updateVelocity(velDiagonal, wishDiagonal, true, WALK_SPEED, dt);
      }

      const speedCardinal = Math.hypot(velCardinal.x, velCardinal.z);
      const speedDiagonal = Math.hypot(velDiagonal.x, velDiagonal.z);

      expect(speedCardinal).toBeCloseTo(WALK_SPEED, 3);
      expect(speedDiagonal).toBeCloseTo(WALK_SPEED, 3);
      expect(speedDiagonal).toBeCloseTo(speedCardinal, 3);
    });
  });

  describe("2. Friction Stopping", () => {
    it("stops within ~0.2 s when keys are released from walking speed", () => {
      const velocity = { x: 0, y: 0, z: -WALK_SPEED };
      const dummyWish = { x: 0, y: 0, z: 0 };
      const dt = 0.0166; // ~60 fps
      const stepsTo02s = Math.ceil(STOP_TIME_S / dt); // ~13 steps (0.216s)

      let prevSpeed = Math.hypot(velocity.x, velocity.z);
      for (let i = 0; i < stepsTo02s; i++) {
        updateVelocity(velocity, dummyWish, false, 0, dt);
        const currentSpeed = Math.hypot(velocity.x, velocity.z);
        // Speed must strictly decrease monotonically
        expect(currentSpeed).toBeLessThanOrEqual(prevSpeed);
        prevSpeed = currentSpeed;
      }

      const finalSpeed = Math.hypot(velocity.x, velocity.z);
      // In ~0.2 s, velocity must have come to a complete stop
      expect(finalSpeed).toBe(0);
      expect(velocity.x).toBe(0);
      expect(velocity.z).toBe(0);
    });

    it("stops within ~0.25 s when keys are released from sprint speed", () => {
      const velocity = { x: SPRINT_SPEED * 0.6, y: 0, z: -SPRINT_SPEED * 0.8 };
      const dummyWish = { x: 0, y: 0, z: 0 };
      const dt = 0.0166;
      const totalSteps = 16; // ~0.26s

      for (let i = 0; i < totalSteps; i++) {
        updateVelocity(velocity, dummyWish, false, 0, dt);
      }

      const finalSpeed = Math.hypot(velocity.x, velocity.z);
      expect(finalSpeed).toBe(0);
    });
  });

  describe("3. Pitch Clamping", () => {
    const limitDeg = DEFAULT_PITCH_LIMIT_DEG; // 70 deg
    const limitRad = (limitDeg * Math.PI) / 180; // ~1.2217 rad

    it("clamps extreme upward pitch (+85 deg) to +70 deg", () => {
      const pitch85 = (85 * Math.PI) / 180;
      const clamped = clampPitch(pitch85, limitDeg);
      expect(clamped).toBeCloseTo(limitRad, 5);
    });

    it("clamps extreme downward pitch (-85 deg) to -70 deg", () => {
      const pitchNeg85 = (-85 * Math.PI) / 180;
      const clamped = clampPitch(pitchNeg85, limitDeg);
      expect(clamped).toBeCloseTo(-limitRad, 5);
    });

    it("preserves pitches within allowable range [-70 deg, +70 deg]", () => {
      const angles = [0, 15, -30, 45, -60, 69.9, -69.9];
      for (const deg of angles) {
        const rad = (deg * Math.PI) / 180;
        expect(clampPitch(rad, limitDeg)).toBeCloseTo(rad, 5);
      }
    });
  });

  describe("4. Yaw Direction and Three.js Convention", () => {
    it("verifies yaw 0 looks strictly along -Z (North)", () => {
      const dir = yawToDirection(0);
      expect(dir.x).toBeCloseTo(0.0, 5);
      expect(dir.y).toBe(0);
      expect(dir.z).toBeCloseTo(-1.0, 5);
    });

    it("verifies positive yaw turns left (counter-clockwise seen from above)", () => {
      // +90 degrees (pi/2) turns left towards -X
      const dirLeft = yawToDirection(Math.PI / 2);
      expect(dirLeft.x).toBeCloseTo(-1.0, 5);
      expect(dirLeft.z).toBeCloseTo(0.0, 5);
    });

    it("verifies negative yaw turns right (clockwise seen from above)", () => {
      // -90 degrees (-pi/2) turns right towards +X
      const dirRight = yawToDirection(-Math.PI / 2);
      expect(dirRight.x).toBeCloseTo(1.0, 5);
      expect(dirRight.z).toBeCloseTo(0.0, 5);
    });

    it("verifies yaw pi (180 deg) looks along +Z (South)", () => {
      const dirSouth = yawToDirection(Math.PI);
      expect(dirSouth.x).toBeCloseTo(0.0, 5);
      expect(dirSouth.z).toBeCloseTo(1.0, 5);
    });
  });

  describe("5. Eye Height Configuration", () => {
    it("confirms EYE_HEIGHT_M is 1.45 m matching captured video height", () => {
      expect(EYE_HEIGHT_M).toBe(1.45);
    });
  });
});
