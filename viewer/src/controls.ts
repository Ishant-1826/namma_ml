import * as THREE from "three";
import {
  OccupancyGrid,
  SceneBounds,
  MoveResult,
  resolveMovement,
  fixSpawnPosition,
  clampToBounds,
} from "./collision";

/**
 * EYE_HEIGHT_M: Eye height in metres above scene.json floor_y.
 * The eye-level video pass was captured at 1.45 m above the floor.
 */
export const EYE_HEIGHT_M = 1.45;

export const DEFAULT_PITCH_LIMIT_DEG = 70;
export const WALK_SPEED = 1.4; // m/s
export const SPRINT_SPEED = 2.8; // m/s (2x walking)
export const ACCELERATION = 12.0; // m/s^2
export const STOP_TIME_S = 0.2; // target stop time in seconds
export const STOP_DECEL = WALK_SPEED / STOP_TIME_S; // 7.0 m/s^2 linear deceleration
export const FRICTION_COEFF = 14.0; // s^-1 exponential damping
export const MAX_DT_S = 0.05; // clamp delta time to avoid lag-spike teleportation
export const DEFAULT_LOOK_SENSITIVITY = 0.002;
export const DEFAULT_TOUCH_LOOK_SENSITIVITY = 0.003;

// Phase 8 Movement Constraints Configuration
export const LIMIT_TO_WALKABLE = true;
export const HEAD_BOB_AMPLITUDE_M = 0.015;
export const PLAYER_COLLISION_RADIUS_M = 0.08; // 0.08 m circle against already-inflated grid

export function isTouchDevice(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(pointer: coarse)").matches || "ontouchstart" in window;
}

export interface ControlsConfig {
  pitchLimitDeg?: number;
  walkSpeed?: number;
  sprintSpeed?: number;
  acceleration?: number;
  frictionCoeff?: number;
  stopDecel?: number;
  maxDt?: number;
  lookSensitivity?: number;
  touchLookSensitivity?: number;
  eyeHeightM?: number;
  playerRadius?: number;
  limitToWalkable?: boolean;
  headBobAmplitude?: number;
  grid?: OccupancyGrid;
  sceneBounds?: SceneBounds;
  joystickContainer?: HTMLElement | null;
  joystickThumb?: HTMLElement | null;
}

/**
 * Clamps pitch angle to [-maxPitchRad, +maxPitchRad].
 */
export function clampPitch(pitchRad: number, maxPitchDeg: number = DEFAULT_PITCH_LIMIT_DEG): number {
  const maxPitchRad = (maxPitchDeg * Math.PI) / 180;
  return Math.max(-maxPitchRad, Math.min(maxPitchRad, pitchRad));
}

/**
 * Returns forward direction vector from yaw angle.
 * Yaw 0 points along -Z.
 * Positive yaw turns left (counter-clockwise seen from above), pointing along -X at +pi/2.
 */
export function yawToDirection(yawRad: number): { x: number; y: number; z: number } {
  return {
    x: -Math.sin(yawRad),
    y: 0,
    z: -Math.cos(yawRad),
  };
}

/**
 * Returns right direction vector from yaw angle.
 * Yaw 0 has right pointing along +X.
 */
export function yawToRightDirection(yawRad: number): { x: number; y: number; z: number } {
  return {
    x: Math.cos(yawRad),
    y: 0,
    z: -Math.sin(yawRad),
  };
}

/**
 * Computes normalised wish direction in the horizontal XZ plane based on 2D input and camera yaw.
 * Normalises diagonal movement so diagonal (e.g. W+D) is not faster than cardinal (e.g. W).
 * Returns true if input is non-zero, false otherwise.
 */
export function computeWishDirection(
  inputX: number,
  inputZ: number,
  yawRad: number,
  outDir: { x: number; y: number; z: number }
): boolean {
  const len = Math.hypot(inputX, inputZ);
  if (len < 1e-5) {
    outDir.x = 0;
    outDir.y = 0;
    outDir.z = 0;
    return false;
  }

  // Normalise diagonal inputs
  const normX = inputX / len;
  const normZ = inputZ / len;

  // Yaw 0 looks along -Z; positive yaw turns left (CCW seen from above)
  const forwardX = -Math.sin(yawRad);
  const forwardZ = -Math.cos(yawRad);
  const rightX = Math.cos(yawRad);
  const rightZ = -Math.sin(yawRad);

  outDir.x = normZ * forwardX + normX * rightX;
  outDir.y = 0;
  outDir.z = normZ * forwardZ + normX * rightZ;
  return true;
}

/**
 * Updates velocity with acceleration towards target speed or friction damping to stop in ~0.2s.
 */
export function updateVelocity(
  velocity: { x: number; y: number; z: number },
  wishDir: { x: number; y: number; z: number },
  isMoving: boolean,
  targetSpeed: number,
  dt: number,
  accel: number = ACCELERATION,
  friction: number = FRICTION_COEFF,
  stopDecel: number = STOP_DECEL
): void {
  const effectiveDt = Math.min(dt, MAX_DT_S);

  if (isMoving && targetSpeed > 0) {
    const targetVx = wishDir.x * targetSpeed;
    const targetVz = wishDir.z * targetSpeed;
    const diffX = targetVx - velocity.x;
    const diffZ = targetVz - velocity.z;
    const diffDist = Math.hypot(diffX, diffZ);
    const maxStep = accel * effectiveDt;

    if (diffDist <= maxStep) {
      velocity.x = targetVx;
      velocity.z = targetVz;
    } else {
      velocity.x += (diffX / diffDist) * maxStep;
      velocity.z += (diffZ / diffDist) * maxStep;
    }
  } else {
    // Damping and linear stopping friction to come to a complete stop in ~0.2s
    const speed = Math.hypot(velocity.x, velocity.z);
    if (speed > 0) {
      const decel = stopDecel * effectiveDt;
      const expDamp = speed * (1 - Math.exp(-friction * effectiveDt));
      const drop = Math.max(decel, expDamp);
      const newSpeed = Math.max(0, speed - drop);
      if (newSpeed < 0.002) {
        velocity.x = 0;
        velocity.z = 0;
      } else {
        const factor = newSpeed / speed;
        velocity.x *= factor;
        velocity.z *= factor;
      }
    }
  }
}

/**
 * FirstPersonControls:
 * Framework-free first-person camera controller with pointer-lock look,
 * touch virtual joystick on left 1/3, touch drag look on right 2/3,
 * WASD horizontal movement, acceleration/friction damping, 2D occupancy collision sliding,
 * soft walkable region limit, and optional head-bob toggle.
 * Reuses Vector3, Euler, and MoveResult instances to ensure ZERO allocations per frame.
 */
export class FirstPersonControls {
  readonly camera: THREE.PerspectiveCamera;
  readonly domElement: HTMLElement;

  readonly pitchLimitDeg: number;
  readonly walkSpeed: number;
  readonly sprintSpeed: number;
  readonly acceleration: number;
  readonly frictionCoeff: number;
  readonly stopDecel: number;
  readonly maxDt: number;
  readonly lookSensitivity: number;
  readonly touchLookSensitivity: number;
  readonly eyeHeightM: number;
  readonly playerRadius: number;
  readonly headBobAmplitude: number;

  public limitToWalkable: boolean;
  public headBobEnabled = false;

  private _fixedEyeHeight = EYE_HEIGHT_M;
  private _spawnPosition = new THREE.Vector3(0, EYE_HEIGHT_M, 0);
  private _spawnYaw = 0;
  private _grid: OccupancyGrid | null = null;
  private _sceneBounds?: SceneBounds;

  // Head bob state
  private _bobTimer = 0;

  // Touch states
  readonly isTouch: boolean;
  private _joystickPointerId: number | null = null;
  private _lookPointerId: number | null = null;
  private _joystickOriginX = 0;
  private _joystickOriginY = 0;
  private _touchMoveX = 0;
  private _touchMoveZ = 0;
  private _lastLookX = 0;
  private _lastLookY = 0;
  private readonly _joystickContainer: HTMLElement | null;
  private readonly _joystickThumb: HTMLElement | null;

  // Pre-allocated objects reused every frame (ZERO heap allocations per frame)
  private readonly _euler = new THREE.Euler(0, 0, 0, "YXZ");
  private readonly _velocity = new THREE.Vector3(0, 0, 0);
  private readonly _wishDir = new THREE.Vector3(0, 0, 0);
  private readonly _moveResult: MoveResult = { x: 0, z: 0, hitX: false, hitZ: false };

  // Input states
  private _keys = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    sprint: false,
  };

  private _isLocked = false;
  public onLockChange?: (locked: boolean) => void;
  public onHeadBobChange?: (enabled: boolean) => void;
  public onWalkableLimitChange?: (enabled: boolean) => void;

  // Bound event handlers for clean teardown
  private readonly _onMouseMove: (e: MouseEvent) => void;
  private readonly _onKeyDown: (e: KeyboardEvent) => void;
  private readonly _onKeyUp: (e: KeyboardEvent) => void;
  private readonly _onPointerLockChange: () => void;
  private readonly _onWindowBlur: () => void;
  private readonly _onPointerDown: (e: PointerEvent) => void;
  private readonly _onPointerMove: (e: PointerEvent) => void;
  private readonly _onPointerUp: (e: PointerEvent) => void;
  private readonly _onPointerCancel: (e: PointerEvent) => void;

  constructor(
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement,
    config: ControlsConfig = {}
  ) {
    this.camera = camera;
    this.domElement = domElement;

    this.pitchLimitDeg = config.pitchLimitDeg ?? DEFAULT_PITCH_LIMIT_DEG;
    this.walkSpeed = config.walkSpeed ?? WALK_SPEED;
    this.sprintSpeed = config.sprintSpeed ?? SPRINT_SPEED;
    this.acceleration = config.acceleration ?? ACCELERATION;
    this.frictionCoeff = config.frictionCoeff ?? FRICTION_COEFF;
    this.stopDecel = config.stopDecel ?? STOP_DECEL;
    this.maxDt = config.maxDt ?? MAX_DT_S;
    this.lookSensitivity = config.lookSensitivity ?? DEFAULT_LOOK_SENSITIVITY;
    this.touchLookSensitivity = config.touchLookSensitivity ?? DEFAULT_TOUCH_LOOK_SENSITIVITY;
    this.eyeHeightM = config.eyeHeightM ?? EYE_HEIGHT_M;
    this.playerRadius = config.playerRadius ?? PLAYER_COLLISION_RADIUS_M;
    this.limitToWalkable = config.limitToWalkable ?? LIMIT_TO_WALKABLE;
    this.headBobAmplitude = config.headBobAmplitude ?? HEAD_BOB_AMPLITUDE_M;

    this._grid = config.grid ?? null;
    this._sceneBounds = config.sceneBounds;
    this._fixedEyeHeight = this.eyeHeightM;

    this._joystickContainer = config.joystickContainer ?? null;
    this._joystickThumb = config.joystickThumb ?? null;

    this.isTouch = isTouchDevice();
    if (this.isTouch) {
      this.domElement.style.touchAction = "none";
    }

    // Mouse look
    this._onMouseMove = (e: MouseEvent) => {
      if (!this._isLocked) return;
      const movementX = e.movementX || 0;
      const movementY = e.movementY || 0;

      // Moving mouse right (movementX > 0) turns view right (decreases yaw)
      this._euler.y -= movementX * this.lookSensitivity;

      // Moving mouse up (movementY < 0 in screen coords) looks up (increases pitch)
      this._euler.x -= movementY * this.lookSensitivity;
      this._euler.x = clampPitch(this._euler.x, this.pitchLimitDeg);

      this.camera.quaternion.setFromEuler(this._euler);
    };

    // Keyboard inputs
    this._onKeyDown = (e: KeyboardEvent) => {
      switch (e.code) {
        case "KeyW":
        case "ArrowUp":
          this._keys.forward = true;
          break;
        case "KeyS":
        case "ArrowDown":
          this._keys.backward = true;
          break;
        case "KeyA":
        case "ArrowLeft":
          this._keys.left = true;
          break;
        case "KeyD":
        case "ArrowRight":
          this._keys.right = true;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          this._keys.sprint = true;
          break;
        case "KeyR":
          this.reset();
          break;
        case "KeyH":
          this.toggleHeadBob();
          break;
      }
    };

    this._onKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case "KeyW":
        case "ArrowUp":
          this._keys.forward = false;
          break;
        case "KeyS":
        case "ArrowDown":
          this._keys.backward = false;
          break;
        case "KeyA":
        case "ArrowLeft":
          this._keys.left = false;
          break;
        case "KeyD":
        case "ArrowRight":
          this._keys.right = false;
          break;
        case "ShiftLeft":
        case "ShiftRight":
          this._keys.sprint = false;
          break;
      }
    };

    // Pointer lock change
    this._onPointerLockChange = () => {
      const locked = document.pointerLockElement === this.domElement;
      this._isLocked = locked;
      if (!locked && !this.isTouch) {
        this.resetKeys();
      }
      this.onLockChange?.(locked);
    };

    // Touch pointer events
    this._onPointerDown = (e: PointerEvent) => {
      if (!this.isTouch) return;

      const isLeftThird = e.clientX < window.innerWidth / 3;
      if (isLeftThird && this._joystickPointerId === null) {
        this._joystickPointerId = e.pointerId;
        try {
          this.domElement.setPointerCapture(e.pointerId);
        } catch {}
        this._joystickOriginX = e.clientX;
        this._joystickOriginY = e.clientY;
        this._touchMoveX = 0;
        this._touchMoveZ = 0;

        if (this._joystickContainer) {
          this._joystickContainer.style.left = `${e.clientX}px`;
          this._joystickContainer.style.top = `${e.clientY}px`;
          this._joystickContainer.classList.remove("hidden");
        }
        if (this._joystickThumb) {
          this._joystickThumb.style.transform = `translate(0px, 0px)`;
        }
      } else if (!isLeftThird && this._lookPointerId === null) {
        this._lookPointerId = e.pointerId;
        try {
          this.domElement.setPointerCapture(e.pointerId);
        } catch {}
        this._lastLookX = e.clientX;
        this._lastLookY = e.clientY;
      }
    };

    this._onPointerMove = (e: PointerEvent) => {
      if (!this.isTouch) return;

      if (e.pointerId === this._joystickPointerId) {
        const dx = e.clientX - this._joystickOriginX;
        const dy = e.clientY - this._joystickOriginY;
        const dist = Math.hypot(dx, dy);
        const maxR = 50; // max joystick radius 50px
        const deadZone = maxR * 0.10; // 10% dead zone = 5px

        const clampedDist = Math.min(dist, maxR);
        const angle = Math.atan2(dy, dx);
        const thumbX = Math.cos(angle) * clampedDist;
        const thumbY = Math.sin(angle) * clampedDist;

        if (this._joystickThumb) {
          this._joystickThumb.style.transform = `translate(${thumbX}px, ${thumbY}px)`;
        }

        if (dist < deadZone) {
          this._touchMoveX = 0;
          this._touchMoveZ = 0;
        } else {
          const ratio = (clampedDist - deadZone) / (maxR - deadZone);
          // Thumb moved right (dx > 0) is strafe right
          this._touchMoveX = (thumbX / clampedDist) * ratio;
          // Thumb pushed forward/up (dy < 0) is move forward (+Z in camera wishDir)
          this._touchMoveZ = -(thumbY / clampedDist) * ratio;
        }
      } else if (e.pointerId === this._lookPointerId) {
        const deltaX = e.clientX - this._lastLookX;
        const deltaY = e.clientY - this._lastLookY;
        this._lastLookX = e.clientX;
        this._lastLookY = e.clientY;

        this._euler.y -= deltaX * this.touchLookSensitivity;
        this._euler.x -= deltaY * this.touchLookSensitivity;
        this._euler.x = clampPitch(this._euler.x, this.pitchLimitDeg);
        this.camera.quaternion.setFromEuler(this._euler);
      }
    };

    const releasePointer = (e: PointerEvent) => {
      if (e.pointerId === this._joystickPointerId) {
        this._joystickPointerId = null;
        this._touchMoveX = 0;
        this._touchMoveZ = 0;
        if (this._joystickContainer) {
          this._joystickContainer.classList.add("hidden");
        }
        if (this._joystickThumb) {
          this._joystickThumb.style.transform = `translate(0px, 0px)`;
        }
        try {
          this.domElement.releasePointerCapture(e.pointerId);
        } catch {}
      } else if (e.pointerId === this._lookPointerId) {
        this._lookPointerId = null;
        try {
          this.domElement.releasePointerCapture(e.pointerId);
        } catch {}
      }
    };

    this._onPointerUp = releasePointer;
    this._onPointerCancel = releasePointer;

    // Window blur: release all keys and virtual joystick so camera never keeps drifting
    this._onWindowBlur = () => {
      this.resetKeys();
      this._joystickPointerId = null;
      this._lookPointerId = null;
      this._touchMoveX = 0;
      this._touchMoveZ = 0;
      if (this._joystickContainer) {
        this._joystickContainer.classList.add("hidden");
      }
    };

    // Add listeners
    document.addEventListener("mousemove", this._onMouseMove, false);
    window.addEventListener("keydown", this._onKeyDown, false);
    window.addEventListener("keyup", this._onKeyUp, false);
    document.addEventListener("pointerlockchange", this._onPointerLockChange, false);
    window.addEventListener("blur", this._onWindowBlur, false);

    if (this.isTouch) {
      this.domElement.addEventListener("pointerdown", this._onPointerDown, false);
      this.domElement.addEventListener("pointermove", this._onPointerMove, false);
      this.domElement.addEventListener("pointerup", this._onPointerUp, false);
      this.domElement.addEventListener("pointercancel", this._onPointerCancel, false);
    }
  }

  get isLocked(): boolean {
    return this._isLocked;
  }

  get currentYawRad(): number {
    return this._euler.y;
  }

  get currentYawDeg(): number {
    return (this._euler.y * 180) / Math.PI;
  }

  get currentPitchDeg(): number {
    return (this._euler.x * 180) / Math.PI;
  }

  get velocity(): THREE.Vector3 {
    return this._velocity;
  }

  setGrid(grid: OccupancyGrid): void {
    this._grid = grid;
  }

  setSceneBounds(bounds: SceneBounds): void {
    this._sceneBounds = bounds;
  }

  toggleHeadBob(): boolean {
    this.headBobEnabled = !this.headBobEnabled;
    this.onHeadBobChange?.(this.headBobEnabled);
    return this.headBobEnabled;
  }

  toggleWalkableLimit(): boolean {
    this.limitToWalkable = !this.limitToWalkable;
    this.onWalkableLimitChange?.(this.limitToWalkable);
    return this.limitToWalkable;
  }

  startTouchSession(): void {
    this._isLocked = true;
    this.onLockChange?.(true);
  }

  requestLock(): void {
    if (this.isTouch) {
      this.startTouchSession();
    } else {
      this.domElement.requestPointerLock();
    }
  }

  unlock(): void {
    if (document.pointerLockElement === this.domElement) {
      document.exitPointerLock();
    }
    this._isLocked = false;
    this.resetKeys();
    this.onLockChange?.(false);
  }

  resetKeys(): void {
    this._keys.forward = false;
    this._keys.backward = false;
    this._keys.left = false;
    this._keys.right = false;
    this._keys.sprint = false;
    this._touchMoveX = 0;
    this._touchMoveZ = 0;
    this._wishDir.set(0, 0, 0);
  }

  setSpawn(spawnX: number, floorY: number, spawnZ: number, spawnYaw: number): void {
    this._fixedEyeHeight = floorY + this.eyeHeightM;
    let finalX = spawnX;
    let finalZ = spawnZ;

    // Spawn fix: if spawn is blocked, move to nearest free cell centre
    if (this._grid) {
      const fix = fixSpawnPosition(
        this._grid,
        spawnX,
        spawnZ,
        this.playerRadius,
        this.limitToWalkable
      );
      finalX = fix.x;
      finalZ = fix.z;
    }

    if (this._sceneBounds) {
      const clamped = clampToBounds(finalX, finalZ, this._sceneBounds);
      finalX = clamped.x;
      finalZ = clamped.z;
    }

    this._spawnPosition.set(finalX, this._fixedEyeHeight, finalZ);
    this._spawnYaw = spawnYaw;
    this.reset();
  }

  reset(): void {
    this.camera.position.copy(this._spawnPosition);
    this._euler.set(0, this._spawnYaw, 0, "YXZ");
    this.camera.quaternion.setFromEuler(this._euler);
    this._velocity.set(0, 0, 0);
    this._bobTimer = 0;
    this.resetKeys();
  }

  update(dt: number): void {
    const effectiveDt = Math.min(dt, this.maxDt);

    const keyX = (this._keys.right ? 1 : 0) - (this._keys.left ? 1 : 0);
    const keyZ = (this._keys.forward ? 1 : 0) - (this._keys.backward ? 1 : 0);

    // Combine keyboard input with touch joystick input
    const inputX = Math.abs(this._touchMoveX) > 1e-4 ? this._touchMoveX : keyX;
    const inputZ = Math.abs(this._touchMoveZ) > 1e-4 ? this._touchMoveZ : keyZ;

    const isMoving = computeWishDirection(inputX, inputZ, this._euler.y, this._wishDir);
    const targetSpeed = this._keys.sprint ? this.sprintSpeed : this.walkSpeed;

    updateVelocity(
      this._velocity,
      this._wishDir,
      isMoving,
      targetSpeed,
      effectiveDt,
      this.acceleration,
      this.frictionCoeff,
      this.stopDecel
    );

    const dx = this._velocity.x * effectiveDt;
    const dz = this._velocity.z * effectiveDt;

    if (this._grid) {
      // 2D Occupancy Grid Collision with Wall Sliding and Anti-Tunneling Sub-stepping
      resolveMovement(
        this._grid,
        this.camera.position.x,
        this.camera.position.z,
        dx,
        dz,
        this.playerRadius,
        this.limitToWalkable,
        this._moveResult
      );

      let newX = this._moveResult.x;
      let newZ = this._moveResult.z;

      if (this._sceneBounds) {
        const clamped = clampToBounds(newX, newZ, this._sceneBounds);
        newX = clamped.x;
        newZ = clamped.z;
      }

      this.camera.position.x = newX;
      this.camera.position.z = newZ;

      // Zero velocity component if hit wall on that axis to prevent sticky sliding snag
      if (this._moveResult.hitX) {
        this._velocity.x = 0;
      }
      if (this._moveResult.hitZ) {
        this._velocity.z = 0;
      }
    } else {
      let newX = this.camera.position.x + dx;
      let newZ = this.camera.position.z + dz;

      if (this._sceneBounds) {
        const clamped = clampToBounds(newX, newZ, this._sceneBounds);
        newX = clamped.x;
        newZ = clamped.z;
      }

      this.camera.position.x = newX;
      this.camera.position.z = newZ;
    }

    // Eye height maintenance and optional Head-bob (Key H)
    const currentSpeed = Math.hypot(this._velocity.x, this._velocity.z);
    if (this.headBobEnabled && currentSpeed > 0.1) {
      this._bobTimer += effectiveDt * 10.0;
      const bobOffset = Math.sin(this._bobTimer) * this.headBobAmplitude;
      this.camera.position.y = this._fixedEyeHeight + bobOffset;
    } else {
      this._bobTimer = 0;
      this.camera.position.y = this._fixedEyeHeight;
    }
  }

  dispose(): void {
    document.removeEventListener("mousemove", this._onMouseMove);
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    document.removeEventListener("pointerlockchange", this._onPointerLockChange);
    window.removeEventListener("blur", this._onWindowBlur);

    if (this.isTouch) {
      this.domElement.removeEventListener("pointerdown", this._onPointerDown);
      this.domElement.removeEventListener("pointermove", this._onPointerMove);
      this.domElement.removeEventListener("pointerup", this._onPointerUp);
      this.domElement.removeEventListener("pointercancel", this._onPointerCancel);
    }
  }
}
