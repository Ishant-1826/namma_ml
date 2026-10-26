import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { FirstPersonControls, EYE_HEIGHT_M } from "./controls";
import { OccupancyGrid, SceneBounds, isPointInUnknownCell } from "./collision";
import { Minimap } from "./minimap";

// Asset base from Vite env (default empty = same origin)
const ASSET_BASE = (import.meta.env.VITE_ASSET_BASE || "").replace(/\/+$/, "");

export interface SceneManifest {
  scene: string;
  units: string;
  coordinate_convention: string;
  bounds: SceneBounds;
  floor_y: number;
  ceiling_y: number;
  spawn_position: [number, number, number];
  spawn_yaw: number;
  spawn_yaw_deg?: number;
  splat_count: number;
  splat_file: string;
  splat_files?: {
    splat: string;
    spz?: string;
  };
  splat_rotation_euler: [number, number, number];
  splat_rotation_note?: string;
}

export type QualityTier = "low" | "med" | "high";

interface QualityConfig {
  name: QualityTier;
  label: string;
  resScale: number;
  sparkLodSplatScale: number;
  sparkLodRenderScale: number;
  sparkMaxStdDev: number;
}

const QUALITY_PRESETS: Record<QualityTier, QualityConfig> = {
  low: {
    name: "low",
    label: "Low (0.5x)",
    resScale: 0.5,
    sparkLodSplatScale: 0.5,
    sparkLodRenderScale: 2.0,
    sparkMaxStdDev: 2.0,
  },
  med: {
    name: "med",
    label: "Med (0.75x)",
    resScale: 0.75,
    sparkLodSplatScale: 0.75,
    sparkLodRenderScale: 1.5,
    sparkMaxStdDev: 2.45,
  },
  high: {
    name: "high",
    label: "High (1.0x)",
    resScale: 1.0,
    sparkLodSplatScale: 1.0,
    sparkLodRenderScale: 1.0,
    sparkMaxStdDev: Math.sqrt(8), // Spark default ~2.828
  },
};

const STORAGE_KEY_QUALITY = "namma_space_quality";

function loadSavedQuality(): QualityTier {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_QUALITY);
    if (saved === "low" || saved === "med" || saved === "high") {
      return saved;
    }
  } catch (err) {
    console.warn("[Viewer] Failed to read quality from localStorage", err);
  }
  return "med";
}

function saveQuality(quality: QualityTier): void {
  try {
    localStorage.setItem(STORAGE_KEY_QUALITY, quality);
  } catch (err) {
    console.warn("[Viewer] Failed to write quality to localStorage", err);
  }
}

// DOM Elements
const canvasContainer = document.getElementById("canvas-container") as HTMLDivElement;
const crosshair = document.getElementById("crosshair") as HTMLDivElement;
const hudUnreconstructed = document.getElementById("hud-unreconstructed") as HTMLDivElement;
const minimapContainer = document.getElementById("minimap-container") as HTMLDivElement;
const joystickContainer = document.getElementById("joystick-container") as HTMLDivElement;
const joystickThumb = document.getElementById("joystick-thumb") as HTMLDivElement;

const startOverlay = document.getElementById("start-overlay") as HTMLDivElement;
const startBtn = document.getElementById("start-btn") as HTMLButtonElement;
const startBtnTitle = document.getElementById("start-btn-title") as HTMLSpanElement;
const startBtnSub = document.getElementById("start-btn-sub") as HTMLSpanElement;
const controlsList = document.getElementById("controls-list") as HTMLDivElement;

const btnQualityLow = document.getElementById("btn-quality-low") as HTMLButtonElement;
const btnQualityMed = document.getElementById("btn-quality-med") as HTMLButtonElement;
const btnQualityHigh = document.getElementById("btn-quality-high") as HTMLButtonElement;

const statsPanel = document.getElementById("stats-panel") as HTMLDivElement;
const statsFpsVal = document.getElementById("stats-fps") as HTMLSpanElement;
const statsFrameTimeVal = document.getElementById("stats-frametime") as HTMLSpanElement;
const statsSplatsVal = document.getElementById("stats-splats") as HTMLSpanElement;
const statsFileVal = document.getElementById("stats-file") as HTMLSpanElement;
const statsLoadTimeVal = document.getElementById("stats-loadtime") as HTMLSpanElement;
const statsPosVal = document.getElementById("stats-pos") as HTMLSpanElement;
const statsYawVal = document.getElementById("stats-yaw") as HTMLSpanElement;
const statsQualityVal = document.getElementById("stats-quality") as HTMLSpanElement;
const statsWalkableVal = document.getElementById("stats-walkable") as HTMLSpanElement;
const statsCollisionVal = document.getElementById("stats-collision") as HTMLSpanElement;
const statsHeadbobVal = document.getElementById("stats-headbob") as HTMLSpanElement;
const statsMinimapVal = document.getElementById("stats-minimap") as HTMLSpanElement;
const statsExploreVal = document.getElementById("stats-explore") as HTMLSpanElement;

const loadingOverlay = document.getElementById("loading-overlay") as HTMLDivElement;
const loadingSubtitle = document.getElementById("loading-subtitle") as HTMLDivElement;
const progressTrack = document.getElementById("progress-track") as HTMLDivElement;
const progressBar = document.getElementById("progress-bar") as HTMLDivElement;

const errorOverlay = document.getElementById("error-overlay") as HTMLDivElement;
const errorMessage = document.getElementById("error-message") as HTMLDivElement;
const errorUrl = document.getElementById("error-url") as HTMLDivElement;
const retryBtn = document.getElementById("retry-btn") as HTMLButtonElement;

// Three.js Core Variables
let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let spark: SparkRenderer;
let controls: FirstPersonControls;
let minimap: Minimap;
let clock: THREE.Clock;

let currentSplatMesh: SplatMesh | null = null;
let currentManifest: SceneManifest | null = null;
let currentOccupancy: OccupancyGrid | null = null;
let activeQualityTier: QualityTier = loadSavedQuality();

let loadedFileName = "—";
let loadDurationMs = 0;
let loadedSplatsCount = 0;
let expectedSplatsCount = 0;

let statsPanelVisible = false;
let lastStatsUpdateTime = 0;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 60;
let currentFrameTimeMs = 16.6;

// Progress Helpers
function updateProgress(fileName: string, event: ProgressEvent) {
  if (event.lengthComputable && event.total > 0) {
    const pct = Math.min(100, Math.round((event.loaded / event.total) * 100));
    progressTrack.classList.remove("indeterminate");
    progressBar.style.width = `${pct}%`;
    const loadedMB = (event.loaded / (1024 * 1024)).toFixed(1);
    const totalMB = (event.total / (1024 * 1024)).toFixed(1);
    loadingSubtitle.textContent = `Downloading ${fileName}: ${pct}% (${loadedMB} / ${totalMB} MB)`;
  } else if (event.loaded > 0) {
    progressTrack.classList.add("indeterminate");
    const loadedMB = (event.loaded / (1024 * 1024)).toFixed(1);
    loadingSubtitle.textContent = `Downloading ${fileName}: ${loadedMB} MB`;
  } else {
    progressTrack.classList.add("indeterminate");
    loadingSubtitle.textContent = `Loading ${fileName}...`;
  }
}

function showLoading(fileName: string) {
  loadingOverlay.classList.remove("hidden");
  errorOverlay.classList.add("hidden");
  progressBar.style.width = "0%";
  progressTrack.classList.add("indeterminate");
  loadingSubtitle.textContent = `Connecting to ${fileName}...`;
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function showError(url: string, message: string) {
  loadingOverlay.classList.add("hidden");
  errorOverlay.classList.remove("hidden");
  errorUrl.textContent = url;
  errorMessage.textContent = message;
}

// Quality & Resolution Scaling
function applyQualitySettings(tier: QualityTier) {
  activeQualityTier = tier;
  saveQuality(tier);
  const cfg = QUALITY_PRESETS[tier];

  // Update button active state
  btnQualityLow.classList.toggle("active", tier === "low");
  btnQualityMed.classList.toggle("active", tier === "med");
  btnQualityHigh.classList.toggle("active", tier === "high");

  // Keep devicePixelRatio capped at 1.5, scaled by quality preset
  const baseDpr = Math.min(window.devicePixelRatio, 1.5);
  const finalPixelRatio = baseDpr * cfg.resScale;
  renderer.setPixelRatio(finalPixelRatio);

  // Apply Spark LOD / Splat Detail API properties if available
  if (spark) {
    const sparkAny = spark as unknown as {
      lodSplatScale?: number;
      lodRenderScale?: number;
      maxStdDev?: number;
    };
    if ("lodSplatScale" in spark) {
      sparkAny.lodSplatScale = cfg.sparkLodSplatScale;
    }
    if ("lodRenderScale" in spark) {
      sparkAny.lodRenderScale = cfg.sparkLodRenderScale;
    }
    if ("maxStdDev" in spark) {
      sparkAny.maxStdDev = cfg.sparkMaxStdDev;
    }
  }

  console.log(`[Viewer] Quality switched to: ${cfg.label} (DPR: ${finalPixelRatio.toFixed(2)})`);
  statsQualityVal.textContent = cfg.label;
}

// Initialize Three.js, SparkRenderer, FirstPersonControls, and Minimap
function initGraphics() {
  scene = new THREE.Scene();
  // Keep dark background color (#0b0d12) so empty space does not look broken
  scene.background = new THREE.Color(0x0b0d12);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 1000);
  clock = new THREE.Clock();

  // WebGLRenderer: antialias false for 3DGS performance
  renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  canvasContainer.replaceChildren(renderer.domElement);

  // SparkRenderer integration into Three.js scene with quality presets
  const initialQuality = QUALITY_PRESETS[activeQualityTier];
  spark = new SparkRenderer({
    renderer,
    maxStdDev: initialQuality.sparkMaxStdDev,
    lodSplatScale: initialQuality.sparkLodSplatScale,
    lodRenderScale: initialQuality.sparkLodRenderScale,
  });
  scene.add(spark);

  // Minimap initialization
  minimap = new Minimap(minimapContainer);

  // Apply active quality resolution scale and Spark LOD settings
  applyQualitySettings(activeQualityTier);

  // FirstPersonControls attached to canvas
  controls = new FirstPersonControls(camera, renderer.domElement, {
    eyeHeightM: EYE_HEIGHT_M,
    pitchLimitDeg: 70,
    joystickContainer,
    joystickThumb,
  });

  // Telemetry callbacks
  controls.onHeadBobChange = (enabled) => {
    statsHeadbobVal.textContent = enabled ? "ON (Key H)" : "OFF (Key H)";
  };
  controls.onWalkableLimitChange = (enabled) => {
    statsWalkableVal.textContent = enabled ? "ON" : "OFF";
  };
  controls.onExploreUnknownChange = (enabled) => {
    statsExploreVal.textContent = enabled ? "ON (Key U)" : "OFF (Key U)";
  };

  // Lock / touch state listener: toggle overlay & crosshair
  controls.onLockChange = (locked) => {
    if (locked) {
      startOverlay.classList.add("hidden");
      crosshair.classList.add("visible");
    } else {
      startOverlay.classList.remove("hidden");
      crosshair.classList.remove("visible");
    }
  };

  // Touch device adaptations (detected via pointer: coarse / ontouchstart)
  if (controls.isTouch) {
    console.log("[Viewer] Touch device detected: configuring touch-friendly UI and virtual joystick");
    if (startBtnTitle) startBtnTitle.textContent = "Tap to Start";
    if (startBtnSub) startBtnSub.textContent = "Left 1/3: Move • Right 2/3: Look";
    if (controlsList) {
      controlsList.innerHTML = `
        <div class="control-row">
          <span class="key-badge">Left 1/3 Screen</span>
          <span class="control-desc">Virtual joystick to move</span>
        </div>
        <div class="control-row">
          <span class="key-badge">Right 2/3 Screen</span>
          <span class="control-desc">Drag to look around</span>
        </div>
        <div class="control-row">
          <span class="key-badge">Tap Anywhere</span>
          <span class="control-desc">Start exploring scene</span>
        </div>
      `;
    }
  }

  // Canvas click / tap
  renderer.domElement.addEventListener("click", () => {
    if (!controls.isLocked) {
      controls.requestLock();
    }
  });

  // Start button click / tap
  startBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    controls.requestLock();
  });

  // Quality button event listeners
  btnQualityLow.addEventListener("click", (e) => {
    e.stopPropagation();
    applyQualitySettings("low");
  });
  btnQualityMed.addEventListener("click", (e) => {
    e.stopPropagation();
    applyQualitySettings("med");
  });
  btnQualityHigh.addEventListener("click", (e) => {
    e.stopPropagation();
    applyQualitySettings("high");
  });

  // Keyboard shortcut listener for Quality (1, 2, 3), F3 Stats, Minimap (B), Head-bob (H), Walkable (L)
  window.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.code === "Digit1" || e.key === "1") {
      applyQualitySettings("low");
    } else if (e.code === "Digit2" || e.key === "2") {
      applyQualitySettings("med");
    } else if (e.code === "Digit3" || e.key === "3") {
      applyQualitySettings("high");
    } else if (e.code === "KeyB") {
      const isVisible = minimap.toggle();
      statsMinimapVal.textContent = isVisible ? "ON (Key B)" : "OFF (Key B)";
    } else if (e.code === "KeyH") {
      // FirstPersonControls internal listener toggles it; update DOM
      statsHeadbobVal.textContent = controls.headBobEnabled ? "ON (Key H)" : "OFF (Key H)";
    } else if (e.code === "KeyL") {
      const limit = controls.toggleWalkableLimit();
      statsWalkableVal.textContent = limit ? "ON" : "OFF";
    } else if (e.code === "KeyU") {
      statsExploreVal.textContent = controls.exploreUnknown ? "ON (Key U)" : "OFF (Key U)";
    } else if (e.code === "F3" || e.key === "F3") {
      e.preventDefault();
      statsPanelVisible = !statsPanelVisible;
      statsPanel.classList.toggle("hidden", !statsPanelVisible);
      if (statsPanelVisible) {
        updateStatsDOM();
      }
    }
  });

  // Window resize handler
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    const baseDpr = Math.min(window.devicePixelRatio, 1.5);
    renderer.setPixelRatio(baseDpr * QUALITY_PRESETS[activeQualityTier].resScale);
  });

  // Render loop: ZERO allocations per frame and stops while tab is hidden
  renderer.setAnimationLoop(() => {
    // If tab is hidden, flush clock delta and return immediately
    if (document.hidden) {
      clock.getDelta();
      return;
    }

    const dt = Math.min(clock.getDelta(), 0.05);
    const now = performance.now();

    controls.update(dt);
    renderer.render(scene, camera);

    // Update unreconstructed area HUD badge (Key U Explore Mode)
    if (controls.exploreUnknown && currentOccupancy) {
      const inUnknown = isPointInUnknownCell(
        currentOccupancy,
        camera.position.x,
        camera.position.z
      );
      hudUnreconstructed.classList.toggle("hidden", !inUnknown);
    } else {
      hudUnreconstructed.classList.add("hidden");
    }

    // Update minimap player heading and dot at most 10 times per second
    minimap.update(camera.position.x, camera.position.z, controls.currentYawRad, now);

    // Frame rate measurement
    frameCount++;
    const elapsedFps = now - lastFpsTime;
    if (elapsedFps >= 500) {
      currentFps = Math.round((frameCount * 1000) / elapsedFps);
      currentFrameTimeMs = elapsedFps / frameCount;
      frameCount = 0;
      lastFpsTime = now;
    }

    // Update stats DOM at most 4 times per second (>= 250 ms)
    if (statsPanelVisible && now - lastStatsUpdateTime >= 250) {
      lastStatsUpdateTime = now;
      updateStatsDOM();
    }
  });
}

function updateStatsDOM(): void {
  statsFpsVal.textContent = `${currentFps} FPS`;
  statsFrameTimeVal.textContent = `${currentFrameTimeMs.toFixed(1)} ms`;
  statsSplatsVal.textContent = expectedSplatsCount > 0
    ? `${loadedSplatsCount.toLocaleString()} / ${expectedSplatsCount.toLocaleString()}`
    : `${loadedSplatsCount.toLocaleString()}`;
  statsFileVal.textContent = loadedFileName;
  statsLoadTimeVal.textContent = loadDurationMs < 1000
    ? `${Math.round(loadDurationMs)} ms`
    : `${(loadDurationMs / 1000).toFixed(2)} s`;
  statsPosVal.textContent = `X: ${camera.position.x.toFixed(2)}, Y: ${camera.position.y.toFixed(2)}, Z: ${camera.position.z.toFixed(2)}`;
  statsYawVal.textContent = `${controls.currentYawDeg.toFixed(1)}° (North = -Z)`;
  statsQualityVal.textContent = QUALITY_PRESETS[activeQualityTier].label;
  statsWalkableVal.textContent = controls.limitToWalkable ? "ON" : "OFF";
  statsCollisionVal.textContent = currentOccupancy ? `ON (r=${controls.playerRadius}m)` : "OFF";
  statsHeadbobVal.textContent = controls.headBobEnabled ? "ON (Key H)" : "OFF (Key H)";
  statsMinimapVal.textContent = minimap.isVisible ? "ON (Key B)" : "OFF (Key B)";
  statsExploreVal.textContent = controls.exploreUnknown ? "ON (Key U)" : "OFF (Key U)";
}

// Fetch Manifest, Load Occupancy Grid, and Load Splat Assets
async function loadScene() {
  const sceneJsonUrl = `${ASSET_BASE}/scene/scene.json`;
  showLoading("scene.json");

  let manifest: SceneManifest;
  try {
    const res = await fetch(sceneJsonUrl);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    manifest = await res.json();
    currentManifest = manifest;
    expectedSplatsCount = manifest.splat_count || 0;
    controls.setSceneBounds(manifest.bounds);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Viewer] Failed to fetch scene manifest from ${sceneJsonUrl}:`, err);
    showError(sceneJsonUrl, `Failed to fetch scene.json: ${msg}`);
    return;
  }

  // Load Occupancy Grid for Phase 8 Movement Constraints and Minimap
  const occJsonUrl = `${ASSET_BASE}/scene/occupancy.json`;
  try {
    console.log(`[Viewer] Loading occupancy grid from ${occJsonUrl}`);
    const occRes = await fetch(occJsonUrl);
    if (occRes.ok) {
      const occData = await occRes.json();
      currentOccupancy = occData as OccupancyGrid;
      controls.setGrid(currentOccupancy);
      minimap.setGrid(currentOccupancy);
      console.log(
        `[Viewer] Occupancy grid loaded: ${occData.width}x${occData.height} cells, ` +
        `cell_size=${occData.cell_size}m, origin=[${occData.origin.join(", ")}], ` +
        `inflation=${occData.inflation_radius_m}m`
      );
    } else {
      console.warn(`[Viewer] Occupancy grid not found at ${occJsonUrl} (HTTP ${occRes.status}). Proceeding without collision.`);
    }
  } catch (occErr) {
    console.warn(`[Viewer] Failed to load occupancy grid from ${occJsonUrl}:`, occErr);
  }

  // Initialize FirstPersonControls spawn:
  // Use spawn_position x and z, eye height for y (floor_y + EYE_HEIGHT_M = 0 + 1.45 = 1.45m), and spawn_yaw
  const spawnX = manifest.spawn_position[0];
  const spawnZ = manifest.spawn_position[2];
  const spawnYaw = manifest.spawn_yaw;
  controls.setSpawn(spawnX, manifest.floor_y, spawnZ, spawnYaw);

  // Identify splat file candidates
  const spzFileName = manifest.splat_files?.spz || "scene.spz";
  const splatFileName = manifest.splat_files?.splat || manifest.splat_file || "scene.splat";

  const spzUrl = `${ASSET_BASE}/scene/${spzFileName}`;
  const splatUrl = `${ASSET_BASE}/scene/${splatFileName}`;

  if (currentSplatMesh) {
    scene.remove(currentSplatMesh);
    currentSplatMesh.dispose();
    currentSplatMesh = null;
  }

  // Attempt 1: Load .spz first
  showLoading(spzFileName);
  let loadedMesh: SplatMesh | null = null;
  let fileLoaded = "";
  let durationMs = 0;
  let lastAttemptedUrl = spzUrl;

  const tStartSpz = performance.now();
  try {
    console.log(`[Viewer] Attempting to load primary format .spz from: ${spzUrl}`);
    const mesh = new SplatMesh({
      url: spzUrl,
      onProgress: (ev) => updateProgress(spzFileName, ev),
    });
    await mesh.initialized;
    loadedMesh = mesh;
    fileLoaded = spzFileName;
    durationMs = performance.now() - tStartSpz;
    console.log(`[Viewer] Successfully loaded .spz in ${durationMs.toFixed(1)} ms`);
  } catch (spzError) {
    console.warn(`[Viewer] .spz loading failed from ${spzUrl}. Falling back to .splat...`, spzError);

    // Attempt 2: Fall back to .splat
    showLoading(splatFileName);
    lastAttemptedUrl = splatUrl;
    const tStartSplat = performance.now();
    try {
      console.log(`[Viewer] Attempting to load fallback format .splat from: ${splatUrl}`);
      const fallbackMesh = new SplatMesh({
        url: splatUrl,
        onProgress: (ev) => updateProgress(splatFileName, ev),
      });
      await fallbackMesh.initialized;
      loadedMesh = fallbackMesh;
      fileLoaded = splatFileName;
      durationMs = performance.now() - tStartSplat;
      console.log(`[Viewer] Successfully loaded fallback .splat in ${durationMs.toFixed(1)} ms`);
    } catch (splatError) {
      console.error(`[Viewer] Both .spz and .splat failed to load.`, splatError);
      const msg = splatError instanceof Error ? splatError.message : String(splatError);
      showError(lastAttemptedUrl, `Failed to load splat files (${spzFileName} and fallback ${splatFileName}): ${msg}`);
      return;
    }
  }

  hideLoading();

  if (!loadedMesh) {
    showError(lastAttemptedUrl, "Unknown error: Splat mesh was not created.");
    return;
  }

  currentSplatMesh = loadedMesh;

  // Apply splat_rotation_euler from scene.json (never negate axes)
  const euler = manifest.splat_rotation_euler || [0, 0, 0];
  currentSplatMesh.rotation.set(euler[0], euler[1], euler[2], "XYZ");
  console.log(`[Viewer] Applied splat_rotation_euler: [${euler.join(", ")}]`);

  // Add splat to scene
  scene.add(currentSplatMesh);

  // Splat count verification
  const count = currentSplatMesh.numSplats ?? (currentSplatMesh.splats ? currentSplatMesh.splats.getNumSplats() : 0);
  loadedSplatsCount = count;
  loadedFileName = fileLoaded;
  loadDurationMs = durationMs;

  console.log(`[Viewer] Loaded Gaussian count: ${count.toLocaleString()} (scene.json expected: ${expectedSplatsCount.toLocaleString()})`);

  // Update stats panel values
  updateStatsDOM();
}

// Setup Retry Button
retryBtn.addEventListener("click", () => {
  if (window.location.search) {
    window.history.replaceState({}, "", window.location.pathname);
  }
  errorOverlay.classList.add("hidden");
  loadScene();
});

// Start Application
initGraphics();
loadScene();
