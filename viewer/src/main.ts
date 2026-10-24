import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { setupDevView, type SceneManifest } from "./devview";

// Asset base from Vite env (default empty = same origin)
const ASSET_BASE = (import.meta.env.VITE_ASSET_BASE || "").replace(/\/+$/, "");

// DOM Elements
const canvasContainer = document.getElementById("canvas-container") as HTMLDivElement;
const hudFormat = document.getElementById("hud-format") as HTMLDivElement;
const hudFile = document.getElementById("hud-file") as HTMLSpanElement;
const hudTime = document.getElementById("hud-time") as HTMLSpanElement;
const hudSplats = document.getElementById("hud-splats") as HTMLSpanElement;
const hudRot = document.getElementById("hud-rot") as HTMLSpanElement;

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
let currentSplatMesh: SplatMesh | null = null;
let currentDevView: ReturnType<typeof setupDevView> | null = null;

// Progress helpers
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

// Initialize Three.js and Spark Scene
function initGraphics() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 1000);

  // Requirement: devicePixelRatio capped at 1.5, antialias false for 3DGS performance
  renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  canvasContainer.replaceChildren(renderer.domElement);

  // SparkRenderer integration into Three.js scene
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  // Window resize handler
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  });

  // Render loop: no per-frame allocations
  renderer.setAnimationLoop(() => {
    if (currentDevView) {
      currentDevView.controls.update();
    }
    renderer.render(scene, camera);
  });
}

// Fetch Manifest and Load Splat Assets
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
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Viewer] Failed to fetch scene manifest from ${sceneJsonUrl}:`, err);
    showError(sceneJsonUrl, `Failed to fetch scene.json: ${msg}`);
    return;
  }

  // Set up Phase 7a Dev-Only Check View
  if (currentDevView) {
    currentDevView.dispose();
  }
  currentDevView = setupDevView(scene, camera, renderer, manifest);

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

  // Apply splat_rotation_euler from scene.json (proper Euler rotation; never negate axes)
  const euler = manifest.splat_rotation_euler || [0, 0, 0];
  currentSplatMesh.rotation.set(euler[0], euler[1], euler[2], "XYZ");
  console.log(`[Viewer] Applied splat_rotation_euler: [${euler.join(", ")}]`);

  // Add splat to scene
  scene.add(currentSplatMesh);

  // Splat count verification
  const count = currentSplatMesh.numSplats ?? (currentSplatMesh.splats ? currentSplatMesh.splats.getNumSplats() : 0);
  const expectedCount = manifest.splat_count;
  console.log(`[Viewer] Loaded Gaussian count: ${count.toLocaleString()} (scene.json expected: ${expectedCount.toLocaleString()})`);

  // Update HUD
  hudFormat.textContent = fileLoaded.endsWith(".spz") ? "SPZ" : "SPLAT";
  hudFormat.style.color = fileLoaded.endsWith(".spz") ? "#00e5ff" : "#f59e0b";
  hudFormat.style.borderColor = fileLoaded.endsWith(".spz") ? "rgba(0, 229, 255, 0.3)" : "rgba(245, 158, 11, 0.3)";
  hudFile.textContent = fileLoaded;
  hudTime.textContent = durationMs < 1000 ? `${Math.round(durationMs)} ms` : `${(durationMs / 1000).toFixed(2)} s`;
  hudSplats.textContent = `${count.toLocaleString()} / ${expectedCount.toLocaleString()}`;
  hudRot.textContent = `[${euler.join(", ")}]`;
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
