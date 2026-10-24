import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export interface SceneBounds {
  min: [number, number, number];
  max: [number, number, number];
}

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

export interface DevViewObjects {
  controls: OrbitControls;
  boundsMesh: THREE.Mesh;
  gridHelper: THREE.GridHelper;
  axesHelper: THREE.AxesHelper;
  spawnGroup: THREE.Group;
  dispose: () => void;
}

/**
 * Sets up Phase 7a Dev-Only Check View:
 * - OrbitControls around center of scene.json bounds.
 * - Camera placed ~8m back (+Z) and ~5m up (+Y).
 * - Floor grid at y = floor_y (1 m cells).
 * - 1 m axes helper at origin (X red, Y green, Z blue).
 * - Labelled marker at spawn_position with arrow along spawn_yaw.
 * - Semi-transparent box showing scene.json bounds.
 * - Dark background colour (#0b0d12).
 *
 * All dev view code is kept in this single file for straightforward deletion in 7b.
 */
export function setupDevView(
  scene: THREE.Scene,
  camera: THREE.PerspectiveCamera,
  renderer: THREE.WebGLRenderer,
  manifest: SceneManifest
): DevViewObjects {
  // 1. Dark background colour (#0b0d12) so empty space does not look like an error
  scene.background = new THREE.Color(0x0b0d12);

  // 2. Compute center and extents of scene.json bounds
  const bMin = new THREE.Vector3(...manifest.bounds.min);
  const bMax = new THREE.Vector3(...manifest.bounds.max);
  const center = new THREE.Vector3().addVectors(bMin, bMax).multiplyScalar(0.5);
  const size = new THREE.Vector3().subVectors(bMax, bMin);

  // 3. Camera placed ~8 m back (+Z) and ~5 m up (+Y) from center so full hall is visible
  camera.position.set(center.x, center.y + 5.0, center.z + 8.0);
  camera.lookAt(center);

  // 4. OrbitControls around center of bounds
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.copy(center);
  controls.update();

  // 5. Floor grid at y = floor_y (1 m cell spacing)
  const maxSpan = Math.max(size.x, size.z);
  const gridCells = Math.max(30, Math.ceil(maxSpan + 10));
  const gridHelper = new THREE.GridHelper(gridCells, gridCells, 0x3d5a80, 0x1f2836);
  gridHelper.position.set(center.x, manifest.floor_y, center.z);
  scene.add(gridHelper);

  // 6. 1 m axes helper at origin (X red, Y green, Z blue)
  const axesHelper = new THREE.AxesHelper(1.0);
  axesHelper.position.set(0, 0, 0);
  scene.add(axesHelper);

  // 7. Labelled marker at spawn_position with arrow along spawn_yaw
  const spawnGroup = new THREE.Group();
  const spawnPos = new THREE.Vector3(...manifest.spawn_position);

  // Spherical pin marker at spawn position
  const markerGeo = new THREE.SphereGeometry(0.12, 16, 16);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
  const markerMesh = new THREE.Mesh(markerGeo, markerMat);
  markerMesh.position.copy(spawnPos);
  spawnGroup.add(markerMesh);

  // Concentric rings at floor level beneath spawn
  const ringGeo = new THREE.RingGeometry(0.25, 0.32, 32);
  ringGeo.rotateX(-Math.PI / 2);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ff88, side: THREE.DoubleSide });
  const ringMesh = new THREE.Mesh(ringGeo, ringMat);
  ringMesh.position.set(spawnPos.x, manifest.floor_y + 0.01, spawnPos.z);
  spawnGroup.add(ringMesh);

  // Vertical stem line connecting floor ring to eye spawn marker
  const stemPoints = [
    new THREE.Vector3(spawnPos.x, manifest.floor_y, spawnPos.z),
    new THREE.Vector3(spawnPos.x, spawnPos.y, spawnPos.z),
  ];
  const stemGeo = new THREE.BufferGeometry().setFromPoints(stemPoints);
  const stemMat = new THREE.LineBasicMaterial({ color: 0x00ff88, transparent: true, opacity: 0.6 });
  const stemLine = new THREE.Line(stemGeo, stemMat);
  spawnGroup.add(stemLine);

  // Direction arrow: yaw 0 looks along -Z, CCW seen from above -> dir = (-sin(yaw), 0, -cos(yaw))
  const yaw = manifest.spawn_yaw;
  const arrowDir = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw)).normalize();
  const arrowHelper = new THREE.ArrowHelper(arrowDir, spawnPos, 0.9, 0x00ff88, 0.22, 0.12);
  spawnGroup.add(arrowHelper);

  // Canvas label sprite
  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 100;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "rgba(11, 13, 18, 0.88)";
    ctx.fillRect(4, 4, 392, 92);
    ctx.strokeStyle = "#00ff88";
    ctx.lineWidth = 2.5;
    ctx.strokeRect(4, 4, 392, 92);

    ctx.fillStyle = "#00ff88";
    ctx.font = "bold 26px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`Spawn (${spawnPos.x.toFixed(2)}, ${spawnPos.y.toFixed(2)}, ${spawnPos.z.toFixed(2)})`, 200, 36);

    ctx.fillStyle = "#e0e6ed";
    ctx.font = "20px monospace";
    const deg = manifest.spawn_yaw_deg ?? (yaw * 180) / Math.PI;
    ctx.fillText(`Yaw: ${deg.toFixed(1)}° (North = -Z)`, 200, 68);
  }
  const texture = new THREE.CanvasTexture(canvas);
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const labelSprite = new THREE.Sprite(spriteMat);
  labelSprite.position.set(spawnPos.x, spawnPos.y + 0.42, spawnPos.z);
  labelSprite.scale.set(1.6, 0.4, 1.0);
  spawnGroup.add(labelSprite);

  scene.add(spawnGroup);

  // 8. Semi-transparent box showing scene.json bounds
  const boxGeo = new THREE.BoxGeometry(size.x, size.y, size.z);
  const boxMat = new THREE.MeshBasicMaterial({
    color: 0x3a86ff,
    transparent: true,
    opacity: 0.08,
    depthWrite: false,
  });
  const boundsMesh = new THREE.Mesh(boxGeo, boxMat);
  boundsMesh.position.copy(center);

  // Wireframe outline for clear perception
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x64b5f6,
    wireframe: true,
    transparent: true,
    opacity: 0.35,
  });
  const wireMesh = new THREE.Mesh(boxGeo, wireMat);
  boundsMesh.add(wireMesh);
  scene.add(boundsMesh);

  return {
    controls,
    boundsMesh,
    gridHelper,
    axesHelper,
    spawnGroup,
    dispose: () => {
      controls.dispose();
      scene.remove(gridHelper);
      scene.remove(axesHelper);
      scene.remove(spawnGroup);
      scene.remove(boundsMesh);
      gridHelper.geometry.dispose();
      axesHelper.geometry.dispose();
      markerGeo.dispose();
      markerMat.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      stemGeo.dispose();
      stemMat.dispose();
      texture.dispose();
      spriteMat.dispose();
      boxGeo.dispose();
      boxMat.dispose();
      wireMat.dispose();
    },
  };
}
