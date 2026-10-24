# Architecture — Namma Space

## Pipeline Overview

```
Capture (phone video)
  │
  ▼
Frame Extraction          ← pipeline/extract_frames.py
  │  (ffmpeg → JPEGs)
  ▼
Structure-from-Motion     ← pipeline/run_sfm.py
  │  (COLMAP ≥ 4.0, global_mapper)
  ▼
Gravity / Scale Align     ← pipeline/align_scene.py
  │  (orient Y-up, floor y=0, metric scale)
  ▼
Gaussian Splat Training   ← pipeline/train_splat.py
  │  (GPU — may run on a remote machine)
  ▼
Post-process / Compress   ← pipeline/postprocess_scene.py
  │  (prune, compress, export .ply / .splat)
  ▼
Web Viewer                ← viewer/  (Vite + TypeScript)
  │  (first-person roaming, loads splat file)
  ▼
Deploy                    (static hosting — Vercel / GitHub Pages / S3)
```

## Directory Layout

```
Namma_Space/
├── AGENTS.md              # standing rules for agents
├── requirements.txt       # Python deps
├── configs/
│   └── default.yaml       # pipeline configuration
├── pipeline/              # Python package — one CLI per stage
│   ├── __init__.py
│   ├── extract_frames.py
│   ├── run_sfm.py
│   ├── align_scene.py
│   ├── train_splat.py
│   ├── postprocess_scene.py
│   └── utils/
│       └── __init__.py
├── viewer/                # Vite + TypeScript web app
├── scripts/               # helper scripts (check_env.py, etc.)
├── notebooks/             # Jupyter exploration
├── data/<scene>/          # raw inputs  (gitignored)
├── outputs/<scene>/       # stage outputs (gitignored)
├── test/                  # tiny sample dataset (committed later)
└── docs/
    ├── ARCHITECTURE.md    # this file
    ├── SOP.md             # step-by-step operating procedure
    └── STATE.md           # current project state
```

## Key Dependencies

| Dependency       | Purpose                        | Notes                              |
|------------------|--------------------------------|------------------------------------|
| FFmpeg           | Video → frames                 | System install required            |
| COLMAP ≥ 4.0     | SfM / MVS                      | Must have `global_mapper`          |
| OpenCV           | Image I/O, undistortion        | `opencv-python`                    |
| NumPy            | Numerics                       |                                    |
| Open3D           | Point-cloud manipulation       |                                    |
| PyYAML           | Config loading                 |                                    |
| tqdm             | Progress bars                  |                                    |
| Pillow           | Image processing               |                                    |
| trimesh          | Mesh I/O                       |                                    |
| scikit-image     | Image metrics (SSIM, etc.)     |                                    |
| pycolmap         | Python bindings for COLMAP     | Optional — CLI fallback available  |
| Node.js + npm    | Viewer build toolchain         | Vite + TypeScript                  |
| CUDA             | GPU training                   | Required for splat training        |

## Scene Convention (Final Output)

- Coordinate system: **right-handed, Y-up**
- Units: **metres**
- Floor plane: **y = 0**

## Splat Compression & Web Formats

To ensure real-time web delivery and minimal network transfer for the roaming 3D viewer, the pipeline supports multiple Gaussian Splatting distribution formats:

### Format Comparison & Converter Research

1. **Web Standard Binary (`.splat`)**:
   - 32 bytes per Gaussian (position: 12B float32, scale: 12B float32, color: 4B uint8 RGBA, rotation: 4B uint8 normalized quaternion).
   - Generated natively in `pipeline/postprocess_scene.py` (`write_splat_binary`).
   - Universal WebGL/WebGPU viewer support (Antimatter15 standard, Three.js, Spark).

2. **Niantic SPZ (`.spz`)**:
   - High-performance compressed format developed by Niantic (Scaniverse).
   - Employs coordinate quantization and Zstandard (ZSTD) compression streams.
   - Natively supported in Three.js / World Labs Spark (`@sparkjsdev/spark`) and PlayCanvas engine.
   - Converted via `@playcanvas/splat-transform` CLI (`splat-transform input.ply output.spz`).
   - Achieves **~29.4x reduction** from raw training PLY and **~1.8x reduction** over 32-byte `.splat`.

3. **PlayCanvas SOG (`.sog`)**:
   - Spatially Ordered Gaussians with WebP texture packing and Morton ordering.
   - Evaluated during Phase 6 converter research: on Windows headless environments, SOG compression relies on `node-webgpu` Dawn backend which requires specific GPU driver adapters (`vulkan-1.dll`/`d3d12`).
   - `.spz` operates CPU-side via Zstandard stream compression and is chosen as the primary high-compression asset.

### Converter Tool & Exact Command

The official converter used is **`@playcanvas/splat-transform`** (v3.10.3):

```bash
# SPZ conversion command (CPU Zstandard bit-packing):
npx -y @playcanvas/splat-transform -w outputs/<scene>/scene_clean.ply outputs/<scene>/scene.spz

# Windows execution:
cmd.exe /c "npx.cmd --yes @playcanvas/splat-transform -w outputs/<scene>/scene_clean.ply outputs/<scene>/scene.spz"
```

### Benchmark File Sizes (`home` scene)

| Asset File | Format | File Size | Reduction vs Raw PLY | Splats |
|---|---|---|---|---|
| `outputs/home/scene_raw.ply` | Binary PLY (SH Deg 3) | 136.40 MB | Baseline (1.0x) | 606,040 |
| `outputs/home/scene_clean.ply` | Binary PLY (Clean) | 61.74 MB | 2.2x (54.7% pruned) | 274,296 |
| `outputs/home/scene.splat` | 32-byte Web Splat | 8.37 MB | 16.3x (93.9% reduction) | 274,296 |
| `outputs/home/scene.spz` | Compressed SPZ (ZSTD) | 4.64 MB | 29.4x (96.6% reduction) | 274,296 |
