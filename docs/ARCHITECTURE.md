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
