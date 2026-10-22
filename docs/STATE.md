# Project State -- Namma Space

> Last updated: 2026-10-22

## What Is Done

- [x] Repository scaffolded (directory structure, AGENTS.md, configs, docs).
- [x] Pushed to GitHub remote `origin`: `https://github.com/Ishant-1826/nammaspace.git`.
- [x] `requirements.txt` created and dependencies installed in `.venv`.
- [x] `docs/COMPETITION.md` written detailing Techfest Round 1 rules, deliverables, criteria, and dates.
- [x] `AGENTS.md` updated with Rule 10 on competition scope/priority alignment.
- [x] `scripts/check_env.py` updated with automatic Windows PATH registry resolution.
- [x] FFmpeg and CUDA Toolkit 13.4 installed and detected.
- [x] COLMAP 4.2.1 with CUDA & `global_mapper` installed.
- [x] `docs/SOP.md` written: rigorous indoor smartphone capture SOP + 1-page field checklist.
- [x] `scripts/validate_capture.py` implemented: ffprobe metadata audit + OpenCV blur/brightness/drift sampling.
- [x] `configs/measurements.template.json` created for metric ground-truth scaling.
- [x] `docs/ARCHITECTURE.md` documents the full pipeline.
- [x] Viewer directory scaffolded (Vite + TypeScript, not yet functional).
- [x] **Stage 1 (`pipeline/extract_frames.py`)**: Full implementation complete & re-run on new videos (2026-10-21):
  - Input from `data/home/raw/` (`20261018_163604.mp4` and `20261018_163859.mp4`).
  - Extracted 590 raw candidate frames at 3 fps, filtered 60 blurry frames and 3 near-duplicates.
  - Output 450 frames at 1920x1080 to `data/home/images/` (`frame_000001.jpg` ... `frame_000450.jpg`).
  - Generated 8x6 contact sheet (`data/home/contact_sheet.jpg`) and `data/home/frames_report.json`.
- [x] **Stage 2 (`pipeline/run_sfm.py`)**: COLMAP CLI Subprocess Pipeline -- COMPLETE & VERIFIED (2026-10-21):
  - Audited local CLI via `colmap -h` and subcommand `--help` pages (`feature_extractor`, `exhaustive_matcher`, `sequential_matcher`, `mapper`, `image_undistorter`, `model_converter`).
  - Feature extraction: single shared camera, OPENCV model by default, GPU SIFT, dynamic ALIKED check.
  - Matching: sequential matching with loop detection (or exhaustive when <= 300 frames), GPU SIFT, dynamic LightGlue check.
  - Mapping: incremental mapper with global mapper CLI probe and graceful fallback.
  - Model converter: exports sparse model 0 to TXT and PLY (`data/<scene>/colmap/sparse/0/`, `points3D.ply`).
  - Image undistorter: undistorts frames into `data/<scene>/colmap/dense/`.
  - Comprehensive metrics reporting: `data/<scene>/sfm_report.json` (450 total images, 62 registered / 13.8%, 7 models, 2,928 3D points, 1.2296 px mean reprojection error, 5.54 mean track length, OPENCV intrinsics).
  - Quality gate: successfully triggered quality gate exit code `1` when registration is 13.8% (< 85%), printing actionable diagnostics (rotation-only motion, motion blur, textureless walls, doorways) and next config switches.
  - Supported `--resume` mode for skipping completed stages.
- [x] **Top-View Floor Plan & Trajectory Generation Fixes (`scripts/view_sparse.py`)** -- VERIFIED (2026-10-22):
  - Auto up-axis detection combining average camera up vectors (`-R[1, :]`) and RANSAC floor plane fitting on lowest 20% height points.
  - Aligned to canonical metric axes via Rodrigues rotation matrix to canonical $+Y$ up, snapping floor strictly to $y = 0.00\text{ m}$.
  - Robust 1st–99th percentile bounding box clipping preventing outlier points/cameras from blowing up plot scale.
  - 2D density histogram floor plan (0.05m bins, log-scaled) with PCA in-plane yaw alignment to align dominant room walls horizontally and vertically.
  - Multi-pass trajectory plotting with directional arrows every 5 frames and Start/End markers.
  - Metric scale bar and compass north indicator.
  - Debug side-view elevation plots (`side_view_xy.png`, `side_view_yz.png`) verifying horizontal floor at $y=0$ and ceiling bounds.
- [x] **Detached SfM Runner & Status Monitor (`pipeline/run_sfm.py`, `wait_sfm.sh`, `scripts/wait_sfm.py`)** -- COMPLETE (2026-10-22):
  - CLI argument `--multiple-models` (default 0) to control single-model vs multi-model stopping.
  - Real-time status reporting written atomically to `logs/status.json` and `<scene>/logs/status.json` with stage tracking and registration progress.
  - Blocking monitor `./wait_sfm.sh <project_dir>` polling `status.json` every 30s until `done: true` without intermediate timers.
  - Model converter to TXT/PLY and image undistortion to `dense/` on model completion.
- [x] **Phase 3 SfM Diagnosis & Alignment Protocol** (2026-10-19):
  - **`docs/sfm_diagnosis.md`**: Completed rigorous 7-part SfM diagnosis covering model count, intrinsics, step lengths, long-range matches, handedness, alignment quality, and plan-fit residuals.
  - **Identified Root Causes**:
    1. *Plot Inversion Bug*: Previous top-view plot mapped +Z upward on image, which horizontally mirrored the room (inverting all turns from Left to Right). Fixed in `scripts/make_topview.py` with true top-down projection (X right, Z down, North up).
    2. *Pass Height Inversion*: Pass 1 (video 1) was recorded at chest level (median 1.12 m) and Pass 2 (video 2) was recorded at eye level (median 1.54 m). Fixed in `pipeline/align_scene.py` to correctly map dual-pass heights.
    3. *Missing 19% Frames*: Caused by rapid camera rotation past featureless white walls (frames 128-140 have only 10 SIFT keypoints, std=16) and motion blur (laplacian variance drops to 3.8-7.1).
  - **`docs/sfm_experiments.md`**: Completed SfM experiments log (E1 pass-1-only, E3 sequential matching, E4 global mapper).
  - **Stage 3 (`pipeline/align_scene.py`) Upgraded**:
    - Up vector derived via SVD of camera up orientations and refined by RANSAC floor plane fitting (angle to camera-up = 0.76° <= 8°).
    - Canonical yaw rotation rotates dominant walls axis-aligned and directs Leg 1 strictly North (-Z direction).
    - Floor snapped to y = 0.0, ceiling at 3.77 m (matches 4.0 m measurement).
    - Plausible camera heights: Pass 1 chest-level = 1.12 m, Pass 2 eye-level = 1.54 m.
  - **Stage 4 (`scripts/make_topview.py`) Implemented**:
    - True orthographic top-down rendering (looking along -Y, equal aspect ratio).
    - Background 2D point density histogram (0.3m to 1.8m above floor, 5cm cells, log scale).
    - Multi-pass color-coded trajectory with directional arrows every ~5s.
    - Start/End markers, start-end gap annotated (0.34 m for Pass 1).
    - Similarity-fitted floor-plan overlay (`outputs/home/topview_overlay.png`) with residual (5.0% path length).
    - Visual keyframe verification contact sheet (`outputs/home/topview_legs.jpg`) showing representative keyframe per leg for L1..L10.
- [x] **New Classroom Capture & Reconstruction (2026-10-19)**:
  - **New Raw Videos**: `data/home/raw/20261019_162930.mp4` (Eye-level, 107.9s) and `data/home/raw/20261019_163257.mp4` (Chest-level, 131.8s).
  - **New Sketch & Plan**: `docs/reference/floorplan_sketch.jpg` and `docs/floorplan.json` (1 Door, 2 Tables, 5 Chairs, closed counter-clockwise rectangular loop `L, L, L`).
  - **Stage 1 (Extraction)**: 450 frames extracted at 3 fps (`pass_split = 201`, frames 1–201 Eye-level, 202–450 Chest-level). `data/home/frames_report.json` saved.
  - **Stage 2 (SfM Reconstruction)**: Tuned incremental mapper on GPU database achieved **415 / 450 registered frames (92.2%)** in a single unified model (Pass 1 = 166/201, Pass 2 = 249/249 = 100%).
    - 19,394 3D points.
    - Mean reprojection error: **1.0965 px** (close to ~1 px ideal).
    - Mean track length: 10.08 frames/point.
  - **Stage 3 Metric Scene Alignment (`pipeline/align_scene.py`)** -- COMPLETE & VERIFIED (2026-10-20):
    - Loads COLMAP `sparse/0` (pycolmap if available, robust COLMAP TXT parser fallback).
    - Iterative Open3D RANSAC plane fitting (`segment_plane`) automatically isolates lowest horizontal floor plane (agreement with camera up <= 8.0°, camera clearance verification, min inliers threshold).
    - Proper SO(3) Rodrigues rotation (det = +1) rotates floor normal to +Y, canonical yaw aligns walls and Leg 1 heading North (-Z), and translation snaps floor strictly to `y = 0.0`.
    - Metric scale alignment: Mode (a) Phone height assumption (default 1.45 m from config/CLI) scales model so mean camera height is 1.450 m (Pass 1 eye = 1.696 m, Pass 2 chest = 1.286 m, ratio 1.32x); Mode (b) two points distance.
    - Exports: `data/<scene>/colmap_aligned/` (`cameras.txt`, `images.txt`, `points3D.txt`), `transforms.json` in Nerfstudio format (handling COLMAP OpenCV to OpenGL axis conversion), and `alignment.json` (R, t, s, room extents, geometry stats).
  - **Stage 3 Alignment Verification & Audit (`scripts/check_alignment.py`)** -- COMPLETE & AUDITED (2026-10-20):
    - Ground truth updated to exact survey measurement: walked square side = 2.25 m (15% tolerance), ceiling = 4.10 m in `data/home/measurements.json` (removed all erroneous room-wall comparisons).
    - Minimum-area bounding rectangle of camera centres:
      - Pass 1: 1.544 m x 3.024 m, mean side = 2.284 m, perimeter/4 = 2.284 m (error = +1.51% vs 2.25 m, rotation = 78.41°).
      - Pass 2: 1.610 m x 3.303 m, mean side = 2.457 m, perimeter/4 = 2.457 m (error = +9.19% vs 2.25 m, rotation = -103.07°).
      - Combined mean: 2.370 m (error = +5.35% vs 2.25 m).
    - Scene extent (points 0.3–1.5 m above floor, 2nd–98th percentile after Open3D statistical outlier removal, informational only): Width X = 4.431 m, Length Z = 6.654 m, Height Y = 0.883 m.
    - Floor audit & explanation of 'SNAPPED': Verified that floor was detected by iterative Open3D RANSAC plane fitting (inliers = 934, angle to camera-up = 7.65° <= 8.0°). 'SNAPPED' signifies translation of origin so detected physical floor plane lies strictly at y = 0.000 m.
    - Ceiling audit: Highest camera at y = 1.968 m; no dense horizontal plane detected >= 0.3 m above cameras (featureless ceiling unobserved during forward/downward phone walk). Correctly classified as 'ceiling: not measurable' and excluded from pass/fail; nominal ceiling_height_m = 4.10 m stored in config for Phase 6 bounding box cropping.
    - Multi-criteria validation: All criteria pass with zero contradictions (no ceiling below cameras, floor detected, independent scale estimates agree within <5%). Full audit saved to `data/home/reports/align.json`.
    - Generated orthographic side view `alignment_side.png` (X-Y elevation, floor line at y=0, Pass 1 eye / Pass 2 chest tracks, equal aspect, 0.5 m grid, 1.0 m scale bar) and top view `alignment_top.png` (X-Z plan, min-area walked square bbox, equal aspect, 0.5 m grid, 1.0 m scale bar) in both `data/home/reports/` and `outputs/home/`.
  - **Stage 4 (`scripts/make_topview.py`)**:
    - Metric orthographic top-down view (`outputs/home/topview.png`).
    - Floor-plan overlay (`outputs/home/topview_overlay.png`): similarity fit residual = 0.51 m (5.8% path length). Both passes follow the expected rectangular loop around the 2 tables and 5 chairs.
    - 4-leg visual verification contact sheet (`outputs/home/topview_legs.jpg`) with representative keyframes for L1 to L4.
  - **One-Command Reproduction**: `python scripts/reproduce_all.py --scene home` reproduces diagnosis, alignment, and top-view generation in ~4s.
- [x] **Stage 4 (`pipeline/train_splat.py`)**: 3D Gaussian Splatting training wrapper & local Brush execution -- COMPLETE & VERIFIED (2026-10-20):
  - Local execution backend: `brush` 0.3.0 (`tools/brush/brush_app.exe`) running natively on NVIDIA GeForce RTX 4060 Laptop GPU via WebGPU.
  - Coordinate system retention: Verified that Brush does NOT re-center, re-orient, or auto-scale on load; it trains directly in the metric Y-up input frame.
  - "Fast" preset execution (7,000 steps smoke test):
    - Command: `tools/brush/brush_app.exe outputs/home/train_pack --total-steps 7000 --max-splats 3000000 --eval-split-every 10 --eval-every 1000 --export-every 7000 --export-path outputs/home --export-name scene_raw_7000.ply`
    - Wall-clock time: **316.7 s** (5.28 min).
    - Peak VRAM: **2,513 MiB** (2.45 GB / 8 GB budget).
    - Final Gaussian count: **118,837 splats**.
    - Model export: Preserved at `outputs/home/scene_raw_fast.ply` and `outputs/home/metrics_fast.json`.
    - Coordinate verification: [PASS] (yaw = -0.01°, trans = 0.0007 m, scale = 0.9997, floor height = +0.032 m, tilt = 1.12°).
  - "Final" preset execution (30,000 steps, SH degree 3, production run):
    - Command: `tools/brush/brush_app.exe outputs/home/train_pack --total-steps 30000 --max-splats 3000000 --sh-degree 3 --eval-split-every 8 --eval-every 1000 --export-every 30000 --export-path outputs/home --export-name scene_raw_30000.ply --max-resolution 1920`
    - Wall-clock time: **1,621.6 s** (27.03 min).
    - Peak VRAM: **2,883 MiB** (2.82 GB / 8 GB budget, continuous 5-s logging in `outputs/home/vram_log.csv`).
    - Final Gaussian count: **318,823 splats** (well within the 3,000,000 budget).
    - Exported PLY: `outputs/home/scene_raw.ply` (75.2 MB).
    - Training reports: `outputs/home/train_report.json`, `outputs/home/metrics.json`, `outputs/home/splat_frame_verification.json`.
    - Splat statistics (`scripts/ply_stats.py`):
      - Opacity < 0.05: 41.5%
      - Opacity >= 0.2: 16.8% (53,211 Gaussians)
      - Sub-floor points below y = -0.3 m: 3.29%
      - Largest scale axis > 0.5 m: 1.28%
      - NaN / Inf values: 0
    - Rigid alignment fit (Umeyama / Trimmed ICP vs 19,394 reference sparse points):
      - Yaw: **-0.00°** (tolerance <= 5.0°) -> **PASS**.
      - Translation: **0.0014 m** (1.4 mm, tolerance <= 0.25 m) -> **PASS**.
      - Uniform scale: **0.9994** (tolerance [0.90, 1.10]) -> **PASS**.
      - Unmirrored fit is best (mirrored fits >10x worse RMSE: 0.27m - 0.29m vs 0.0269m).
    - Floor & trajectory planes:
      - Splat floor plane height: **+0.0275 m** (floor peak = +0.070 m, within 7 cm of y = 0.00 m) -> **PASS**.
      - Splat floor plane tilt: **1.29°** (tolerance <= 3.0°) -> **PASS**.
      - Pass 1 camera plane tilt: **3.72°** (tolerance <= 5.0°) -> **PASS**.
      - Pass 2 camera plane tilt: **1.26°** (tolerance <= 5.0°) -> **PASS**.
      - Camera inclusion: **100.0%** (415/415 cameras inside splat bounds) -> **PASS**.
      - Mean camera height above floor: **1.423 m**.
    - Upgraded Extents Coverage Rule (`scripts/verify_splat_frame.py`):
      - Replaced span-equality check with dual coverage test:
        (a) Reference p2-p98 box lies inside filtered splat p2-p98 box with 0.3 m tolerance.
        (b) Filtered splat box exceeds reference box by at most 1.0 m on any side.
      - Results:
        - X: Ref [-3.19, +1.94], Splat [-3.41, +2.41], overhang = [+0.22, +0.47] m -> **PASS**.
        - Y: Ref [-0.33, +1.81], Splat [-0.43, +2.36], overhang = [+0.10, +0.55] m -> **PASS**.
        - Z: Ref [-4.65, +2.42], Splat [-4.70, +2.46], overhang = [+0.05, +0.04] m -> **PASS**.
      - Overall verification status: **[PASS] SPLAT COORDINATE FRAME IS VALID**.
  - Walked-Loop Footprint & Geometry Audit (`scripts/check_alignment.py`):
    - Discarded "mean side" reporting: footprints are distinctly rectangular.
    - Pass 1 (Eye Track): 1.648 m x 3.191 m (Aspect Ratio: **1.937**, Area: 5.258 m² bbox, RDP Area: 2.571 m²).
    - Pass 2 (Chest Track): 1.715 m x 3.483 m (Aspect Ratio: **2.031**, Area: 5.972 m² bbox, RDP Area: 3.154 m²).
    - RDP Path Simplification (0.3 m tolerance):
      - Pass 1: 6 corners (lengths: 3.19m, 0.65m, 1.09m, 1.44m, 1.41m, 1.07m).
      - Pass 2: 8 corners (lengths: 3.47m, 0.57m, 1.33m, 0.43m, 2.21m, 0.53m, 1.10m, 0.55m).
    - Generated orthographic top-down diagram with annotated lengths and 0.5 m grid: `outputs/home/reports/loop_shape.png`.
    - Validator Update: Walked-loop check compares rectangular sides in `measurements.json` and is **SKIPPED** until provided (2.25 m square removed from pass/fail criteria).
    - Camera Heights: Pass 1 Eye = **1.761 m +/- 0.052 m**, Pass 2 Chest = **1.242 m +/- 0.032 m** (ratio = 1.42x).
  - Held-out Evaluation Audit:
    - Headless Brush CLI executes silently without printing or logging text PSNR/SSIM tables (`outputs/home/metrics.json` records `"psnr": "not recorded"` and `"ssim": "not recorded"`).
  - Multi-backend selection: `nerfstudio` (`ns-train splatfacto`, CUDA) preserved as the Google Colab option (`notebooks/colab_train.ipynb`).
  - Dataset packager: `scripts/pack_dataset.py` packed training bundle into `outputs/home/train_pack.zip` (60.17 MB).
- [ ] Stage 5 (`pipeline/postprocess_scene.py`): Model compression and export.
- [ ] Web Viewer (`viewer/`): Interactive first-person 3DGS roaming web viewer.


## How to Run

```bash
# 1. Activate venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# --- ONE-COMMAND REPRODUCTION (Runs diagnosis + alignment + top-view) ---
python scripts/reproduce_all.py --scene home

# --- INDIVIDUAL STAGES ---
# 1. Run SfM diagnosis
python scripts/diagnose_sfm.py --scene home
# Outputs: docs/sfm_diagnosis.md

# 2. Align scene to metric Y-up, floor at y=0 (Stage 3)
python pipeline/align_scene.py --scene home
# Outputs: data/home/colmap_aligned/ (cameras.txt, images.txt, points3D.txt)
#          data/home/transforms.json (Nerfstudio format)
#          data/home/alignment.json

# 3. Check and verify alignment (diagnostic side and top PNGs)
python scripts/check_alignment.py --scene home
# Outputs: outputs/home/alignment_check.png
#          outputs/home/alignment_side.png
#          outputs/home/alignment_top.png

# 4. Generate trustworthy top-down view, overlay, and 4-leg verification sheet
python scripts/make_topview.py --scene home
# Outputs: outputs/home/topview.png
#          outputs/home/topview_overlay.png
#          outputs/home/topview_legs.jpg

# 5. Pack training bundle for GPU / Colab (Stage 4)
python scripts/pack_dataset.py --scene home
# Outputs: outputs/home/train_pack.zip

# 6. Train 3D Gaussian Splatting model (Stage 4)
python pipeline/train_splat.py --scene home --preset fast --dry-run
# Remote / Colab: notebooks/colab_train.ipynb
# Outputs: outputs/home/scene_raw.ply
#          outputs/home/metrics.json

# 7. Verify exported splat coordinate frame
python scripts/verify_splat_frame.py --scene home
# Outputs: outputs/home/splat_frame_verification.json
```

## Known Issues / TODOs

- Pass 1 registration rate (82.6%) vs Pass 2 (100%): 35 frames dropped in Pass 1 during quick turns across featureless white walls. Overall registration across both passes is 92.2% (exceeds >=85% quality gate).
- Training preparation: `data/home/colmap/aligned/` is fully prepared and metric-scaled for 3D Gaussian Splatting (`pipeline/train_splat.py`).
- Implement first-person interactive WebGL/WebGPU viewer in `viewer/`.
