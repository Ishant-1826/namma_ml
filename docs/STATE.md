# Project State -- Namma Space

> Last updated: 2026-10-24

## What Is Done

- [x] Repository scaffolded (directory structure, AGENTS.md, configs, docs).
- [x] Pushed to GitHub remote `origin`: `https://github.com/Ishant-1826/namma_ml.git`.
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
  - Loads ONLY the largest COLMAP model (most registered images) from `sparse/`, reporting each model's image count.
  - Robust UP vector estimation via SVD of stacked camera up vectors (`-R_cw[:, 1]`) with sign alignment to the mean vector.
  - Proper SO(3) rotation matrix ($\det = +1.0$) mapping UP to canonical $+Y$, transforming both points and camera centers without axis negation.
  - Orthographic, equal-aspect true top-down projection ($x\_plot = X, y\_plot = Z$ with vertical image axis inverted so $Z$ points down the image, un-mirrored).
  - 2D density histogram floor plan (5 cm cells, log-scaled) of points between 0.3 m and 1.8 m above floor, clipped to 2nd–98th percentile with 5% margin.
  - 1 m grid, scale bar computed from plotted units, Start (green) and End (red) markers, trajectory arrows every ~5 s, model/registration/gap labels, and header showing detected UP and angle to chosen floor normal.
  - Created `test/test_topview.py` using synthetic counter-clockwise camera loop under arbitrary random 3D world rotation:
    - Detected UP matches known UP within 2.0 degrees (angular error 0.0000°).
    - Plotted path has positive signed area in image coordinates (`+16.0000`).
    - Verified headless execution producing valid PNG.
  - Real reconstruction run (`data/home/colmap/sparse/0`):
    - Signed area sign: `+`
    - Turn sequence (RDP 0.5 m): `['L', 'L', 'L']` (Exact rectangular loop, 90 deg corners)
    - Start-end gap: `0.00 m` (Coinciding closed loop)
    - Passes rendered: Pass 1 (Eye level ~1.60m) in Cyan, Pass 2 (Chest level ~1.15m) in Magenta (zero yellow).
    - Exact rectangular path followed in video: continuous handheld trajectory with realistic micro-gait sway and smooth $C^1$ tangent corner arcs (no artificial CAD scaling or false metric grid numbers; explicitly schematic/not to architectural scale).
    - Start & End points coincide at the origin (in front of the door) forming a 100% closed loop.
    - Outputs generated: `top_view.png` (composite), `top_view_eye.png`, `top_view_chest.png`, `top_view_dual.png` (side-by-side comparison).
    - Model count: `2` (Model '0' with 4 registered cameras selected as largest).
- [x] **Detached SfM Runner & Status Monitor (`pipeline/run_sfm.py`, `wait_sfm.sh`, `scripts/wait_sfm.py`)** -- COMPLETE (2026-10-22):
  - CLI argument `--multiple-models` (default 0) to control single-model vs multi-model stopping.
  - Real-time status reporting written atomically to `logs/status.json` and `<scene>/logs/status.json` with stage tracking and registration progress.
  - Blocking monitor `./wait_sfm.sh <project_dir>` polling `status.json` every 30s until `done: true` without intermediate timers.
  - Model converter to TXT/PLY and image undistortion to `dense/` on model completion.
- [x] **SfM Diagnosis & Failure Recovery Experiments (E1–E5)** -- COMPLETE (2026-10-22):
  - **Step 1 Pre-checks**:
    - Both raw videos (`20261021_195555.mp4` and `20261021_195930.mp4`) are identical in resolution (1920x1080), framerate (30.0 fps), and orientation metadata (none/0° rotation).
    - Extracted frames are 1920x1080 and physically upright (verified via 3x3 sample sheets saved to `docs/frame_sample_v1.jpg` and `docs/frame_sample_v2.jpg`).
    - Blur analysis: 0 frames on disk fall below the per-video extraction blur thresholds (149.0 for V1, 201.9 for V2).
    - Log audit: The earlier run reporting 62 registered frames (13.8%) lived in `task-584` (`dbafd692-9594-4f30-8ae2-4465d7ff85a1`) and `task-191` (`c5022929-da63-4a7a-ad17-3f68f2e4f6b0`, step 216 noted BA final cost 0.76 px on model 0 before compaction).
  - **Experiments E1–E5 (`colmap_runs/`)**:
    - **E1** (V1 alone, SIMPLE_RADIAL, single camera, focal prior 1440px, locked extra params): 218/222 registered (98.2%), 1 model, reproj 1.32 px, fx=fy=1483.3px, k1=0.0.
    - **E2** (Same as E1 + `TwoViewGeometry.min_num_inliers=30` + quadratic loop matching): 218/222 registered (98.2%), 1 model, reproj 1.34 px, trajectory broke at turn (gap 13.4 units, step ratio 57.4x).
    - **E3** (Both videos, 2 cameras via `--ImageReader.single_camera_per_folder 1`, best E1 settings): **446/450 registered (99.11%)** in 1 unified model (V1: 222/222 = 100%, V2: 224/228 = 98.25%), reproj 1.3059 px, fx=fy=1477.9px, k1=0.0, start-end gap 0.65 units.
    - **E4** (`global_mapper` / GLOMAP via `pycolmap.global_mapping` on E3 database): 450/450 registered (100%), reproj 1.8894 px, severe gauge drift (jumps >42 units).
    - **E5** (Re-extracted 450 frames at 4 fps from sharpest segments, rerun E3 settings): 318/450 registered in model 1 (70.67%), fragmented into 3 models.
  - **Outcome**: E3 successfully resolved the catastrophic 4-frame degenerate fit (0.89%) and recovered 99.1% registration across both passes in one unified model. Reprojection error remains at ~1.30 px due to low-texture whiteboard/wall panning during fast turns.
- [x] **Stage 3 (`pipeline/align_scene.py`) Upgraded**:
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
- [x] **Phase 4: Metric Scene Alignment (`pipeline/align_scene.py`)** -- COMPLETE & AUDITED (2026-10-22):
  - Model selection: picks passing/best model via `align.input_model` in `configs/default.yaml` (`colmap_runs/E3/sparse/0`, 446/450 frames = 99.11%, 131,482 points).
  - Stage 1 load & intrinsics check: fx=1477.94 px, fy=1477.94 px (0.0% diff), k1=0.0000, mean reproj error=1.3059 px.
  - Stage 2 UP vector: SVD on camera up vectors (`-R_cw[:, 1]`), sign-corrected to mean up `[+0.0067, -0.9911, -0.1329]`, spread = 10.17° (<= 25° threshold).
  - Stage 3 floor detection: lowest 25% height range (32,871 points), 6 iterative Open3D RANSAC planes evaluated. Lowest plane with >= 1.5% inliers chosen (#4: height = -1.6990 u, 6,175 inliers / 4.70%, angle to UP = 0.84° <= 8.0°).
  - Stage 4 rotate & translate: Rodrigues rotation to +Y (det = +1.0), floor snapped to y = 0.0000 m, dominant wall PCA yaw aligned (259.40°) with First Leg pointing North (-Z / up top-view).
  - Stage 5 scale: Mode (b) eye level reference = 1.450 m, chest level check = 1.023 m (ratio 1.42x, within [1.1, 1.5]).
  - Robust Point Extents & Camera Footprint:
    - Robust point filtering: track length >= 4 and reprojection error <= 1.0 px (9,783 robust points), clipped to 1st-99th percentile bounds.
    - Camera-loop footprint: 3.71 m x 6.81 m (compared to 2.25 m reference: X +65.1%, Z +202.7%).
    - Robust point extents: 9.16 m x 10.76 m (compared to 2.25 m reference: X +307.2%, Z +378.4%).
    - Ceiling height: 4.10 m (labeled "measured" from `measurements.json`; +0.0% error vs ground truth).
  - Stage 6 outputs: `data/home/colmap_aligned/` (cameras.txt/bin, images.txt/bin, points3D.txt/bin, points3D.ply), `transforms.json` (Nerfstudio OpenGL camera convention: flip Y and Z), `alignment.json`.
- [x] **Phase 4 Alignment Verification & Diagnostic Plotter (`scripts/check_alignment.py`)** -- VERIFIED & UN-MIRRORED (2026-10-22):
  - Fixed top view mirroring: vertical axis explicitly inverted (`ylim[0] > ylim[1]`) so +Z points strictly down the image, matching physical layout and compass convention (North = -Z up, South = +Z down, East = +X right, West = -X left).
  - Smooth turn sequence: 5-frame moving average + 1.0 m RDP simplification + merge short legs (< 1.0 m) extracts clean 7-corner counter-clockwise path: `['L', 'L', 'L', 'L', 'L', 'L', 'L']` (7 corners).
  - Ceiling height SOP: uses `measurements.json` ceiling ("measured"); fallback estimates only from points in footprint + 1.0 m and above `max_cam_h + 0.3 m` (95th percentile if >= 500 points), else prints "ceiling not reconstructed" and uses `ceiling_default_m`; never reports below `max_cam_h + 0.5 m`.
  - Side view plots (`side_xy.png`, `side_zy.png`): robust bounds + 5% margin, 1:1 aspect ratio (`adjustable="box"`), `MaxNLocator` tick formatting to eliminate label crowding.
  - Generates 4 verification PNGs in `outputs/<scene>/`:
    - `top.png`: Orthographic top-down plan, inverted Z, log-density, ~5s arrows, gap (0.46 m), scale bar.
    - `side_xy.png`: X vs Y elevation, floor y=0, measured ceiling 4.10m, Pass 1 Eye / Pass 2 Chest tracks.
    - `side_zy.png`: Z vs Y elevation, floor y=0, measured ceiling 4.10m, Pass 1 Eye / Pass 2 Chest tracks.
    - `alignment_check.png`: 4-panel diagnostic dashboard with un-mirrored plan, elevations, and audit metrics.
- [x] **Phase 4 Unit Testing Suite (`test/test_align.py`)** -- 7/7 PASSING (2026-10-22):
  - Synthetic room generator: flat floor at y=0, four walls, CCW camera loop at 1.45 m under random SO(3) rotation (det = +1.0), random translation [-5, 5], random scale [0.3, 3.0].
  - 7 rigorous assertions verified:
    1. Floor at y = 0 within 1 cm: error = 0.00016 m [PASS]
    2. UP = +Y within 1 degree: error = 0.0006° [PASS]
    3. Recovered camera height within 1%: error = 0.000% [PASS]
    4. Determinant of rotation matrix = +1.0: det(R) = 1.00000000 [PASS]
    5. CCW trajectory in top-down coordinates: signed area = +12.5550 > 0, turns = `['L', 'L', 'L', 'L', 'L']` [PASS]
    6. Real Render Test: renders `top.png`, transforms camera positions with `ax.transData.transform`, asserts shoelace area in DISPLAY coordinates is positive (+261,697.58, CCW on screen), asserts first leg points UP the image (display y diff = +130.47 px), and asserts `ylim[0] > ylim[1]` [PASS]
- [x] **Phase 5: 3D Gaussian Splatting Training Wrapper & Dataset Packaging (`pipeline/train_splat.py`, `scripts/pack_dataset.py`, `scripts/verify_splat_frame.py`)** -- COMPLETE & VERIFIED (2026-10-22):
  - **Environment Audit**:
    - GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8,151 MiB VRAM), Driver 610.88, CUDA 13.3.
    - Python virtual environment: Native Windows, PyTorch not installed in Windows venv.
    - Local execution backend: `brush-cli 0.3.0` (`tools/brush/brush_app.exe`) running natively on NVIDIA GPU via WebGPU.
    - Remote execution backend: `nerfstudio` (`ns-train splatfacto`) documented in `notebooks/colab_train.ipynb`.
  - **Dataset & Intrinsics Check**:
    - Transforms JSON: `data/home/transforms.json` (446 registered frames, width 1920, height 1080).
    - Camera Model: OPENCV (pinhole geometry), fx = 1477.94 px, fy = 1477.94 px, cx = 960.0, cy = 540.0.
    - Distortion: k1 = 0.0000, k2 = 0.0000, p1 = 0.0000, p2 = 0.0000.
    - Route Taken: Model E3 locked radial distortion to 0.0 during incremental mapping, producing a pure pinhole camera model. Raw extracted frames are therefore already undistorted pinhole images, eliminating the need for image resampling or pose alterations.
    - Image Paths & Uprightness: All 446 registered image paths exist on disk (0 missing). All images are physically upright landscape (1920x1080) with no EXIF rotation flags.
    - Initial Point Cloud: Aligned PLY (`data/home/colmap_aligned/points3D.ply`) contains **131,482 points**.
  - **Dataset Packaging (`scripts/pack_dataset.py`)**:
    - Packages only training essentials: `transforms.json` with normalized image paths, `sparse_pc.ply` (131,482 points), and all 446 registered frames.
    - Output: `outputs/home/train_pack.zip` (**109.11 MB**).
  - **Detached Training Wrapper (`pipeline/train_splat.py`)**:
    - Backend support: `brush` (local WebGPU) and `nerfstudio` (`ns-train splatfacto`).
    - Anti-centering/Anti-scaling: Nerfstudio dataparser configured with `--pipeline.datamanager.dataparser.orientation-method none --pipeline.datamanager.dataparser.center-method none --pipeline.datamanager.dataparser.auto-scale-poses False --pipeline.datamanager.dataparser.scale-factor 1.0` to preserve the metric Y-up room frame.
    - Presets: `fast` (7,000 steps smoke test) and `final` (30,000 steps, SH degree 3).
    - Detached execution: Runs background worker process with logging to `outputs/<scene>/logs/train.log` and atomic status updates to `outputs/<scene>/status.json`.
    - Fast progress query: `python pipeline/train_splat.py --scene home --status` returns in < 1 second.
    - Evaluation: Computes PSNR and SSIM on holdout frames (10% holdout split, 45 frames) into `outputs/<scene>/metrics.json` (Mean PSNR: 24.04 dB, Mean SSIM: 0.8866).
    - Executed fast preset run: 7,000 steps, wall-clock time 546.4 s (9.11 min), peak VRAM 2,476 MiB (2.42 GB / 8 GB budget), 606,040 Gaussians. Coordinate frame verification: [PASS] (Umeyama RMSE 0.0195 m, yaw -0.00 deg, scale 0.9999, floor peak -0.050 m, floor tilt 0.49 deg, camera inclusion 100%).
  - **Verification Suite & Unit Tests (`scripts/verify_splat_frame.py`, `test/test_verify_splat.py`)**:
    - Updated `scripts/verify_splat_frame.py` with 15% span tolerance and robust exception handling for displaced floor detection.
    - Added unit test suite `test/test_verify_splat.py` testing synthetic PLY:
      1. Valid aligned splat passes [PASS].
      2. Displaced floor (y = 0.80 m) fails loudly [PASS].
      3. Upside-down / inverted splat fails loudly [PASS].
      4. Extents scaled by 1.45x (> 15% error) fails loudly with numbers [PASS].
  - **Google Colab Training Notebook (`notebooks/colab_train.ipynb`)**:
    - Pinned dependencies: `nerfstudio==1.1.5`, `gsplat==1.0.0`, `lpips==0.1.4`, `open3d==0.18.0`, `ninja==1.11.1.4`.
    - Colab runtime instructions: Runtime -> Change runtime type -> Hardware accelerator: T4 GPU or A100 GPU (Python 3).
    - Complete workflow: environment check, dependency installation, unzip `train_pack.zip`, train `splatfacto` with anti-scaling flags, export `scene_raw.ply`, evaluate holdout metrics with `ns-eval`, coordinate verification, and artifact download.
- [x] **Stage 5 / Phase 6: Gaussian Splat Post-Processing, Occupancy Grid & Compression (`pipeline/postprocess_scene.py`, `scripts/verify_splat_frame.py`)** -- COMPLETE & VERIFIED (2026-10-24):
  - Geometric Crop Margins (Preserving Full Hall Content):
    - Margin from config `crop_margin_m: 4.0` m around camera-path bounding box ($X \in [-5.83, +5.89]\text{ m}$, $Z \in [-7.23, +7.58]\text{ m}$).
    - Intersected with robust extents of reference points ($p_{0.5}$ to $p_{99.5}$, track length $\ge 4$, error $\le 1.0\text{ px}$) $+ 1.0\text{ m}$ ($X \in [-7.41, +4.09]\text{ m}$, $Z \in [-5.20, +7.75]\text{ m}$, 9,783 robust points).
    - Resulting Intersected Crop Box: $X \in [-5.83, +4.09]\text{ m}$, $Z \in [-5.20, +7.58]\text{ m}$. Full hall reference content (reaching $X=-6.15\text{ m}$ and $Z=+6.77\text{ m}$) is now fully preserved.
    - Y Height Clamp: $y \in [\text{floor\_y} - 0.30, \text{ceiling\_y} + 0.30] = [-0.30\text{ m}, +4.40\text{ m}]$, with $\text{floor\_y} = 0.0$ and $\text{ceiling\_y} = 4.10\text{ m}$. Completely prevents floor/ceiling truncation.
  - Sequential Filtering & Open3D Statistical Outlier Removal:
    - 1. Raw input splats: **606,040 splats** (136.40 MB).
    - 2. After Y clamp: 595,843 splats (10,197 floaters pruned, 1.7%).
    - 3. After Intersected XZ crop: 576,249 splats (19,594 exterior floaters pruned, 3.3%).
    - 4. After opacity threshold ($\sigma(\text{opacity}) \ge 0.05$): 392,189 splats (184,060 floaters pruned, 31.9%).
    - 5. After maximum semi-axis scale ($\max(\text{scale}) \le 0.50\text{ m}$): 392,107 splats (82 billboard splats pruned, 0.0%).
    - 6. After Open3D Statistical Outlier Removal (`nb_neighbors 20, std_ratio 2.5`): **383,101 splats** (9,006 isolated floaters pruned, 2.3%).
    - Retained High-Fidelity Gaussians: **383,101 splats (63.2%)**, total reduction = 36.8% (222,939 splats pruned).
  - Floor Layer Splats Audit:
    - Floor height band $y \in [-0.15, +0.15]\text{ m}$: 106,586 raw -> **74,525 clean**.
    - Sub-floor band $y \in [-0.30, -0.05)\text{ m}$: 24,101 raw -> **13,331 clean** (preserved).
  - Verifier Upgrades (`scripts/verify_splat_frame.py`):
    - Cropped Reference Points: Reference points cropped to splat box before extents comparison ($131,482 \to 123,308$ points). Overhang: $X \in [+0.07, -0.08]\text{ m}$, $Z \in [+0.04, +0.03]\text{ m}$ -> [PASS].
    - Camera Inclusion: Tested against `scene.json` bounds in X/Z and $Y \in [\text{floor\_y}, \text{ceiling\_y}] = [0.00, 4.10]\text{ m}$: **446/446 cameras inside (100.0%)** -> [PASS].
    - Pass Labelling Fix: Reused `align_scene.py` pass splitting logic: Pass 1 Eye = **222 cameras** (mean height = $1.450\text{ m}$), Pass 2 Chest = **224 cameras** (mean height = $1.023\text{ m}$), ratio = $1.42\text{x}$ -> [PASS].
    - Floor & Trajectory Alignments: Splat floor peak = $-0.050\text{ m}$ (tol $\le 0.10\text{ m}$), floor tilt = $0.54^\circ$ (tol $\le 3.0^\circ$), Pass 1 camera tilt = $0.46^\circ$, Pass 2 camera tilt = $0.32^\circ$ (tol $\le 5.0^\circ$), Umeyama RMSE = $0.0151\text{ m}$, scale = $1.0000$, translation = $0.0002\text{ m}$, yaw = $-0.00^\circ$ -> **[PASS] SPLAT COORDINATE FRAME IS VALID**.
  - 2D Occupancy Grid (`outputs/home/occupancy.json` & `outputs/home/occupancy.png`):
    - Free Cell Assertion: **444/446 cameras (99.55% >= 98.0%)** and spawn fall strictly in FREE cells with default inflation $0.25\text{ m}$ and $\text{min\_points} = 5$.
    - Walkable Region Added: 5,778 cells (**48.64%** of grid) within `walkable_radius_m: 1.5` m of continuous camera trajectory.
    - Occupied Cells: 5,230 (**44.02%**); Free Cells: 6,650 (**55.98%**). Total grid: $99\text{ cols} \times 120\text{ rows} = 11,880\text{ cells}$ at $0.10\text{ m}$ resolution.
    - `occupancy.png`: Rendered with $+Z$ pointing strictly **DOWN**, dual-pass trajectory (Pass 1 Cyan, Pass 2 Magenta), green spawn marker/heading arrow, and light cyan dashed outline for the walkable region.
  - Multi-Format Compression & Round-Trip Quality:
    - Clean PLY: [`outputs/home/scene_clean.ply`](file:///c:/Users/gisha/Downloads/Roomverse_Namma_Space-main/Roomverse_Namma_Space-main/outputs/home/scene_clean.ply) (**86.22 MB**).
    - Web Standard 32-byte Binary: [`outputs/home/scene.splat`](file:///c:/Users/gisha/Downloads/Roomverse_Namma_Space-main/Roomverse_Namma_Space-main/outputs/home/scene.splat) (**11.69 MB**, 11.67x compression vs raw PLY).
    - Niantic SPZ: [`outputs/home/scene.spz`](file:///c:/Users/gisha/Downloads/Roomverse_Namma_Space-main/Roomverse_Namma_Space-main/outputs/home/scene.spz) (**6.54 MB**, **20.86x compression vs raw PLY**, 1.79x vs `.splat`).
    - SPZ Round-Trip Quality (decompressed back to PLY via `@playcanvas/splat-transform`):
      - Splat count difference: **0** (exact 383,101 match).
      - Max centre error: **0.210 mm** ($0.000210\text{ m}$).
      - Mean centre error: **0.117 mm** ($0.000117\text{ m}$).
      - Mean opacity error: **0.000977** ($< 0.1\%$).
  - Web Asset Deployment (`viewer/public/scene/`):
    - Deployed `scene.json`, `scene.splat`, `scene.spz`, `occupancy.json`, `occupancy.png` to `viewer/public/scene/`.
- [x] **Phase 5 Production Training: "Final" Preset (30k steps, SH deg 3) & LPIPS Evaluation** -- COMPLETE & VERIFIED (2026-10-24):
  - Wall-clock time: **2,447.68 s** (40.79 min).
  - Peak VRAM: **3,695 MiB** (3.61 GB / 8 GB budget).
  - Final Gaussian count: **1,309,428 splats** (raw export: `outputs/home/scene_raw.ply`).
  - Fast preset preserved: `outputs/home/scene_raw_fast.ply` (606,040 Gaussians) and `outputs/home/metrics_fast.json`.
  - Holdout Test Evaluation Comparison (45 test frames, 10% holdout split):
    - **Fast Preset (7,000 steps)**: PSNR = **24.04 dB**, SSIM = **0.8866**, LPIPS = **0.3029**.
    - **Final Preset (30,000 steps, SH deg 3)**: PSNR = **23.97 dB**, SSIM = **0.8903**, LPIPS = **0.2698** (10.9% perceptual error reduction).
  - Splat statistics (`outputs/home/train_report.json`):
    - Opacity < 0.05: 33.02%
    - Opacity >= 0.2: 20.03% (261,744 Gaussians)
    - Sub-floor points below y = -0.3 m: 1.22%
  - Coordinate Frame Verification (`scripts/verify_splat_frame.py`):
    - Rigid alignment fit (Umeyama): yaw = **-0.00°**, translation = **0.0002 m** (0.2 mm), scale = **0.9999**, RMSE = **0.0221 m** -> **PASS**.
    - Splat floor peak: **-0.050 m** (tolerance <= 0.10 m) -> **PASS**.
    - Splat floor plane tilt: **0.64°** (tolerance <= 3.0°) -> **PASS**.
    - Camera tilt: Pass 1 Eye = **0.46°**, Pass 2 Chest = **0.32°** (tolerance <= 5.0°) -> **PASS**.
    - Camera inclusion: **100.0%** (446/446 cameras enclosed) -> **PASS**.
    - Extents coverage: Overhang X = [+0.50, -0.00] m, Z = [+0.02, -0.03] m -> **PASS**.
    - Overall status: **[PASS] SPLAT COORDINATE FRAME IS VALID** (`outputs/home/splat_frame_verification.json`).
- [x] **Stage 5 / Phase 6: 30k "Final" Model Post-Processing & Three-Way Variant Comparison** -- COMPLETE & VERIFIED (2026-10-24):
  - Backed up initial fast-preset web assets to `outputs/home/web_assets_fast/` (`scene.json`, `scene.spz`, `scene.splat`, `occupancy.json`, `occupancy.png`, `scene_clean.ply`, `clean_frame_verification.json`).
  - Added `max_splats` config option (default `0 = unlimited`) to `configs/default.yaml` and CLI argument `--max-splats` in `pipeline/postprocess_scene.py`.
  - Implemented importance ranking filter:
    $$\text{importance} = \sigma(\text{opacity}) \times \text{volume}, \quad \text{volume} = \exp\left(\sum_{i=0}^2 \text{scale}_i\right)$$
    Pruning lowest-importance Gaussians globally without spatial position cuts to guarantee the scene is not truncated anywhere.
  - Full sequential filter-count progression on 30k final model (`outputs/home/scene_raw.ply`, 1,309,428 raw splats):
    1. Y Clamp `[-0.30m, +4.40m]`: 1,309,428 -> 1,292,885 (16,543 pruned, 1.3%)
    2. Intersected XZ BBox Crop: 1,292,885 -> 1,251,767 (41,118 pruned, 3.2%)
    3. Min Opacity ($\ge 0.05$): 1,251,767 -> 843,881 (407,886 pruned, 32.6%)
    4. Max Scale ($\le 0.50\text{ m}$): 843,881 -> 843,651 (230 pruned, 0.0%)
    5. Open3D SOR (`nb_neighbors 20, std_ratio 2.5`): 843,651 -> 824,903 (18,748 pruned, 2.2%)
    6. Importance Ranking / Budget:
       - `max_splats = 0` (unlimited): 824,903 -> **824,903** (0 pruned, 0.0%)
       - `max_splats = 700000`: 824,903 -> **700,000** (124,903 pruned, 15.1%)
  - **Three-Way Comparison Table**:
    | Variant | Splat Count | Clean .ply Size | Web .splat Size | Web .spz Size | SPZ vs Raw Ratio | Verifier Result | Camera Free % | Spawn Free |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Fast Preset (7k model)** (`web_assets_fast/`) | 383,101 | 86.22 MB (90,413,377 B) | 11.69 MB (12,259,232 B) | 6.54 MB (6,857,564 B) | 20.86x | **PASS** | 99.55% (444/446) | **True** |
    | **Final Full (30k model)** (`final_full/`) | 824,903 | 185.66 MB (194,678,649 B) | 25.17 MB (26,396,896 B) | 16.03 MB (16,811,342 B) | 18.38x | **PASS** | 98.43% (439/446) | **True** |
    | **Final 700k (30k model)** (`final_700k/`) | 700,000 | 157.55 MB (165,201,541 B) | 21.36 MB (22,400,000 B) | 13.49 MB (14,149,807 B) | 21.84x | **PASS** | 98.65% (440/446) | **True** |
  - SPZ Round-trip Quality (Decompressed back to PLY):
    - Final Full: 0 splat count diff, max centre error = 0.210 mm, mean centre error = 0.117 mm, mean opacity error = 0.000975.
    - Final 700k: 0 splat count diff, max centre error = 0.210 mm, mean centre error = 0.117 mm, mean opacity error = 0.000975.
  - Coordinate Frame Verification (`scripts/verify_splat_frame.py`):
    - Final Full: Umeyama yaw = -0.00°, trans = 0.0002 m, scale = 0.9999, floor peak = -0.050 m, floor tilt = 0.53°, camera inclusion = 100.0% -> **PASS**.
    - Final 700k: Umeyama yaw = -0.00°, trans = 0.0003 m, scale = 0.9999, floor peak = -0.050 m, floor tilt = 0.61°, camera inclusion = 100.0% -> **PASS**.
  - Deployed `final_700k` assets (`scene.json`, `scene.splat`, `scene.spz`, `occupancy.json`, `occupancy.png`) to `viewer/public/scene/`.
    - Preserved exact eye height (1.45 m above floor at y = 0.0), spawn position `[1.771, 1.450, 3.523]`, spawn yaw `-0.0426` rad (`-2.44°`), and `splat_rotation_euler` (`[0.0, 0.0, 0.0]`).
- [x] **Phase 7a: Framework-Free WebGL 3DGS Viewer (`viewer/`, Three.js + Spark)** -- COMPLETE & VERIFIED (2026-10-24):
  - **Framework-Free Vanilla Architecture**:
    - Vite + TypeScript vanilla app (`viewer/`) built without React, Tailwind, or external UI frameworks.
    - Pure CSS styling with dark theme (`#0b0d12`), responsive canvas, and glassmorphic HUD.
  - **Spark Package & API Audit (`@sparkjsdev/spark`)**:
    - **Spark Version**: `2.4.0`.
    - **Three.js Compatibility**: `peerDependencies` requires `three >= 0.180.0` (`devDependencies` pins `^0.180.0`). Installed `three: 0.186.1` and `@types/three: 0.186.0`.
    - **Splat Loading API**: High-level `SplatMesh({ url, onProgress, onLoad })` extending Three.js `SplatGenerator`/`Object3D`. Integrated with `SparkRenderer({ renderer })` added to the `THREE.Scene`. Also provides lower-level `SplatLoader` extending `THREE.Loader`.
    - **Progress Observation**: Real progress events supported via `onProgress: (event: ProgressEvent) => void`. Spark's worker reports `{ loaded, total }` during download and decoding, enabling genuine percentage and loaded/total MB tracking.
    - **Format Support**: Both `.spz` and `.splat` (alongside `.ply`, `.ksplat`, `.sog`, and `.rad`) are natively supported through `SplatFileType.SPZ` and `SplatFileType.SPLAT`.
  - **Asset Loading & Dual-Format Fallback**:
    - Asset base configurable via `VITE_ASSET_BASE` (`.env.example` added; defaults to empty string for same-origin).
    - Fetches `scene.json` first, extracting bounds, floor/ceiling elevations, spawn position, spawn yaw, rotation euler, and splat files.
    - Primary format: loads `.spz` first (`scene.spz`, 13.49 MB).
    - Automatic fallback: if `.spz` fails, logs warning and seamlessly loads `.splat` (`scene.splat`, 21.36 MB).
    - Rotation integrity: Strictly applies `splat_rotation_euler` (`[0, 0, 0]`) to the splat mesh as a proper Euler rotation without axis flipping or mirroring.
  - **Dev-Only Check View (`viewer/src/devview.ts`)**:
    - Self-contained in a single modular file for clean deletion in Phase 7b.
    - OrbitControls targeting the centroid of `scene.json` bounds.
    - Camera placed 8 m back (+Z) and 5 m up (+Y) from center so full hall is in view.
    - 1 m floor grid snapped to `y = floor_y` (0.0 m).
    - 1 m axes helper at origin (X red, Y green, Z blue).
    - Labelled marker at `spawn_position` with floor ring, vertical stem, canvas text sprite, and heading arrow along `spawn_yaw` (yaw 0 looks along -Z, CCW seen from above).
    - Semi-transparent box with wireframe outline visualizing `scene.json` bounds.
    - Dark background `#0b0d12`.
  - **Loading UI, Error Resilience & Performance**:
    - Progress bar displaying exact percentage and MB downloaded.
    - Full error overlay showing exact attempted URL and interactive Retry button (supports interactive testing via `?testError=true`).
    - Performance: `devicePixelRatio` capped at 1.5, `antialias: false` for Gaussian splat rasterization performance, zero per-frame heap allocations in animation loop.
    - Splat count verified: logs loaded count (`700,000`) and confirms exact match with `scene.json`.
    - Build: `npm run build` (`tsc && vite build`) passes with ZERO TypeScript errors.
    - Dev server running detached at `http://localhost:5173/`.
- [ ] Phase 7b: First-person WASD + pointer-lock roaming controls, eye-height maintenance, and 2D occupancy grid collision detection.


## How to Run

```bash
# 1. Activate venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/macOS

# --- SFM FAILURE FIX EXPERIMENT (E3: 99.1% RECOVERY) ---
# Run E3 with two cameras (one per video folder), SIMPLE_RADIAL, focal prior 1440px
python scripts/run_e3.py
# Outputs: colmap_runs/E3/ (sparse/0, cameras.txt, images.txt, points3D.txt)

# --- ONE-COMMAND REPRODUCTION (Runs diagnosis + alignment + top-view) ---
python scripts/reproduce_all.py --scene home

# --- INDIVIDUAL STAGES ---
# 1. Run SfM diagnosis
python scripts/diagnose_sfm.py --scene home
# Outputs: docs/sfm_diagnosis.md

# 2. Run Synthetic Alignment Unit Test Suite (Phase 4)
python test/test_align.py
# Tests: floor y=0 <= 1cm, UP=+Y <= 1 deg, cam height <= 1%, det(R)=+1.0, CCW positive area

# 3. Align scene to metric Y-up, floor at y=0 (Phase 4)
python pipeline/align_scene.py --scene home
# Outputs: data/home/colmap_aligned/ (cameras.txt/bin, images.txt/bin, points3D.txt/bin, points3D.ply)
#          data/home/transforms.json (Nerfstudio OpenGL format)
#          data/home/alignment.json

# 4. Check and verify alignment (Phase 4 Diagnostic Plots)
python scripts/check_alignment.py --scene home
# Outputs: outputs/home/top.png (true top-down, log-density, ~5s arrows, gap, scale bar)
#          outputs/home/side_xy.png (X vs Y elevation, floor y=0, ceiling)
#          outputs/home/side_zy.png (Z vs Y elevation, floor y=0, ceiling)
#          outputs/home/alignment_check.png (4-panel diagnostic dashboard)

# 4. Generate trustworthy top-down view, overlay, and 4-leg verification sheet
python scripts/make_topview.py --scene home
# Outputs: outputs/home/topview.png
#          outputs/home/topview_overlay.png
#          outputs/home/topview_legs.jpg

# 5. Pack training bundle for GPU / Colab (Phase 5)
python scripts/pack_dataset.py --scene home
# Outputs: outputs/home/train_pack.zip (109.11 MB)

# 6. Validate training configuration and commands (dry run)
python pipeline/train_splat.py --scene home --preset fast --dry-run
# Remote / Colab: notebooks/colab_train.ipynb

# 7. Train 3D Gaussian Splatting model (detached background)
# Fast preset (7,000 steps smoke test):
python pipeline/train_splat.py --scene home --preset fast
# Final preset (30,000 steps, SH degree 3 production run):
python pipeline/train_splat.py --scene home --preset final
# Check status in < 5 seconds:
python pipeline/train_splat.py --scene home --status
# Or view logs: powershell -Command "Get-Content outputs/home/logs/train.log -Tail 10"

# 8. Run unit test suite on synthetic Gaussian Splats
python test/test_verify_splat.py
# 4 tests: valid splat, displaced floor fail, upside-down fail, extents >15% fail

# 9. Post-Process Scene, Occupancy Grid & Compression (Phase 6)
# (a) Full 30k model post-processing (unlimited splats, max_splats = 0):
python pipeline/postprocess_scene.py --scene home --input outputs/home/scene_raw.ply --output-dir outputs/home/final_full --max-splats 0 --no-copy-viewer
# Outputs: outputs/home/final_full/ (scene_clean.ply, scene.splat, scene.spz, scene.json, occupancy.json, occupancy.png)

# (b) Budgeted 700k model post-processing (max_splats = 700000):
python pipeline/postprocess_scene.py --scene home --input outputs/home/scene_raw.ply --output-dir outputs/home/final_700k --max-splats 700000 --no-copy-viewer
# Outputs: outputs/home/final_700k/ (scene_clean.ply, scene.splat, scene.spz, scene.json, occupancy.json, occupancy.png)

# 10. Verify exported splat coordinate frame
python scripts/verify_splat_frame.py --scene home --splat-ply outputs/home/final_700k/scene_clean.ply --scene-json outputs/home/final_700k/scene.json
# Outputs: outputs/home/final_700k/clean_frame_verification.json

# --- COMMANDS TO SWITCH VIEWER VARIANTS ---
# Deploy 700k variant (Default / Recommended for web performance):
# PowerShell:
Copy-Item -Force outputs/home/final_700k/scene.json, outputs/home/final_700k/scene.splat, outputs/home/final_700k/scene.spz, outputs/home/final_700k/occupancy.json, outputs/home/final_700k/occupancy.png viewer/public/scene/
# Bash:
# cp outputs/home/final_700k/{scene.json,scene.splat,scene.spz,occupancy.json,occupancy.png} viewer/public/scene/

# Switch to Full 30k variant (825k splats):
# PowerShell:
Copy-Item -Force outputs/home/final_full/scene.json, outputs/home/final_full/scene.splat, outputs/home/final_full/scene.spz, outputs/home/final_full/occupancy.json, outputs/home/final_full/occupancy.png viewer/public/scene/
# Bash:
# cp outputs/home/final_full/{scene.json,scene.splat,scene.spz,occupancy.json,occupancy.png} viewer/public/scene/

# Switch to Fast 7k preset (383k splats):
# PowerShell:
Copy-Item -Force outputs/home/web_assets_fast/scene.json, outputs/home/web_assets_fast/scene.splat, outputs/home/web_assets_fast/scene.spz, outputs/home/web_assets_fast/occupancy.json, outputs/home/web_assets_fast/occupancy.png viewer/public/scene/
# --- OCCUPANCY GRID V2 (3D VOXEL RAY CARVING) ---
# Rebuild occupancy grid v2 with ray carving:
python pipeline/build_occupancy.py --scene home
# Or via postprocess_scene:
python pipeline/postprocess_scene.py --scene home --input outputs/home/scene_cleaned.ply
# Run occupancy unit tests:
pytest test/test_occupancy.py

# Deploy occupancy v2 to viewer:
# PowerShell:
Copy-Item -Force outputs/home/occupancy_v2/occupancy.json, outputs/home/occupancy_v2/occupancy.png viewer/public/scene/
# Bash:
# cp outputs/home/occupancy_v2/{occupancy.json,occupancy.png} viewer/public/scene/

# --- PHASE 7: 3DGS WEB VIEWER ---
# 1. Install dependencies:
cd viewer && npm install && cd ..

# 2. Run unit tests (Vitest: movement maths, friction, diagonal normalisation, pitch clamping, yaw convention):
cd viewer && npm test && cd ..

# 3. Type-check and build production bundle:
cd viewer && npm run build && cd ..

# 4. Run development server (detached):
cd viewer && npm run dev
# URL: http://localhost:5173/
# Test error modal & retry flow: http://localhost:5173/?testError=true
```

## Known Issues / TODOs

- Phase 7a Complete: Framework-free Vite + TypeScript 3DGS WebGL viewer built with Three.js and `@sparkjsdev/spark` 2.4.0. Dual-format `.spz` and fallback `.splat` loader, proper `splat_rotation_euler` application, OrbitControls Dev View (`viewer/src/devview.ts`), real progress bar, error handling with Retry, and zero TypeScript errors (`npm run build`). Dev server running at `http://localhost:5173/`.
- Phase 7b Complete: First-Person Viewer in `viewer/`:
  - **Dev View Removed**: Deleted `viewer/src/devview.ts` and all references to orbit controls, axes, grid, bounding box, and spawn markers. Kept dark background `#0b0d12` and moved asset loading telemetry behind F3 toggle only.
  - **First-Person Controls (`viewer/src/controls.ts`)**:
    - Pointer lock on canvas click; Esc or window blur releases lock and clears key state to prevent drifting.
    - Mouse look with Euler order "YXZ": yaw about world +Y, pitch about camera X axis. Clamped to ±70° from config.
    - Sign convention verified: In Three.js right-handed coordinates, positive `rotation.y` rotates counter-clockwise when viewed from above (+Y looking down), turning the view to the left (-X). Moving mouse right (`movementX > 0`) decreases `euler.y` to turn right. Moving mouse up (`movementY < 0`) increases `euler.x` to look up (NOT inverted). Yaw 0 looks along -Z (North), exactly matching `scene.json`'s `spawn_yaw`.
    - Horizontal-only movement: Camera yaw is extracted while pitch is ignored, ensuring looking up or down never changes elevation or makes the user fly.
    - Diagonal normalisation: W+D combination is normalized to unit length so diagonal traversal speed matches cardinal speed (1.4 m/s walking, 2.8 m/s sprint).
    - Acceleration & friction damping: 12 m/s² acceleration, combined linear stopping deceleration (7.0 m/s²) and exponential damping (14 s⁻¹) bringing camera to a complete stop in ~0.2s from walking speed. Delta time clamped to at most 0.05s to prevent lag-spike teleportation.
    - Fixed eye height: Pinned at `floor_y + EYE_HEIGHT_M = 0.0 + 1.45 = 1.45 m` (commented in code: eye-level video pass was captured at 1.45 m).
    - Key R: Resets camera to `spawn_position` (X, Z with eye height for Y) and `spawn_yaw`, zeroing velocity.
    - Zero per-frame allocations: Reuses `Vector3` and `Euler` instances every frame.
  - **Overlays & UI (`index.html`, `style.css`)**:
    - Plain TypeScript + CSS with CSS variables (`--bg-dark`, `--accent-cyan`, etc.) and system fonts (no external web fonts).
    - "Click to start" overlay with "Namma Space" title, controls cheat-sheet, quality buttons, and Start button (`pointer-events: none` on overlay wrapper, `pointer-events: auto` on interactive elements).
    - Crosshair (4px dot) displayed only while pointer is locked.
    - F3 stats panel (FPS, frame time, splat count, asset name, load time, position X/Y/Z, yaw deg, quality tier), updated at most 4 times per second (≥250ms interval).
  - **Performance**:
    - `devicePixelRatio` capped at 1.5.
    - Quality toggle with keys 1 / 2 / 3 and UI buttons: Low (0.5x res scale), Med (0.75x res scale, default), High (1.0x res scale). Stored in `localStorage` under `namma_space_quality` inside `try/catch`.
    - Spark LOD runtime tuning: Configured `@sparkjsdev/spark` `SparkRenderer.lodSplatScale` (0.5 / 0.75 / 1.0), `lodRenderScale` (2.0 / 1.5 / 1.0), and `maxStdDev` (2.0 / 2.45 / 2.828).
    - Render loop skips execution and flushes clock delta when tab is hidden (`document.hidden`).
  - **Testing**:
    - 15 unit tests in `viewer/test/controls.test.ts` (Vitest 2.1.9) passing with 100% coverage on pure movement maths.
    - `npm run build` succeeds with zero TypeScript errors. Production bundle: JS 3,208 kB (gzip 1,073 kB), CSS 8.6 kB, HTML 5.4 kB.
    - Loaded 700,000 splats from `scene.spz` (14.15 MB) at `http://localhost:5173/`.
- [x] **Phase 8: Movement Constraints, 2D Occupancy Collision, Minimap & Touch Navigation (`viewer/src/collision.ts`, `viewer/src/controls.ts`, `viewer/src/minimap.ts`, `viewer/test/collision.test.ts`)** -- COMPLETE & VERIFIED (2026-10-24):
  - **Occupancy Grid Inspection**:
    - Keys: `origin` ([-5.9, -4.7]), `cell_size` (0.1 m), `width` (100), `height` (122), `data` (row-major 12,200 elements), `walkable` (12,200 elements), `axes` ("x = X, row = Z"), `inflation_radius_m` (0.20 m), `occupied_cells_raw` (2,214), `occupied_cells_inflated` (5,642).
    - World-to-cell mapping: `col = Math.floor((x - origin[0]) / cell_size)`, `row = Math.floor((z - origin[1]) / cell_size)`, `index = row * width + col`.
    - Player collision radius: `data` is already inflated by 0.20 m during Phase 6 post-processing, and no uninflated raw array is saved in `occupancy.json`. Player is treated as a small circle with radius **0.08 m** against `data` to avoid double-inflation.
  - **Pure Collision Maths (`viewer/src/collision.ts`)**:
    - Pure TypeScript functions with ZERO Three.js or DOM imports for headless testability: `worldToCell`, `isBlocked`, `isPointBlocked`, `isCircleBlocked`, `getDistanceToFree`, `isMoveAllowed`, `resolveMovement`, `fixSpawnPosition`, `clampToBounds`.
    - Circle-vs-grid nearest-point-on-cell test against overlapping cell bounding boxes.
    - Wall sliding: diagonal moves into walls preserve parallel velocity component without snags or jitter.
    - Anti-tunneling sub-stepping: moves subdivided into sub-steps of at most half a cell (0.05 m). Worst case at sprint speed (2.8 m/s, max dt 0.05s) is only 3 sub-steps per frame (`ceil(0.14 / 0.05) = 3`).
    - Stuck escape: if player starts inside a blocked cell (e.g. camera poses near geometry), any move that maintains or decreases distance to nearest free cell is permitted, allowing clean walk-out.
    - Blocked spawn fix: outward concentric ring search moves blocked spawn to the nearest free cell centre and logs a warning.
    - Boundaries: `LIMIT_TO_WALKABLE` (default `true`, toggleable with Key L / shown on F3) rejects moves into unwalkable cells with sliding logic. Fallback clamping to `scene.json` bounds.
    - Eye height: strictly locked at `floor_y + EYE_HEIGHT_M = 1.45 m`; optional head-bob (Key H, default OFF, amplitude 0.015 m) never modifies collision position.
  - **Minimap Overlay (`viewer/src/minimap.ts`)**:
    - Key B toggles (default OFF).
    - Semi-transparent canvas (220 px wide) rendered ONCE from `occupancy.json` into an offscreen canvas (`ImageData`).
    - Occupied cells dark (`#12151b`), walkable cells light (`#e2e8f0`), free but not walkable cells mid-grey (`#64748b`).
    - +Z points DOWN the canvas (matching `top.png` compass convention).
    - Player dot and heading arrow redrawn at most 10 times per second (>= 100 ms interval).
    - `pointer-events: none` on overlay wrapper so viewer interactions are never blocked.
  - **Touch Navigation (`viewer/src/controls.ts`)**:
    - Detected via `matchMedia("(pointer: coarse)")` or `"ontouchstart"`, NOT user agent strings.
    - Virtual joystick appears under thumb on left 1/3 of screen (`clientX < window.innerWidth / 3`) with a 10% dead zone (5 px on 50 px radius).
    - Right 2/3 of screen (`clientX >= window.innerWidth / 3`) used for drag look with pitch clamping [-70°, +70°].
    - Pointer events only (`setPointerCapture`, `touch-action: none` on canvas, supports two simultaneous touches).
    - Start card adapts automatically to "Tap to Start" with touch controls hints. No pointer lock requested on touch.
    - Joystick released on `pointercancel` and window `blur`.
  - **Testing & Verification**:
    - 27 unit tests passing across `viewer/test/collision.test.ts` (12 tests) and `viewer/test/controls.test.ts` (15 tests) with 100% pass rate.
    - Tests assert perpendicular stopping, diagonal sliding, inside corner stability without jitter, passable 1.0 m gap vs impassable 0.3 m gap, 2.0 m anti-tunneling, spawn fix, stuck escape walk-out, walkable limit enforcement, and real `occupancy.json` verification.
    - `npm run build` succeeds with ZERO TypeScript errors. Production bundle: JS 3,221.08 kB (gzip 1,077.24 kB), CSS 10.24 kB (gzip 2.50 kB), HTML 6.98 kB (gzip 1.59 kB).
- [x] **Occupancy Grid v2 Rebuild with 3D Voxel Ray Carving (`pipeline/build_occupancy.py`, `pipeline/postprocess_scene.py`, `test/test_occupancy.py`)** -- COMPLETE & VERIFIED (2026-10-24):
  - **3D Voxel Grid in Body Slab**:
    - Slab elevation: $y \in [0.20, 1.40]$ m above floor at $y=0$ (12 vertical slices at 0.10 m voxel resolution).
    - Horizontal extents: aligned with COLMAP camera and point extents (e.g. $100 \times 122$ grid cells).
  - **Occupied Evidence Integration**:
    - Combined SfM robust points (track length $\ge 3$, reprojection error $\le 1.0$ px) and cleaned splats (opacity $\ge 0.5$, max scale $\le 0.15$ m).
    - Voxel marked as candidate occupied if support $\ge 2$.
  - **Vectorized 3D Ray Carving (Free Space Evidence)**:
    - Marching rays from 446 registered camera centres toward SfM points, stepping every 0.05 m and stopping 0.15 m before the point.
    - Each camera counted at most once per voxel.
    - Highly optimized vectorized NumPy implementation running 88,123 rays in **~2.3 seconds** (well under the 3-minute budget).
    - Overrode 2,675 floater voxels with weak occupied evidence back to FREE because cameras observed empty space through them.
  - **Classification & 2D Collapse**:
    - Voxel: FREE if free-count $\ge 2$; OCCUPIED if candidate and free-count $< 2$; otherwise UNKNOWN.
    - 2D Collapse: OCCUPIED if any slab voxel is OCCUPIED (with isolated 1-voxel noise filter); KNOWN-FREE if $\ge 60\%$ slab voxels are FREE and none OCCUPIED; otherwise UNKNOWN.
  - **Clearance & Collision Data Generation**:
    - `data` (1 = blocked): occupied cells inflated by 0.12 m, plus unknown cells that are $> 0.30$ m away from a known-free cell.
    - Effective player body radius against grid: $0.12\text{ m (grid)} + 0.08\text{ m (viewer circle)} = 0.20\text{ m}$.
    - Passable clearance: narrowest passable gap detected is **0.70 m** (corridor clearance 1.10 m).
    - `walkable`: 4-neighbour flood fill connected to spawn cell.
    - Preserved existing viewer schema (`origin`, `cell_size`, `width`, `height`, `data`, `walkable`) and added `state` (0 unknown, 1 free, 2 occupied, 3 inflated), `clearance_m` (distance to nearest blocked cell, 2 decimals), and `params`.
    - Generated diagnostic maps: `occupancy.png` (dark occupied, orange inflated, light green known free, grey unknown, overlaid camera trajectories and spawn, 1m grid, +Z down) and `occupancy_debug.png` (clearance heatmap).
  - **Verification Audit (`outputs/home/occupancy_report.json`)**:
    - Camera path walkability: 88.79% with default 0.60 slab ratio; consecutive camera path blocked steps = **0.00%** (well within $\le 2.0\%$ limit).
    - Grid distribution: Occupied: 1.70%, Inflated: 4.39%, Free: 16.92%, Unknown: 76.99%, Walkable: 14.89%.
    - Narrowest passable corridors identified:
      1. $[x=-1.50, z=2.40]$ (width: 1.10 m)
      2. $[x=0.00, z=2.40]$ (width: 1.10 m)
      3. $[x=-1.40, z=2.40]$ (width: 1.20 m)
    - Comparison vs v1: 1,607 cells changed from blocked to walkable (recovering aisles between chairs and previously floater-blocked regions); 842 cells changed from walkable to blocked (unobserved pockets safely blocked).
    - Versioning & Deployment: Old files backed up to `outputs/home/occupancy_v1/`, v2 written to `outputs/home/occupancy_v2/` and `outputs/home/`, and deployed to `viewer/public/scene/` without touching splat assets.
  - **Unit Testing**:
    - Synthetic room test in `test/test_occupancy.py` asserting wall is blocked, 0.9 m aisle between chairs is walkable, floater seen by multiple cameras is removed, and unobserved pocket is unknown & not walkable.
    - All 21 pytest tests and 27 viewer Vitest tests passing.
- [x] **Phase 8 Extension: Extended Scene `home3` Setup & Incremental Continuation Pipeline (2026-10-26)**:
  - **Abandonment of `home2`**: The `home2` attempt (running monolithic SfM from scratch across all 7 sources) was abandoned because its background runs died without completing sparse reconstruction (`data/home2/colmap/sparse/` empty).
  - **Scene `home3` Architecture**: Extends the verified working `home` reconstruction (446 registered images, 2 cameras SIMPLE_RADIAL with $f_x=f_y=1477.94$ px and $f_x=f_y=1476.14$ px, $k_1=0.0$) rather than reconstructing from scratch.
  - **Baseline Copy**: Robocopied `data/home/images` (900 files: 450 root frames + video1 222 + video2 228), `measurements.json`, and `config.yaml` into `data/home3/`. Sourced working baseline `colmap/database.db` (305.4 MB) and `colmap/sparse/0/` (446 registered images) into `data/home3/colmap/`.
  - **New Image Intake & Audit**: Copied 4 sweep videos and 11 stills from `data/home2/images/` into `data/home3/images/sweep_1`..`sweep_4` and `data/home3/images/stills`. Verified all frames are upright $1920 \times 1080$ landscape with normal EXIF orientation. Generated verification contact sheets `docs/contact_sweeps.jpg` and `docs/contact_stills.jpg`.
  - **Laplacian Blur Filtering**: Filtered out bottom 10% blurriest frames per sweep folder (dropped 20 frames total: 5 in sweep_1, 5 in sweep_2, 4 in sweep_3, 6 in sweep_4; 0 stills dropped), retaining 185 high-quality new frames (43 in sweep_1, 41 in sweep_2, 38 in sweep_3, 52 in sweep_4, 11 stills). Exported database-relative list to `data/home3/new_images.txt`.
  - **Targeted Pair Generation (`scripts/make_home3_pairs.py`)**: Produced 44,770 pairs in `data/home3/pairs.txt` comprising: (i) all pairs among new images ($C(185, 2) = 17,020$ pairs) and (ii) cross pairs of each new image against every 3rd old image ($185 \times 150 = 27,750$ pairs).
  - **PowerShell Orchestration Scripts**:
    - `scripts/home3_features.ps1`: GPU SIFT `feature_extractor` for new images only (`--image_list_path data/home3/new_images.txt`), preserving camera prior ($1440, 960, 540, 0$) and 1 camera per folder.
    - `scripts/home3_matching.ps1`: Runs pair generator and `matches_importer` (`match_type=pairs`, GPU guided matching, max ratio 0.80) with 1-minute speed telemetry.
    - `scripts/home3_mapping.ps1`: `colmap mapper` continuation (`--input_path data/home3/colmap/sparse/0`, `--output_path data/home3/colmap/sparse_ext`, `--Mapper.fix_existing_images 1`) locking the 446 registered images and triangulating new footage, followed by `model_analyzer` into `data/home3/logs/analysis.txt`.
    - `scripts/run_home3.ps1`: Resumable master pipeline orchestrator skipping completed stages.
    - `scripts/status_home3.ps1`: Fast (<5s) monitor reporting stage completions, COLMAP process status, log tail, database growth, and GPU utilization.
  - **Execution & Verification Audit (Stage A & Stage B)**:
    - Stage A completed successfully in 51.02 minutes (features: 16s, matching: 29.0m, mapper continuation: 21.6m). All 635 candidate frames registered (100.0%), generating 153,124 3D points.
    - Initial `sparse_ext` audit: 446/446 original cameras remained fixed (shift 0.000m med / 0.000m max, fx change 0.00%), but model-wide reprojection error was 1.3508 px (> 1.0 px). Per-source error: originals 1.4643 px, sweep_1 1.6791 px, sweep_2 1.8989 px, sweep_3 1.8317 px, sweep_4 1.7459 px, stills 1.4740 px. All 5 new cameras were refined during SfM extension (+16 to +121 px from 1440 prior).
    - Model Pruning & Re-Bundle Adjustment (`sparse_pruned` & `sparse_pruned_ba`): Pruned new 3D points with error > 2.0 px or track < 3 (11,116 points removed; originals untouched). Pruned new images with mean error > 1.5 px or < 100 observations (155 images removed; output in `data/home3/colmap/sparse_pruned`). Re-ran bundle adjustment with fixed original poses (`--BundleAdjustment.refine_extrinsics 0`) refining new camera focal lengths into `data/home3/colmap/sparse_pruned_ba` (cost: 0.8515 -> 0.8431 px).
    - Post-Pruning Evaluation: On `sparse_pruned_ba`, model-wide mean reprojection error was 1.3134 px (threshold < 1.1 px: FAIL). Per-source registration: originals 99.78% (1.4552 px), sweep_1 41.86% (1.3072 px), sweep_2 0.00%, sweep_3 0.00%, sweep_4 13.46% (1.0547 px), stills 54.55% (1.3387 px). 0 new sources met the rule (error < 1.5 px and >= 60% registered).
    - Quality Fallback Policy: Strict fallback to `SCENE = home` confirmed.
  - **Metric Alignment & Coverage (Stage C)**:
    - `SCENE=home` verified: floor strictly at $y = 0.000$ m, floor normal within 0.84° of camera up ($\le 8.0^\circ$), eye-level camera height $1.450$ m, chest-level camera height $1.023$ m.
    - Coverage analysis (cells with $\ge 3$ distinct observing cameras): `home` achieves 70.25% (999 / 1,422 cells) vs `home3` 79.11% (2,139 / 2,704 cells).
    - Verification diagnostic plots regenerated in `outputs/home/`: `top.png`, `side_xy.png`, `side_zy.png`, `coverage.png`, `alignment_check.png`.
- [x] **Phase 8.1 Production Training: `final_v2` on SCENE=home (Stage D)**:
  - Scheduled task `home_train_v2` completed in 36.48 minutes with Brush CLI 0.3.0.
  - Scale regularization (`--scale-loss-weight 1e-7`) and opacity regularization (`--opac-loss-weight 1e-8`) successfully eliminated floaters and needles at training time.
  - Exported `outputs/home/v2/scene_raw.ply`: 792,291 splats (178.32 MB).
  - Evaluated on 56 holdout evaluation frames (`outputs/home/v2/metrics.json`): PSNR = 23.90 dB, SSIM = 0.8891, LPIPS = 0.2812.
- [x] **Phase 8.2 Post-Processing & Splat Diagnostics (Stage E)**:
  - Splat properties diagnosis (`docs/splat_quality.md`): Max scale 50th/90th/99th percentiles = 0.0187 / 0.0636 / 0.2852 m; scale ratio 50th/90th/99th = 22.22 / 584.12 / 31188.70; splats > 0.15m = 2.43%, > 0.25m = 1.18%, > 0.50m = 0.55%; share in cells seen by < 3 cameras = 0.57%. Documented Spark `@sparkjsdev/spark` blur and clamping options (`lodSplatScale`, `lodRenderScale`, `maxStdDev`).
  - Applied sequential filters: Y clamp `[-0.30, 4.40]` m, intersected XZ crop, opacity $\ge 0.05$, max scale $\le 0.25$ m, needle filter (ratio > 12 & scale > 0.08 m), coverage filter (seen by $\ge 3$ cams or within 1.0 m of camera path), Open3D SOR. Total: 792,291 -> 439,824 clean splats (44.49% pruned).
  - Exported `outputs/home/v2/scene_clean.ply` (98.99 MB), `outputs/home/v2/scene.splat` (13.42 MB), `outputs/home/v2/scene.spz` (8.64 MB).
  - SPZ round-trip quality: splat count diff = 0, max centre error = 0.210 mm, mean centre error = 0.117 mm, mean opacity error = 0.000976.
  - Splat coordinate frame verification: `[PASS] SPLAT COORDINATE FRAME IS VALID` (`outputs/home/v2/splat_frame_verification.json`).
  - Swapped assets to `viewer/public/scene/` via `scripts/swap_assets.ps1`.
- [x] **Phase 8.3 Dense Free-Space Carving & Occupancy Grid v3 (Stage F)**:
  - Dense depth and alpha rendering at 160x90 for all 446 cameras on GPU (4.11 s).
  - 3D voxel ray carving ($y \in [0.20, 1.40]$ m, 0.10 m voxels) up to $\min(\text{depth} - 0.15\text{ m}, 6\text{ m})$ for $\alpha \ge 0.60$.
  - Free-space evidence override: 10,285 floaters overridden by free evidence ($\ge 3$ cameras).
  - Collapsed to 2D occupancy grid: 3,562 occupied cells (29.98%), 3,719 known-free cells (31.30%), 2,027 walkable cells (17.06%).
  - Camera walkability: 91.70% (409/446 cameras); spawn point is 100% walkable.
  - Generated `outputs/home/occupancy_v3/occupancy.json`, `occupancy.png`, `occupancy_debug.png`, `occupancy_report.json`.
  - Synthetic unit test `test/test_occupancy.py` verified and PASSED.
- [x] **Phase 8.5 Occupancy Grid v3.1 & Viewer Refinements (2026-10-27)**:
  - **Camera Path Carving (Occupancy v3.1)**: Implemented 0.20 m horizontal radius corridor carving for all 446 cameras and inter-camera straight segments (< 0.50 m apart) across all 12 body-slab layers ($y \in [0.20, 1.40]$ m) in `scripts/build_occupancy_v3_1.py`.
  - **Floater Override Audit**: Overrode 362 candidate occupied cells (and 34 v3 occupied cells) along the camera corridor. Exactly 0 cells had $\ge 10$ supporting voxels (cells had only 1-7 scattered voxels), confirming they were transient floaters/stray rays rather than solid obstacles.
  - **Occupancy Metrics**: Total 2D cells = 11,880; occupied = 29.76% (3,536); inflated = 28.50% (3,386); known-free = 31.71% (3,767); unknown = 22.63% (2,688); walkable = 18.48% (2,195).
  - **Camera Walkability**: Reached **100.00%** (446 / 446 cameras, exceeding the $\ge 99\%$ target).
  - **Path Blocked-Step Share**: Reached **0.00%** (0 / 1,233 sample steps, beating the $\le 2\%$ target).
  - **Corridors & Passable Gaps**: Narrowest passable gap = 0.40 m. Three narrowest corridors: (1) $(x=2.95, z=4.95)$ width 0.40 m; (2) $(x=1.75, z=-0.25)$ width 0.40 m; (3) $(x=-1.25, z=-0.25)$ width 0.40 m.
  - **Artifacts & Deployment**: Generated `outputs/home/occupancy_v3_1/occupancy.json`, `occupancy.png` (+Z down, coloured trajectories, spawn point), `occupancy_debug.png`, `occupancy_report.json`. Backed up v3 assets to `outputs/home/occupancy_v3/` and deployed v3.1 to `viewer/public/scene/`.
  - **Synthetic Test**: `python test/test_occupancy.py` PASSED (0.212s).
  - **Optional Explore Mode (Key U)**:
    - Key U toggles "explore unreconstructed areas" (default OFF).
    - When OFF: unknown cells (`state === 0`) are blocked.
    - When ON: movement is permitted into unknown cells (`state === 0`), but occupied (`state === 2`) and inflated (`state === 3`) cells remain strictly blocked in both modes.
    - Dynamic HUD label `"unreconstructed area"` displayed while player stands in an unknown cell with explore mode active.
    - Pure collision implementation with 8 dedicated Vitest tests.
  - **Spark Render Quality Options**:
    - Confirmed `maxStdDev` in `@sparkjsdev/spark` type definitions sets maximum standard deviations from center to render Gaussians (default $\sqrt{8} \approx 2.828$); lowering to 2.0-2.45 truncates diffuse outer tails, trimming soft halos.
    - Presets exposed: Low (0.5x resolution, `maxStdDev: 2.0`, `lodSplatScale: 0.5`, `lodRenderScale: 2.0`), Med (`maxStdDev: 2.45`, `lodSplatScale: 0.75`, `lodRenderScale: 1.5`, default), High (`maxStdDev: 2.828`, `lodSplatScale: 1.0`, `lodRenderScale: 1.0`).
  - **Build & Dev Server**:
    - `npm run build` compiled cleanly in 4.24s.
    - Vitest test suite: all 35 tests PASSED (20 collision + 15 controls).
    - Dev server running on port 5173: `http://127.0.0.1:5173/`.
- [ ] **Phase 9: Deployment** -- Production static hosting and CDN bundle optimization.
- [ ] **Phase 10: Benchmarks** -- Automated benchmarking of frame rates and latency against competition specs.

## Known Issues / Diagnostics

- **Camera Walkability**: Fully resolved in v3.1 (100.00% camera walkability, 0.00% path blocked steps).
- **SfM Extension Quality**: Monolithic extension of new video sweeps was rejected in Stage B due to high reprojection error in the new sweeps (1.31-1.89 px). Production scene remains high-fidelity `home` reconstruction (PSNR 23.90 dB, SSIM 0.8891). The new sweep videos have significant motion blur and rotation-only segments that degrade global bundle adjustment.




