# Q0 Truth Report: Scene Composition, home3 SfM Registration & Coverage Analysis

## 1. Images Used in outputs/home/v2 Training (final_v2)
- **Source Manifest**: `outputs/home/train_pack/transforms.json`
- **Total Registered Frames**: 446 frames (from original handheld classroom sweep `data/home/images/frame_000001.jpg` to `frame_000450.jpg`, excluding 4 unregistered frames).
- **Counts per Source Folder**:
  - `originals`: 446 frames (100%)
  - `sweep_1`: 0 frames
  - `sweep_2`: 0 frames
  - `sweep_3`: 0 frames
  - `sweep_4`: 0 frames
  - `stills`: 0 frames
- **Split Configuration**: `--eval-split-every 8` (every 8th frame held out for validation).
  - **Train Set**: 390 frames
  - **Held-Out Set**: 56 frames (`indices: 0, 8, 16, ..., 440`)
  - **Held-Out Filenames (56 total)**:
    - `images/frame_000001.jpg`
    - `images/frame_000009.jpg`
    - `images/frame_000017.jpg`
    - `images/frame_000025.jpg`
    - `images/frame_000033.jpg`
    - `images/frame_000041.jpg`
    - `images/frame_000049.jpg`
    - `images/frame_000057.jpg`
    - `images/frame_000065.jpg`
    - `images/frame_000073.jpg`
    - `images/frame_000081.jpg`
    - `images/frame_000089.jpg`
    - `images/frame_000097.jpg`
    - `images/frame_000105.jpg`
    - `images/frame_000113.jpg`
    - `images/frame_000121.jpg`
    - `images/frame_000129.jpg`
    - `images/frame_000137.jpg`
    - `images/frame_000145.jpg`
    - `images/frame_000153.jpg`
    - `images/frame_000161.jpg`
    - `images/frame_000169.jpg`
    - `images/frame_000177.jpg`
    - `images/frame_000185.jpg`
    - `images/frame_000193.jpg`
    - `images/frame_000201.jpg`
    - `images/frame_000209.jpg`
    - `images/frame_000217.jpg`
    - `images/frame_000225.jpg`
    - `images/frame_000233.jpg`
    - `images/frame_000241.jpg`
    - `images/frame_000249.jpg`
    - `images/frame_000257.jpg`
    - `images/frame_000265.jpg`
    - `images/frame_000273.jpg`
    - `images/frame_000281.jpg`
    - `images/frame_000289.jpg`
    - `images/frame_000297.jpg`
    - `images/frame_000305.jpg`
    - `images/frame_000313.jpg`
    - `images/frame_000321.jpg`
    - `images/frame_000329.jpg`
    - `images/frame_000337.jpg`
    - `images/frame_000345.jpg`
    - `images/frame_000353.jpg`
    - `images/frame_000361.jpg`
    - `images/frame_000369.jpg`
    - `images/frame_000377.jpg`
    - `images/frame_000385.jpg`
    - `images/frame_000393.jpg`
    - `images/frame_000401.jpg`
    - `images/frame_000409.jpg`
    - `images/frame_000417.jpg`
    - `images/frame_000425.jpg`
    - `images/frame_000433.jpg`
    - `images/frame_000441.jpg`

---

## 2. home3 Colmap Evaluation (`sparse_ext` vs `sparse_pruned_ba`)

### (a) Initial Extended Model (`data/home3/colmap/sparse_ext`)
- **Total Registered Images**: 635 / 635 (100.00%)
- **Total 3D Points**: 153,124
- **Model-Wide Mean Reprojection Error**: 1.3508 px

| Source | Reg / Total | Reg % | 3D Points | Mean Reproj Error | Meets Rule (<1.5 px & >=60%)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **originals** | 450 / 450 | 100.00% | 129,501 | 1.4643 px | Baseline retained |
| **sweep_1** | 43 / 43 | 100.00% | 9,730 | 1.6791 px | **FAIL** (1.68 px > 1.5 px) |
| **sweep_2** | 41 / 41 | 100.00% | 3,058 | 1.8989 px | **FAIL** (1.90 px > 1.5 px) |
| **sweep_3** | 38 / 38 | 100.00% | 10,892 | 1.8317 px | **FAIL** (1.83 px > 1.5 px) |
| **sweep_4** | 52 / 52 | 100.00% | 11,137 | 1.7459 px | **FAIL** (1.75 px > 1.5 px) |
| **stills** | 11 / 11 | 100.00% | 5,501 | 1.4740 px | **PASS** (< 1.5 px, 100%) |

*Only 1 new source (`stills`) passed the <1.5 px error threshold; requirement is >= 2 sources.*

### (b) Pruned & Bundle-Adjusted Model (`data/home3/colmap/sparse_pruned_ba`)
- **Point Pruning**: Removed new points with error > 2.0 px or track length < 3 (11,116 points removed).
- **Image Pruning**: Removed new images with mean error > 1.5 px or < 100 observations (155 images removed).
- **Bundle Adjustment**: Fixed all 446 original camera poses (`--BundleAdjustment.refine_extrinsics 0`), refining new cameras' focal lengths.
- **Total Registered Images**: 480 / 635 (75.59%)
- **Total 3D Points**: 136,234
- **Model-Wide Mean Reprojection Error**: 1.3134 px (Target: < 1.1 px: **FAIL**)
- **Originals Drift**: Median shift 0.000000 m, Max shift 0.000000 m (446/446 identical).

| Source | Reg / Total | Reg % | 3D Points | Mean Reproj Error | Meets Rule (<1.5 px & >=60%)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **originals** | 449 / 450 | 99.78% | 126,902 | 1.4552 px | Baseline retained |
| **sweep_1** | 18 / 43 | 41.86% | 6,032 | 1.3072 px | **FAIL** (< 60% registered) |
| **sweep_2** | 0 / 41 | 0.00% | 0 | 0.0000 px | **FAIL** (< 60% registered) |
| **sweep_3** | 0 / 38 | 0.00% | 0 | 0.0000 px | **FAIL** (< 60% registered) |
| **sweep_4** | 7 / 52 | 13.46% | 1,609 | 1.0547 px | **FAIL** (< 60% registered) |
| **stills** | 6 / 11 | 54.55% | 1,691 | 1.3387 px | **FAIL** (< 60% registered) |

### (c) Exact Decision Recorded in `docs/PROGRESS.md`
> **Result**: Exactly 0 new sources passed (requirement: >= 2 sources with error < 1.5 px and >= 60% registered), and model-wide reprojection error is 1.3134 px (>= 1.1 px).  
> **Decision**: Take fallback to **`SCENE = home`** as planned. Continue to Stage C/D with `SCENE = home`.

---

## 3. Coverage Comparison: 0.5 m Cells Seen by >= 3 Cameras
- **home Bounds**: Based on aligned hall bounding box (`camera_footprint_m: 3.71m x 6.81m`, robust room extents: `9.16m x 10.76m`).
- **home Coverage**:
  - Total 0.5m grid cells in hall bounds: 1,422 cells
  - Cells seen by >= 3 cameras: **999 cells (70.25%)**
- **home3 Coverage** (extended coverage area across hall):
  - Total 0.5m grid cells in expanded hall bounds: 2,704 cells
  - Cells seen by >= 3 cameras: **2,139 cells (79.11%)**
- **Takeaway**: While `home3` physically observed 79.11% of the wider hall bounds, the fast rotational panning in sweeps 1–4 generated high reprojection errors (1.68–1.90 px) and motion blur that could not meet the strict precision gate without severe image pruning (leaving 0% to 42% registered).

---

## 4. Q1 Gate Evaluation
- **Requirement**: "at least 2 new sources registered at 60 percent or more in Q0b; otherwise write why and go to Q2 with SCENE=home".
- **Outcome**: In `sparse_pruned_ba`, registered shares are:
  - `sweep_1`: 41.86% (< 60%)
  - `sweep_2`: 0.00% (< 60%)
  - `sweep_3`: 0.00% (< 60%)
  - `sweep_4`: 13.46% (< 60%)
  - `stills`: 54.55% (< 60%)
- **Gate Result**: **0 new sources** meet the $\ge 60\%$ registration requirement.
- **Proceeding To**: Q2 with **`SCENE = home`**.
