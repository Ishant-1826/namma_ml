# SfM Experiments Log — Namma Space / home

> Pass criteria: ≥95% frames in ONE model; reprojection error < 1 px; start-end gap < 0.5 m;
> turn sequence = `L,R,L,L,R,L,L,R,L` (±1 spurious); P1↔P2 median dist ≤ 0.4 m;
> plan-fit residual < 15% path length, no mirror; floor normal within 8° of camera-up.

| Exp | Description | Reg% | Models | Reproj err (px) | Start-end gap (m) | Turn seq P1 | P1↔P2 dist (m) | Plan residual | Notes | Pass? |
|-----|-------------|------|--------|----------------|------------------|-------------|----------------|--------------|-------|-------|
| Baseline | Exhaustive match + incremental mapper, all 400 frames | 81.0% | 1 | 1.299 | 0.517 | R,L,R,R,L,R,L,L,L | 0.291 | 13.4% | Camera heights inverted (P1=1.20m, P2=1.47m): scale wrong | ✗ |
| E1 | Pass 1 alone (207 frames), same exhaustive DB, `init_min_tri=8`, `abs_pose_min_inliers=20` | 81.6% (169/207) | **2** | 1.220 (model0) | TBD | TBD | N/A | TBD | 2 sub-models: 169+12. Blank wall at frames 127-140 is the structural break. Exhaustive DB aliases prevent bridging. | ✗ |
| E3 | Sequential match (overlap=15, loop-detect OFF, guided, max_ratio=0.80), full 400 frames mapper | 25.5% (max model 102/400) | **9** | ~1.18 | N/A | Fragmented | N/A | N/A | Severe fragmentation into 9 sub-models (max 102 imgs). Pure sequential matching has 0 cross-pass links and breaks at low-texture/turn segments. | ✗ |


## Protocol Evaluation (October 22, 2026)

Pass criteria per experiment:
- At least 90% of frames of the video(s) used are in ONE model
- Mean reprojection error under 1.0 px
- fx and fy within 5% of each other, |k1| < 0.3
- No frame-to-frame step larger than 5x the median step

| Experiment | Setting Changed | Registered % per Video | Models | Reproj Err (px) | fx | fy | k1 | Start-End Gap | Largest Step | Step Ratio | Pass? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | Video 1 alone, SIMPLE_RADIAL, single camera, focal prior 1440px (0.75x longest side), locked cx/cy/k1, sequential overlap 15, loop detection OFF, guided matching ON | V1: 98.20% (218/222)<br>V2: N/A | 1 | 1.3209 | 1483.31 | 1483.31 | 0.0000 | 0.9183 | 1.1836 | 10.23x | ✗ (Reproj 1.32px > 1.0px; Step ratio 10.2x > 5x) |
| **E2** | Same as E1 + stricter geometric verification (`min_num_inliers=30`) and quadratic sequence loop matching | V1: 98.20% (218/222)<br>V2: N/A | 1 | 1.3417 | 1449.71 | 1449.71 | 0.0000 | 13.4295 | 1.9013 | 57.36x | ✗ (Reproj 1.34px > 1.0px; Step ratio 57.4x > 5x) |
| **E3** | Add Video 2 as a SECOND camera (`--ImageReader.single_camera_per_folder 1`), keeping settings from best of E1/E2 (E1), reconstruct both together | V1: 100.00% (222/222)<br>V2: 98.25% (224/228)<br>Total: 99.11% (446/450) | 1 | 1.3059 | 1477.94 | 1477.94 | 0.0000 | 0.6501 | 1.2541 | 10.47x | ✗ (Reproj 1.31px > 1.0px; Step ratio 10.5x > 5x) |
| **E4** | `global_mapper` (GLOMAP via `pycolmap.global_mapping`) on best database (E3) | V1: 100.00% (222/222)<br>V2: 100.00% (228/228)<br>Total: 100.00% (450/450) | 1 | 1.8894 | 1400.83 | 1400.83 | -0.0373 | 14.4813 | 42.5303 | 391.29x | ✗ (Reproj 1.89px > 1.0px; Step ratio 391.3x > 5x; translation jumps) |
| **E5** | Re-extract frames at 4 fps from sharpest segments (sliding window + duplicate filtering) and rerun best settings (E3) | V1: 100.00% (225/225)<br>V2: 41.33% (93/225)<br>Total: 70.67% (318/450) | 3 | 1.3490 | 1474.88 | 1474.88 | 0.0000 | 9.5576 | 1.1715 | 1.80x | ✗ (Model 1 Reg 70.7% < 90%; Reproj 1.35px > 1.0px; 3 sub-models) |
