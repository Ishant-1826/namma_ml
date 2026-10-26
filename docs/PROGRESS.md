# Pipeline Progress Tracking -- Namma Space

## RULES

R1. Never modify data/home, data/home2, outputs/home or any earlier output folder;
    new results go into new folders (home3, v2, v3). Never git push. Never guess
    COLMAP, ns-train or tool flags: read --help for every command you use.
R2. LONG JOBS. Anything that can take over 30 seconds must run as a Windows
    scheduled task so it is independent of this terminal, never as a child of it:
      schtasks /create /tn <name> /tr "powershell -NoProfile -ExecutionPolicy Bypass
        -File <script.ps1>" /sc once /st 00:00 /f
      schtasks /run /tn <name>
    Every script writes to its own log file under logs/ with timestamps, writes a
    <stage>.done file only on success, is resumable (skips finished work), and
    writes its PID or task name to a file. Delete the task when it is done.
R3. NEVER WAIT. Do not sleep, loop or use timers or any "wait" tool. After
    launching a job, run ONE status check (under 5 seconds), write the status to
    docs/PROGRESS.md, and end your turn with a one-line message saying what is
    running, the estimated finish, and "send 'continue' to re-check". When I send
    "continue": run one status check; if the job finished, validate it and go to the
    next stage; if it is still running, say so in one line; if it died (no process, no
    .done, log not updated for 10 minutes), find the cause in the log, relaunch with
    resume, and tell me.
R4. NATIVE COMMANDS IN POWERSHELL. COLMAP prints normal progress to stderr, and
    PowerShell turns stderr text into a fatal NativeCommandError when
    $ErrorActionPreference='Stop'. All scripts must dot-source scripts/common.ps1
    with a function Invoke-Native($Exe, [string[]]$CmdArgs, $Log) that sets
    $ErrorActionPreference='Continue' around the call, runs it with 2>&1, appends
    every line to $Log, and throws ONLY if $LASTEXITCODE is not 0 (printing the last
    20 log lines). Do not use a parameter named $Args. Judge failure by exit code and
    by log lines starting with E or F, never by stderr text alone.
R5. POWER. At the start run powercfg /change standby-timeout-ac 0 and
    powercfg /change hibernate-timeout-ac 0, tell me you did, and restore both to
    30 minutes at the end. Tell me to keep the laptop plugged in with the lid open.
R6. FALLBACK CLOCK. If a stage takes more than 3 times its estimate, kill it and
    take the fallback for that stage. Overall hard limit: I submit on 30 Oct, so
    the final assets must be deployed to viewer/public/scene/ with a working viewer
    before then; always keep the previous assets in outputs/home/web_assets_v1/.
R7. After every stage update docs/PROGRESS.md and docs/STATE.md. UI stays
    framework-free. Do not change the viewer except for asset files and,
    if Spark exposes them, rendering options (stage 7).

---

## Stage Progress Table

| Stage | Status | Started | Finished | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Stage 1: Safety & Asset Backup** | Completed | 2026-10-26 16:53 | 2026-10-26 16:54 | Power set to 0 timeout; processes checked; web_assets_v1 backed up; scripts/swap_assets.ps1 written |
| **Stage A: Build & Extend SfM (home3)** | Completed | 2026-10-26 16:58 | 2026-10-26 17:49 | All 635 frames registered (100%); 153,124 points; runtime 51.02m (feat: 16s, match: 29m, map: 21.6m) |
| **Stage B: Verify home3** | Failed (Fallback) | 2026-10-26 17:50 | 2026-10-26 17:53 | 446/446 orig registered; shift 0.00m med / 0.00m max; fx diff 0.0%; reproj 1.3508px (>1.0px fail) -> SCENE=home |
| **Stage C: Align & Prepare Training Data** | Completed | 2026-10-26 17:54 | 2026-10-26 17:57 | SCENE=home verified: floor y=0.0, tilt 0.84°<=8°, cam h 1.450m; coverage cells >=3: home 70.25% (999/1422) vs home3 79.11% (2139/2704) |
| **Stage D: Train final_v2** | Completed | 2026-10-26 18:20 | 2026-10-26 18:56 | Brush 30k steps (36.48m); 792,291 raw splats (178.32MB); Holdout metrics: PSNR 23.90 dB, SSIM 0.8891, LPIPS 0.2812 |
| **Stage E: Postprocess v2** | Completed | 2026-10-26 19:17 | 2026-10-26 19:26 | Splat diagnostics (docs/splat_quality.md); 792k -> 439,824 splats; SPZ 8.64MB (0.117mm error); frame verification PASS; assets swapped |
| **Stage F: Occupancy v3** | Completed | 2026-10-26 19:32 | 2026-10-26 19:35 | Dense 160x90 depth carving (4.11s); 10,285 floaters overridden; cam walkable 91.70%; synthetic aisle unit test PASSED |
| **Stage F (v3.1): Occupancy v3.1** | Completed | 2026-10-27 00:03 | 2026-10-27 00:04 | Camera path carving (r=0.20m, 446 cams + segments <0.5m); 362 cand / 34 v3 floater cells overridden (0 with >=10 voxels); cam walkable 100.00% (446/446); path blocked-step 0.00%; synthetic test PASSED; assets deployed |
| **Stage G: Final Check & Report** | Completed | 2026-10-26 19:35 | 2026-10-26 19:36 | Vite production build PASSED (4.66s); Vitest 27/27 tests PASSED; powercfg restored to 30 min |
| **Stage H: Viewer Refinements** | Completed | 2026-10-27 00:12 | 2026-10-27 00:15 | Key U explore mode (HUD label 'unreconstructed area'); Spark Low/Med/High quality presets (maxStdDev 2.0/2.45/2.828); Vitest 35/35 PASSED; Vite build PASSED (4.24s); dev server scheduled on port 5173 |

---

## Stage B: Verification Audit Numbers & Pruning Evaluation

### 1. Registration per Source (Initial `data/home3/colmap/sparse_ext`)
| Source | Reg / Total | Reg % | 3D Points | Mean Reproj Error |
| :--- | :---: | :---: | :---: | :---: |
| **originals** | 450 / 450 | 100.00% | 129,501 | 1.4643 px |
| **sweep_1** | 43 / 43 | 100.00% | 9,730 | 1.6791 px |
| **sweep_2** | 41 / 41 | 100.00% | 3,058 | 1.8989 px |
| **sweep_3** | 38 / 38 | 100.00% | 10,892 | 1.8317 px |
| **sweep_4** | 52 / 52 | 100.00% | 11,137 | 1.7459 px |
| **stills** | 11 / 11 | 100.00% | 5,501 | 1.4740 px |
| **Model-Wide** | 635 / 635 | 100.00% | 153,124 | 1.3508 px |

### 2. Camera Intrinsics (7 Cameras, 4 Decimals)
| Cam ID | Source | Model | fx | Principal Point (cx, cy) | k1 | Refined vs Prior (1440.0) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | video1 | SIMPLE_RADIAL | 1477.9196 | (960.0000, 540.0000) | 0.0000 | Original baseline (1477.94) |
| 2 | video2 | SIMPLE_RADIAL | 1476.0312 | (960.0000, 540.0000) | 0.0000 | Original baseline (1476.14) |
| 3 | sweep_1 | SIMPLE_RADIAL | 1485.6599 | (960.0000, 540.0000) | 0.0000 | **REFINED** (+45.6599 px from 1440.0) |
| 4 | sweep_2 | SIMPLE_RADIAL | 1560.9540 | (960.0000, 540.0000) | 0.0000 | **REFINED** (+120.9540 px from 1440.0) |
| 5 | sweep_3 | SIMPLE_RADIAL | 1456.7076 | (960.0000, 540.0000) | 0.0000 | **REFINED** (+16.7076 px from 1440.0) |
| 6 | sweep_4 | SIMPLE_RADIAL | 1508.1588 | (960.0000, 540.0000) | 0.0000 | **REFINED** (+68.1588 px from 1440.0) |
| 7 | stills | SIMPLE_RADIAL | 1477.8034 | (960.0000, 540.0000) | 0.0000 | **REFINED** (+37.8034 px from 1440.0) |

*All 5 new cameras were refined during SfM extension (none remained at the 1440.0 prior).*

### 3. Pruning & Re-Bundle Adjustment (`sparse_pruned` & `sparse_pruned_ba`)
- **Point Pruning**: Pruned new points only with reprojection error > 2.0 px or track length < 3 (removed 11,116 points; original points untouched).
- **Image Pruning**: Pruned new images with mean error > 1.5 px or < 100 observations (removed 155 new images; output in `data/home3/colmap/sparse_pruned`).
- **Bundle Adjustment**: Fixed all 446 original image poses (`--BundleAdjustment.refine_extrinsics 0`), refining new cameras' focal lengths (`data/home3/colmap/sparse_pruned_ba`). Cost decreased from 0.8515 px to 0.8431 px.

### 4. Stage B Criteria Re-Evaluation on `sparse_pruned_ba`
| Criterion | Target | Measured Value | Result |
| :--- | :---: | :---: | :---: |
| 446 originals unchanged | 446 / 446 | 446 / 446 (shift 0.000m med / 0.000m max) | **PASS** |
| Model-wide mean reprojection error | < 1.1 px | **1.3134 px** | **FAIL** |
| At least TWO new sources with error < 1.5 px & reg >= 60% | >= 2 sources | **0 sources** | **FAIL** |

#### Per-Source Breakdown on `sparse_pruned_ba`
| Source | Reg / Total | Reg % | 3D Points | Mean Reproj Error | Meets Rule (<1.5 px & >=60%)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **originals** | 449 / 450 | 99.78% | 126,902 | 1.4552 px | Baseline retained |
| **sweep_1** | 18 / 43 | 41.86% | 6,032 | 1.3072 px | **FAIL** (< 60% registered) |
| **sweep_2** | 0 / 41 | 0.00% | 0 | 0.0000 px | **FAIL** (< 60% registered) |
| **sweep_3** | 0 / 38 | 0.00% | 0 | 0.0000 px | **FAIL** (< 60% registered) |
| **sweep_4** | 7 / 52 | 13.46% | 1,609 | 1.0547 px | **FAIL** (< 60% registered) |
| **stills** | 6 / 11 | 54.55% | 1,691 | 1.3387 px | **FAIL** (< 60% registered) |

### 5. Final Decision
- **Result**: Exactly 0 new sources passed (requirement: >= 2 sources with error < 1.5 px and >= 60% registered), and model-wide reprojection error is 1.3134 px (>= 1.1 px).
- **Decision**: Take fallback to **`SCENE = home`** as planned. Continue to Stage C/D with `SCENE = home`.



