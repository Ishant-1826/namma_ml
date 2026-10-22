# SfM Experiments Log — Namma Space / home

> Pass criteria: ≥95% frames in ONE model; reprojection error < 1 px; start-end gap < 0.5 m;
> turn sequence = `L,R,L,L,R,L,L,R,L` (±1 spurious); P1↔P2 median dist ≤ 0.4 m;
> plan-fit residual < 15% path length, no mirror; floor normal within 8° of camera-up.

| Exp | Description | Reg% | Models | Reproj err (px) | Start-end gap (m) | Turn seq P1 | P1↔P2 dist (m) | Plan residual | Notes | Pass? |
|-----|-------------|------|--------|----------------|------------------|-------------|----------------|--------------|-------|-------|
| Baseline | Exhaustive match + incremental mapper, all 400 frames | 81.0% | 1 | 1.299 | 0.517 | R,L,R,R,L,R,L,L,L | 0.291 | 13.4% | Camera heights inverted (P1=1.20m, P2=1.47m): scale wrong | ✗ |
| E1 | Pass 1 alone (207 frames), same exhaustive DB, `init_min_tri=8`, `abs_pose_min_inliers=20` | 81.6% (169/207) | **2** | 1.220 (model0) | TBD | TBD | N/A | TBD | 2 sub-models: 169+12. Blank wall at frames 127-140 is the structural break. Exhaustive DB aliases prevent bridging. | ✗ |
| E3 | Sequential match (overlap=15, loop-detect OFF, guided, max_ratio=0.80), full 400 frames mapper | 25.5% (max model 102/400) | **9** | ~1.18 | N/A | Fragmented | N/A | N/A | Severe fragmentation into 9 sub-models (max 102 imgs). Pure sequential matching has 0 cross-pass links and breaks at low-texture/turn segments. | ✗ |
| E4 | global_mapper on exhaustive DB (10,022 pairs) | 91.8% (367/400) | 1 | 3.558 | >1000m | Degenerate | N/A | >100% | Severe gauge drift and translation explosion (jumps >13,000 units) due to lack of focal length priors and weak viewing graph. | ✗ |

