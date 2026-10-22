# SfM Diagnosis — Namma Space / home

> Generated: 2026-10-20T10:13:45
> Model: `data/home/colmap/sparse/0/`  (+ aligned at `data/home/colmap/aligned/`)

---

## (a) Models and Registration

**Number of sub-models in `sparse/`:** 1

- Model `0`: **415 images**

**Total registered:** 415 / 450 = **92.2%** (pass criterion: ≥95% in ONE model)

| Pass | Registered | Total on disk | % |
|------|-----------|--------------|---|
| Pass 1 (eye-level, frames 1–201) | 166 | 201 | 82.6% |
| Pass 2 (chest-level, frames 202–450) | 249 | 249 | 100.0% |

> **FAIL**: 
> Both passes in ONE model: YES

---

## (b) Camera Intrinsics

**Number of distinct camera entries:** 1

- `Camera 1: SIMPLE_RADIAL 1920x1080 | f=1449.72px cx=960.0 cy=540.0 k1=-0.016650`

**Shared single camera:** YES

> COLMAP SIMPLE_RADIAL: params = [f, cx, cy, k1]. Both videos are 1920×1080 30fps,
> same phone, same zoom — sharing one camera model is appropriate here.

---

## (c) Step Lengths and Yaw Jumps (Aligned Model)

### 10 Largest Frame-to-Frame Jumps

| From | To | Distance (m) | Δyaw (°) |
|------|-----|-------------|---------|
| `frame_000152.jpg` | `frame_000185.jpg` | 1.224 | 40.9 |
| `frame_000427.jpg` | `frame_000428.jpg` | 0.794 | 1.5 |
| `frame_000071.jpg` | `frame_000075.jpg` | 0.716 | 39.4 |
| `frame_000319.jpg` | `frame_000320.jpg` | 0.323 | 18.6 |
| `frame_000013.jpg` | `frame_000014.jpg` | 0.197 | 3.0 |
| `frame_000027.jpg` | `frame_000028.jpg` | 0.183 | 1.1 |
| `frame_000032.jpg` | `frame_000033.jpg` | 0.165 | 2.6 |
| `frame_000020.jpg` | `frame_000021.jpg` | 0.154 | 1.1 |
| `frame_000247.jpg` | `frame_000248.jpg` | 0.152 | 2.0 |
| `frame_000212.jpg` | `frame_000213.jpg` | 0.151 | 4.4 |

### Yaw Jumps > 40° in One Step

Pass 1: **1** events | Pass 2: **0** events

| Pass | From | To | Δyaw (°) |
|------|------|-----|---------|
| P1 | `frame_000152.jpg` | `frame_000185.jpg` | 40.9 |

---

## (d) Long-Range Matches (|Δframe| > 30)

| Category | Pairs | Notes |
|----------|-------|-------|
| Within Pass 1 (|Δ|>30) | 610 | Loop closures or false aliases |
| Within Pass 2 (|Δ|>30) | 0 | Loop closures or false aliases |
| Cross-pass | 132 | Geometric links between passes |

**Suspicious (< 20 inliers) within-pass long-range pairs:** P1=24, P2=0

Top suspicious Pass 1 pairs (low inlier count):
  - `frame_000085.jpg` ↔ `frame_000004.jpg`: 15 inliers
  - `frame_000099.jpg` ↔ `frame_000064.jpg`: 15 inliers
  - `frame_000106.jpg` ↔ `frame_000064.jpg`: 15 inliers
  - `frame_000113.jpg` ↔ `frame_000032.jpg`: 15 inliers
  - `frame_000124.jpg` ↔ `frame_000032.jpg`: 15 inliers

> **Note:** Many low-inlier long-range pairs within a pass indicate
> perceptual aliasing (identical seat rows matching wrong locations).

---

## (e) Handedness and Turn Sequence

| Pass | Points | Signed area (XZ) | Winding | RDP pts | Turn sequence |
|------|--------|-----------------|---------|---------|--------------|
| Pass 1 | 166 | 1.671 | CCW ✓ | 5 | `L,L,L` |
| Pass 2 | 249 | 2.218 | CCW ✓ | 5 | `L,L,L` |
| **Expected** | — | > 0 | CCW | — | `L,R,L,L,R,L,L,R,L` |

**Start-end gap:**
- Pass 1: 0.890 m
- Pass 2: 0.452 m
- Combined first-to-last: 0.806 m  *(pass criterion: < 0.5 m)*

**Cross-pass median nearest-neighbour distance:** 0.493 m  *(pass criterion: ≤ 0.4 m)*

---

## (f) Alignment Quality

**Mean camera-up vector (aligned model):** [-0.0245, +0.9973, +0.0689]
**Angle to world +Y:** 4.20°  *(pass criterion: ≤ 8°)*

### Floor Plane Candidates

| Candidate | Normal | Inlier frac | Angle to cam-up | Accept? |
|-----------|--------|-------------|----------------|---------|
| Lowest 20% of cloud | [-0.0011,+1.0000,+0.0036] | 0.540 | 3.98° | YES |

### Per-Pass Camera Heights

| Pass | Median height (m) | Expected range (m) | Status |
|------|------------------|-------------------|--------|
| Pass 1 (eye-level) | 1.666 | 1.4 – 1.8 | ✓ |
| Pass 2 (chest-level) | 1.272 | 1.0 – 1.4 | ✓ |
| **P1/P2 ratio** | 1.310 | ~1.3 | ✓ |

**align_report.json:** scale = 0.294789652207842 m/unit (method: phone_height)

---

## (g) Plan-Fit Residual

**Similarity fit (reconstruction XZ → sketch XY):**
- Fit RMS: 1.8719 sketch units
- Plan path length: 12.0000 sketch units
- **Residual: 15.6%**  *(pass criterion: < 15%, no mirror)*
- Mirror detected: **False**  *(must be False)*

---

## Conclusion

**Root causes identified:**

- (i) Bad SfM: 92.2% registered < 95% threshold
- (i) Bad SfM: reprojection error 1.097 > 1 px

### Pass / Fail Summary

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Sub-models | 1 | 1 | ✓ |
| Overall registration | 92.2% | ≥95% | ✗ |
| Pass 1 registration | 82.6% | ≥95% | ✗ |
| Pass 2 registration | 100.0% | ≥95% | ✓ |
| Reprojection error | 1.097 px | < 1.0 px | ✗ |
| Start-end gap | 0.806 m | < 0.5 m | ✗ |
| Turn sequence P1 | `L,L,L` | `L,R,L,L,R,L,L,R,L` | ✗ |
| Turn sequence P2 | `L,L,L` | `L,R,L,L,R,L,L,R,L` | ✗ |
| P1/P2 cross-dist | 0.493 m | ≤ 0.4 m | ✗ |
| Plan-fit residual | 15.6% | < 15% | ✗ |
| No mirror | True | True | ✓ |
| Floor normal angle | 4.20° | ≤ 8° | ✓ |

### Recommended Next Step

Based on the above, the recommended experiment order is:

1. **E1**: Reconstruct Pass 1 alone (frames 1–201) to verify single-pass quality
2. **E3**: Sequential matching (overlap=15, guided, loop-detect OFF) to suppress chair aliasing
3. **E2**: Per-folder cameras if E1 shows intrinsics mismatch
4. **E4**: global_mapper on the best DB
5. **E5/E6**: Re-extract / ALIKED only if blur or SIFT quality is the bottleneck
