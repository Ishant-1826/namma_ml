# 3D Gaussian Splat Quality & Diagnostics Report

## 1. Raw Splat Properties (final_v2, SCENE=home)
- **Total Raw Splats**: 792,291
- **Scale Regularisation**: `--scale-loss-weight 1e-7`
- **Opacity Regularisation**: `--opac-loss-weight 1e-8`

### Max Scale Distribution
- **50th Percentile (Median)**: 0.0187 m (1.87 cm)
- **90th Percentile**: 0.0636 m (6.36 cm)
- **99th Percentile**: 0.2852 m (28.52 cm)

### Scale Anisotropy (Max / Min Scale Ratio)
- **50th Percentile**: 22.22
- **90th Percentile**: 584.12
- **99th Percentile**: 31188.70

### Floater & Needle Splat Counts
- **Splats > 0.15 m semi-axis**: 19,282 (2.43%)
- **Splats > 0.25 m semi-axis**: 9,328 (1.18%)
- **Splats > 0.50 m semi-axis**: 4,380 (0.55%)
- **Splats in cells seen by < 3 cameras**: 4,525 (0.57%)

---

## 2. Spark Renderer (`@sparkjsdev/spark`) Blur & Clamping Options
In the Three.js vanilla viewer (`viewer/src/main.ts`), the following Spark renderer properties control splat blur, LOD scaling, and radial clamping:

| Option | Type | Default | Balanced | High / Cinematic | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `lodSplatScale` | `number` | `0.50` | `0.75` | `1.00` | Controls splat spatial scaling factor in screen-space rasterisation. Lower values reduce splat overlap and eliminate haze. |
| `lodRenderScale` | `number` | `2.00` | `1.50` | `1.00` | Downsampling ratio for intermediate splat sorting passes. `1.0` is native pixel resolution. |
| `maxStdDev` | `number` | `2.00` | `2.45` | `2.828` ($\sqrt{8}$) | Gaussian standard deviation cutoff radius. Clamping at $2.0\sigma$ chops diffuse outer tails of large Gaussians, preventing blurry halo artifacts. |

---

## 3. Post-Processing Filter Progression
The following multi-stage filter pipeline was applied to `outputs/home/v2/scene_raw.ply`:
1. **Y Height Clamp `[-0.30, 4.40]` m**: Removes spurious sub-floor and high ceiling floaters while preserving floor and ceiling slabs.
2. **Intersected XZ BBox Crop**: Intersects camera trajectory bounding box (+4.0 m margin) with robust reference point extents (p0.5 to p99.5 + 1.0 m margin).
3. **Min Opacity Filter (`sig_op >= 0.05`)**: Eliminates invisible or faint haze splats.
4. **Max Scale Filter (`max_scale <= 0.25 m`)**: Eliminates giant floaters and puffy splats.
5. **Needle Filter (`ratio > 12 AND max_scale > 0.08 m`)**: Prunes razor-sharp, distorted needle splats.
6. **Camera Coverage Filter**: Removes splats in 0.5m grid cells seen by fewer than 3 cameras, except those within 1.0 m of the camera trajectory.
7. **Statistical Outlier Removal (SOR)**: Open3D k-NN outlier removal (`nb_neighbors=20, std_ratio=2.5`) on splat centroids.
