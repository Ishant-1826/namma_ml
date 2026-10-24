# SfM Diagnosis — Namma Space / home

> Date: 2026-10-22  
> Scene: `home`  
> Primary Model: `data/home/colmap/sparse/0/`  
> Secondary Model: `data/home/colmap/sparse/old_4cam/`  
> Database: `data/home/colmap/database.db`  
> Input Frames: 450 total (`data/home/images/`) across 2 videos (`data/home/frames_report.json`)

---

## 1. Number of COLMAP Models & Registered % per Video

- **Total sparse sub-models in `data/home/colmap/sparse/`:** **2**
  - **Model `0`** (primary / selected largest): **4 images**, 781 3D points
  - **Model `old_4cam`**: **2 images**, 62 3D points

### Registration Rate by Video (Model `0`)

The 450 input frames originate from two consecutive continuous handheld video captures (split at frame 222):
- **Video 1 (`20261021_195555.mp4` / Pass 1, Eye Level, frames 1–222):**
  - Total frames on disk: **222**
  - Registered frames: **4** (`frame_000074.jpg`, `frame_000080.jpg`, `frame_000085.jpg`, `frame_000090.jpg`)
  - Registration percentage: **1.80%** (4 / 222)
- **Video 2 (`20261021_195930.mp4` / Pass 2, Chest Level, frames 223–450):**
  - Total frames on disk: **228**
  - Registered frames: **0**
  - Registration percentage: **0.00%** (0 / 228)
- **Overall Registration:**
  - Total registered: **4 / 450** = **0.89%** (Severe Failure; threshold is $\ge 85\%$)

| Model Name | Registered Images | 3D Points | Video 1 (Eye, 222 f) | Video 2 (Chest, 228 f) | Overall % |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Model `0`** (Primary) | **4** | **781** | **4 (1.80%)** | **0 (0.00%)** | **0.89%** |
| **Model `old_4cam`** | **2** | **62** | **2 (0.90%)** | **0 (0.00%)** | **0.44%** |

---

## 2. Cameras: Model, Image Size, Focal Length, Distortion, & Sharing

### Camera Parameters in Primary Model (`sparse/0/cameras.txt`)

- **Number of distinct camera entries:** **1** (`camera_id = 1`)
- **Image Size:** **1920 × 1080 px** (width = 1920, height = 1080)
- **Camera Model:** **`OPENCV`** (8 parameters)
- **Focal Lengths:**
  - $f_x =$ **9264.2891 px**
  - $f_y =$ **2071.7869 px**
  - Ratio $f_x / f_y =$ **4.47 : 1** (Unphysical; smartphone sensors possess square pixels where $f_x \approx f_y \approx 1400\text{–}1500\text{ px}$)
- **Principal Point:**
  - $c_x =$ **960.0 px**
  - $c_y =$ **540.0 px** (exact geometric image center)
- **Distortion Parameters:**
  - Radial $k_1 =$ **+4.531015**
  - Radial $k_2 =$ **-25.093274**
  - Tangential $p_1 =$ **+1.041986**
  - Tangential $p_2 =$ **+0.157974**
  - *Analysis:* These distortion coefficients are pathological orders of magnitude larger than normal physical smartphone optics ($|k_1| < 0.1, |k_2| < 0.05$). This indicates catastrophic focal length divergence and degenerate parameter overfitting during two-view initialization.

### Camera Sharing
- **Do both videos share one camera?**
  - **YES**. In `data/home/colmap/database.db`, all 450 images (from both Video 1 and Video 2) are mapped to a single camera (`camera_id = 1`).

---

## 3. Step Lengths & Yaw Jumps

Because only 4 frames are registered in Model `0` (`frame_000074.jpg`, `frame_000080.jpg`, `frame_000085.jpg`, `frame_000090.jpg`), there are exactly **3 consecutive frame-to-frame steps** in the entire reconstruction.

### Consecutive Step Lengths (Descending Order)

| Rank | From Frame | To Frame | Distance (Model Units) | $\Delta\text{yaw}$ (deg) | 3D Rotation (deg) | Frame Gap |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | `frame_000074.jpg` | `frame_000080.jpg` | **8.4703** | 4.74° | 9.32° | +6 frames |
| 2 | `frame_000085.jpg` | `frame_000090.jpg` | **7.2959** | 0.50° | 9.76° | +5 frames |
| 3 | `frame_000080.jpg` | `frame_000085.jpg` | **6.5232** | 1.33° | 8.25° | +5 frames |

*(Note: There are only 3 consecutive steps available; all 3 are reported above).*

### Camera Centre Coordinates in Model 0:
- `frame_000074.jpg`: $C = [-0.4064, -4.4687, +8.5245]$
- `frame_000080.jpg`: $C = [-0.0953, -0.9357, +0.8325]$
- `frame_000085.jpg`: $C = [+0.7155, +5.4044, -0.4705]$
- `frame_000090.jpg`: $C = [+0.5016, +12.6964, -0.3619]$

### Yaw Changes $> 40^\circ$ in One Step
- **Zero frames** (**0** events).
- Maximum $\Delta\text{yaw}$ between any consecutive registered frame is **4.74°** (between `frame_000074.jpg` and `frame_000080.jpg`).

---

## 4. Loop Closure Distance (First to Last Camera)

- **Pass 1 (Video 1, frames 1–222):**
  - First registered camera: `frame_000074.jpg` ($C = [-0.4064, -4.4687, +8.5245]$)
  - Last registered camera: `frame_000090.jpg` ($C = [+0.5016, +12.6964, -0.3619]$)
  - Euclidean distance: **19.3502 model units**
  - *Analysis:* Across only 16 frames (~5.3 seconds of walking along one wall), the estimated camera translation drifted by 19.35 model units instead of closing. The trajectory is entirely open.
- **Pass 2 (Video 2, frames 223–450):**
  - **N/A** (**0.00 m** / no frames registered).
- **Combined / Cross-Pass Loop Closure:**
  - **N/A** (Video 2 has 0 registered cameras; no cross-pass loop closure exists).

---

## 5. Mean Camera Height Above Floor per Pass

In Model `0`, the camera up vectors estimate an UP direction of $\mathbf{u} = [+0.0308, -0.9757, +0.2168]$. Projecting points and camera centres along this vertical axis reveals:

- **Estimated Floor Elevation along UP:** $-9.1874$ model units (1st percentile of 3D points)
- **Pass 1 Individual Camera Heights Above Floor:**
  - `frame_000074.jpg`: $+15.3836$ model units
  - `frame_000080.jpg`: $+10.2780$ model units
  - `frame_000085.jpg`: $+3.8343$ model units
  - `frame_000090.jpg`: $-3.2637$ model units
- **Pass 1 Mean Camera Height Above Floor:** **6.5580 model units** (unscaled, unaligned)
  - *Vertical Drift:* Camera height drops precipitously by **18.6473 model units** in just 16 frames (drifting from $+15.38$ down into the floor at $-3.26$), demonstrating a non-physical vertical trajectory.
- **Pass 2 Mean Camera Height Above Floor:** **N/A** (0 cameras registered).

---

## 6. Verdict: (i) Bad SfM, (ii) Bad Alignment or Scale, or (iii) Plot Only

### **VERDICT: (i) Bad SfM**

### Evidence & Detailed Root Cause Analysis:

1. **Extreme Registration Failure (0.89% registered):**
   - 446 of the 450 frames failed to register.
   - Video 2 completely failed to initialize or register (0 / 228 frames).
   - Video 1 registered only 4 isolated frames (4 / 222 frames).
   - This falls catastrophically short of the required $\ge 85\%$ registration threshold.

2. **Catastrophic Two-View Degeneracy & Intrinsics Divergence:**
   - COLMAP initialized two-view geometry on a nearly planar, low-parallax segment between frames 74–90.
   - The estimated camera model diverged into non-physical values: $f_x \approx 9264\text{ px}$ vs $f_y \approx 2072\text{ px}$ (a $4.47 : 1$ aspect ratio distortion), paired with violent radial/tangential distortion ($k_1 = +4.53, k_2 = -25.09, p_1 = +1.04$).
   - On smartphone video with uniform square pixels, $f_x$ must equal $f_y \approx 1400\text{–}1500\text{ px}$ with $|k_1| < 0.1$.

3. **Unphysical Translation Explosions & Vertical Drift:**
   - Consecutive step lengths are $8.47, 7.30, 6.52$ model units per frame. At 3 fps walking speed, physical steps are $\sim 0.3\text{ m}$.
   - Vertical camera position drifts by 17 units in 16 frames, plunging through the floor plane.

4. **Why Alignment or Plotting Is Not the Blame:**
   - (ii) *Alignment or scale* cannot fix a model that only contains 4 corrupt frames spanning 16 video frames.
   - (iii) *Plotting* accurately renders the true mathematical output of the reconstruction; the failure lies entirely within the COLMAP feature matching and incremental mapper initialization.
