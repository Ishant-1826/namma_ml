# Standard Operating Procedure (SOP): Smartphone Indoor Capture

**Target:** Reconstruct a single indoor space (room + adjoining corridor) with metric fidelity using smartphone video, COLMAP SfM, and 3D Gaussian Splatting (3DGS).

---

## 1. Why Rigid Capture Discipline Is Mandatory

Structure-from-Motion (SfM) solves for camera poses and scene structure by matching feature points across overlapping frames. Gaussian Splatting subsequently fits 3D Gaussians to minimize photometric re-projection error. Both steps fail catastrophically if:
1. **The camera rotates in place without translation:** Zero-baseline rotation produces collinear rays, causing depth ambiguity, degenerate epipolar geometry, and split SfM models.
2. **Motion blur is present:** Blurry frames destroy high-frequency keypoints (SIFT), causing frame dropouts.
3. **Auto-Exposure / White Balance hunts:** Changing photometric response causes Gaussian splats to develop "floaters" and hazy halos.
4. **Dynamic elements or reflections move:** Mirrors, TVs, and moving people violate the static scene assumption (Lambertian surfaces).

---

## 2. Pre-Capture Checklist: Device & Space

### 2.1 Smartphone Hardware & App Settings
Use a dedicated manual video app if possible (e.g., Blackmagic Camera, ProTake, Open Camera, or native Pro/Manual Video mode):

- [ ] **Lens:** Physically clean front and rear lenses with a microfiber cloth.
- [ ] **Camera Sensor:** Fixed **1x main sensor only**. 
  - *Do NOT use ultra-wide* (severe barrel distortion and optical softness at edges harm COLMAP calibration).
  - *Do NOT use telephoto* (narrow field of view hurts overlap).
  - *NEVER use digital zoom* (destroys pixel-to-ray geometry).
- [ ] **Resolution & Frame Rate:** **4K (3840×2160) at 30 fps or 60 fps**. (Minimum acceptable: 1080p).
- [ ] **Video Stabilisation:** **TURN OFF** all software/electronic video stabilisation (EIS). 
  - *Reason:* EIS dynamically crops and warps frames frame-by-frame, continuously altering the camera intrinsics and principal point, which breaks the pinhole camera model. Hardware optical stabilisation (OIS) is acceptable if it cannot be disabled.
- [ ] **HDR & Tone Mapping:** **TURN OFF** HDR video (Dolby Vision, HDR10+) and dynamic tone mapping. Record in standard Rec.709 8-bit SDR.
- [ ] **Exposure & Shutter Speed:** Set shutter speed to **$\ge 1/120\text{ s}$** (ideally $1/200\text{ s}$ or faster) to freeze motion blur during walking. Adjust ISO cleanly (keep ISO $< 800$ to avoid sensor noise).
- [ ] **Locks (Crucial):**
  - **Lock Auto-Exposure (AE-L)** on a neutral mid-tone.
  - **Lock Focus (AF-L)** at ~2.0 to 2.5 metres (hyperfocal for indoor rooms).
  - **Lock White Balance (AWB-L)** to a fixed Kelvin temperature (e.g., 4500K–5500K matching indoor lights).

### 2.2 Environmental Control
- [ ] **Lighting:** Turn **ON** all room and corridor lights. Eliminate flickering lights. Draw sheer curtains over windows if bright sunlight creates extreme contrast or moving sun patches.
- [ ] **Reflective Surfaces:** 
  - Cover full-length mirrors or large reflective closet doors with non-reflective sheets or tape temporary paper markers/post-its on them.
  - Turn **OFF** televisions, computer monitors, and glowing screens.
  - Close bathroom/reflective glass doors or apply matte tape marks.
- [ ] **Static Scene Guarantee:** Close all exterior doors and windows. Ensure zero drafts (no fluttering curtains). Keep all pets and people completely out of the space. Do not touch or move furniture once capture begins.

---

## 3. Movement & Trajectory Protocol

```
┌──────────────────────────────────────── Corridor ─────────────────────────────────────┐
│  ◄─────────────────────────── [Pass 1 & 2 Return Loop] ──────────────────────────     │
│  ──────────────────────────── [Pass 1 & 2 Forward] ────────────► [Turnaround Arc]    │
└─────────────────────────────────┬─── Doorway ───┬─────────────────────────────────────┘
                                  │   (≥70% ovlp) │
                                  │   ▲ Look-back │
┌─────────────────────────────────┴───   Arc      ┴─────────────────────────────────────┐
│                                                                                       │
│     ┌────────────────────────────────────────────────────────┐                        │
│     │                                                        │                        │
│     │       Pass 1: Eye-level (~1.6m), tilted ~15° down      │                        │
│     │       Pass 2: Chest-level (~1.2m), tilted ~20° down    │       ROOM             │
│     │       Pace: ~0.5 m/s ("Ninja Walk")                    │                        │
│     │                                                        │                        │
│     └──────────────────────── [Corner Sweeps] ───────────────┘                        │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Walking Technique ("The Ninja Walk")
- Hold the phone firmly with both hands, elbows bent and braced against your ribcage.
- Walk with bent knees, rolling heel-to-toe at a steady **$\sim 0.5\text{ m/s}$** pace.
- **The Cardinal Rule:** **NEVER rotate the camera in place.** To turn a corner or change direction, translate your body along a smooth arc while pointing the camera inward toward the scene.

### 3.2 Two-Pass Elevation Strategy
1. **Pass 1 — Eye-Level ($\sim 1.6\text{ m}$):**
   - Walk the perimeter of the room $\sim 1\text{ m}$ away from walls.
   - Point the camera slightly downwards ($\sim 10^\circ - 15^\circ$) so both the horizon/wall features and the upper floor are visible.
2. **Pass 2 — Chest/Waist-Level ($\sim 1.2\text{ m}$):**
   - Follow a complementary inward path at chest height, angled down $\sim 20^\circ$.
   - Captures lower boundaries of furniture (table legs, under-sofa gaps, baseboards) and floor plane constraints.

### 3.3 Corner Sweeps & Featureless Walls
- When reaching a room corner, do not stop and swivel. Walk a small semicircular arc facing the corner, keeping corner seams in view.
- If a wall is plain white/featureless, maintain an oblique angle ($30^\circ - 45^\circ$) so the ceiling cornice, baseboard, doorframes, and floor texture provide SIFT keypoints.

### 3.4 Doorway & Corridor Transition Protocol
Transitions between rooms and corridors are the #1 cause of SfM graph splitting. Execute this exact sequence:
1. Approach the open doorway slowly on the room side.
2. Pass through the doorway with high frame overlap ($\ge 70\%$).
3. **The Look-Back Arc:** Once 1 metre inside the corridor, smoothly sweep in a lateral arc looking backward through the open doorway into the room. This firmly links the room keypoint cluster with the corridor keypoint cluster.
4. Walk down the corridor along one side, loop around at the corridor end in a wide arc, and walk back along the opposite wall.
5. Re-enter the room and face the original starting view.

### 3.5 Loop Closure
Every capture sequence **must finish exactly where it started** in both position and orientation. A closed visual loop allows COLMAP bundle adjustment to distribute drift evenly and avoid "banana" warping. Total video length should be **3 to 8 minutes** (maximum 12 minutes).

---

## 4. Supplementary Detail Stills & Ground-Truth Scale

### 4.1 Supplemental Detail Stills
For tight alcoves, intricate furniture, or shadowed corners where video might underexpose:
- Take 10–20 high-resolution still photos from varying viewpoints.
- Keep the same locked camera settings. Store these in `data/<scene>/stills/`.

### 4.2 Ground-Truth Physical Measurements
SfM and Gaussian Splatting reconstruct scenes up to an arbitrary scale factor. To convert the scene into true metric units (metres, floor at $y=0$):
1. Use a laser measure or standard tape measure.
2. Record:
   - **Dimension 1:** Main room longest wall-to-wall distance (metres).
   - **Dimension 2:** Main room perpendicular wall-to-wall distance (metres).
   - **Dimension 3:** Floor-to-ceiling clear height (metres).
3. Save into `data/<scene>/measurements.json`:

```json
{
  "scene_name": "living_room_corridor",
  "ceiling_height_m": 2.75,
  "measurements": [
    {
      "name": "room_length_north_south",
      "description": "Baseboard to baseboard from north window wall to south corridor wall",
      "metres": 5.42
    },
    {
      "name": "room_width_east_west",
      "description": "Baseboard to baseboard from east bookcase wall to west kitchen wall",
      "metres": 3.86
    }
  ]
}
```

---

## 5. Go / No-Go Validation & Common Failure Modes

Run the validation script immediately after recording:
```bash
python scripts/validate_capture.py --video data/<scene>/input.mp4
```

| Issue / Symptom | Root Cause | Fix / Preventive Action |
|---|---|---|
| **COLMAP fails at doorway (split model)** | Insufficient baseline or zero overlap across door threshold | Re-record with the "look-back arc" protocol; ensure doorway is fully open; walk smoothly through threshold. |
| **Hazy floaters / splat halos** | Auto-Exposure / AWB hunted during capture | Re-record with AE-L and AWB-L locked; ensure lights are not turned on/off mid-capture. |
| **Dropouts / unregistered frames** | Motion blur from fast walking or low shutter speed | Shutter speed must be $\ge 1/120\text{ s}$; walk slower ($\le 0.5\text{ m/s}$); hold phone with two hands braced. |
| **Distorted geometry / curvature** | EIS software stabilization enabled | Disable software stabilization in camera settings; use a gimbal or two-hand ninja walk. |
| **Degenerate reconstruction** | Swivelling / panning in place without translating body | Never rotate in place; always walk along an arc to maintain triangulation baseline. |

---

## 6. One-Page Field Capture Checklist (Printable)

```
================================================================================
                    NAMMA SPACE — ONE-PAGE FIELD CHECKLIST
================================================================================
Date: _______________  Scene Name: ___________________  Operator: ______________

[ ] 1. SPACE PREPARATION
    [ ] All indoor lights turned ON (room + corridor).
    [ ] Windows / blinds drawn if harsh sunlight or shadows.
    [ ] Mirrors and large TV/monitor screens covered or taped with matte markers.
    [ ] All doors along the trajectory opened wide and wedged in place.
    [ ] Pets, family members, and loose moving items cleared from the space.

[ ] 2. CAMERA & DEVICE SETTINGS
    [ ] Clean physical camera lens with microfiber cloth.
    [ ] Selected 1x Main Camera sensor (NO ultra-wide, NO telephoto, NO zoom).
    [ ] Set resolution to 4K (3840x2160) at 30 fps or 60 fps.
    [ ] Software/Electronic Video Stabilisation (EIS) turned OFF.
    [ ] HDR / Dolby Vision / Dynamic Tone Mapping turned OFF (standard SDR).
    [ ] Shutter speed set to >= 1/120 s (manual mode); clean ISO (< 800).
    [ ] Auto-Exposure LOCKED (AE-L) on a representative midtone.
    [ ] Auto-Focus LOCKED (AF-L) at ~2.2 m hyperfocal distance.
    [ ] White Balance LOCKED (AWB-L) to a fixed temperature.

[ ] 3. CAPTURE TRAJECTORY & MOVEMENT
    [ ] Start at marked position A in the room. Note direction.
    [ ] Two-hand grip with elbows tucked into ribcage ("Ninja Walk", ~0.5 m/s).
    [ ] Pass 1: Eye level (~1.6 m), tilted slightly down (~15°).
    [ ] Pass 2: Chest level (~1.2 m), capturing low furniture & floor boundary.
    [ ] Sweeping arcs around all corners (NO pure rotation in place).
    [ ] Doorway: slow transit with >=70% overlap + look-back arc into room.
    [ ] Corridor: walk down one side, turnaround arc at end, return other side.
    [ ] Loop Closure: return to exact starting point A and match start orientation.
    [ ] Total elapsed recording time between 3 and 8 minutes (< 12 min).

[ ] 4. GROUND TRUTH MEASUREMENTS (data/<scene>/measurements.json)
    [ ] Distance 1 (Longest wall-to-wall): ________ metres.
    [ ] Distance 2 (Perpendicular wall):   ________ metres.
    [ ] Ceiling clear height:              ________ metres.

[ ] 5. IMMEDIATE VALIDATION GATE
    [ ] Copy video to data/<scene>/input.mp4.
    [ ] Run: python scripts/validate_capture.py --video data/<scene>/input.mp4
    [ ] Result: [ ] PASS   [ ] WARN (acceptable)   [ ] FAIL (re-record required)
================================================================================
```
