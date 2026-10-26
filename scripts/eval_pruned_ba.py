import pycolmap, json, numpy as np
from collections import defaultdict

rec_orig = pycolmap.Reconstruction('data/home3/colmap/sparse/0')
rec_ba = pycolmap.Reconstruction('data/home3/colmap/sparse_pruned_ba')

with open('data/home/alignment.json') as f:
    align = json.load(f)
metric_scale = align.get('s', 1.0)

orig_image_names = set(img.name.replace('\\', '/') for img in rec_orig.images.values())

# 1. Shifts of 446 originals
orig_shifts = []
matched_orig = 0
for img in rec_orig.images.values():
    name = img.name.replace('\\', '/')
    img_ba = rec_ba.find_image_with_name(name)
    if img_ba and img_ba.has_pose:
        matched_orig += 1
        c_orig = img.projection_center()
        c_ba = img_ba.projection_center()
        shift_m = float(np.linalg.norm(c_orig - c_ba)) * metric_scale
        orig_shifts.append(shift_m)

median_shift_m = float(np.median(orig_shifts)) if orig_shifts else 999.0
max_shift_m = float(np.max(orig_shifts)) if orig_shifts else 999.0

# 2. Model-wide reprojection error
model_mean_reproj = rec_ba.compute_mean_reprojection_error()

# 3. Per-source stats
total_expected = {
    'originals': 450,
    'sweep_1': 43,
    'sweep_2': 41,
    'sweep_3': 38,
    'sweep_4': 52,
    'stills': 11
}

source_reg = defaultdict(int)
source_errors = defaultdict(list)
source_points = defaultdict(set)

for img in rec_ba.images.values():
    if not img.has_pose:
        continue
    name = img.name.replace('\\', '/')
    src = 'originals' if name in orig_image_names or name.startswith('video') else None
    if not src:
        for s in ['sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']:
            if s in name:
                src = s
                break
    if not src:
        src = 'other'

    source_reg[src] += 1
    cam = rec_ba.cameras[img.camera_id]
    rigid = img.cam_from_world()

    for p2d in img.points2D:
        if p2d.has_point3D():
            pid = p2d.point3D_id
            source_points[src].add(pid)
            p3d = rec_ba.points3D[pid]
            p_cam = rigid * p3d.xyz
            p_img = cam.img_from_cam(p_cam)
            if p_img is not None:
                err = np.linalg.norm(p_img - p2d.xy)
                source_errors[src].append(err)

print('================================================================================')
print('  EVALUATION OF sparse_pruned_ba (AFTER PRUNING & BA)')
print('================================================================================')
print(f'Matched original cameras: {matched_orig} / 446')
print(f'Originals median shift   : {median_shift_m:.6f} m (Threshold: < 0.05 m)')
print(f'Originals max shift      : {max_shift_m:.6f} m (Threshold: < 0.25 m)')
print(f'Model-wide mean reproj   : {model_mean_reproj:.4f} px (Threshold: < 1.1 px)')
print('--------------------------------------------------------------------------------')
print(f'{"Source":12s} | {"Reg / Tot":12s} | {"Reg %":7s} | {"3D Points":10s} | {"Mean Reproj":12s} | {"Source Pass?"}')
print('--------------------------------------------------------------------------------')

passing_new_sources = []
for s in ['originals', 'sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']:
    reg_cnt = source_reg[s]
    tot_cnt = total_expected.get(s, 0)
    pct = (reg_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
    pts_cnt = len(source_points[s])
    mean_err = float(np.mean(source_errors[s])) if source_errors[s] else 0.0

    if s == 'originals':
        src_pass = (reg_cnt >= 446 and mean_err < 1.5)
        pass_str = 'PASS' if src_pass else 'FAIL'
    else:
        src_pass = (pct >= 60.0 and mean_err < 1.5)
        pass_str = 'PASS' if src_pass else 'FAIL (<60% or >=1.5px)'
        if src_pass:
            passing_new_sources.append(s)

    print(f'{s:12s} | {reg_cnt:3d} / {tot_cnt:3d}    | {pct:6.2f}% | {pts_cnt:10d} | {mean_err:8.4f} px | {pass_str}')

print('--------------------------------------------------------------------------------')
print(f'Passing new sources count: {len(passing_new_sources)} (need >= 2): {passing_new_sources}')

keep_home3 = (
    matched_orig == 446 and
    median_shift_m < 0.05 and
    max_shift_m < 0.25 and
    model_mean_reproj < 1.1 and
    len(passing_new_sources) >= 2
)

decision = 'home3' if keep_home3 else 'home'
print('================================================================================')
print(f'DECISION: SCENE = {decision}')
if not keep_home3:
    reasons = []
    if model_mean_reproj >= 1.1:
        reasons.append(f'Model-wide reproj error {model_mean_reproj:.4f}px >= 1.1px')
    if len(passing_new_sources) < 2:
        reasons.append(f'Passing new sources count {len(passing_new_sources)} < 2')
    print('Reason: ' + '; '.join(reasons))
print('================================================================================')
