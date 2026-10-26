import os
import pycolmap
import numpy as np
from collections import defaultdict

rec = pycolmap.Reconstruction('data/home3/colmap/sparse_ext')

source_names = ['originals', 'sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']
total_expected = {
    'originals': 450,
    'sweep_1': 43,
    'sweep_2': 41,
    'sweep_3': 38,
    'sweep_4': 52,
    'stills': 11
}

def get_source(name):
    name = name.replace('\\', '/')
    if name.startswith('video1/') or name.startswith('video2/') or name.startswith('frame_'):
        return 'originals'
    for s in ['sweep_1', 'sweep_2', 'sweep_3', 'sweep_4', 'stills']:
        if s in name:
            return s
    return 'other'

source_images = defaultdict(list)
for img in rec.images.values():
    if img.has_pose:
        src = get_source(img.name)
        source_images[src].append(img)

# Compute per-source points and reprojection error
source_points = defaultdict(set)
source_errors = defaultdict(list)
image_errors = {}
image_obs_count = {}

for img in rec.images.values():
    if not img.has_pose:
        continue
    src = get_source(img.name)
    cam = rec.cameras[img.camera_id]
    rigid = img.cam_from_world()
    
    img_errs = []
    for p2d in img.points2D:
        if p2d.has_point3D():
            pid = p2d.point3D_id
            p3d = rec.points3D[pid]
            source_points[src].add(pid)
            
            p_cam = rigid * p3d.xyz
            p_img = cam.img_from_cam(p_cam)
            if p_img is not None:
                err = np.linalg.norm(p_img - p2d.xy)
                img_errs.append(err)
                source_errors[src].append(err)
                
    image_errors[img.image_id] = float(np.mean(img_errs)) if img_errs else 0.0
    image_obs_count[img.image_id] = len(img_errs)

print("=" * 80)
print("  REGISTRATION PER SOURCE (sparse_ext)")
print("=" * 80)
print(f"{'Source':12s} | {'Reg / Total':12s} | {'Reg %':7s} | {'3D Points':10s} | {'Mean Reproj Error':18s}")
print("-" * 80)

for s in source_names:
    reg_cnt = len(source_images[s])
    tot_cnt = total_expected.get(s, 0)
    pct = (reg_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
    pts_cnt = len(source_points[s])
    mean_err = np.mean(source_errors[s]) if source_errors[s] else 0.0
    print(f"{s:12s} | {reg_cnt:3d} / {tot_cnt:3d}    | {pct:6.2f}% | {pts_cnt:10d} | {mean_err:8.4f} px")

print("=" * 80)
model_mean_err = rec.compute_mean_reprojection_error()
print(f"Model-Wide Mean Reprojection Error: {model_mean_err:.4f} px")
print(f"Total Unique 3D Points: {rec.num_points3D():,}")
print("=" * 80)

print("\n" + "=" * 80)
print("  CAMERA INTRINSICS (7 CAMERAS, 4 DECIMALS)")
print("=" * 80)
print(f"{'Cam ID':6s} | {'Source':10s} | {'Model':13s} | {'fx':10s} | {'cx':10s} | {'cy':10s} | {'k1':8s} | {'Refined from 1440 Prior?'}")
print("-" * 80)

cam_source_map = {
    1: 'video1',
    2: 'video2',
    3: 'sweep_1',
    4: 'sweep_2',
    5: 'sweep_3',
    6: 'sweep_4',
    7: 'stills'
}

for cam_id in sorted(rec.cameras.keys()):
    cam = rec.cameras[cam_id]
    p = cam.params
    fx = p[0]
    cx = p[1]
    cy = p[2]
    k1 = p[3]
    src = cam_source_map.get(cam_id, 'unknown')
    
    if cam_id in [1, 2]:
        status = "Original (1477.94 / 1476.14)"
    else:
        diff = abs(fx - 1440.0)
        if diff > 1.0:
            status = f"REFINED (delta: {fx - 1440.0:+.4f} px)"
        else:
            status = "AT PRIOR (1440.0000)"
            
    print(f"{cam_id:6d} | {src:10s} | {cam.model_name:13s} | {fx:10.4f} | {cx:10.4f} | {cy:10.4f} | {k1:8.4f} | {status}")

print("=" * 80)
