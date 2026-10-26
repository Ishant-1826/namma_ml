import cv2
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
home3_img = repo_root / 'data' / 'home3' / 'images'
docs_dir = repo_root / 'docs'
docs_dir.mkdir(exist_ok=True)

# 1. Contact sheet for stills (11 images -> 4 cols x 3 rows)
stills = sorted(list((home3_img / 'stills').glob('*.jpg')))
thumb_w, thumb_h = 480, 270
grid_cols, grid_rows = 4, 3
stills_sheet = np.zeros((grid_rows * thumb_h, grid_cols * thumb_w, 3), dtype=np.uint8)

for idx, p in enumerate(stills):
    r = idx // grid_cols
    c = idx % grid_cols
    im = cv2.imread(str(p))
    thumb = cv2.resize(im, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    cv2.putText(thumb, f'still_{idx+1}: {p.name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    stills_sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

cv2.imwrite(str(docs_dir / 'contact_stills.jpg'), stills_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('Wrote docs/contact_stills.jpg')

# 2. Contact sheet for sweeps (4 sweeps x 6 samples each -> 6 cols x 4 rows)
sweep_folders = ['sweep_1', 'sweep_2', 'sweep_3', 'sweep_4']
sweeps_sheet = np.zeros((4 * thumb_h, 6 * thumb_w, 3), dtype=np.uint8)

for r, sf in enumerate(sweep_folders):
    s_imgs = sorted(list((home3_img / sf).glob('*.jpg')))
    sample_indices = np.linspace(0, len(s_imgs) - 1, 6, dtype=int)
    for c, idx in enumerate(sample_indices):
        p = s_imgs[idx]
        im = cv2.imread(str(p))
        thumb = cv2.resize(im, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f'{sf} #{idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        sweeps_sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

cv2.imwrite(str(docs_dir / 'contact_sweeps.jpg'), sweeps_sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print('Wrote docs/contact_sweeps.jpg')

# 3. Drop 10% blurriest frames per sweep folder
drop_summary = {}
for sf in sweep_folders:
    s_imgs = sorted(list((home3_img / sf).glob('*.jpg')))
    blur_scores = []
    for p in s_imgs:
        im = cv2.imread(str(p))
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_scores.append((score, p))
    
    # Sort by blur score ascending (lowest variance = blurriest)
    blur_scores.sort(key=lambda x: x[0])
    num_to_drop = int(round(len(s_imgs) * 0.10))
    dropped = blur_scores[:num_to_drop]
    kept = blur_scores[num_to_drop:]
    
    for score, p in dropped:
        p.unlink()
        
    drop_summary[sf] = {
        'initial': len(s_imgs),
        'dropped': num_to_drop,
        'remaining': len(kept),
        'dropped_scores': [round(s, 2) for s, _ in dropped],
        'min_kept_score': round(kept[0][0], 2) if kept else 0
    }

print('\n=== Blur Filtering Summary ===')
for sf, s in drop_summary.items():
    print(f"{sf}: started with {s['initial']}, dropped {s['dropped']} blurriest (scores: {s['dropped_scores']}), {s['remaining']} remaining (min kept score: {s['min_kept_score']})")

# 4. Write data/home3/new_images.txt
all_new_images = []
for folder in sweep_folders + ['stills']:
    p_folder = home3_img / folder
    remaining = sorted(list(p_folder.glob('*.jpg')))
    for p in remaining:
        rel_path = f"{folder}/{p.name}"
        all_new_images.append(rel_path)

new_images_txt = repo_root / 'data' / 'home3' / 'new_images.txt'
with open(new_images_txt, 'w', encoding='utf-8') as f:
    for line in all_new_images:
        f.write(line + '\n')

print(f'\nWrote {len(all_new_images)} new images to {new_images_txt}')
