import os
import cv2
import numpy as np
from pathlib import Path

images_dir = Path("data/home/images")
all_frames = sorted(images_dir.glob("frame_*.jpg"))

v1_frames = [f for f in all_frames if int(f.stem.split("_")[1]) <= 222]
v2_frames = [f for f in all_frames if int(f.stem.split("_")[1]) > 222]

def make_sample_sheet(frames, output_path, title):
    # Select 9 evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int)
    selected = [frames[i] for i in indices]
    
    thumb_w, thumb_h = 640, 360
    grid = np.zeros((thumb_h * 3, thumb_w * 3, 3), dtype=np.uint8)
    
    for idx, fpath in enumerate(selected):
        img = cv2.imread(str(fpath))
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        
        # Add label
        label = fpath.name
        cv2.putText(thumb, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(thumb, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        row = idx // 3
        col = idx % 3
        grid[row*thumb_h:(row+1)*thumb_h, col*thumb_w:(col+1)*thumb_w] = thumb
        
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved {output_path} (size: {grid.shape[1]}x{grid.shape[0]})")

out_v1 = Path("docs/frame_sample_v1.jpg")
out_v2 = Path("docs/frame_sample_v2.jpg")

make_sample_sheet(v1_frames, out_v1, "Video 1")
make_sample_sheet(v2_frames, out_v2, "Video 2")
