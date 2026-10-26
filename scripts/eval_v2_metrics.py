import os
import sys
import re
import json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips

def main():
    v2_dir = Path("outputs/home/v2")
    train_pack = Path("outputs/home/train_pack")
    eval_dirs = sorted(v2_dir.glob("eval_*"), key=lambda d: int(re.search(r"\d+", d.name).group()) if re.search(r"\d+", d.name) else 0, reverse=True)
    if not eval_dirs:
        print("No eval directories found!")
        sys.exit(1)
        
    latest_eval_dir = eval_dirs[0]
    print(f"Using eval directory: {latest_eval_dir}")
    eval_files = sorted(list(latest_eval_dir.glob("*.png")) + list(latest_eval_dir.glob("*.jpg")))
    print(f"Found {len(eval_files)} eval render files.")

    with open(train_pack / "transforms.json", "r", encoding="utf-8") as f:
        tf_data = json.load(f)
    frames = tf_data.get("frames", [])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LPIPS model on {device}...")
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    lpips_model.eval()

    psnr_list, ssim_list, lpips_list = [], [], []

    for eval_file in eval_files:
        nums = re.findall(r"\d+", eval_file.stem)
        if not nums:
            continue
        fnum = int(nums[-1])
        # Find corresponding frame in transforms.json
        matched_gt = None
        for frame in frames:
            if f"{fnum:04d}" in frame["file_path"] or f"{fnum:06d}" in frame["file_path"]:
                gt_path = train_pack / frame["file_path"]
                if gt_path.is_file():
                    matched_gt = gt_path
                    break
        if matched_gt is None:
            continue

        pred_img = np.array(Image.open(eval_file).convert("RGB"))
        gt_img = np.array(Image.open(matched_gt).convert("RGB"))
        if gt_img.shape != pred_img.shape:
            pred_img_pil = Image.fromarray(pred_img).resize((gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
            pred_img = np.array(pred_img_pil)

        cur_psnr = peak_signal_noise_ratio(gt_img, pred_img)
        cur_ssim = structural_similarity(gt_img, pred_img, channel_axis=2)
        
        t_pred = torch.from_numpy(pred_img).permute(2, 0, 1).float().to(device) / 127.5 - 1.0
        t_gt = torch.from_numpy(gt_img).permute(2, 0, 1).float().to(device) / 127.5 - 1.0
        with torch.no_grad():
            cur_lpips = lpips_model(t_pred.unsqueeze(0), t_gt.unsqueeze(0)).item()

        psnr_list.append(cur_psnr)
        ssim_list.append(cur_ssim)
        lpips_list.append(cur_lpips)

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    print("=" * 60)
    print("  EVALUATION METRICS (final_v2 on SCENE=home)")
    print("=" * 60)
    print(f"Evaluated Frames : {len(psnr_list)}")
    print(f"Mean PSNR        : {mean_psnr:.2f} dB (Previous baseline: 23.97 dB)")
    print(f"Mean SSIM        : {mean_ssim:.4f} (Previous baseline: 0.8903)")
    print(f"Mean LPIPS       : {mean_lpips:.4f} (Previous baseline: 0.2698)")
    print("=" * 60)

    metrics = {
        "scene": "home",
        "model": "final_v2",
        "steps": 30000,
        "frames_evaluated": len(psnr_list),
        "psnr": round(mean_psnr, 2),
        "ssim": round(mean_ssim, 4),
        "lpips": round(mean_lpips, 4),
        "baseline_comparison": {
            "baseline_psnr": 23.97,
            "baseline_ssim": 0.8903,
            "baseline_lpips": 0.2698,
            "psnr_delta": round(mean_psnr - 23.97, 2),
            "ssim_delta": round(mean_ssim - 0.8903, 4),
            "lpips_delta": round(mean_lpips - 0.2698, 4)
        }
    }

    metrics_file = v2_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
