#!/usr/bin/env python3
"""Generate pairs for home3 matching.

Pair criteria:
  (i) Every pair among the NEW images: C(N_new, 2)
  (ii) Each NEW image against every 3rd OLD image from the existing database.

Output format for COLMAP matches_importer:
  image_name1 image_name2
"""

import argparse
import itertools
import sqlite3
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate image pairs for home3 matching.")
    parser.add_argument(
        "--database",
        type=str,
        default="data/home3/colmap/database.db",
        help="Path to COLMAP database.db containing old images.",
    )
    parser.add_argument(
        "--new-images",
        type=str,
        default="data/home3/new_images.txt",
        help="Path to new_images.txt listing relative paths of new images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/home3/pairs.txt",
        help="Path to output pairs.txt file.",
    )
    parser.add_argument(
        "--step-old",
        type=int,
        default=3,
        help="Subsample step for old images (default: 3, every 3rd image).",
    )
    args = parser.parse_args()

    new_images_path = Path(args.new_images)
    if not new_images_path.is_file():
        print(f"[ERROR] New images file not found: {new_images_path}", file=sys.stderr)
        sys.exit(1)

    with open(new_images_path, "r", encoding="utf-8") as f:
        new_images = [line.strip().replace("\\", "/") for line in f if line.strip()]

    num_new = len(new_images)
    if num_new == 0:
        print(f"[ERROR] No new images found in {new_images_path}", file=sys.stderr)
        sys.exit(1)

    # Read old images from database
    db_path = Path(args.database)
    if not db_path.is_file():
        print(f"[ERROR] Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # Get all images currently in DB that are not in new_images
    cur.execute("SELECT name FROM images ORDER BY image_id")
    db_imgs = [row[0].replace("\\", "/") for row in cur.fetchall()]
    conn.close()

    new_images_set = set(new_images)
    old_images = [img for img in db_imgs if img not in new_images_set]

    # Subsample every 3rd old image
    step = max(1, args.step_old)
    sampled_old = old_images[::step]

    # 1. Pairs among new images
    new_pairs = list(itertools.combinations(new_images, 2))

    # 2. Cross pairs: each new image against each sampled old image
    cross_pairs = []
    for new_img in new_images:
        for old_img in sampled_old:
            cross_pairs.append((new_img, old_img))

    total_pairs = len(new_pairs) + len(cross_pairs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for im1, im2 in new_pairs:
            f.write(f"{im1} {im2}\n")
        for im1, im2 in cross_pairs:
            f.write(f"{im1} {im2}\n")

    print("==================================================")
    print("  home3 Matching Pair Generator")
    print("==================================================")
    print(f"  New images count               : {num_new}")
    print(f"  Old images in database         : {len(old_images)}")
    print(f"  Old images sampled (every {step}rd) : {len(sampled_old)}")
    print(f"  (i)  Pairs among new images    : {len(new_pairs):,}")
    print(f"  (ii) Pairs new vs sampled old  : {len(cross_pairs):,}")
    print(f"  TOTAL PAIRS TO MATCH           : {total_pairs:,}")
    print(f"  Output saved to                : {out_path.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
