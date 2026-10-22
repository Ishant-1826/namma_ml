import sys
from pathlib import Path
import tempfile
import unittest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from pipeline.extract_frames import (
    compute_blur_score,
    filter_blur_sliding_window,
    is_near_duplicate,
    resize_longest_edge,
    build_contact_sheet,
)


class TestExtractFrames(unittest.TestCase):
    """Test suite for frame extraction algorithms using synthetic images."""

    def setUp(self):
        # Create a sharp synthetic test image (high-contrast grid)
        self.sharp_img = np.zeros((360, 640, 3), dtype=np.uint8)
        for y in range(0, 360, 20):
            cv2.line(self.sharp_img, (0, y), (639, y), (255, 255, 255), 2)
        for x in range(0, 640, 20):
            cv2.line(self.sharp_img, (x, 0), (x, 359), (255, 255, 255), 2)

        # Mildly blurred copy
        self.medium_img = cv2.GaussianBlur(self.sharp_img, (7, 7), 0)

        # Heavily blurred copy
        self.heavy_blur_img = cv2.GaussianBlur(self.sharp_img, (25, 25), 0)

        # Flat featureless image
        self.flat_img = np.full((360, 640, 3), 128, dtype=np.uint8)

    def test_compute_blur_score_ranking(self):
        """Variance of Laplacian must rank sharp > medium > heavy > flat."""
        s_score = compute_blur_score(self.sharp_img)
        m_score = compute_blur_score(self.medium_img)
        h_score = compute_blur_score(self.heavy_blur_img)
        f_score = compute_blur_score(self.flat_img)

        self.assertGreater(s_score, m_score)
        self.assertGreater(m_score, h_score)
        self.assertGreater(h_score, f_score)
        self.assertEqual(f_score, 0.0)

    def test_filter_blur_sliding_window_selection(self):
        """Window of N=3 must pick the sharpest candidate in each window."""
        # 6 frames: [Window 1: 10, 200, 50], [Window 2: 180, 20, 15]
        scores = [10.0, 200.0, 50.0, 180.0, 20.0, 15.0]

        # Window 1 sharpest is index 1 (score 200)
        # Window 2 sharpest is index 3 (score 180)
        kept, thresh, dropped = filter_blur_sliding_window(
            scores, window_size=3, percentile_threshold=15.0
        )

        self.assertIn(1, kept)
        self.assertIn(3, kept)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)

    def test_filter_blur_percentile_dropout(self):
        """Windows whose best frame falls below the 15th percentile must be dropped."""
        # 20 frames: 17 sharp frames (score 100-150) and 3 blurry frames (score 2-5)
        scores = [120.0 + i for i in range(17)] + [4.0, 2.0, 5.0]
        # The 15th percentile will be around ~120, so the degraded window [17, 18, 19]
        # with best score 5.0 is well below the 15th percentile and must be dropped.

        kept, thresh, dropped = filter_blur_sliding_window(
            scores, window_size=3, percentile_threshold=15.0
        )

        # The last window (indices 18, 19) or [18, 19] must be dropped
        last_indices = {17, 18, 19}
        self.assertTrue(last_indices.isdisjoint(kept))
        self.assertGreater(dropped, 0)

    def test_near_duplicate_filter(self):
        """Identical or near-identical frames must be flagged as duplicates."""
        # Identical image -> near duplicate
        self.assertTrue(is_near_duplicate(self.sharp_img, self.sharp_img))

        # Add tiny imperceptible noise
        noisy = self.sharp_img.copy()
        noise = np.random.randint(-2, 3, noisy.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        self.assertTrue(is_near_duplicate(self.sharp_img, noisy, ssim_threshold=0.98))

        # Inverted image -> NOT duplicate
        inverted = 255 - self.sharp_img
        self.assertFalse(is_near_duplicate(self.sharp_img, inverted, ssim_threshold=0.98))

    def test_resize_longest_edge(self):
        """Images larger than max_dim must be resized; smaller ones left intact."""
        # Landscape 3840x2160 -> 1920x1080
        big_landscape = np.zeros((2160, 3840, 3), dtype=np.uint8)
        resized_l = resize_longest_edge(big_landscape, max_dim=1920)
        self.assertEqual(resized_l.shape[:2], (1080, 1920))

        # Portrait 2160x3840 -> 1080x1920
        big_portrait = np.zeros((3840, 2160, 3), dtype=np.uint8)
        resized_p = resize_longest_edge(big_portrait, max_dim=1920)
        self.assertEqual(resized_p.shape[:2], (1920, 1080))

        # Small image (640x360) -> unchanged
        resized_s = resize_longest_edge(self.sharp_img, max_dim=1920)
        self.assertEqual(resized_s.shape[:2], (360, 640))

    def test_contact_sheet_dimensions(self):
        """Contact sheet generator must produce image with correct dimensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sample_paths = []
            for i in range(12):
                p = tmp_path / f"img_{i:03d}.jpg"
                cv2.imwrite(str(p), self.sharp_img)
                sample_paths.append(p)

            sheet_out = tmp_path / "contact_sheet.jpg"
            cols = 8
            rows = 6
            cell_w = 240
            cell_h = 135

            build_contact_sheet(
                sample_paths,
                sheet_out,
                cols=cols,
                rows=rows,
                cell_w=cell_w,
                cell_h=cell_h,
            )

            self.assertTrue(sheet_out.is_file())
            sheet = cv2.imread(str(sheet_out))
            self.assertIsNotNone(sheet)
            self.assertEqual(sheet.shape[1], cols * cell_w)  # 1920
            self.assertEqual(sheet.shape[0], rows * cell_h)  # 810


if __name__ == "__main__":
    unittest.main()
