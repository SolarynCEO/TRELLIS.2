import unittest
import warnings

import numpy as np
from PIL import Image

from trellis2.pipelines.rembg.preprocess_utils import frame_and_premultiply_rgba


class TestRembgPreprocessUtils(unittest.TestCase):
    def test_dark_cat_like_low_coverage_mask_falls_back_to_full_frame(self) -> None:
        width, height = 1024, 594
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = 40
        rgba[250:270, 500:520, :3] = 220
        rgba[250:270, 500:520, 3] = 255
        image = Image.fromarray(rgba, mode="RGBA")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            framed, meta = frame_and_premultiply_rgba(
                image,
                warning_context="test-dark-cat",
                min_mask_coverage=0.02,
                min_crop_size_px=128,
            )

        self.assertTrue(meta["fallback_used"])
        self.assertIn("mask coverage", meta["fallback_reason"])
        self.assertEqual(framed.size, (width, height))
        self.assertTrue(any("mask coverage" in str(w.message) for w in caught))

    def test_tiny_crop_size_falls_back_to_full_frame(self) -> None:
        width, height = 400, 400
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = 180
        rgba[195:205, 195:205, :3] = 255
        rgba[195:205, 195:205, 3] = 255
        image = Image.fromarray(rgba, mode="RGBA")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            framed, meta = frame_and_premultiply_rgba(
                image,
                warning_context="test-crop-collapse",
                min_mask_coverage=0.0,
                min_crop_size_px=256,
            )

        self.assertTrue(meta["fallback_used"])
        self.assertIn("crop size", meta["fallback_reason"])
        self.assertEqual(framed.size, (width, height))
        self.assertTrue(any("crop size" in str(w.message) for w in caught))

    def test_large_mask_keeps_center_crop(self) -> None:
        width, height = 300, 200
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = 255
        rgba[20:180, 40:240, 3] = 255
        image = Image.fromarray(rgba, mode="RGBA")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            framed, meta = frame_and_premultiply_rgba(
                image,
                warning_context="test-normal",
                min_mask_coverage=0.01,
                min_crop_size_px=64,
            )

        self.assertFalse(meta["fallback_used"])
        self.assertEqual(framed.size, (198, 198))
        self.assertEqual(len(caught), 0)


if __name__ == "__main__":
    unittest.main()
