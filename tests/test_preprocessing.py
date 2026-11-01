"""Unit tests for image loading, validation, and preprocessing consistency."""

import unittest
import numpy as np
import torch
from PIL import Image
from src.utils.image_processing import prepare_image


class TestPreprocessing(unittest.TestCase):
    def test_rgb_image_preprocessing(self):
        # Create a synthetic 100x150 RGB image
        img = Image.new("RGB", (100, 150), color=(120, 80, 40))
        display_img, tensor = prepare_image(img)

        # Check display image format
        self.assertEqual(display_img.mode, "RGB")

        # Check tensor shape: (1, 3, 224, 224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

        # Normalization check: should not be strictly in [0, 1] due to ImageNet mean/std
        # ImageNet mean is around 0.45, std around 0.22, so pixel values should fall roughly in [-2.5, 2.5]
        self.assertTrue(tensor.min() > -3.0)
        self.assertTrue(tensor.max() < 3.0)

    def test_grayscale_conversion_to_rgb(self):
        # Grayscale image (single channel 'L')
        gray_img = Image.new("L", (80, 80), color=128)
        display_img, tensor = prepare_image(gray_img)

        self.assertEqual(display_img.mode, "RGB")
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_rgba_conversion_to_rgb(self):
        # RGBA image with alpha channel
        rgba_img = Image.new("RGBA", (90, 90), color=(100, 150, 200, 255))
        display_img, tensor = prepare_image(rgba_img)

        self.assertEqual(display_img.mode, "RGB")
        self.assertEqual(tensor.shape, (1, 3, 224, 224))


if __name__ == '__main__':
    unittest.main()
