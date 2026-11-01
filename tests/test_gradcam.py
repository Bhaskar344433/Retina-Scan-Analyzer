"""Unit tests for Grad-CAM layer identification, hook registration, and heatmap generation."""

import unittest
import torch
import numpy as np
from src.models.cnn import create_cnn_model
from src.visualization.interpretability import (
    GradCAM,
    compute_grad_cam,
    visualize_grad_cam,
    get_model_target_layer,
)


class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.resnet = create_cnn_model("resnet50", num_classes=5, pretrained=False)
        self.densenet = create_cnn_model("densenet121", num_classes=5, pretrained=False)

    def test_target_layer_detection(self):
        res_layer = get_model_target_layer(self.resnet, "resnet50")
        self.assertIn("layer4", res_layer)

        dense_layer = get_model_target_layer(self.densenet, "densenet121")
        self.assertTrue("denseblock4" in dense_layer or "norm5" in dense_layer)
        # Ensure it does NOT pick denseblock1
        self.assertNotIn("denseblock1", dense_layer)

    def test_compute_gradcam_resnet(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        cam = compute_grad_cam(self.resnet, dummy_input)

        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(len(cam.shape), 2)
        self.assertTrue(0.0 <= cam.min() <= cam.max() <= 1.0)

    def test_compute_gradcam_densenet(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        cam = compute_grad_cam(self.densenet, dummy_input)

        self.assertIsInstance(cam, np.ndarray)
        self.assertEqual(len(cam.shape), 2)
        self.assertTrue(0.0 <= cam.min() <= cam.max() <= 1.0)

    def test_visualize_gradcam_overlay(self):
        synthetic_img = np.full((224, 224, 3), 100, dtype=np.uint8)
        synthetic_cam = np.linspace(0, 1, 224 * 224).reshape(224, 224)

        overlay = visualize_grad_cam(synthetic_img, synthetic_cam, alpha=0.5)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_invalid_target_layer_raises_error(self):
        with self.assertRaises(ValueError):
            GradCAM(self.resnet, target_layer="non_existent_conv_layer_999")


if __name__ == '__main__':
    unittest.main()
