"""Unit tests for checkpoint verification and model loading."""

import unittest
from pathlib import Path
import torch
from src.models.cnn import create_cnn_model


class TestCheckpointLoading(unittest.TestCase):
    def setUp(self):
        self.results_dir = Path("experiments/results_real")

    def test_resnet50_checkpoint(self):
        ckpt_path = self.results_dir / "resnet50" / "model_best.pth"
        if not ckpt_path.is_file():
            self.skipTest("ResNet50 checkpoint not found.")

        model = create_cnn_model("resnet50", num_classes=5, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertIn("model_state_dict", checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Run single forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (1, 5))

    def test_densenet121_checkpoint(self):
        ckpt_path = self.results_dir / "densenet121" / "model_best.pth"
        if not ckpt_path.is_file():
            self.skipTest("DenseNet121 checkpoint not found.")

        model = create_cnn_model("densenet121", num_classes=5, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertIn("model_state_dict", checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
