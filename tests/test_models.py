"""Unit tests for CNN models and parameter configurations."""

import unittest
import torch
from src.models.cnn import create_cnn_model, CNNBackbone


class TestCNNModels(unittest.TestCase):
    def test_resnet50_instantiation_and_forward(self):
        model = create_cnn_model("resnet50", num_classes=5, pretrained=False)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.backbone_name, "resnet50")

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5))

    def test_densenet121_instantiation_and_forward(self):
        model = create_cnn_model("densenet121", num_classes=5, pretrained=False)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.backbone_name, "densenet121")

        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5))

    def test_vgg16_instantiation_and_forward(self):
        model = create_cnn_model("vgg16", num_classes=5, pretrained=False)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.backbone_name, "vgg16")

        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))

    def test_invalid_backbone_raises_error(self):
        with self.assertRaises(ValueError):
            create_cnn_model("nonexistent_arch", num_classes=5, pretrained=False)


if __name__ == '__main__':
    unittest.main()
