"""Unit tests for dataset split counts, labels, and wrapper integrity."""

import unittest
from pathlib import Path
import numpy as np
import torch
from experiments.real_data_experiment import RetinaMNISTDataset


class TestDatasetIntegrity(unittest.TestCase):
    def setUp(self):
        self.npz_path = Path("data/medmnist/retinamnist.npz")

    def test_local_retinamnist_archive(self):
        if not self.npz_path.is_file():
            self.skipTest("RetinaMNIST archive not found locally.")

        data = np.load(self.npz_path)

        # Check required keys
        expected_keys = {"train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"}
        self.assertTrue(expected_keys.issubset(set(data.keys())))

        # Verify exact counts per MedMNIST v2 specifications
        # Train: 1,080 | Val: 120 | Test: 400
        self.assertEqual(data["train_images"].shape, (1080, 28, 28, 3))
        self.assertEqual(data["train_labels"].shape, (1080, 1))
        self.assertEqual(data["val_images"].shape, (120, 28, 28, 3))
        self.assertEqual(data["val_labels"].shape, (120, 1))
        self.assertEqual(data["test_images"].shape, (400, 28, 28, 3))
        self.assertEqual(data["test_labels"].shape, (400, 1))

        # Check label range: integer values in {0, 1, 2, 3, 4}
        all_labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]])
        unique_labels = set(np.unique(all_labels))
        self.assertEqual(unique_labels, {0, 1, 2, 3, 4})

    def test_wrapper_resizing(self):
        # Create a synthetic mini dataset to test RetinaMNISTDataset wrapper
        synthetic_samples = [
            (Image_np, np.array([idx % 5]))
            for idx, Image_np in enumerate([np.full((28, 28, 3), 128, dtype=np.uint8) for _ in range(5)])
        ]

        class DummyMedMNIST:
            def __len__(self):
                return len(synthetic_samples)
            def __getitem__(self, idx):
                return synthetic_samples[idx]

        wrapper = RetinaMNISTDataset(DummyMedMNIST(), resize_to=224)
        self.assertEqual(len(wrapper), 5)

        img_tensor, label_tensor = wrapper[0]
        self.assertEqual(img_tensor.shape, (3, 224, 224))
        self.assertEqual(label_tensor.dtype, torch.long)


if __name__ == '__main__':
    unittest.main()
