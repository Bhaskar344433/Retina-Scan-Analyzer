"""Unit tests for classification and medical grading evaluation metrics."""

import unittest
import numpy as np
from src.utils.metrics import compute_classification_metrics


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        y_prob = np.eye(5)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 1.0)
        self.assertAlmostEqual(metrics['quadratic_weighted_kappa'], 1.0)
        self.assertAlmostEqual(metrics['f1_macro'], 1.0)
        self.assertAlmostEqual(metrics['precision_macro'], 1.0)
        self.assertAlmostEqual(metrics['recall_macro'], 1.0)

    def test_imbalanced_predictions_and_qwk(self):
        # 10 samples with ordinal off-by-one errors
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertTrue(0.0 <= metrics['accuracy'] <= 1.0)
        self.assertTrue(0.0 <= metrics['balanced_accuracy'] <= 1.0)
        self.assertTrue(-1.0 <= metrics['quadratic_weighted_kappa'] <= 1.0)
        self.assertEqual(len(metrics['confusion_matrix']), 5)

    def test_majority_class_baseline_behavior(self):
        # Simulate predicting majority class (Grade 0) on imbalanced distribution
        y_true = np.array([0, 0, 0, 0, 1, 2, 3, 4])  # 4 out of 8 are class 0 (50% accuracy)
        y_pred = np.zeros(8, dtype=int)

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics['accuracy'], 0.5)
        # Balanced accuracy should be 1/5 = 0.2 because only class 0 has recall 1.0, classes 1-4 have recall 0.0
        self.assertAlmostEqual(metrics['balanced_accuracy'], 0.2)
        # QWK should be near 0 for degenerate constant prediction
        self.assertAlmostEqual(metrics['quadratic_weighted_kappa'], 0.0)

    def test_support_counts(self):
        y_true = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4])
        y_pred = np.array([0, 1, 1, 2, 2, 3, 3, 4, 0])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertEqual(metrics['support_class_0'], 2)
        self.assertEqual(metrics['support_class_1'], 1)
        self.assertEqual(metrics['support_class_2'], 3)
        self.assertEqual(metrics['support_class_3'], 1)
        self.assertEqual(metrics['support_class_4'], 2)


if __name__ == '__main__':
    unittest.main()
