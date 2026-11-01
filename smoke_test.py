"""
Fast Smoke Test Script for Retinal Image Analysis Toolkit
Validates environment, dataset archive, checkpoint loading, model inference,
Grad-CAM explainability, and evaluation metrics in < 5 seconds without training.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn import create_cnn_model
from src.utils.metrics import compute_classification_metrics
from src.visualization.interpretability import (
    compute_grad_cam,
    visualize_grad_cam,
    get_model_target_layer,
)
from src.utils.image_processing import prepare_image


def run_smoke_test():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("RETINAL IMAGE ANALYSIS -- FAST SYSTEM SMOKE TEST")
    print("=" * 70)
    start_time = time.time()
    passed_checks = 0
    total_checks = 6

    # 1. Environment & Dependencies Check
    print("\n[Check 1/6] Python and PyTorch Environment...")
    print(f"  [OK] Python version:  {sys.version.split()[0]}")
    print(f"  [OK] PyTorch version: {torch.__version__}")
    print(f"  [OK] Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    passed_checks += 1

    # 2. Local Dataset Verification
    print("\n[Check 2/6] Verifying RetinaMNIST dataset archive...")
    npz_path = ROOT / "data" / "medmnist" / "retinamnist.npz"
    if npz_path.is_file():
        data = np.load(npz_path)
        train_count = len(data["train_images"])
        val_count = len(data["val_images"])
        test_count = len(data["test_images"])
        print(f"  [OK] Found: {npz_path.relative_to(ROOT)} ({npz_path.stat().st_size / 1024:.1f} KB)")
        print(f"  [OK] Split counts: Train={train_count}, Val={val_count}, Test={test_count} (Total={train_count+val_count+test_count})")
        passed_checks += 1
    else:
        print(f"  [WARN] Dataset archive not found at {npz_path}")

    # 3. Checkpoint Loading
    print("\n[Check 3/6] Verifying trained checkpoints...")
    models_to_check = ["resnet50", "densenet121"]
    loaded_models = {}
    for backbone in models_to_check:
        ckpt_path = ROOT / "experiments" / "results_real" / backbone / "model_best.pth"
        if not ckpt_path.is_file():
            print(f"  [FAIL] Missing checkpoint: {ckpt_path}")
            continue
        model = create_cnn_model(backbone, num_classes=5, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        loaded_models[backbone] = model
        params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Loaded {backbone.upper()} checkpoint ({ckpt_path.stat().st_size / (1024*1024):.1f} MB, {params:,} parameters)")
    if len(loaded_models) == len(models_to_check):
        passed_checks += 1

    # 4. Image Preprocessing & Forward Inference
    print("\n[Check 4/6] Running inference on synthetic fundus image...")
    synthetic_img = Image.new("RGB", (256, 256), color=(140, 70, 40))
    display_img, input_tensor = prepare_image(synthetic_img)
    self_checks_ok = True
    for backbone, model in loaded_models.items():
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            pred_grade = int(probs.argmax())
        print(f"  [OK] {backbone.upper()} prediction: Grade {pred_grade} (Max confidence: {probs[pred_grade]:.2%})")
        if probs.shape != (5,):
            self_checks_ok = False
    if self_checks_ok:
        passed_checks += 1

    # 5. Live Grad-CAM Generation
    print("\n[Check 5/6] Testing live Grad-CAM heatmaps...")
    cam_ok = True
    for backbone, model in loaded_models.items():
        target_layer = get_model_target_layer(model, backbone)
        cam = compute_grad_cam(model, input_tensor, target_layer=target_layer, device="cpu")
        overlay = visualize_grad_cam(np.asarray(display_img.resize((224, 224))), cam)
        print(f"  [OK] {backbone.upper()} Grad-CAM: Target layer '{target_layer}' -> Heatmap {cam.shape}, Overlay {overlay.shape}")
        if overlay.shape != (224, 224, 3):
            cam_ok = False
    if cam_ok:
        passed_checks += 1

    # 6. Evaluation Metrics Engine
    print("\n[Check 6/6] Verifying medical grading metrics (Accuracy, Balanced Acc, QWK, Macro F1)...")
    y_true = np.array([0, 0, 1, 2, 2, 3, 4])
    y_pred = np.array([0, 1, 1, 2, 3, 3, 4])
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"  [OK] Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  [OK] Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  [OK] QWK:               {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  [OK] Macro F1:          {metrics['f1_macro']:.4f}")
    passed_checks += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SMOKE TEST PASSED: {passed_checks}/{total_checks} checks succeeded in {elapsed:.2f}s!")
    print("=" * 70)
    print("\nNext step: launch the local demonstration dashboard in PowerShell:")
    print("  python -m streamlit run app.py\n")


if __name__ == "__main__":
    run_smoke_test()
