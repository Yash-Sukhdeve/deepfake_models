"""
Test script for visualization.py module.
Tests all plotting functions with dummy data.
"""

import numpy as np
import sys
from pathlib import Path
import visualization as viz

# Create output directory
output_dir = Path("test_plots")
output_dir.mkdir(exist_ok=True)

print("="*60)
print("Testing Visualization Module")
print("="*60)

# Generate dummy data
np.random.seed(42)
n_bonafide = 1000
n_spoof = 1000

# Simulate score distributions (bonafide higher, spoof lower)
bonafide_scores = np.random.normal(2.0, 1.5, n_bonafide)
spoof_scores = np.random.normal(-2.0, 1.5, n_spoof)

# Simulate FRR/FAR curves
n_points = 100
frr = np.logspace(-3, 0, n_points)
far = np.logspace(-3, 0, n_points)[::-1] * 0.8  # Descending
eer = 0.01  # 1% EER
eer_threshold = 0.5

print("\nTest 1: DET Curve")
try:
    viz.plot_det_curve(
        frr=frr,
        far=far,
        eer=eer,
        eer_threshold=eer_threshold,
        save_path=output_dir / "test_det_curve.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 2: ROC Curve")
try:
    # Simulate ROC data
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - np.exp(-5 * fpr)  # Exponential curve
    roc_auc = 0.95

    viz.plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        save_path=output_dir / "test_roc_curve.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 3: Score Distributions")
try:
    viz.plot_score_distributions(
        bonafide_scores=bonafide_scores,
        spoof_scores=spoof_scores,
        eer_threshold=eer_threshold,
        save_path=output_dir / "test_score_distributions.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 4: Per-Attack EER Bar Chart")
try:
    # Simulate per-attack EERs
    attack_eers = {
        f'A{i:02d}': np.random.uniform(0.5, 3.0)
        for i in range(7, 20)
    }
    overall_eer = 1.2

    viz.plot_per_attack_eer(
        attack_eers=attack_eers,
        overall_eer=overall_eer,
        save_path=output_dir / "test_per_attack_eer.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 5: Confusion Matrix")
try:
    # Simulate predictions
    y_true = np.concatenate([np.ones(n_bonafide), np.zeros(n_spoof)])
    y_pred = (np.concatenate([bonafide_scores, spoof_scores]) > eer_threshold).astype(int)

    viz.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path=output_dir / "test_confusion_matrix.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 6: t-DCF Curve")
try:
    # Simulate t-DCF values
    cm_thresholds = np.linspace(-5, 5, 200)
    tdcf_values = 0.5 + 0.5 * np.exp(-((cm_thresholds - 0.5)**2) / 2)
    min_tdcf = np.min(tdcf_values)
    min_tdcf_threshold = cm_thresholds[np.argmin(tdcf_values)]

    viz.plot_tdcf_curve(
        tdcf_values=tdcf_values,
        cm_thresholds=cm_thresholds,
        min_tdcf=min_tdcf,
        min_tdcf_threshold=min_tdcf_threshold,
        save_path=output_dir / "test_tdcf_curve.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 7: Score Scatter Plot")
try:
    # Simulate attack types
    attack_types = np.random.choice([f'A{i:02d}' for i in range(7, 13)], n_spoof)

    viz.plot_score_scatter(
        bonafide_scores=bonafide_scores,
        spoof_scores=spoof_scores,
        attack_types=attack_types,
        eer_threshold=eer_threshold,
        save_path=output_dir / "test_score_scatter.png",
        max_points=1000
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\nTest 8: Score Box Plots")
try:
    # Simulate scores by attack
    scores_by_attack = {
        f'A{i:02d}': np.random.normal(-2 + i*0.2, 1.5, 200)
        for i in range(7, 13)
    }

    viz.plot_score_boxplots(
        scores_by_attack=scores_by_attack,
        bonafide_scores=bonafide_scores[:200],
        save_path=output_dir / "test_score_boxplots.png"
    )
    print("✓ PASS")
except Exception as e:
    print(f"✗ FAIL: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All visualization tests PASSED!")
print(f"Test plots saved to: {output_dir.absolute()}")
print("="*60)
