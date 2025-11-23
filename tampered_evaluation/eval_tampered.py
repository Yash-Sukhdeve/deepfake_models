#!/usr/bin/env python3
"""
Unified evaluation script for audio tampering detection.
Supports both XLS-R + SLS and AASIST models on Trans-Splicing and Semantic datasets.

Usage:
    # XLS-R on Trans-Splicing Dataset
    python eval_tampered.py --model xlsr --dataset trans_splicing

    # AASIST on Semantic
    python eval_tampered.py --model aasist --model_variant pretrained --dataset semantic

    # All evaluations
    python eval_tampered.py --model all --dataset all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Constants
SAMPLE_RATE = 16000
CUT_LENGTH = 64600  # ~4 seconds at 16kHz

# Paths
BASE_DIR = Path("/home/lab2208/Documents/deepfake_models")
TAMPERED_DIR = BASE_DIR / "tampered_evaluation"

# Model paths
XLSR_MODEL_PATH = BASE_DIR / "xls_r_sls/SLSforASVspoof-2021-DF/best_model_4epochs_2.97EER.pth"
AASIST_PRETRAINED_PATH = BASE_DIR / "aasist/models/weights/AASIST.pth"
AASIST_TRAINED_PATH = BASE_DIR / "aasist/exp_result/LA_AASIST_ep100_bs24/weights/best.pth"


class TamperedDataset(Dataset):
    """Dataset for tampered audio evaluation."""

    def __init__(self, protocol_path, dataset_type="parker"):
        self.samples = []
        self.dataset_type = dataset_type

        with open(protocol_path, 'r') as f:
            protocol = json.load(f)

        for entry in protocol:
            file_path = Path(entry["file_path"])
            if file_path.exists():
                self.samples.append({
                    "file_path": str(file_path),
                    "label": 1 if entry["label"] == "bonafide" else 0,  # 1=bonafide, 0=spoof
                    "category": entry.get("category", entry.get("tamper_type", "unknown")),
                    "metadata": entry
                })

        print(f"Loaded {len(self.samples)} samples from {dataset_type} dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio, sr = sf.read(sample["file_path"])

        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Pad or truncate to CUT_LENGTH
        if len(audio) < CUT_LENGTH:
            audio = np.pad(audio, (0, CUT_LENGTH - len(audio)), 'constant')
        else:
            audio = audio[:CUT_LENGTH]

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "label": sample["label"],
            "category": sample["category"],
            "file_path": sample["file_path"]
        }


def load_xlsr_model(device):
    """Load XLS-R + SLS model."""
    xlsr_dir = BASE_DIR / "xls_r_sls/SLSforASVspoof-2021-DF"
    sys.path.insert(0, str(xlsr_dir))

    # Change to XLS-R directory for relative path resolution
    original_dir = os.getcwd()
    os.chdir(xlsr_dir)

    from model import Model
    model = Model(None, device)
    state_dict = torch.load(XLSR_MODEL_PATH, map_location=device)

    # Handle DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # Restore original directory
    os.chdir(original_dir)

    return model


def load_aasist_model(device, variant="pretrained"):
    """Load AASIST model."""
    aasist_dir = BASE_DIR / "aasist"
    sys.path.insert(0, str(aasist_dir))

    # Change to AASIST directory
    original_dir = os.getcwd()
    os.chdir(aasist_dir)

    from importlib import import_module

    model_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    module = import_module("models.AASIST")
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)

    model_path = AASIST_PRETRAINED_PATH if variant == "pretrained" else AASIST_TRAINED_PATH
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Restore original directory
    os.chdir(original_dir)

    return model


def compute_eer(bonafide_scores, spoof_scores):
    """Compute Equal Error Rate."""
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return float('nan'), float('nan')

    y_true = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    y_score = np.concatenate([bonafide_scores, spoof_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = interp1d(fpr, thresholds)(eer)
    except:
        eer = 0.5
        eer_threshold = 0.0

    return eer * 100, eer_threshold


def compute_detection_rate(scores, threshold=0.5):
    """Compute detection rate (% classified as spoof)."""
    # For detection rate, scores below threshold are classified as spoof
    detected = np.sum(scores < threshold)
    return (detected / len(scores)) * 100


def evaluate_model(model, dataloader, model_type, device):
    """Evaluate model on dataset."""
    all_scores = []
    all_labels = []
    all_categories = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            audio = batch["audio"].to(device)
            labels = batch["label"].numpy()
            categories = batch["category"]
            paths = batch["file_path"]

            # Forward pass
            output = model(audio)

            if model_type == "xlsr":
                # XLS-R outputs log probabilities, get bonafide probability
                probs = torch.exp(output)
                scores = probs[:, 1].cpu().numpy()  # Higher = more likely bonafide
            else:
                # AASIST outputs (last_hidden, output), get output and softmax
                if isinstance(output, tuple):
                    output = output[1]  # Get the classification output
                probs = F.softmax(output, dim=1)
                scores = probs[:, 1].cpu().numpy()  # Higher = more likely bonafide

            all_scores.extend(scores)
            all_labels.extend(labels)
            all_categories.extend(categories)
            all_paths.extend(paths)

    return np.array(all_scores), np.array(all_labels), all_categories, all_paths


def run_evaluation(model_name, model_variant, dataset_name, device, batch_size=10):
    """Run complete evaluation."""
    results = {}

    # Load protocol
    if dataset_name == "trans_splicing":
        protocol_path = TAMPERED_DIR / "trans_splicing/protocol.json"
        has_bonafide = False
    else:  # semantic
        protocol_path = TAMPERED_DIR / "semantic/protocol.json"
        has_bonafide = True

    # Load dataset
    dataset = TamperedDataset(protocol_path, dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    print(f"\nLoading {model_name} model ({model_variant})...")
    if model_name == "xlsr":
        model = load_xlsr_model(device)
        model_type = "xlsr"
    else:
        model = load_aasist_model(device, model_variant)
        model_type = "aasist"

    # Evaluate
    scores, labels, categories, paths = evaluate_model(model, dataloader, model_type, device)

    # Compute metrics
    results["total_samples"] = len(scores)
    results["model"] = f"{model_name}_{model_variant}"
    results["dataset"] = dataset_name

    if has_bonafide:
        # Compute EER for semantic dataset
        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]
        eer, threshold = compute_eer(bonafide_scores, spoof_scores)
        results["eer"] = float(eer)
        results["threshold"] = float(threshold) if not np.isnan(threshold) else 0.0
        results["bonafide_count"] = len(bonafide_scores)
        results["spoof_count"] = len(spoof_scores)
        results["bonafide_mean_score"] = float(np.mean(bonafide_scores))
        results["spoof_mean_score"] = float(np.mean(spoof_scores))

        # Detection rate at EER threshold
        if not np.isnan(threshold):
            results["detection_rate"] = float(compute_detection_rate(spoof_scores, threshold))
    else:
        # Compute detection rate for parker (all samples are spoof)
        # Use 0.5 as default threshold
        results["detection_rate_05"] = compute_detection_rate(scores, 0.5)
        results["mean_score"] = float(np.mean(scores))
        results["std_score"] = float(np.std(scores))

        # Per-category breakdown
        results["per_category"] = {}
        unique_cats = list(set(categories))
        for cat in unique_cats:
            cat_mask = np.array([c == cat for c in categories])
            cat_scores = scores[cat_mask]
            results["per_category"][cat] = {
                "count": int(np.sum(cat_mask)),
                "detection_rate_05": compute_detection_rate(cat_scores, 0.5),
                "mean_score": float(np.mean(cat_scores)),
                "std_score": float(np.std(cat_scores))
            }

    # Save scores
    output_dir = TAMPERED_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    score_file = output_dir / f"scores_{dataset_name}_{model_name}_{model_variant}.txt"
    with open(score_file, 'w') as f:
        for path, score, label in zip(paths, scores, labels):
            f.write(f"{Path(path).name} {score:.6f} {label}\n")

    results_file = output_dir / f"results_{dataset_name}_{model_name}_{model_variant}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*60)
    print(f"RESULTS: {results['model']} on {results['dataset']}")
    print("="*60)
    print(f"Total samples: {results['total_samples']}")

    if "eer" in results:
        print(f"\nEER: {results['eer']:.2f}%")
        print(f"Threshold: {results['threshold']:.4f}")
        print(f"Bonafide: {results['bonafide_count']} (mean score: {results['bonafide_mean_score']:.4f})")
        print(f"Spoof: {results['spoof_count']} (mean score: {results['spoof_mean_score']:.4f})")
        if "detection_rate" in results:
            print(f"Detection Rate at EER threshold: {results['detection_rate']:.2f}%")
    else:
        print(f"\nDetection Rate (@0.5): {results['detection_rate_05']:.2f}%")
        print(f"Mean score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

        if "per_category" in results:
            print("\nPer-Category Results:")
            for cat, cat_results in sorted(results["per_category"].items()):
                print(f"  {cat}: {cat_results['detection_rate_05']:.2f}% "
                      f"(n={cat_results['count']}, μ={cat_results['mean_score']:.4f})")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate tampering detection")
    parser.add_argument("--model", type=str, choices=["xlsr", "aasist", "all"],
                        default="xlsr", help="Model to evaluate")
    parser.add_argument("--model_variant", type=str, choices=["pretrained", "trained"],
                        default="pretrained", help="AASIST variant")
    parser.add_argument("--dataset", type=str, choices=["trans_splicing", "semantic", "all"],
                        default="trans_splicing", help="Dataset to evaluate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine what to evaluate
    models = ["xlsr", "aasist"] if args.model == "all" else [args.model]
    datasets = ["trans_splicing", "semantic"] if args.dataset == "all" else [args.dataset]
    variants = ["pretrained", "trained"] if args.model == "aasist" or args.model == "all" else [args.model_variant]

    all_results = []

    for dataset in datasets:
        for model in models:
            if model == "xlsr":
                results = run_evaluation(model, "best", dataset, device, args.batch_size)
                print_results(results)
                all_results.append(results)
            else:  # aasist
                for variant in variants:
                    results = run_evaluation(model, variant, dataset, device, args.batch_size)
                    print_results(results)
                    all_results.append(results)

    # Save summary
    summary_file = TAMPERED_DIR / "results/evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {TAMPERED_DIR / 'results'}")
    return all_results


if __name__ == "__main__":
    main()
