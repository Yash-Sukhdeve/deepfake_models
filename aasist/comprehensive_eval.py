"""
Comprehensive evaluation script for AASIST anti-spoofing model.

This script performs complete evaluation including:
- Score generation
- Metric computation (EER, t-DCF, per-attack breakdown)
- Visualization generation (DET curves, score distributions, etc.)
- Results organization and reporting

Usage:
    python comprehensive_eval.py --checkpoint path/to/model.pth --dataset 2019
    python comprehensive_eval.py --checkpoint path/to/model.pth --dataset 2021

Author: Research Lab
Date: 2025-10-31
"""

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

# Import local modules
from data_utils import Dataset_ASVspoof2019_devNeval, genSpoof_list
from evaluation import calculate_tDCF_EER, compute_det_curve, compute_eer
import visualization as viz


def get_model(model_config: dict, device: torch.device):
    """Load model architecture."""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f"Model parameters: {nb_params:,}")
    return model


def generate_scores(
    model,
    data_loader: DataLoader,
    device: torch.device,
    save_path: Path,
    trial_path: Path,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate prediction scores for all samples.

    Returns:
    --------
    scores : np.ndarray
        Prediction scores
    labels : np.ndarray
        True labels (1=bonafide, 0=spoof)
    sources : np.ndarray
        Attack types for spoof samples
    """
    model.eval()

    # Read trial file
    with open(trial_path, "r") as f:
        all_trial_lines = f.readlines()

    fname_list = []
    score_list = []

    if verbose:
        print(f"\nGenerating scores for {len(data_loader)} batches...")

    start_time = time.time()

    for i, (batch_x, utt_id) in enumerate(data_loader):
        if verbose and i % 100 == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / len(data_loader) * 100
            print(f"  Progress: {i+1}/{len(data_loader)} ({progress:.1f}%) - {elapsed:.1f}s")

        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    if verbose:
        total_time = time.time() - start_time
        print(f"✓ Score generation complete in {total_time:.1f}s")

    # Create a mapping of filename to trial line
    trial_dict = {}
    for trl in all_trial_lines:
        parts = trl.strip().split(' ')
        _, utt_id, _, src, key = parts
        trial_dict[utt_id] = (trl, src, key)

    assert len(fname_list) == len(score_list), \
        f"Mismatch: {len(fname_list)} files, {len(score_list)} scores"

    # Write scores to file
    scores = []
    labels = []
    sources = []

    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            if fn not in trial_dict:
                print(f"WARNING: {fn} not found in trial file, skipping")
                continue

            trl, src, key = trial_dict[fn]

            # Write to file
            fh.write(f"{fn} {src} {key} {sco}\n")

            # Collect for analysis
            scores.append(sco)
            labels.append(1 if key == 'bonafide' else 0)
            sources.append(src)

    if verbose:
        print(f"✓ Scores saved to {save_path}")

    return np.array(scores), np.array(labels), np.array(sources)


def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    asv_score_file: Path,
    cm_score_file: Path,
    output_dir: Path
) -> Dict:
    """
    Compute all evaluation metrics.

    Returns:
    --------
    metrics : dict
        Dictionary containing all computed metrics
    """
    print("\n" + "="*60)
    print("Computing Metrics")
    print("="*60)

    metrics = {}

    # Separate bonafide and spoof scores
    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    print(f"Bonafide samples: {len(bonafide_scores)}")
    print(f"Spoof samples: {len(spoof_scores)}")

    # 1. Compute EER and DET curve
    print("\n1. Computing EER and DET curve...")
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    eer, eer_threshold = compute_eer(bonafide_scores, spoof_scores)

    metrics['eer'] = eer * 100  # Convert to percentage
    metrics['eer_threshold'] = eer_threshold
    metrics['frr'] = frr
    metrics['far'] = far
    metrics['det_thresholds'] = thresholds

    print(f"   EER: {metrics['eer']:.4f}%")
    print(f"   EER Threshold: {eer_threshold:.4f}")

    # 2. Compute FAR and FRR at EER threshold
    print("\n2. Computing FAR/FRR at EER threshold...")
    far_at_eer = np.sum(spoof_scores >= eer_threshold) / len(spoof_scores)
    frr_at_eer = np.sum(bonafide_scores < eer_threshold) / len(bonafide_scores)

    metrics['far_at_eer'] = far_at_eer * 100
    metrics['frr_at_eer'] = frr_at_eer * 100

    print(f"   FAR at EER: {metrics['far_at_eer']:.4f}%")
    print(f"   FRR at EER: {metrics['frr_at_eer']:.4f}%")

    # 3. Compute ROC and AUC
    print("\n3. Computing ROC curve and AUC...")
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    metrics['roc_thresholds'] = roc_thresholds
    metrics['auc'] = roc_auc

    print(f"   AUC: {roc_auc:.6f}")

    # 4. Compute t-DCF (if ASV scores available)
    print("\n4. Computing t-DCF...")
    if asv_score_file.exists():
        eer_tdcf, min_tdcf = calculate_tDCF_EER(
            cm_scores_file=str(cm_score_file),
            asv_score_file=str(asv_score_file),
            output_file=str(output_dir / "tdcf_eer_results.txt"),
            printout=False
        )
        metrics['tdcf'] = min_tdcf
        print(f"   min t-DCF: {min_tdcf:.6f}")
    else:
        print(f"   ASV scores not found at {asv_score_file}, skipping t-DCF")
        metrics['tdcf'] = None

    # 5. Compute per-attack EER
    print("\n5. Computing per-attack EER...")
    attack_types = [f'A{i:02d}' for i in range(7, 20)]
    attack_eers = {}

    for attack in attack_types:
        attack_mask = sources == attack
        if np.sum(attack_mask) > 0:
            attack_spoof_scores = scores[attack_mask]
            attack_eer = compute_eer(bonafide_scores, attack_spoof_scores)[0]
            attack_eers[attack] = attack_eer * 100
            print(f"   {attack}: {attack_eers[attack]:.4f}%")

    metrics['attack_eers'] = attack_eers

    # 6. Organize scores by attack type
    print("\n6. Organizing scores by attack type...")
    scores_by_attack = {}
    for attack in attack_types:
        attack_mask = sources == attack
        if np.sum(attack_mask) > 0:
            scores_by_attack[attack] = scores[attack_mask]

    metrics['scores_by_attack'] = scores_by_attack
    metrics['bonafide_scores'] = bonafide_scores
    metrics['spoof_scores'] = spoof_scores
    metrics['attack_types'] = sources[labels == 0]

    print("\n" + "="*60)
    print("Metrics Computation Complete")
    print("="*60)

    return metrics


def generate_all_plots(metrics: Dict, output_dir: Path, dataset_name: str):
    """Generate all visualization plots."""
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    # 1. DET Curve
    print("\n1. Generating DET curve...")
    viz.plot_det_curve(
        frr=metrics['frr'],
        far=metrics['far'],
        eer=metrics['eer'] / 100,
        eer_threshold=metrics['eer_threshold'],
        save_path=plot_dir / f"det_curve_{dataset_name}.png",
        title=f"DET Curve - {dataset_name}"
    )

    # 2. ROC Curve
    print("2. Generating ROC curve...")
    viz.plot_roc_curve(
        fpr=metrics['fpr'],
        tpr=metrics['tpr'],
        roc_auc=metrics['auc'],
        save_path=plot_dir / f"roc_curve_{dataset_name}.png",
        title=f"ROC Curve - {dataset_name}"
    )

    # 3. Score Distributions
    print("3. Generating score distributions...")
    viz.plot_score_distributions(
        bonafide_scores=metrics['bonafide_scores'],
        spoof_scores=metrics['spoof_scores'],
        eer_threshold=metrics['eer_threshold'],
        save_path=plot_dir / f"score_distributions_{dataset_name}.png",
        title=f"Score Distributions - {dataset_name}"
    )

    # 4. Per-Attack EER
    print("4. Generating per-attack EER chart...")
    viz.plot_per_attack_eer(
        attack_eers=metrics['attack_eers'],
        overall_eer=metrics['eer'],
        save_path=plot_dir / f"per_attack_eer_{dataset_name}.png",
        title=f"EER by Attack Type - {dataset_name}"
    )

    # 5. Confusion Matrix
    print("5. Generating confusion matrix...")
    y_true = np.concatenate([
        np.ones(len(metrics['bonafide_scores'])),
        np.zeros(len(metrics['spoof_scores']))
    ])
    y_pred = (np.concatenate([
        metrics['bonafide_scores'],
        metrics['spoof_scores']
    ]) >= metrics['eer_threshold']).astype(int)

    viz.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        save_path=plot_dir / f"confusion_matrix_{dataset_name}.png",
        title=f"Confusion Matrix - {dataset_name}"
    )

    # 6. Score Scatter Plot
    print("6. Generating score scatter plot...")
    viz.plot_score_scatter(
        bonafide_scores=metrics['bonafide_scores'],
        spoof_scores=metrics['spoof_scores'],
        attack_types=metrics['attack_types'],
        eer_threshold=metrics['eer_threshold'],
        save_path=plot_dir / f"score_scatter_{dataset_name}.png",
        title=f"Score Scatter by Attack - {dataset_name}",
        max_points=5000
    )

    # 7. Box Plots
    print("7. Generating box plots...")
    viz.plot_score_boxplots(
        scores_by_attack=metrics['scores_by_attack'],
        bonafide_scores=metrics['bonafide_scores'],
        save_path=plot_dir / f"score_boxplots_{dataset_name}.png",
        title=f"Score Distributions by Attack - {dataset_name}"
    )

    print("\n" + "="*60)
    print("Visualization Generation Complete")
    print(f"Plots saved to: {plot_dir}")
    print("="*60)


def save_results_summary(metrics: Dict, output_dir: Path, dataset_name: str, args):
    """Save results summary to text file."""
    summary_file = output_dir / f"results_summary_{dataset_name}.txt"

    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"AASIST Evaluation Results - {dataset_name}\n")
        f.write("="*70 + "\n\n")

        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-"*70 + "\n")
        f.write("PRIMARY METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"EER (Equal Error Rate):     {metrics['eer']:.4f}%\n")
        f.write(f"EER Threshold:              {metrics['eer_threshold']:.6f}\n")
        if metrics['tdcf'] is not None:
            f.write(f"min t-DCF:                  {metrics['tdcf']:.6f}\n")
        f.write(f"AUC (Area Under ROC):       {metrics['auc']:.6f}\n")
        f.write(f"FAR at EER:                 {metrics['far_at_eer']:.4f}%\n")
        f.write(f"FRR at EER:                 {metrics['frr_at_eer']:.4f}%\n\n")

        f.write("-"*70 + "\n")
        f.write("PER-ATTACK TYPE EER\n")
        f.write("-"*70 + "\n")
        for attack in sorted(metrics['attack_eers'].keys()):
            f.write(f"{attack}:  {metrics['attack_eers'][attack]:6.4f}%\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\n✓ Results summary saved to {summary_file}")


def main(args):
    """Main evaluation function."""
    print("\n" + "="*70)
    print("AASIST Comprehensive Evaluation")
    print("="*70)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU - this will be slow!")

    # Load model configuration
    checkpoint_path = Path(args.checkpoint)
    config_path = checkpoint_path.parent.parent / "config.conf"

    if not config_path.exists():
        print(f"\nConfig not found at {config_path}")
        print("Using default AASIST configuration...")
        model_config = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "in_channels": 1,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
    else:
        with open(config_path, "r") as f:
            config = json.loads(f.read())
            model_config = config["model_config"]

    # Define model
    print(f"\nLoading model from: {checkpoint_path}")
    model = get_model(model_config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully")

    # Setup paths based on dataset
    data_root = Path(args.data_path)

    if args.dataset == "2019":
        trial_path = data_root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        audio_path = data_root / "ASVspoof2019_LA_eval"
        asv_score_file = data_root / "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
        dataset_name = "ASVspoof2019"
    elif args.dataset == "2021":
        trial_path = data_root / "ASVspoof2021.LA.cm.eval.trl.txt"
        audio_path = data_root / "ASVspoof2021_LA_eval"
        asv_score_file = data_root / "keys/LA/ASV/ASVTorch_Kaldi/score.txt"
        dataset_name = "ASVspoof2021"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create output directory
    output_dir = Path(args.output_dir) / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Get file list
    print(f"\nReading trial file: {trial_path}")
    file_list = genSpoof_list(dir_meta=trial_path, is_train=False, is_eval=True)
    print(f"Total files: {len(file_list)}")

    # Limit to subset if requested
    if args.max_files > 0 and args.max_files < len(file_list):
        print(f"Limiting to first {args.max_files} files for testing")
        file_list = file_list[:args.max_files]

    # Create dataset and dataloader
    eval_set = Dataset_ASVspoof2019_devNeval(
        list_IDs=file_list,
        base_dir=audio_path
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # Generate scores
    cm_score_file = output_dir / f"cm_scores_{dataset_name}.txt"
    scores, labels, sources = generate_scores(
        model=model,
        data_loader=eval_loader,
        device=device,
        save_path=cm_score_file,
        trial_path=trial_path,
        verbose=True
    )

    # Compute metrics
    metrics = compute_all_metrics(
        scores=scores,
        labels=labels,
        sources=sources,
        asv_score_file=asv_score_file,
        cm_score_file=cm_score_file,
        output_dir=output_dir
    )

    # Generate plots
    generate_all_plots(metrics, output_dir, dataset_name)

    # Save summary
    save_results_summary(metrics, output_dir, dataset_name, args)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nResults Location: {output_dir.absolute()}")
    print(f"\nKey Metrics:")
    print(f"  EER:       {metrics['eer']:.4f}%")
    if metrics['tdcf'] is not None:
        print(f"  min t-DCF: {metrics['tdcf']:.6f}")
    print(f"  AUC:       {metrics['auc']:.6f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive AASIST evaluation with metrics and visualizations"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["2019", "2021"],
        required=True,
        help="Dataset to evaluate on: 2019 or 2021"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to dataset root directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Maximum number of files to evaluate (for testing, -1 for all)"
    )

    args = parser.parse_args()

    # Auto-detect data path if not specified
    if args.data_path is None:
        if args.dataset == "2019":
            args.data_path = "/home/lab2208/Documents/deepfake_models/data/asvspoof/asvspoof2019/LA"
        else:  # 2021
            args.data_path = "/home/lab2208/Documents/deepfake_models/data/asvspoof/asvspoof2021"

    main(args)
