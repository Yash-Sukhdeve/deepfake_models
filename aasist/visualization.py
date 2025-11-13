"""
Visualization utilities for ASVspoof anti-spoofing evaluation.

This module provides publication-ready plotting functions for anti-spoofing
model evaluation including DET curves, ROC curves, score distributions, and
per-attack analysis.

Author: Research Lab
Date: 2025-10-31
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_curve, auc, confusion_matrix


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Seaborn style
sns.set_palette("husl")


def plot_det_curve(
    frr: np.ndarray,
    far: np.ndarray,
    eer: float,
    eer_threshold: float,
    save_path: Union[str, Path],
    title: str = "Detection Error Tradeoff (DET) Curve",
    show_legend: bool = True
) -> None:
    """
    Plot Detection Error Tradeoff (DET) curve.

    The DET curve plots False Rejection Rate (FRR) vs False Acceptance Rate (FAR)
    on logarithmic scales, making it easier to visualize performance across
    different operating points.

    Parameters:
    -----------
    frr : np.ndarray
        False Rejection Rate values
    far : np.ndarray
        False Acceptance Rate values
    eer : float
        Equal Error Rate (as fraction, e.g., 0.01 for 1%)
    eer_threshold : float
        Threshold value at EER point
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title
    show_legend : bool
        Whether to show legend

    Example:
    --------
    >>> frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    >>> eer, eer_thresh = compute_eer(bonafide_scores, spoof_scores)
    >>> plot_det_curve(frr, far, eer, eer_thresh, 'results/det_curve.png')
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot DET curve
    ax.plot(far * 100, frr * 100, 'b-', linewidth=2, label='DET Curve')

    # Mark EER point
    ax.plot(eer * 100, eer * 100, 'ro', markersize=10,
            label=f'EER = {eer*100:.3f}%')

    # Diagonal line (EER line)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=1)

    # Log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Labels and title
    ax.set_xlabel('False Acceptance Rate (%)')
    ax.set_ylabel('False Rejection Rate (%)')
    ax.set_title(title)

    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)

    if show_legend:
        ax.legend(loc='upper right')

    # Set reasonable axis limits
    ax.set_xlim([0.01, 100])
    ax.set_ylim([0.01, 100])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ DET curve saved to {save_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Union[str, Path],
    title: str = "Receiver Operating Characteristic (ROC) Curve"
) -> None:
    """
    Plot ROC curve with AUC score.

    Parameters:
    -----------
    fpr : np.ndarray
        False Positive Rate values
    tpr : np.ndarray
        True Positive Rate values
    roc_auc : float
        Area Under the Curve
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title

    Example:
    --------
    >>> fpr, tpr, _ = roc_curve(y_true, y_scores)
    >>> roc_auc = auc(fpr, tpr)
    >>> plot_roc_curve(fpr, tpr, roc_auc, 'results/roc_curve.png')
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC (AUC = {roc_auc:.4f})')

    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1,
            label='Random Classifier')

    # Labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    ax.legend(loc='lower right')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")


def plot_score_distributions(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    eer_threshold: float,
    save_path: Union[str, Path],
    title: str = "Score Distributions",
    bins: int = 50
) -> None:
    """
    Plot overlapping histograms of bonafide and spoof score distributions.

    Parameters:
    -----------
    bonafide_scores : np.ndarray
        Scores for bonafide (genuine) samples
    spoof_scores : np.ndarray
        Scores for spoof samples
    eer_threshold : float
        Threshold at EER point (marked with vertical line)
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title
    bins : int
        Number of histogram bins

    Example:
    --------
    >>> plot_score_distributions(bonafide, spoof, threshold, 'results/scores.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(bonafide_scores, bins=bins, alpha=0.6, color='green',
            label=f'Bonafide (n={len(bonafide_scores)})', density=True)
    ax.hist(spoof_scores, bins=bins, alpha=0.6, color='red',
            label=f'Spoof (n={len(spoof_scores)})', density=True)

    # Mark EER threshold
    ax.axvline(eer_threshold, color='black', linestyle='--', linewidth=2,
               label=f'EER Threshold = {eer_threshold:.3f}')

    # Labels and title
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Score distributions saved to {save_path}")


def plot_per_attack_eer(
    attack_eers: Dict[str, float],
    overall_eer: float,
    save_path: Union[str, Path],
    title: str = "EER by Attack Type",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot bar chart of EER values for each attack type.

    Parameters:
    -----------
    attack_eers : Dict[str, float]
        Dictionary mapping attack type (e.g., 'A07') to EER value (as %)
    overall_eer : float
        Overall EER across all attacks (as %)
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size (width, height)

    Example:
    --------
    >>> attack_eers = {'A07': 0.5, 'A08': 1.2, 'A09': 0.8}
    >>> plot_per_attack_eer(attack_eers, 0.83, 'results/per_attack_eer.png')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort attacks by name
    attacks = sorted(attack_eers.keys())
    eers = [attack_eers[a] for a in attacks]

    # Create bar chart
    bars = ax.bar(range(len(attacks)), eers, color='steelblue', alpha=0.8)

    # Color bars based on performance relative to overall EER
    for i, (attack, eer) in enumerate(zip(attacks, eers)):
        if eer > overall_eer * 1.5:  # Significantly worse
            bars[i].set_color('red')
            bars[i].set_alpha(0.7)
        elif eer > overall_eer:  # Worse than average
            bars[i].set_color('orange')
            bars[i].set_alpha(0.7)
        else:  # Better than average
            bars[i].set_color('green')
            bars[i].set_alpha(0.7)

    # Add overall EER line
    ax.axhline(overall_eer, color='black', linestyle='--', linewidth=2,
               label=f'Overall EER = {overall_eer:.3f}%')

    # Labels
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('EER (%)')
    ax.set_title(title)
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for i, v in enumerate(eers):
        ax.text(i, v + max(eers)*0.02, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-attack EER plot saved to {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
    labels: List[str] = ['Bonafide', 'Spoof']
) -> None:
    """
    Plot confusion matrix.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels (0=spoof, 1=bonafide)
    y_pred : np.ndarray
        Predicted labels (0=spoof, 1=bonafide)
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title
    labels : List[str]
        Class labels for display

    Example:
    --------
    >>> plot_confusion_matrix(y_true, y_pred, 'results/confusion_matrix.png')
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def plot_tdcf_curve(
    tdcf_values: np.ndarray,
    cm_thresholds: np.ndarray,
    min_tdcf: float,
    min_tdcf_threshold: float,
    save_path: Union[str, Path],
    title: str = "Tandem Detection Cost Function (t-DCF) Curve"
) -> None:
    """
    Plot t-DCF values across different CM thresholds.

    Parameters:
    -----------
    tdcf_values : np.ndarray
        Normalized t-DCF values
    cm_thresholds : np.ndarray
        CM threshold values
    min_tdcf : float
        Minimum t-DCF value
    min_tdcf_threshold : float
        Threshold at minimum t-DCF
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title

    Example:
    --------
    >>> plot_tdcf_curve(tdcf, thresholds, min_tdcf, min_thresh, 'results/tdcf.png')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot t-DCF curve
    ax.plot(cm_thresholds, tdcf_values, 'b-', linewidth=2, label='t-DCF')

    # Mark minimum t-DCF
    ax.plot(min_tdcf_threshold, min_tdcf, 'ro', markersize=10,
            label=f'min t-DCF = {min_tdcf:.4f}')

    # Reference line at t-DCF = 1 (worse than no CM)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5,
               label='No CM baseline')

    ax.set_xlabel('CM Threshold')
    ax.set_ylabel('Normalized t-DCF')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ t-DCF curve saved to {save_path}")


def plot_score_scatter(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    attack_types: np.ndarray,
    eer_threshold: float,
    save_path: Union[str, Path],
    title: str = "Score Scatter Plot by Attack Type",
    max_points: int = 5000
) -> None:
    """
    Plot scatter plot of scores colored by attack type.

    Parameters:
    -----------
    bonafide_scores : np.ndarray
        Scores for bonafide samples
    spoof_scores : np.ndarray
        Scores for spoof samples
    attack_types : np.ndarray
        Attack type labels for spoof samples (e.g., 'A07', 'A08', ...)
    eer_threshold : float
        Decision threshold
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title
    max_points : int
        Maximum number of points to plot (for performance)

    Example:
    --------
    >>> plot_score_scatter(bonafide, spoof, attacks, thresh, 'results/scatter.png')
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Subsample if too many points
    if len(bonafide_scores) > max_points:
        idx = np.random.choice(len(bonafide_scores), max_points, replace=False)
        bonafide_scores = bonafide_scores[idx]

    if len(spoof_scores) > max_points:
        idx = np.random.choice(len(spoof_scores), max_points, replace=False)
        spoof_scores = spoof_scores[idx]
        attack_types = attack_types[idx]

    # Plot bonafide
    ax.scatter(range(len(bonafide_scores)), bonafide_scores,
               c='green', alpha=0.3, s=10, label='Bonafide')

    # Plot spoof by attack type
    unique_attacks = np.unique(attack_types)
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_attacks)))

    offset = len(bonafide_scores)
    for i, attack in enumerate(unique_attacks):
        mask = attack_types == attack
        scores = spoof_scores[mask]
        ax.scatter(range(offset, offset + len(scores)), scores,
                  c=[colors[i]], alpha=0.5, s=10, label=attack)
        offset += len(scores)

    # Decision threshold
    ax.axhline(eer_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {eer_threshold:.3f}')

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best', ncol=3, markerscale=3)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Score scatter plot saved to {save_path}")


def plot_score_boxplots(
    scores_by_attack: Dict[str, np.ndarray],
    bonafide_scores: np.ndarray,
    save_path: Union[str, Path],
    title: str = "Score Distributions by Attack Type"
) -> None:
    """
    Plot box plots of score distributions for each attack type.

    Parameters:
    -----------
    scores_by_attack : Dict[str, np.ndarray]
        Dictionary mapping attack type to score array
    bonafide_scores : np.ndarray
        Bonafide scores for reference
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title

    Example:
    --------
    >>> scores = {'A07': arr1, 'A08': arr2, ...}
    >>> plot_score_boxplots(scores, bonafide, 'results/boxplots.png')
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    all_data = []
    labels = []

    # Add bonafide first
    all_data.append(bonafide_scores)
    labels.append('Bonafide')

    # Add each attack type
    for attack in sorted(scores_by_attack.keys()):
        all_data.append(scores_by_attack[attack])
        labels.append(attack)

    # Create box plot
    bp = ax.boxplot(all_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Color bonafide differently
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)

    # Color other boxes
    for i in range(1, len(bp['boxes'])):
        bp['boxes'][i].set_facecolor('lightcoral')
        bp['boxes'][i].set_alpha(0.7)

    ax.set_xlabel('Class / Attack Type')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Rotate labels if many attacks
    if len(labels) > 10:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Box plots saved to {save_path}")


def plot_training_curves(
    metrics_log: str,
    save_path: Union[str, Path],
    title: str = "Training Curves"
) -> None:
    """
    Plot training curves from metric log file.

    Parameters:
    -----------
    metrics_log : str
        Path to metric_log.txt file
    save_path : str or Path
        Path to save the figure
    title : str
        Plot title

    Example:
    --------
    >>> plot_training_curves('exp_result/LA_AASIST/metric_log.txt', 'results/training.png')
    """
    # Parse log file
    epochs = []
    losses = []
    dev_eers = []
    dev_tdcfs = []

    with open(metrics_log, 'r') as f:
        for line in f:
            if line.startswith('epoch:'):
                parts = line.strip().split(', ')
                epoch = int(parts[0].split(':')[1])
                loss = float(parts[1].split(':')[1])
                dev_eer = float(parts[2].split(':')[1])
                dev_tdcf = float(parts[3].split(':')[1])

                epochs.append(epoch)
                losses.append(loss)
                dev_eers.append(dev_eer)
                dev_tdcfs.append(dev_tdcf)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(epochs, losses, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # EER curve
    axes[1].plot(epochs, dev_eers, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('EER (%)')
    axes[1].set_title('Development EER')
    axes[1].grid(True, alpha=0.3)

    # t-DCF curve
    axes[2].plot(epochs, dev_tdcfs, 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('t-DCF')
    axes[2].set_title('Development t-DCF')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {save_path}")


if __name__ == "__main__":
    print("Visualization module for ASVspoof evaluation")
    print("This module provides functions for plotting evaluation metrics")
    print("\nAvailable functions:")
    print("  - plot_det_curve()")
    print("  - plot_roc_curve()")
    print("  - plot_score_distributions()")
    print("  - plot_per_attack_eer()")
    print("  - plot_confusion_matrix()")
    print("  - plot_tdcf_curve()")
    print("  - plot_score_scatter()")
    print("  - plot_score_boxplots()")
    print("  - plot_training_curves()")
