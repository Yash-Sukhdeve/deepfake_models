# AASIST Evaluation Guide

Complete guide for evaluating AASIST anti-spoofing models with comprehensive metrics and visualizations.

## Overview

This evaluation system provides:
- **Comprehensive metrics**: EER, t-DCF, FAR, FRR, AUC, per-attack breakdowns
- **Publication-ready plots**: DET curves, ROC curves, score distributions, confusion matrices
- **Multiple datasets**: ASVspoof2019 and ASVspoof2021 support
- **Automated reporting**: Text summaries and LaTeX-ready tables

## Quick Start

###1. Basic Evaluation

```bash
# Evaluate on ASVspoof2019
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019

# Evaluate on ASVspoof2021
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2021
```

### 2. Test on Subset

```bash
# Test with first 100 files
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019 \
    --max_files 100
```

### 3. Custom Settings

```bash
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019 \
    --batch_size 16 \
    --num_workers 4 \
    --output_dir my_results
```

## Metrics Computed

### Primary Metrics

1. **EER (Equal Error Rate)**
   - Point where FAR = FRR
   - Reported as percentage
   - Lower is better
   - **Benchmark**: AASIST achieves 0.83% on ASVspoof2019

2. **min t-DCF (minimum tandem Detection Cost Function)**
   - ASVspoof-specific metric
   - Evaluates CM in context of ASV system
   - Range: [0, ∞], values > 1 indicate poor performance
   - **Benchmark**: AASIST achieves 0.0275 on ASVspoof2019

3. **AUC (Area Under ROC Curve)**
   - Overall classification performance
   - Range: [0, 1], higher is better
   - Threshold-independent metric

### Secondary Metrics

4. **FAR/FRR at EER**
   - False Acceptance Rate and False Rejection Rate at EER threshold
   - Verifies EER calculation

5. **Per-Attack Type EER**
   - Individual EER for each spoofing algorithm
   - Identifies model strengths/weaknesses
   - ASVspoof2019 has 13 attack types (A07-A19)
   - ASVspoof2021 has 19 attack types (A01-A19)

## Visualizations Generated

### 1. DET Curve
- **File**: `det_curve_*.png`
- False Rejection Rate vs False Acceptance Rate
- Logarithmic scales
- EER point marked
- **Standard in anti-spoofing papers**

### 2. ROC Curve
- **File**: `roc_curve_*.png`
- True Positive Rate vs False Positive Rate
- Shows AUC score
- Compares against random classifier

### 3. Score Distributions
- **File**: `score_distributions_*.png`
- Overlapping histograms for bonafide and spoof
- Decision threshold marked
- Shows class separation

### 4. Per-Attack EER Chart
- **File**: `per_attack_eer_*.png`
- Bar chart of EER by attack type
- Color-coded by performance
- Overall EER reference line

### 5. Confusion Matrix
- **File**: `confusion_matrix_*.png`
- 2x2 classification matrix
- Shows true/false positives/negatives
- Annotated with counts

### 6. Score Scatter Plot
- **File**: `score_scatter_*.png`
- Individual sample scores
- Color-coded by attack type
- Decision threshold line

### 7. Box Plots
- **File**: `score_boxplots_*.png`
- Score distribution statistics by attack
- Shows median, quartiles, outliers
- Bonafide vs each spoof type

## Output Directory Structure

```
results/
├── asvspoof2019/
│   ├── cm_scores_ASVspoof2019.txt          # Raw scores
│   ├── results_summary_ASVspoof2019.txt    # Text summary
│   ├── tdcf_eer_results.txt                # Detailed metrics
│   └── plots/
│       ├── det_curve_ASVspoof2019.png
│       ├── roc_curve_ASVspoof2019.png
│       ├── score_distributions_ASVspoof2019.png
│       ├── per_attack_eer_ASVspoof2019.png
│       ├── confusion_matrix_ASVspoof2019.png
│       ├── score_scatter_ASVspoof2019.png
│       └── score_boxplots_ASVspoof2019.png
│
└── asvspoof2021/
    └── ... (same structure)
```

## Dataset Paths

Default paths (auto-detected):
- **ASVspoof2019**: `/home/lab2208/Documents/df_detection/data/asvspoof/asvspoof2019/LA`
- **ASVspoof2021**: `/home/lab2208/Documents/df_detection/data/asvspoof/asvspoof2021`

Override with `--data_path`:
```bash
python comprehensive_eval.py \
    --checkpoint model.pth \
    --dataset 2019 \
    --data_path /custom/path/to/data
```

## Performance Tips

### GPU Memory Issues

If you encounter CUDA out of memory errors:
```bash
# Reduce batch size
python comprehensive_eval.py --batch_size 4 ...

# Or use CPU (much slower)
CUDA_VISIBLE_DEVICES="" python comprehensive_eval.py ...
```

### Speed Optimization

```bash
# Use more workers for data loading
python comprehensive_eval.py --num_workers 8 ...

# Increase batch size if GPU allows
python comprehensive_eval.py --batch_size 32 ...
```

## Benchmark Results

### AASIST on ASVspoof2019 LA

| Metric | Published | Expected |
|--------|-----------|----------|
| EER | 0.83% | < 1.5% |
| min t-DCF | 0.0275 | < 0.05 |
| AUC | ~0.999 | > 0.995 |

### Per-Attack Performance (Published)

Best performance: A07, A08, A09 (TTS systems) < 0.5%
Worst performance: A17 (Voice Conversion) ~2-3%

## Troubleshooting

### Common Issues

**1. FileNotFoundError for trial files**
- Check data path with `--data_path`
- Verify dataset structure matches expected format

**2. CUDA out of memory**
- Reduce `--batch_size`
- Close other GPU processes
- Use `nvidia-smi` to check GPU usage

**3. Slow evaluation**
- Increase `--num_workers`
- Increase `--batch_size` if GPU allows
- Check disk I/O (audio file loading)

**4. Metrics seem wrong**
- Verify correct checkpoint is loaded
- Check if model was trained on same dataset
- Confirm data preprocessing matches training

## Scientific Citations

When using these evaluation metrics in publications:

**t-DCF metric:**
```
@inproceedings{kinnunen2018t,
  title={t-DCF: a detection cost function for the tandem assessment of spoofing countermeasures and automatic speaker verification},
  author={Kinnunen, Tomi and Lee, Kong Aik and Delgado, H{\'e}ctor and Evans, Nicholas and Todisco, Massimiliano and Sahidullah, Md and Yamagishi, Junichi and Reynolds, Douglas A},
  booktitle={Proc. Odyssey},
  year={2018}
}
```

**AASIST model:**
```
@article{jung2022aasist,
  title={AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2022}
}
```

## Advanced Usage

### Using Visualization Module Standalone

```python
import visualization as viz
import numpy as np

# Plot custom DET curve
frr, far = ...  # Your data
viz.plot_det_curve(frr, far, eer=0.01, eer_threshold=0.5,
                   save_path='my_det.png')

# Plot score distributions
viz.plot_score_distributions(bonafide_scores, spoof_scores,
                             threshold, save_path='scores.png')
```

### Processing Existing Score Files

If you already have score files, you can compute metrics without re-running inference:

```python
from evaluation import calculate_tDCF_EER

eer, tdcf = calculate_tDCF_EER(
    cm_scores_file='my_scores.txt',
    asv_score_file='asv_scores.txt',
    output_file='metrics.txt'
)
```

## Contact and Support

For issues or questions:
- Check this guide first
- Review error messages carefully
- Check GPU memory and data paths
- Verify dataset integrity

## Version History

- **2025-10-31**: Initial comprehensive evaluation system
  - Full metric computation
  - All visualization functions
  - Support for ASVspoof2019 and 2021
  - Publication-ready outputs
