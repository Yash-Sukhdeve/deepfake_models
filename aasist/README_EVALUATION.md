# AASIST Evaluation System

Comprehensive evaluation framework for AASIST anti-spoofing models with publication-ready metrics and visualizations.

## 🎯 Quick Start

```bash
# Activate environment
source ../../venv/bin/activate

# Evaluate on ASVspoof2019
python comprehensive_eval.py \
    --checkpoint exp_result/LA_AASIST_ep100_bs24/weights/best.pth \
    --dataset 2019

# Evaluate on ASVspoof2021
python comprehensive_eval.py \
    --checkpoint exp_result/LA_AASIST_ep100_bs24/weights/best.pth \
    --dataset 2021
```

## 📊 What You Get

### Metrics
- ✅ **EER** (Equal Error Rate)
- ✅ **min t-DCF** (Tandem Detection Cost Function)
- ✅ **AUC** (Area Under ROC Curve)
- ✅ **FAR/FRR** at EER threshold
- ✅ **Per-attack EER** breakdown (13-19 attack types)

### Visualizations (7 plots)
- ✅ **DET Curve** - Standard in anti-spoofing papers
- ✅ **ROC Curve** - With AUC score
- ✅ **Score Distributions** - Bonafide vs Spoof
- ✅ **Per-Attack Bar Chart** - Identify weaknesses
- ✅ **Confusion Matrix** - Classification performance
- ✅ **Score Scatter Plot** - By attack type
- ✅ **Box Plots** - Distribution statistics

### Reports
- ✅ **Text Summary** - All metrics in one file
- ✅ **Raw Scores** - For further analysis
- ✅ **Detailed Metrics** - t-DCF breakdown

## 📁 Files Overview

```
├── visualization.py           # Plotting functions (9 types)
├── comprehensive_eval.py      # Main evaluation script
├── evaluation.py             # Metric computation (from AASIST)
├── data_utils.py             # Dataset loading (from AASIST)
├── EVALUATION_GUIDE.md       # User guide with examples
├── docs/
│   └── EVALUATION_METHODOLOGY.md  # Scientific documentation
└── test_visualization.py     # Unit tests for plots
```

## 🚀 Installation

### Dependencies
```bash
pip install matplotlib seaborn scikit-learn pandas scipy numpy torch
```

All dependencies are listed in `../../requirements.txt`.

### Verification
```bash
# Test visualization module
python test_visualization.py

# Should output: "All visualization tests PASSED!"
```

## 📖 Usage Examples

### Basic Evaluation
```bash
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019
```

### Test Mode (100 files)
```bash
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019 \
    --max_files 100
```

### Custom Settings
```bash
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019 \
    --batch_size 16 \
    --num_workers 4 \
    --output_dir my_results
```

### GPU Memory Constrained
```bash
python comprehensive_eval.py \
    --checkpoint path/to/model.pth \
    --dataset 2019 \
    --batch_size 4  # Reduce batch size
```

## 📊 Expected Results

### AASIST on ASVspoof2019 LA (Published)
| Metric | Value |
|--------|-------|
| EER | 0.83% |
| min t-DCF | 0.0275 |
| AUC | ~0.999 |

Your results should be close to these benchmarks if using the properly trained AASIST model.

### ASVspoof2021 LA
Expected performance drop of 2-5% EER due to:
- Unknown attack algorithms
- Codec variations
- More challenging conditions

## 🔬 Scientific Documentation

Detailed methodology documentation available in:
- [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) - User-focused guide
- [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md) - Scientific rigor

### Key Papers to Cite

**AASIST Model:**
```bibtex
@article{jung2022aasist,
  title={AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2022}
}
```

**t-DCF Metric:**
```bibtex
@inproceedings{kinnunen2018t,
  title={t-DCF: a detection cost function for the tandem assessment of spoofing countermeasures and automatic speaker verification},
  author={Kinnunen, Tomi and others},
  booktitle={Proc. Odyssey},
  year={2018}
}
```

**ASVspoof2019 Dataset:**
```bibtex
@article{nautsch2021asvspoof,
  title={ASVspoof 2019: Spoofing countermeasures for the detection of synthesized, converted and replayed speech},
  author={Nautsch, Andreas and others},
  journal={IEEE TASLP},
  year={2021}
}
```

## 🎨 Visualization Examples

All plots are:
- Publication-ready (300 DPI)
- Vector graphics where possible
- Properly labeled with units
- Color-blind friendly palettes
- Consistent styling

### Using Visualization Module Standalone

```python
import visualization as viz
import numpy as np

# Your data
bonafide_scores = np.array([...])
spoof_scores = np.array([...])

# Generate plots
viz.plot_score_distributions(
    bonafide_scores=bonafide_scores,
    spoof_scores=spoof_scores,
    eer_threshold=0.5,
    save_path='my_plot.png'
)
```

See [`test_visualization.py`](test_visualization.py) for complete examples of all 8 plot types.

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python comprehensive_eval.py --batch_size 4 ...

# Solution 2: Use CPU (slower)
CUDA_VISIBLE_DEVICES="" python comprehensive_eval.py ...
```

### FileNotFoundError
```bash
# Check data paths
ls /home/lab2208/Documents/df_detection/data/asvspoof/

# Specify custom path
python comprehensive_eval.py --data_path /your/path ...
```

### Slow Evaluation
```bash
# Increase workers (if CPU has capacity)
python comprehensive_eval.py --num_workers 8 ...

# Increase batch size (if GPU has memory)
python comprehensive_eval.py --batch_size 32 ...
```

### Metrics Seem Wrong
1. Verify checkpoint is correct model
2. Check model was trained on same dataset
3. Confirm preprocessing matches training
4. Test on small subset first (--max_files 100)

## 📂 Output Structure

```
results/
├── asvspoof2019/
│   ├── cm_scores_ASVspoof2019.txt
│   ├── results_summary_ASVspoof2019.txt
│   ├── tdcf_eer_results.txt
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

## 🔍 Understanding the Metrics

### EER (Equal Error Rate)
- Where False Accept Rate = False Reject Rate
- **Lower is better**
- Most commonly reported metric

### min t-DCF
- Accounts for application costs
- Standard in ASVspoof challenges
- **Lower is better**
- Values > 1.0 = worse than no CM

### AUC (Area Under ROC)
- Overall classification ability
- **Higher is better** (max = 1.0)
- Threshold-independent

### Per-Attack EER
- Shows which attacks are hard to detect
- Guides model improvements
- Essential for research papers

## 🚦 Performance Interpretation

| Level | EER | min t-DCF | Quality |
|-------|-----|-----------|---------|
| Excellent | < 1% | < 0.03 | State-of-the-art |
| Good | 1-3% | 0.03-0.10 | Competitive |
| Moderate | 3-10% | 0.10-0.30 | Baseline |
| Poor | > 10% | > 0.30 | Needs work |

## 🤝 Contributing

To add new visualization types:
1. Add function to `visualization.py`
2. Add test to `test_visualization.py`
3. Update `comprehensive_eval.py` to call it
4. Update documentation

## 📜 License

This evaluation framework follows the AASIST license. See original repository for details.

## ✉️ Contact

For questions about evaluation methodology or issues:
1. Check [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
2. Review [docs/EVALUATION_METHODOLOGY.md](docs/EVALUATION_METHODOLOGY.md)
3. Check error messages and troubleshooting section

## 🎓 Educational Use

This system is designed for:
- Research reproducibility
- Thesis/dissertation work
- Benchmark comparisons
- Anti-spoofing education
- Model analysis and debugging

## ⏱️ Performance Notes

### Evaluation Time
- **ASVspoof2019 (71K files)**: ~30-45 minutes (batch_size=8, GPU)
- **ASVspoof2021 (148K files)**: ~60-90 minutes (batch_size=8, GPU)
- **Test subset (100 files)**: ~1 minute

### Memory Requirements
- **GPU**: ~2-4 GB VRAM (batch_size=8-16)
- **RAM**: ~8-16 GB
- **Disk**: ~500 MB per evaluation (scores + plots)

## 🔄 Version History

- **v1.0 (2025-10-31)**: Initial comprehensive evaluation system
  - Full metric computation (EER, t-DCF, AUC, per-attack)
  - 7 visualization types
  - Support for ASVspoof2019 and 2021
  - Publication-ready outputs
  - Complete documentation

## 🙏 Acknowledgments

- AASIST model: Jung et al., NAVER Corp
- ASVspoof challenge organizers
- Original evaluation scripts from ASVspoof2019/2021

---

**Made with ❤️ for reproducible anti-spoofing research**
