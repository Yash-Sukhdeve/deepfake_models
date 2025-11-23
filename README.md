# Audio Deepfake Detection

A comprehensive implementation of state-of-the-art audio deepfake detection systems, featuring AASIST and XLS-R + SLS models with interactive GUI applications.

## Key Features

- **Two Detection Models**: AASIST (graph attention networks) and XLS-R + SLS (self-supervised learning)
- **Interactive GUIs**: Web-based interfaces for single file analysis and batch processing
- **Tampering Evaluation**: Custom datasets for trans-splicing and semantic tampering detection
- **Benchmark Results**: Reproduced published results on ASVspoof 2019/2021 datasets

## Performance Summary

| Model | ASVspoof 2019 LA | ASVspoof 2021 LA | Trans-Splicing Detection |
|-------|------------------|------------------|--------------------------|
| **XLS-R + SLS** | **0.26% EER** | **2.97% EER** | **95.45%** |
| AASIST | 0.83% EER | 48.27% EER | 41.72% |

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/deepfake_models.git
cd deepfake_models

# Create conda environment
conda create -n deepfake_detection python=3.8
conda activate deepfake_detection

# Install PyTorch with CUDA
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
pip install soundfile librosa numpy pandas matplotlib scipy scikit-learn gradio torchcontrib

# For XLS-R model (fairseq)
cd xls_r_sls/SLSforASVspoof-2021-DF
pip install -e fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1/
```

### 2. Download Pretrained Models

**AASIST**: Already included in `aasist/models/weights/AASIST.pth`

**XLS-R + SLS**: Download XLS-R 300M backbone:
```bash
# Place xlsr2_300m.pt in xls_r_sls/SLSforASVspoof-2021-DF/
# Trained model: best_model_4epochs_2.97EER.pth
```

### 3. Run GUI Applications

**AASIST Detection Interface:**
```bash
cd aasist
python gradio_app.py --port 7860
# Open http://127.0.0.1:7860
```

**XLS-R Detection Interface:**
```bash
cd xls_r_sls/SLSforASVspoof-2021-DF
python gradio_app.py --port 7861
# Open http://127.0.0.1:7861
```

## Repository Structure

```
deepfake_models/
├── aasist/                          # AASIST model
│   ├── config/                      # Training configurations
│   ├── models/                      # Model architecture & weights
│   ├── main.py                      # Training script
│   ├── gradio_app.py               # Simple GUI
│   └── gradio_app_multitab.py      # Multi-tab GUI
├── xls_r_sls/                       # XLS-R + SLS model
│   └── SLSforASVspoof-2021-DF/
│       ├── model.py                 # Model architecture
│       ├── train_LA.sh             # Training script
│       └── gradio_app.py           # GUI application
├── tampered_evaluation/             # Tampering detection
│   ├── trans_splicing/             # Trans-splicing dataset
│   ├── semantic/                    # Semantic tampering dataset
│   └── eval_tampered.py            # Unified evaluation script
├── figures/                         # Visualizations
├── PROJECT_REPORT.md               # Detailed documentation
└── README.md                        # This file
```

## Datasets

### Required Datasets

1. **ASVspoof 2019 LA**: [Download from Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336)
2. **ASVspoof 2021 LA**: [Download from Zenodo](https://zenodo.org/record/4837263)

### Dataset Preparation

```bash
# Expected directory structure:
data/asvspoof/
├── asvspoof2019/LA/
│   ├── ASVspoof2019_LA_train/flac/
│   ├── ASVspoof2019_LA_dev/flac/
│   ├── ASVspoof2019_LA_eval/flac/
│   └── ASVspoof2019_LA_cm_protocols/
└── asvspoof2021/
    ├── ASVspoof2021_LA_eval/flac/
    └── keys/LA/CM/
```

## Training

### Train AASIST

```bash
cd aasist

# Edit config/AASIST.conf to set your dataset path
# Set "database_path": "/path/to/data/asvspoof/asvspoof2019/LA/"

python main.py --config config/AASIST.conf
```

### Train XLS-R + SLS

```bash
cd xls_r_sls/SLSforASVspoof-2021-DF

# Edit train_LA.sh to set paths
bash train_LA.sh

# Important: Stop after 2-4 epochs for best results
```

## Evaluation

### Evaluate on ASVspoof

```bash
# AASIST
cd aasist
python main.py --eval --config config/AASIST.conf

# XLS-R + SLS
cd xls_r_sls/SLSforASVspoof-2021-DF
python eval_LA.py --model_path best_model_4epochs_2.97EER.pth
```

### Evaluate on Tampering Datasets

```bash
cd tampered_evaluation

# XLS-R on Trans-Splicing
python eval_tampered.py --model xlsr --dataset trans_splicing

# AASIST on Trans-Splicing
python eval_tampered.py --model aasist --dataset trans_splicing

# All evaluations
python eval_tampered.py --model all --dataset all
```

## GUI Features

### AASIST Multi-Tab Interface

1. **Single Detection**: Upload audio and get instant predictions
2. **Batch Processing**: Process multiple files with CSV export
3. **Model Comparison**: Compare predictions from different models
4. **Training Monitor**: View training progress
5. **Dataset Explorer**: Browse ASVspoof datasets

### XLS-R Interface

- Upload or record audio
- Real-time deepfake detection
- Color-coded results (green=real, red=fake)
- Waveform visualization

## Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Comprehensive project documentation
- **[TAMPERING_RESULTS.md](TAMPERING_RESULTS.md)**: Detailed tampering evaluation results
- **[AUDIO_TAMPERING_TECHNIQUES.md](AUDIO_TAMPERING_TECHNIQUES.md)**: Tampering methodology

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB VRAM |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |

## Citations

If you use this code, please cite:

```bibtex
@article{jung2022aasist,
  title={AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and others},
  journal={IEEE/ACM TASLP},
  year={2022}
}

@inproceedings{zhang2024audio,
  title={Audio Deepfake Detection with Self-supervised XLS-R and SLS classifier},
  author={Zhang, Qishan and Wen, Shuangbing and Hu, Tao},
  booktitle={ACM Multimedia},
  year={2024}
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size in config |
| Fairseq import error | Reinstall fairseq from source |
| Audio format error | Convert to WAV/FLAC, 16kHz mono |
| Model not loading | Check checkpoint path |

## License

This project is for research purposes. See individual model repositories for specific licenses.

## Acknowledgments

- ASVspoof Challenge organizers
- AASIST authors (CLOVA AI Research)
- XLS-R + SLS authors
