#!/bin/bash
# ============================================================================
# Audio Deepfake Detection Project - One-Command Setup Script
# ============================================================================
# This script sets up the complete environment for reproducing the experiments.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requirements:
#   - Anaconda/Miniconda installed
#   - NVIDIA GPU with CUDA support (recommended)
#   - ~50GB disk space for datasets and models
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "Audio Deepfake Detection Project Setup"
echo "============================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Create conda environment from exported environment.yml
echo "[1/6] Creating conda environment 'xls_r_test'..."
if conda env list | grep -q "xls_r_test"; then
    echo "      Environment 'xls_r_test' already exists. Skipping creation."
else
    conda env create -f environment.yml
    echo "      Environment created successfully."
fi

# Activate environment
echo "[2/6] Activating environment..."
conda activate xls_r_test

# Verify GPU availability
echo "[3/6] Checking GPU availability..."
python -c "import torch; print(f'      PyTorch version: {torch.__version__}'); print(f'      CUDA available: {torch.cuda.is_available()}'); print(f'      GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check model files
echo "[4/6] Checking model checkpoints..."
MODELS_OK=true

if [ -f "aasist/models/weights/AASIST.pth" ]; then
    echo "      AASIST pretrained: OK"
else
    echo "      AASIST pretrained: MISSING"
    MODELS_OK=false
fi

if [ -f "xls_r_sls/SLSforASVspoof-2021-DF/best_model_4epochs_2.97EER.pth" ]; then
    echo "      XLS-R + SLS best: OK"
else
    echo "      XLS-R + SLS best: MISSING"
    MODELS_OK=false
fi

if [ -f "xls_r_sls/pretrained_models/xlsr2_300m.pt" ]; then
    echo "      XLS-R 300M pretrained: OK"
else
    echo "      XLS-R 300M pretrained: MISSING"
    echo "      Download from: https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr"
    MODELS_OK=false
fi

# Check datasets
echo "[5/6] Checking datasets..."
DATASETS_OK=true

if [ -d "data/asvspoof/asvspoof2019/LA" ]; then
    echo "      ASVspoof 2019 LA: OK"
else
    echo "      ASVspoof 2019 LA: MISSING"
    echo "      Download from: https://datashare.ed.ac.uk/handle/10283/3336"
    DATASETS_OK=false
fi

if [ -d "data/asvspoof/asvspoof2021/ASVspoof2021_LA_eval" ]; then
    echo "      ASVspoof 2021 LA: OK"
else
    echo "      ASVspoof 2021 LA: MISSING"
    echo "      Download from: https://zenodo.org/record/4837263"
    DATASETS_OK=false
fi

if [ -d "data/asvspoof/asvspoof2021/keys/LA" ]; then
    echo "      ASVspoof 2021 LA keys: OK"
else
    echo "      ASVspoof 2021 LA keys: MISSING"
    echo "      Download from: https://www.asvspoof.org/index2021.html"
    DATASETS_OK=false
fi

# Summary
echo ""
echo "[6/6] Setup Summary"
echo "============================================================================"
if [ "$MODELS_OK" = true ] && [ "$DATASETS_OK" = true ]; then
    echo "STATUS: READY"
    echo ""
    echo "Quick Start Commands:"
    echo ""
    echo "  # Activate environment"
    echo "  conda activate xls_r_test"
    echo ""
    echo "  # Verify XLS-R results (2.97% EER)"
    echo "  cd xls_r_sls/SLSforASVspoof-2021-DF"
    echo "  python evaluate_2021_LA_fixed.py scores_LA_epoch2.txt \\"
    echo "      ../../data/asvspoof/asvspoof2021/keys eval"
    echo ""
    echo "  # Run AASIST GUI"
    echo "  cd aasist && python gradio_app.py"
    echo ""
    echo "  # Run XLS-R GUI"
    echo "  cd xls_r_sls/SLSforASVspoof-2021-DF && python gradio_app.py"
else
    echo "STATUS: INCOMPLETE"
    echo ""
    echo "Please download missing models/datasets before running experiments."
    echo "See PROJECT_REPORT.md Section 8 for download links and instructions."
fi
echo "============================================================================"
