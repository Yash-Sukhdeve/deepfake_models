#!/bin/bash
# Training Configuration for XLS-R + SLS on ASVspoof 2021 LA Track
# Target: 2.87% EER on ASVspoof 2021 LA evaluation

set -e

# Activate environment
source ~/anaconda3/bin/activate xls_r_sls

# Paths
REPO_DIR="/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF"
DATABASE_PATH="/home/lab2208/Documents/deepfake_models/data/asvspoof"
PROTOCOLS_PATH="/home/lab2208/Documents/deepfake_models/data/asvspoof"

# Training hyperparameters (from paper)
BATCH_SIZE=5         # Paper uses 5 for DF, using same for safety
NUM_EPOCHS=50        # Paper specification
LR=0.000001          # Paper specification (1e-6)
SEED=1234            # Reproducibility

# Output directory
OUTPUT_DIR="${REPO_DIR}/experiments/la_track"
mkdir -p "${OUTPUT_DIR}/models"
mkdir -p "${OUTPUT_DIR}/scores"
mkdir -p "${OUTPUT_DIR}/logs"

echo "=========================================="
echo "XLS-R + SLS Training - LA Track"
echo "=========================================="
echo "Database: ${DATABASE_PATH}"
echo "Training: ASVspoof 2019 LA"
echo "Evaluation: ASVspoof 2021 LA"
echo "Batch size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LR}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

cd "${REPO_DIR}"

# Start training
python main.py \
    --database_path "${DATABASE_PATH}" \
    --protocols_path "${PROTOCOLS_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --track LA \
    --comment "la_track_reproduction" \
    2>&1 | tee "${OUTPUT_DIR}/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Training Complete!"
echo "Logs saved to: ${OUTPUT_DIR}/logs/"
echo "=========================================="
