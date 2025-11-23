#!/bin/bash
# Dataset Download Script for XLS-R + SLS Experiment
# Downloads ASVspoof 2021 DF, In-the-Wild, and XLS-R checkpoint

set -e  # Exit on error

DATA_DIR="/home/lab2208/Documents/deepfake_models/data/asvspoof"
PRETRAINED_DIR="/home/lab2208/Documents/deepfake_models/xls_r_sls/pretrained_models"
LOG_FILE="/home/lab2208/Documents/deepfake_models/xls_r_sls/download_log.txt"

mkdir -p "$DATA_DIR/asvspoof2021"
mkdir -p "$DATA_DIR/in_the_wild"
mkdir -p "$PRETRAINED_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "XLS-R Dataset Download Script" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Function to log and execute
log_exec() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 1. Download XLS-R 300M Checkpoint (fastest, 1.2GB)
log_exec "Downloading XLS-R 300M checkpoint..."
cd "$PRETRAINED_DIR"
if [ ! -f "xlsr2_300m.pt" ]; then
    wget -c https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt \
        -O xlsr2_300m.pt.tmp 2>&1 | tee -a "$LOG_FILE"
    mv xlsr2_300m.pt.tmp xlsr2_300m.pt
    log_exec "✅ XLS-R checkpoint downloaded successfully"
else
    log_exec "✅ XLS-R checkpoint already exists, skipping"
fi

# 2. Download ASVspoof 2021 DF Dataset (CRITICAL, ~50GB)
log_exec "Downloading ASVspoof 2021 DF dataset..."
cd "$DATA_DIR/asvspoof2021"

# Check if already downloaded
if [ -d "ASVspoof2021_DF_eval" ] && [ "$(find ASVspoof2021_DF_eval -name '*.flac' | wc -l)" -gt 100000 ]; then
    log_exec "✅ ASVspoof 2021 DF already exists, skipping"
else
    log_exec "Downloading DF dataset from Zenodo (this will take 1-3 hours)..."

    # Try Zenodo link
    wget -c "https://zenodo.org/record/4835108/files/ASVspoof2021_DF_eval.tar.gz" \
        -O ASVspoof2021_DF_eval.tar.gz 2>&1 | tee -a "$LOG_FILE" || {
        log_exec "⚠️  Direct Zenodo download failed, trying alternative..."
        # Alternative: use zenodo API or manual instruction
        log_exec "❌ Please manually download from: https://zenodo.org/record/4835108"
        exit 1
    }

    log_exec "Extracting DF dataset..."
    tar -xzf ASVspoof2021_DF_eval.tar.gz
    log_exec "✅ ASVspoof 2021 DF extracted successfully"

    # Verify
    DF_COUNT=$(find ASVspoof2021_DF_eval -name '*.flac' | wc -l)
    log_exec "Found $DF_COUNT FLAC files in DF dataset"
fi

# 3. Download DF protocol/keys
log_exec "Downloading DF keys..."
if [ ! -f "DF-keys-full.tar.gz" ]; then
    wget -c "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz" \
        -O DF-keys-full.tar.gz 2>&1 | tee -a "$LOG_FILE" || {
        log_exec "⚠️  Could not download DF keys from official site"
        log_exec "Please manually download from: https://www.asvspoof.org/index2021.html"
    }

    if [ -f "DF-keys-full.tar.gz" ]; then
        tar -xzf DF-keys-full.tar.gz
        log_exec "✅ DF keys extracted"
    fi
else
    log_exec "✅ DF keys already exist"
fi

# 4. Download In-the-Wild Dataset (~10GB, optional but requested)
log_exec "Downloading In-the-Wild dataset..."
cd "$DATA_DIR/in_the_wild"

if [ -d "release_in_the_wild" ] && [ "$(find release_in_the_wild -name '*.wav' -o -name '*.flac' | wc -l)" -gt 5000 ]; then
    log_exec "✅ In-the-Wild dataset already exists, skipping"
else
    log_exec "Attempting to download In-the-Wild from deepfake-total.com..."

    # Note: This URL may require authentication or may have changed
    wget -c "https://deepfake-total.com/downloads/in_the_wild.zip" \
        -O in_the_wild.zip 2>&1 | tee -a "$LOG_FILE" || {
        log_exec "⚠️  Direct download failed"
        log_exec "Alternative: Try Kaggle mirror or manual download"
        log_exec "See: https://github.com/deepfake-total/in_the_wild"
        log_exec "⏭️  Continuing without In-the-Wild (not critical for main experiment)"
    }

    if [ -f "in_the_wild.zip" ]; then
        log_exec "Extracting In-the-Wild..."
        unzip -q in_the_wild.zip
        log_exec "✅ In-the-Wild extracted successfully"

        ITW_COUNT=$(find . -name '*.wav' -o -name '*.flac' | wc -l)
        log_exec "Found $ITW_COUNT audio files in In-the-Wild"
    fi
fi

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "DOWNLOAD SUMMARY" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
log_exec "XLS-R checkpoint: $([ -f "$PRETRAINED_DIR/xlsr2_300m.pt" ] && echo "✅" || echo "❌")"
log_exec "ASVspoof 2021 DF: $([ -d "$DATA_DIR/asvspoof2021/ASVspoof2021_DF_eval" ] && echo "✅" || echo "❌")"
log_exec "DF Keys: $([ -d "$DATA_DIR/asvspoof2021/keys" ] || [ -f "$DATA_DIR/asvspoof2021/ASVspoof2021.DF.cm.eval.trl.txt" ] && echo "✅" || echo "⚠️")"
log_exec "In-the-Wild: $([ -d "$DATA_DIR/in_the_wild/release_in_the_wild" ] && echo "✅" || echo "⏭️ (optional)")"
echo "========================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

log_exec "🎉 Dataset download process complete!"
log_exec "Check $LOG_FILE for full details"
