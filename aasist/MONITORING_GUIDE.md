# 📊 AASIST Training Monitoring Guide

**Safe Monitoring:** All commands below are READ-ONLY and will NOT disturb the training process.

---

## 🔍 Quick Status Check

### 1. **View Latest Training Progress**
```bash
tail -20 training_output.log
```
Shows the last 20 lines of training output including:
- Current epoch number
- Loss value
- Dev EER (validation error rate)
- Dev t-DCF (detection cost)

**Example output:**
```
Start training epoch000
Scores saved to exp_result/LA_AASIST_ep100_bs24/metrics/dev_score.txt
DONE.
Loss:0.63666, dev_eer: 11.961, dev_tdcf:0.29096
Start training epoch001
```

---

### 2. **Count Completed Epochs**
```bash
grep "^epoch:" exp_result/LA_AASIST_ep100_bs24/metric_log.txt | wc -l
```
Returns the number of completed epochs.

**Or view all epoch metrics:**
```bash
cat exp_result/LA_AASIST_ep100_bs24/metric_log.txt
```
Shows detailed metrics for all completed epochs.

---

### 3. **Follow Training in Real-Time**
```bash
tail -f training_output.log
```
Continuously displays new log entries as they appear.
**Press Ctrl+C to stop watching** (training continues unaffected).

---

### 4. **Check Saved Checkpoints**
```bash
ls -lht exp_result/LA_AASIST_ep100_bs24/weights/
```
Lists all saved model checkpoints sorted by time (newest first).

**Example output:**
```
total 5.2M
-rw-rw-r-- 1 lab2208 lab2208 1.3M Nov 14 13:45 epoch_23_0.785.pth
-rw-rw-r-- 1 lab2208 lab2208 1.3M Nov 14 13:30 epoch_21_0.902.pth
-rw-rw-r-- 1 lab2208 lab2208 1.3M Nov 14 13:15 epoch_13_0.982.pth
-rw-rw-r-- 1 lab2208 lab2208 1.3M Nov 14 13:45 best.pth
```

Checkpoints are named: `epoch_{N}_{EER}.pth` where:
- **N** = epoch number
- **EER** = validation error rate (lower is better)

---

### 5. **GPU Status**
```bash
nvidia-smi
```
Shows:
- GPU utilization (should be ~90-100% during training)
- Memory usage (should be ~12-14GB)
- Temperature
- Power consumption

**Compact view:**
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
```

---

### 6. **Training Process Status**
```bash
ps aux | grep "python.*main.py" | grep -v grep
```
Confirms training process is still running.

**Example output:**
```
lab2208  1348295  171  2.6 27231232 1733812 ?    Rl   11:59   0:08 python -u main.py --config config/AASIST.conf
```
- High CPU% (>100%) is normal (multi-threading)
- Memory ~1.7GB is expected

---

## 🎯 Using the Training Monitor Script

### Basic Usage
```bash
python monitor_training.py
```
Updates every 30 seconds showing:
- GPU usage
- Latest metrics
- Saved checkpoints
- Estimated progress

### Custom Update Interval
```bash
# Update every 5 minutes (300 seconds)
python monitor_training.py --interval 300

# Update every minute
python monitor_training.py --interval 60
```

**Press Ctrl+C to stop** (training continues).

---

## 📈 Understanding the Metrics

### Loss
- **Starting:** ~0.6-0.8 (high)
- **Target:** <0.02 (low)
- **Trend:** Should decrease steadily

### Dev EER (Equal Error Rate)
- **Starting:** ~10-15% (high)
- **Target:** <1% (benchmark: 0.83%)
- **Best:** Lower is better
- **Trend:** Should decrease rapidly in first 20 epochs, then slowly

### Dev t-DCF (Detection Cost Function)
- **Starting:** ~0.3 (high)
- **Target:** <0.03 (benchmark: 0.0275)
- **Best:** Lower is better

---

## 📊 Current Training Status (as of latest check)

```bash
# Quick one-liner for current status
echo "Epoch:" $(tail -1 training_output.log | grep -oP 'epoch\d+' | grep -oP '\d+' || echo "0") " | GPU: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
```

**Latest observed:**
- **Epoch 0:** Loss: 0.637, Dev EER: 11.961%, Dev t-DCF: 0.291
- **Epoch 1:** Started (in progress)
- **Status:** Training progressing normally ✅

---

## 🕐 Timeline Estimate

### Epoch Timing
- **Epoch 0:** ~3-4 minutes (initial setup overhead)
- **Subsequent epochs:** ~2-3 minutes each
- **Total time:** 100 epochs × 2.5 min = ~250 minutes = **4-5 hours**

**Updated estimate: Training will complete by ~16:00-17:00 EST (4-5pm today)**

### Checkpoint Milestones to Watch For
- **Epoch ~10-15:** First significant checkpoint (EER ~2-5%)
- **Epoch ~20-30:** EER should reach ~1-2%
- **Epoch ~40-50:** EER approaching benchmark (~0.8-1%)
- **Epoch ~90-100:** SWA averaging begins, final model creation

---

## 🚨 Troubleshooting

### If training seems stuck:
```bash
# Check if process is still running
ps aux | grep python.*main.py

# Check GPU usage (should be >80%)
nvidia-smi

# Check last modified time of log
ls -lh training_output.log
```

### If no progress for >10 minutes:
The training may have crashed. Check:
```bash
tail -50 training_output.log  # Look for error messages
```

### Common issues:
- **Out of memory:** Reduce batch_size in config (currently 24)
- **CUDA error:** Check GPU with `nvidia-smi`
- **Dataset error:** Verify dataset path in config

---

## 📁 Key Files to Monitor

| File | Purpose | How to View |
|------|---------|-------------|
| `training_output.log` | Live training output | `tail -f training_output.log` |
| `exp_result/LA_AASIST_ep100_bs24/metric_log.txt` | Epoch metrics log | `cat metric_log.txt` |
| `exp_result/LA_AASIST_ep100_bs24/weights/` | Saved checkpoints | `ls -lh weights/` |
| `exp_result/LA_AASIST_ep100_bs24/metrics/` | Evaluation scores | `ls metrics/` |

---

## ✅ What to Look For

### Good Signs ✅
- GPU utilization >80%
- Loss decreasing over epochs
- Dev EER decreasing (especially first 30 epochs)
- New checkpoint files appearing
- Log file size increasing

### Warning Signs ⚠️
- GPU utilization <50%
- Loss increasing or not changing
- No new log entries for >5 minutes
- Python process not in `ps aux` output

---

## 🎯 Quick Reference Commands

```bash
# Monitor training live
tail -f training_output.log

# Count completed epochs
grep "epoch:" exp_result/LA_AASIST_ep100_bs24/metric_log.txt | wc -l

# Best checkpoint so far
ls -lht exp_result/LA_AASIST_ep100_bs24/weights/ | head -3

# GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Training process CPU usage
ps aux | grep "python.*main.py" | grep -v grep | awk '{print $3 "% CPU, " $4 "% MEM"}'

# Latest metrics
tail -5 training_output.log
```

---

## 📞 Next Steps After Training Completes

When you see **"Start final evaluation"** in the log:
1. Wait for SWA model creation (`swa.pth`)
2. Training will exit automatically
3. Run comprehensive evaluation (see TRAINING_STATUS.md)
4. Launch Gradio GUI for testing

---

**Last Updated:** 2025-11-14 12:05 EST
**Current Status:** ✅ Epoch 1 in progress
**All monitoring is non-invasive and safe during training!**
