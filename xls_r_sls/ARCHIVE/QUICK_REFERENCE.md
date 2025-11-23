# XLS-R + SLS Quick Reference Card

## 🚀 Current Status
**TRAINING COMPLETE** ✅ - Early stopping after 4 epochs (70 minutes) - Ready for evaluation

## ⚡ Quick Commands

### Monitor Training
```bash
python /home/lab2208/Documents/deepfake_models/xls_r_sls/monitor_training.py
```

### View Recent Training Output
```bash
tail -50 /home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF/training_output_LA.log
```

### Check Training Process
```bash
ps aux | grep train_LA | grep -v grep
```

### Activate Environment
```bash
source ~/anaconda3/bin/activate xls_r_test
```

## 📁 Key Locations
- **Project Root**: `/home/lab2208/Documents/deepfake_models/xls_r_sls/`
- **Repository**: `SLSforASVspoof-2021-DF/`
- **Training Log**: `SLSforASVspoof-2021-DF/training_output_LA.log`
- **State Save**: `PROJECT_STATE.md` (complete project state)
- **Monitor**: `monitor_training.py`

## 🎯 Target Metrics
- **ASVspoof 2021 LA**: 2.87% EER (paper benchmark)
- **Model**: XLS-R 300M + SLS (340M parameters)
- **Training**: 50 epochs on ASVspoof 2019 LA

## 📊 Training Results
- Epochs Completed: 4 (early stopping)
- Best Model: epoch_2.pth
- Duration: 70 minutes
- Early Stopping: Triggered after epoch 3

## 🔧 Troubleshooting
- Environment won't activate? Check: `conda env list`
- Training stopped? Check log and restart with `nohup ./train_LA.sh &`
- Out of memory? Reduce batch_size in `train_LA.sh`

## 📞 Next Steps
1. ✅ Training complete (early stopping)
2. ➡️ Evaluate best model (epoch_2.pth) on ASVspoof 2021 LA
3. Compare EER results with target (2.87%)
4. Analyze if early stopping was appropriate
5. Generate analysis report

---
**Last Updated**: 2025-11-18 (Post-Training)
