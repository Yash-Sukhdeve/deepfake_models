# Gradio Multi-Tab Interface - Quick Start

## One-Line Launch

```bash
cd /home/lab2208/Documents/deepfake_models/aasist && python gradio_app_multitab.py
```

Then open: http://127.0.0.1:7860

## 6 Tabs at a Glance

| Tab | Icon | Purpose | Input | Output |
|-----|------|---------|-------|--------|
| **Single Detection** | 🎙️ | Analyze one audio file | Upload/record audio | Prediction + 4 visualizations |
| **Batch Processing** | 📦 | Process multiple files | Multiple audio files | CSV results + statistics |
| **Model Comparison** | ⚖️ | Compare 2 models | Audio + 2 model paths | Side-by-side comparison |
| **Training Monitor** | 📈 | Monitor training progress | Log file path | Status + progress bar |
| **Dataset Explorer** | 🗂️ | Browse ASVspoof data | Dataset path + filters | Statistics + samples |
| **Evaluation** | 🔬 | Benchmark metrics | Model + protocol | EER/tDCF (WIP) |

## Common Commands

### Launch Options
```bash
# Local only (default)
python gradio_app_multitab.py

# Public share link
python gradio_app_multitab.py --share

# Custom port
python gradio_app_multitab.py --port 8080

# External access (0.0.0.0)
python gradio_app_multitab.py --server_name 0.0.0.0
```

### Quick Test
```bash
# Test interface creation without launching
python -c "from gradio_app_multitab import create_interface; create_interface(); print('✓ OK')"
```

## Tab 1: Single Detection

**Fastest Way**:
1. Upload audio → 2. Click "Analyze" → 3. See results

**Model Paths** (auto-detected):
- `exp_result/LA_AASIST_ep100_bs24/weights/best.pth` (AASIST trained)
- `../xls_r_sls/SLSforASVspoof-2021-DF/models/model_LA_*/epoch_2.pth` (XLS-R+SLS)

## Tab 2: Batch Processing

**Workflow**:
1. Drag & drop multiple audio files
2. Select model
3. Click "Process Batch"
4. Download CSV from `batch_results/batch_results_TIMESTAMP.csv`

**CSV Columns**: filename, prediction, confidence, spoof_score, bonafide_score, duration

## Tab 3: Model Comparison

**Use When**:
- Testing AASIST vs XLS-R+SLS
- Ensemble decisions (if models agree → high confidence)
- Model validation

**Outputs**:
- ✅ Agreement or ⚠️ Disagreement
- Score differences
- Side-by-side bar chart

## Tab 4: Training Monitor

**Default Path**: `/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF/training_output_LA.log`

**Shows**:
- Training status (running/completed/early stopped)
- Progress: batch X/Y (Z%)
- Speed: iterations/second
- ETA: estimated time remaining

## Tab 5: Dataset Explorer

**Default Path**: `/home/lab2208/Documents/deepfake_models/data/asvspoof/asvspoof2019/LA/ASVspoof2019_LA_train`

**Filters**:
- Attack Type: All, -, A07-A19
- Label: All, bonafide, spoof

**Outputs**:
- Total samples, bonafide/spoof counts
- Per-attack distributions
- Sample table (first 100)

## Tab 6: Scientific Evaluation

**Status**: 🚧 Coming Soon

**Planned**: Full EER/tDCF evaluation on ASVspoof 2021

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `cd aasist` before running |
| Model not found | Check paths in dropdowns |
| CUDA OOM | Switch to CPU or use AASIST-L |
| Audio not loading | Install `pip install librosa` |
| Port in use | Use `--port 8080` |

## Performance

| Hardware | Speed (single file) | Speed (100 files) |
|----------|---------------------|-------------------|
| RTX 4080 | 0.2-0.5s | 20-50s |
| CPU (i7) | 2-5s | 200-500s |

## File Structure

```
aasist/
├── gradio_app_multitab.py          # Multi-tab interface (NEW)
├── gradio_app.py                   # Original single-tab interface
├── inference.py                    # Backend functions
├── GRADIO_MULTITAB_GUIDE.md        # Full documentation
└── GRADIO_QUICK_START.md           # This file
```

## Next Steps

1. **Test Single Detection**: Upload a sample audio
2. **Try Batch Processing**: Process multiple files
3. **Compare Models**: See AASIST vs XLS-R+SLS differences
4. **Monitor Training**: Check XLS-R+SLS training status
5. **Explore Dataset**: Browse ASVspoof samples

## Support

- **Full Guide**: `GRADIO_MULTITAB_GUIDE.md`
- **Main README**: `../README.md`
- **Issues**: Check console output for errors

---

**Version**: 1.0.0 | **Last Updated**: 2025-11-18
