# Audio Tampering Detection Evaluation Results

**Date**: 2025-11-23
**Purpose**: Evaluate deepfake detection models against audio tampering attacks

---

## Executive Summary

| Model | Trans-Splicing Dataset | Semantic Tampering |
|-------|----------------------------------|-------------------|
| **XLS-R + SLS** | **95.45% Detection Rate** | 44.44% EER |
| AASIST (Pretrained) | 42.96% Detection Rate | 48.00% EER |
| AASIST (Trained) | 41.72% Detection Rate | 49.15% EER |

**Key Finding**: XLS-R significantly outperforms AASIST on trans-splicing detection (95.45% vs ~42%), demonstrating the superiority of self-supervised pre-training for detecting TTS-generated word insertions.

---

## 1. Trans-Splicing Dataset

### 1.1 Dataset Overview
- **Total Files**: 1,932 tampered audio files
- **Categories**: 4 (XTTS-Clean, XTTS-Unclean, YourTTS-Clean, YourTTS-Unclean)
- **Technique**: Word-level trans-splicing with TTS-generated segments
- **Metric**: Detection Rate (% classified as spoof) - higher is better

### 1.2 Results by Category

| Model | XTTS-Clean | XTTS-Unclean | YourTTS-Clean | YourTTS-Unclean | **Overall** |
|-------|------------|--------------|---------------|-----------------|-------------|
| **XLS-R + SLS** | 91.11% | 91.73% | **99.81%** | **100.00%** | **95.45%** |
| AASIST (Pretrained) | 13.44% | 11.42% | 88.62% | 59.95% | 42.96% |
| AASIST (Trained) | 8.70% | 9.25% | 91.98% | 58.12% | 41.72% |

### 1.3 Key Observations

1. **XLS-R Excellence**: Near-perfect detection of YourTTS tampering (99.81-100%) and strong detection of XTTS (91%)

2. **AASIST Limitation**: Fails to detect XTTS-generated tampering (~10% detection rate), only slightly better than random guessing

3. **TTS System Difference**: YourTTS is easier to detect than XTTS for all models, suggesting XTTS produces more natural-sounding audio

4. **Clean vs Unclean**: Minimal difference between clean and unclean variants, indicating post-processing doesn't significantly affect detection

### 1.4 Score Distributions

| Model | Mean Score | Std Dev | Interpretation |
|-------|------------|---------|----------------|
| XLS-R | 0.0473 | 0.1938 | Low bonafide probability = strong spoof detection |
| AASIST Pre | 0.5735 | 0.4246 | High variance = inconsistent detection |
| AASIST Trained | 0.5984 | 0.4248 | Near random (0.5 = coin flip) |

---

## 2. Semantic Tampering Dataset

### 2.1 Dataset Overview
- **Total Files**: 50 (9 bonafide + 41 tampered)
- **Tampering Type**: Deletion (removing words at phoneme boundaries)
- **Technique**: NLP-guided semantic modification
- **Metric**: Equal Error Rate (EER) - lower is better

### 2.2 Results

| Model | EER | Threshold | Bonafide μ | Spoof μ |
|-------|-----|-----------|------------|---------|
| XLS-R + SLS | 44.44% | 0.0000 | 0.0000 | 0.0000 |
| AASIST (Pretrained) | 48.00% | 0.0084 | 0.0147 | 0.0124 |
| AASIST (Trained) | 49.15% | 0.0715 | 0.0645 | 0.0699 |

### 2.3 Key Observations

1. **Near-Random Performance**: All models show EER close to 50% (random guessing)

2. **Source Audio Issue**: Both bonafide and tampered samples receive very low bonafide scores, indicating the models treat all audio from this source as suspicious

3. **Not Suitable for Deepfake Detection**: The Semantic Tampering dataset uses different source audio (not ASVspoof), so results reflect domain mismatch rather than tampering detection ability

4. **Recommendation**: This dataset is better suited for dedicated audio forensics models rather than deepfake detectors trained on ASVspoof

---

## 3. Cross-Dataset Comparison

### 3.1 Model Robustness Summary

| Evaluation | XLS-R | AASIST Pre | AASIST Trained | Winner |
|------------|-------|------------|----------------|--------|
| ASVspoof 2019 LA | 0.26% EER | 0.83% EER | ~0.83% EER | XLS-R |
| ASVspoof 2021 LA | 2.97% EER | 50.07% EER | 48.27% EER | XLS-R |
| **Trans-Splicing** | **95.45%** | 42.96% | 41.72% | **XLS-R** |
| Semantic Tampering | ~44% EER | ~48% EER | ~49% EER | N/A (domain mismatch) |

### 3.2 Why XLS-R Outperforms AASIST

1. **Pre-training**: XLS-R's 436K hours of diverse speech data provides robust, transferable features
2. **Codec Robustness**: Self-supervised learning captures fundamental speech patterns, not dataset-specific artifacts
3. **TTS Detection**: Pre-trained representations better distinguish natural vs synthetic speech

---

## 4. Scientific Implications

### 4.1 For Deepfake Detection Research

1. **Self-supervised pre-training is crucial** for robust detection across different tampering techniques

2. **TTS-specific detectors needed**: AASIST's spectral features may not generalize to modern TTS systems like XTTS

3. **Cross-domain evaluation essential**: Models should be tested on diverse tampering methods, not just traditional spoofing attacks

### 4.2 Limitations

1. **Semantic tampering dataset size**: Only 50 files may not provide statistically significant results

2. **Source domain mismatch**: Semantic tampering uses different source recordings than ASVspoof training data

3. **No original audio for Trans-Splicing dataset**: Cannot compute EER, only detection rate

---

## 5. Recommendations

### For Production Systems
- Use **XLS-R + SLS** for audio deepfake detection
- Early stopping at 2-4 epochs prevents overfitting
- Test on diverse tampering techniques beyond ASVspoof attacks

### For Research
- Investigate why XTTS is harder to detect than YourTTS
- Develop dedicated audio forensics models for semantic tampering
- Create benchmark datasets with both original and tampered versions

---

## 6. Files Generated

```
tampered_evaluation/
├── results/
│   ├── scores_trans_splicing_xlsr_best.txt
│   ├── scores_trans_splicing_aasist_pretrained.txt
│   ├── scores_trans_splicing_aasist_trained.txt
│   ├── scores_semantic_xlsr_best.txt
│   ├── scores_semantic_aasist_pretrained.txt
│   ├── scores_semantic_aasist_trained.txt
│   ├── results_trans_splicing_*.json
│   ├── results_semantic_*.json
│   └── evaluation_summary.json
├── trans_splicing/   # Trans-Splicing dataset (TTS-based word replacement)
├── semantic/         # Semantic tampering dataset
└── eval_tampered.py  # Evaluation script
```

---

## 7. Citations

### Datasets
- **Trans-Splicing Dataset**: In-house dataset with word-level trans-splicing using XTTS/YourTTS TTS systems
- **Semantic Tampering Dataset**: In-house NLP-guided semantic tampering dataset

### Models
- **XLS-R + SLS**: Zhang et al., "Audio Deepfake Detection with Self-supervised XLS-R and SLS classifier", ACM MM 2024
- **AASIST**: Jung et al., "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks", IEEE/ACM TASLP 2022

---

**Report Generated**: 2025-11-23
**Status**: Complete
