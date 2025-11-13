# AASIST Evaluation Methodology

Scientific documentation of evaluation metrics and methodology for anti-spoofing research.

## Table of Contents
1. [Evaluation Protocol](#evaluation-protocol)
2. [Metrics Definitions](#metrics-definitions)
3. [Statistical Foundations](#statistical-foundations)
4. [Visualization Methods](#visualization-methods)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Scientific Rigor](#scientific-rigor)

---

## Evaluation Protocol

### Dataset Specifications

#### ASVspoof2019 LA (Logical Access)
- **Training set**: 25,380 utterances (2,580 bonafide, 22,800 spoof)
- **Development set**: 24,844 utterances (2,548 bonafide, 22,296 spoof)
- **Evaluation set**: 71,237 utterances (7,355 bonafide, 63,882 spoof)
- **Attack types**: 13 types (A07-A19)
  - A07-A13: Text-to-Speech (TTS) systems
  - A14-A19: Voice Conversion (VC) systems
- **Sampling rate**: 16 kHz
- **Audio format**: FLAC

#### ASVspoof2021 LA
- **Evaluation set**: 181,566 utterances (148,176 eval subset, 33,390 progress subset)
- **Attack types**: 19 types (A01-A19), superset of 2019
- **New challenges**:
  - More attack algorithms
  - Codec variations (alaw, ulaw, gsm)
  - Transmission conditions (ita_tx, sin_tx, loc_tx)
- **Purpose**: Test generalization to unseen attacks

### Evaluation Procedure

1. **Score Generation**
   - Model outputs: logits for [spoof, bonafide] classes
   - Score extraction: `score = softmax(logits)[1]` (bonafide class probability)
   - Higher scores = more likely bonafide
   - Lower scores = more likely spoof

2. **Threshold Selection**
   - Development set used for threshold tuning
   - EER threshold commonly chosen for deployment
   - Operating point can be adjusted based on application requirements

3. **Metric Computation**
   - Evaluate on held-out test set
   - Compare against baseline systems
   - Report confidence intervals where applicable

---

## Metrics Definitions

### 1. Equal Error Rate (EER)

**Definition**: The error rate where False Acceptance Rate equals False Rejection Rate.

**Mathematical Formulation**:
```
FAR(θ) = #{spoof samples with score ≥ θ} / #{total spoof samples}
FRR(θ) = #{bonafide samples with score < θ} / #{total bonafide samples}
EER = FAR(θ*) = FRR(θ*) where θ* is the EER threshold
```

**Computation**:
1. Compute FAR and FRR for all possible thresholds
2. Find threshold θ* where |FAR(θ) - FRR(θ)| is minimized
3. EER = average of FAR(θ*) and FRR(θ*)

**Interpretation**:
- Lower is better
- Range: [0%, 100%]
- 0% = perfect classification
- 50% = random guessing for balanced classes
- **AASIST benchmark**: 0.83% on ASVspoof2019 LA

**Limitations**:
- Single operating point metric
- Doesn't show performance across all thresholds
- Sensitive to class imbalance
- Should be complemented with DET/ROC curves

### 2. Tandem Detection Cost Function (t-DCF)

**Definition**: Cost function for cascaded CM+ASV systems accounting for application-specific costs and priors.

**Mathematical Formulation**:
```
t-DCF(θ) = C₁ × P_miss,CM(θ) + C₂ × P_fa,CM(θ)

where:
C₁ = P_tar × (C_miss,CM - C_miss,ASV × P_miss,ASV) - P_non × C_fa,ASV × P_fa,ASV
C₂ = C_fa,CM × P_spoof × (1 - P_miss,spoof,ASV)

Normalized: t-DCF_norm(θ) = t-DCF(θ) / min(C₁, C₂)
```

**Parameters** (ASVspoof2019 standard):
- `P_spoof = 0.05` - Prior probability of spoofing attack
- `P_tar = 0.0495` - Prior probability of target speaker
- `P_non = 0.0005` - Prior probability of nontarget speaker
- `C_miss = C_miss,CM = C_miss,ASV = 1` - Cost of false rejection
- `C_fa = C_fa,CM = C_fa,ASV = 10` - Cost of false acceptance

**Interpretation**:
- Lower is better
- Range: [0, ∞]
- t-DCF > 1.0 = worse than no countermeasure
- t-DCF ≈ 0.0 = perfect CM (extremely rare)
- **AASIST benchmark**: 0.0275 on ASVspoof2019 LA

**Advantages**:
- Application-aware (considers ASV system)
- Accounts for cost asymmetry
- Incorporates prior probabilities
- Standard metric in ASVspoof challenges

**Reference**:
Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification", Odyssey 2018.

### 3. Area Under ROC Curve (AUC)

**Definition**: Area under the Receiver Operating Characteristic curve.

**Computation**:
```
AUC = ∫₀¹ TPR(FPR) d(FPR)

where:
TPR (True Positive Rate) = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)
```

**Interpretation**:
- Range: [0, 1]
- 0.5 = random classifier
- 1.0 = perfect classifier
- **AASIST benchmark**: ~0.999 on ASVspoof2019 LA
- Threshold-independent metric
- Represents probability that random bonafide sample scores higher than random spoof sample

**Advantages**:
- Single number summary
- Threshold-independent
- Intuitive probabilistic interpretation

**Limitations**:
- May be overly optimistic for imbalanced datasets
- Doesn't indicate performance at specific operating points
- Less common in anti-spoofing literature (DET curves preferred)

### 4. False Acceptance Rate (FAR)

**Definition**: Proportion of spoof samples incorrectly accepted as bonafide.

**Formula**:
```
FAR(θ) = FP / (FP + TN) = #{spoof with score ≥ θ} / #{total spoof}
```

**Interpretation**:
- Security risk metric
- High FAR = attackers can easily fool system
- Critical for security-sensitive applications

### 5. False Rejection Rate (FRR)

**Definition**: Proportion of bonafide samples incorrectly rejected as spoof.

**Formula**:
```
FRR(θ) = FN / (FN + TP) = #{bonafide with score < θ} / #{total bonafide}
```

**Interpretation**:
- Usability metric
- High FRR = legitimate users frequently rejected
- Critical for user experience

### 6. Per-Attack Type EER

**Definition**: EER computed separately for each spoofing algorithm against all bonafide samples.

**Purpose**:
- Identify model weaknesses
- Guide targeted improvements
- Understand which attacks are hardest to detect
- Essential for research publications

**Analysis**:
- Compare performance across attack types
- Identify systematic failures
- TTS vs VC performance differences
- Known vs unknown attacks (2021 dataset)

---

## Statistical Foundations

### Confusion Matrix

```
                  Predicted
                Spoof  Bonafide
Actual Spoof     TN      FP
       Bonafide  FN      TP
```

### Derived Metrics

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN) = TPR
Specificity = TN / (TN + FP)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Detection Error Tradeoff (DET) Curve

**Definition**: Plot of FRR vs FAR on log-log scale.

**Advantages over ROC**:
- Better for anti-spoofing (low error rates)
- Logarithmic scale shows detail in low-error region
- Standard in speaker recognition and anti-spoofing
- EER is diagonal point on DET curve

**Construction**:
1. Compute (FAR, FRR) pairs for all thresholds
2. Plot on log-log axes
3. Mark EER point where FAR = FRR
4. Compare multiple systems on same plot

---

## Visualization Methods

### 1. DET Curve Specifications

- **Axes**: Log scale (0.01% to 100%)
- **Grid**: Logarithmic, alpha=0.3
- **EER point**: Red marker at diagonal
- **Line**: Solid, width=2
- **Diagonal reference**: Dashed, represents EER line

### 2. Score Distribution Specifications

- **Bins**: 50 (adjustable based on data)
- **Overlay**: Semi-transparent (alpha=0.6)
- **Colors**: Green=bonafide, Red=spoof
- **Threshold**: Vertical black dashed line
- **Y-axis**: Density (normalized histogram)

### 3. Per-Attack Bar Chart

- **Color coding**:
  - Green: EER < overall EER (better)
  - Orange: overall EER < EER < 1.5× overall
  - Red: EER > 1.5× overall (significantly worse)
- **Reference line**: Overall EER (black dashed)
- **Values**: Labeled on top of each bar

### 4. Confusion Matrix

- **Colormap**: Blues (sequential)
- **Annotations**: Absolute counts
- **Normalization**: Optional, specify per row/column/all
- **Labels**: ["Bonafide", "Spoof"]

---

## Interpretation Guidelines

### Expected Performance Ranges

#### Excellent (State-of-the-Art)
- EER < 1%
- min t-DCF < 0.03
- AUC > 0.998

#### Good (Competitive)
- EER: 1-3%
- min t-DCF: 0.03-0.10
- AUC: 0.95-0.998

#### Moderate (Baseline)
- EER: 3-10%
- min t-DCF: 0.10-0.30
- AUC: 0.90-0.95

#### Poor (Needs Improvement)
- EER > 10%
- min t-DCF > 0.30
- AUC < 0.90

### Red Flags

1. **Perfect scores (EER ≈ 0%)**:
   - May indicate data leakage
   - Verify train/test split
   - Check for duplicates

2. **Inconsistent metrics**:
   - Low EER but high t-DCF → Check ASV scores
   - High AUC but high EER → Imbalanced classes or threshold issues

3. **Single attack very high EER**:
   - Systematic failure mode
   - May need targeted data augmentation
   - Check for artifacts in that attack type

4. **Large gap between dev and eval**:
   - Overfitting
   - Distribution shift
   - Need better generalization

### Cross-Dataset Analysis

**ASVspoof2019 → 2021 Performance Drop**:
- Expected: ~2-5% absolute EER increase
- Reasons:
  - Unknown attacks in 2021
  - Codec variations
  - More challenging conditions
- Indicates generalization capability

---

## Scientific Rigor

### Reporting Standards

When publishing results, always report:

1. **Primary metrics**: EER and min t-DCF
2. **Per-attack breakdown**: Essential for reproducibility
3. **Dataset details**: Train/dev/eval splits
4. **Model details**: Architecture, parameters, training setup
5. **Computational requirements**: GPU hours, memory
6. **Code availability**: GitHub repository
7. **Checkpoint availability**: For reproducibility

### Statistical Significance

For comparing systems:
- Report confidence intervals (bootstrap recommended)
- Perform significance tests (McNemar's test, DeLong test for AUC)
- Multiple runs with different random seeds
- Report mean and standard deviation

### Reproducibility Checklist

- [ ] Random seeds fixed and documented
- [ ] Data preprocessing steps documented
- [ ] Model architecture fully specified
- [ ] Hyperparameters listed
- [ ] Training procedure detailed
- [ ] Evaluation protocol followed exactly
- [ ] Results on standard benchmarks included
- [ ] Code and checkpoints shared

### Citation Requirements

Always cite:
1. ASVspoof challenge papers
2. t-DCF methodology paper
3. Dataset papers
4. Baseline/comparison papers

---

## References

1. Nautsch, A., et al. "ASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech." IEEE TASLP, 2021.

2. Kinnunen, T., et al. "t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification." Odyssey 2018.

3. Jung, J., et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." ICASSP 2022.

4. Yamagishi, J., et al. "ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection." ASVspoof 2021 Workshop.

5. Martin, A., et al. "The DET Curve in Assessment of Detection Task Performance." Eurospeech 1997.

---

## Appendix: Formula Definitions

### Binary Classification Metrics

```
Given predictions ŷ and true labels y:

True Positives (TP) = Σ(ŷ=bonafide ∧ y=bonafide)
True Negatives (TN) = Σ(ŷ=spoof ∧ y=spoof)
False Positives (FP) = Σ(ŷ=bonafide ∧ y=spoof)
False Negatives (FN) = Σ(ŷ=spoof ∧ y=bonafide)

FAR = FP / (FP + TN) = FP / N_spoof
FRR = FN / (FN + TP) = FN / N_bonafide
TPR = TP / (TP + FN) = 1 - FRR
FPR = FP / (FP + TN) = FAR
TNR = TN / (TN + FP) = 1 - FAR
```

### EER Computation Algorithm

```python
def compute_eer(bonafide_scores, spoof_scores):
    # Combine scores and labels
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)),
                             np.zeros(len(spoof_scores))])

    # Sort by score
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    # Compute cumulative FAR and FRR
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    # FRR: bonafide samples below threshold
    frr = np.cumsum(sorted_labels) / n_bonafide

    # FAR: spoof samples above threshold
    far = 1 - np.cumsum(1 - sorted_labels) / n_spoof

    # Find EER point
    abs_diff = np.abs(frr - far)
    eer_idx = np.argmin(abs_diff)
    eer = (frr[eer_idx] + far[eer_idx]) / 2
    eer_threshold = sorted_scores[eer_idx]

    return eer, eer_threshold
```

---

*Document Version: 1.0*
*Last Updated: 2025-10-31*
*Author: Research Lab*
