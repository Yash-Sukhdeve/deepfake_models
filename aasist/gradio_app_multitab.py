#!/usr/bin/env python3
"""
Comprehensive AASIST Deepfake Detection System - Multi-Tab Interface
6 tabs: Single Detection | Batch Processing | Model Comparison | Training Monitor | Dataset Explorer | Scientific Evaluation

Usage:
    python gradio_app_multitab.py
    python gradio_app_multitab.py --share  # Create public link
    python gradio_app_multitab.py --port 7860
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Import inference functions
from inference import load_model, load_audio, predict, predict_batch

warnings.filterwarnings("ignore")

# Global model cache
MODEL_CACHE = {}
CURRENT_TRAINING_LOG = None


def get_model(model_path: str, device: str = "cuda"):
    """Get model from cache or load it."""
    if model_path not in MODEL_CACHE:
        try:
            MODEL_CACHE[model_path] = load_model(model_path, device)
            print(f"✓ Model loaded: {model_path}")
        except Exception as e:
            print(f"✗ Error loading model {model_path}: {e}")
            return None
    return MODEL_CACHE[model_path]


def create_spectrogram(audio: np.ndarray, sr: int = 16000):
    """Create spectrogram visualization."""
    from scipy import signal

    fig, ax = plt.subplots(figsize=(10, 4))
    f, t, Sxx = signal.spectrogram(audio, fs=sr, nperseg=512)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                        shading='gouraud', cmap='viridis')

    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Audio Spectrogram', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 8000])
    plt.colorbar(pcm, ax=ax, label='Power (dB)')
    plt.tight_layout()
    return fig


def create_waveform_plot(audio: np.ndarray, sr: int = 16000):
    """Create waveform visualization."""
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, linewidth=0.5, color='steelblue', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time[-1]])
    plt.tight_layout()
    return fig


def create_prediction_chart(bonafide_score: float, spoof_score: float, prediction: str):
    """Create bar chart for prediction scores."""
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = ['Bonafide\n(Real)', 'Spoof\n(Fake)']
    scores = [bonafide_score, spoof_score]

    colors = ['#2ecc71' if prediction == 'bonafide' else '#95a5a6',
              '#e74c3c' if prediction == 'spoof' else '#95a5a6']

    bars = ax.bar(labels, scores, color=colors, alpha=0.85, edgecolor='black', linewidth=2.5)

    ax.set_ylim([0, 1])
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_title('Detection Results', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Threshold')
    ax.legend()

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{score:.1%}', ha='center', va='bottom',
               fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_chunk_scores_plot(chunk_scores: list):
    """Create line plot for chunk-wise scores."""
    if len(chunk_scores) <= 1:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    chunks_idx = np.arange(1, len(chunk_scores) + 1)
    ax.plot(chunks_idx, chunk_scores, marker='o', linestyle='-',
           linewidth=2.5, markersize=9, color='steelblue',
           markeredgecolor='darkblue', markeredgewidth=1.5)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Threshold')
    ax.fill_between(chunks_idx, 0.5, chunk_scores,
                    where=np.array(chunk_scores) > 0.5,
                    alpha=0.3, color='red', label='Spoof regions')
    ax.fill_between(chunks_idx, 0.5, chunk_scores,
                    where=np.array(chunk_scores) <= 0.5,
                    alpha=0.3, color='green', label='Bonafide regions')

    ax.set_xlabel('Chunk Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spoof Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Chunk Analysis ({len(chunk_scores)} chunks)',
                fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig


# ============================================================================
# TAB 1: SINGLE AUDIO DETECTION
# ============================================================================

def analyze_audio(audio_input, model_selection, device_selection="cuda"):
    """
    Main analysis function for single audio detection.

    Returns:
        Tuple of (result_text, waveform_plot, spectrogram, prediction_chart, chunk_plot)
    """
    if audio_input is None:
        return ("⚠️ Please upload or record an audio file.", None, None, None, None)

    try:
        # Handle Gradio audio input
        if isinstance(audio_input, tuple):
            sr, audio = audio_input
            audio = audio.astype(np.float32)

            if audio.max() > 1.0:
                audio = audio / 32768.0

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

        elif isinstance(audio_input, str):
            audio = load_audio(audio_input)
            sr = 16000
        else:
            return ("⚠️ Invalid audio input format.", None, None, None, None)

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
            except:
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sr)
                audio = signal.resample(audio, num_samples)
            sr = 16000

        duration = len(audio) / sr

        # Load model
        model = get_model(model_selection, device_selection)
        if model is None:
            return ("⚠️ Failed to load model.", None, None, None, None)

        # Run prediction
        prediction, confidence, detailed = predict(model, audio, device_selection)

        # Create visualizations
        waveform_fig = create_waveform_plot(audio, sr)
        spectrogram_fig = create_spectrogram(audio, sr)
        prediction_fig = create_prediction_chart(
            detailed['bonafide_score'],
            detailed['spoof_score'],
            prediction
        )

        chunk_fig = None
        if detailed['num_chunks'] > 1:
            chunk_fig = create_chunk_scores_plot(detailed['chunk_scores'])

        # Format results
        result_icon = "🟢" if prediction == "bonafide" else "🔴"
        result_text = f"""
# {result_icon} Detection Results

## **Prediction: {prediction.upper()}**

### Confidence Scores:
- **Overall Confidence:** {confidence*100:.2f}%
- **Spoof Score:** {detailed['spoof_score']:.4f} ({detailed['spoof_score']*100:.2f}%)
- **Bonafide Score:** {detailed['bonafide_score']:.4f} ({detailed['bonafide_score']*100:.2f}%)

### Audio Information:
- **Duration:** {duration:.2f} seconds
- **Samples:** {len(audio):,}
- **Sample Rate:** {sr} Hz
- **Number of Chunks:** {detailed['num_chunks']}
"""

        if detailed['num_chunks'] > 1:
            result_text += f"- **Score Std Dev:** {detailed['score_std']:.4f}\n"

        result_text += f"""
---
### Interpretation:
"""

        if prediction == "bonafide":
            result_text += """
✅ **This audio appears to be GENUINE/REAL.**

The model detected natural speech characteristics consistent with human vocal production.
"""
        else:
            result_text += """
⚠️ **This audio appears to be SPOOFED/FAKE.**

The model detected artifacts or patterns consistent with synthesized, converted, or replayed speech.
"""

        if confidence < 0.7:
            result_text += """
⚠️ **Note:** Confidence is relatively low. Consider manual verification.
"""

        return (result_text, waveform_fig, spectrogram_fig, prediction_fig, chunk_fig)

    except Exception as e:
        error_msg = f"⚠️ **Error during analysis:**\n\n```\n{str(e)}\n```"
        return (error_msg, None, None, None, None)


# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================

def process_batch(files_input, model_selection, device_selection="cuda"):
    """
    Process multiple audio files and return results table and summary.

    Returns:
        (summary_text, results_dataframe, distribution_plot, results_csv_file)
    """
    if files_input is None or len(files_input) == 0:
        return ("⚠️ Please upload audio files.", None, None, None)

    try:
        # Load model
        model = get_model(model_selection, device_selection)
        if model is None:
            return ("⚠️ Failed to load model.", None, None, None)

        # Process all files
        results = []
        for file_obj in files_input:
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)

            audio = load_audio(file_path)
            if audio is None:
                results.append({
                    "filename": Path(file_path).name,
                    "prediction": "ERROR",
                    "confidence": 0,
                    "spoof_score": 0,
                    "bonafide_score": 0,
                    "duration": 0
                })
                continue

            prediction, confidence, detailed = predict(model, audio, device_selection)

            results.append({
                "filename": Path(file_path).name,
                "prediction": prediction.upper(),
                "confidence": f"{confidence*100:.2f}%",
                "spoof_score": f"{detailed['spoof_score']:.4f}",
                "bonafide_score": f"{detailed['bonafide_score']:.4f}",
                "duration": f"{len(audio)/16000:.2f}s"
            })

        # Create dataframe
        df = pd.DataFrame(results)

        # Calculate statistics
        total_files = len(results)
        bonafide_count = sum(1 for r in results if r['prediction'] == 'BONAFIDE')
        spoof_count = sum(1 for r in results if r['prediction'] == 'SPOOF')
        error_count = sum(1 for r in results if r['prediction'] == 'ERROR')

        # Summary text
        summary = f"""
# 📦 Batch Processing Results

## Summary Statistics:
- **Total Files Processed:** {total_files}
- **Bonafide (Real):** {bonafide_count} ({bonafide_count/total_files*100:.1f}%)
- **Spoof (Fake):** {spoof_count} ({spoof_count/total_files*100:.1f}%)
- **Errors:** {error_count}

---
### Distribution:
"""

        # Create distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Bonafide', 'Spoof']
        sizes = [bonafide_count, spoof_count]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)

        if sum(sizes) > 0:
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                  autopct='%1.1f%%', shadow=True, startangle=90,
                  textprops={'fontsize': 14, 'fontweight': 'bold'})
            ax.set_title(f'Detection Distribution (n={total_files})',
                        fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save results to CSV
        output_dir = Path("batch_results")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"batch_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        return (summary, df, fig, str(csv_path))

    except Exception as e:
        error_msg = f"⚠️ **Error during batch processing:**\n\n```\n{str(e)}\n```"
        return (error_msg, None, None, None)


# ============================================================================
# TAB 3: MODEL COMPARISON
# ============================================================================

def compare_models(audio_input, model1_path, model2_path, device_selection="cuda"):
    """
    Compare predictions from two models side-by-side.

    Returns:
        (comparison_text, comparison_table, comparison_chart, consensus_text)
    """
    if audio_input is None:
        return ("⚠️ Please upload an audio file.", None, None, "")

    try:
        # Load audio
        if isinstance(audio_input, tuple):
            sr, audio = audio_input
            audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        elif isinstance(audio_input, str):
            audio = load_audio(audio_input)
        else:
            return ("⚠️ Invalid audio input.", None, None, "")

        # Load both models
        model1 = get_model(model1_path, device_selection)
        model2 = get_model(model2_path, device_selection)

        if model1 is None or model2 is None:
            return ("⚠️ Failed to load models.", None, None, "")

        # Get predictions from both models
        pred1, conf1, detailed1 = predict(model1, audio, device_selection)
        pred2, conf2, detailed2 = predict(model2, audio, device_selection)

        # Create comparison table
        comparison_data = {
            "Metric": ["Prediction", "Confidence", "Spoof Score", "Bonafide Score"],
            "Model 1": [
                pred1.upper(),
                f"{conf1*100:.2f}%",
                f"{detailed1['spoof_score']:.4f}",
                f"{detailed1['bonafide_score']:.4f}"
            ],
            "Model 2": [
                pred2.upper(),
                f"{conf2*100:.2f}%",
                f"{detailed2['spoof_score']:.4f}",
                f"{detailed2['bonafide_score']:.4f}"
            ],
            "Agreement": [
                "✅" if pred1 == pred2 else "⚠️",
                f"{abs(conf1-conf2)*100:.2f}% diff",
                f"{abs(detailed1['spoof_score']-detailed2['spoof_score']):.4f} diff",
                f"{abs(detailed1['bonafide_score']-detailed2['bonafide_score']):.4f} diff"
            ]
        }

        df = pd.DataFrame(comparison_data)

        # Create comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ['Bonafide\nScore', 'Spoof\nScore', 'Confidence']
        model1_scores = [detailed1['bonafide_score'], detailed1['spoof_score'], conf1]
        model2_scores = [detailed2['bonafide_score'], detailed2['spoof_score'], conf2]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax.bar(x - width/2, model1_scores, width, label='Model 1',
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, model2_scores, width, label='Model 2',
                      color='coral', alpha=0.8, edgecolor='black')

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Consensus analysis
        consensus = "**AGREEMENT**" if pred1 == pred2 else "**DISAGREEMENT**"
        consensus_icon = "✅" if pred1 == pred2 else "⚠️"

        consensus_text = f"""
# {consensus_icon} Model Consensus: {consensus}

**Model 1:** {pred1.upper()} ({conf1*100:.2f}% confidence)
**Model 2:** {pred2.upper()} ({conf2*100:.2f}% confidence)

**Confidence Difference:** {abs(conf1-conf2)*100:.2f}%
**Score Difference:** {abs(detailed1['spoof_score']-detailed2['spoof_score']):.4f}
"""

        if pred1 != pred2:
            consensus_text += "\n⚠️ **Models disagree!** Manual review recommended."

        comparison_summary = f"""
# ⚖️ Model Comparison Results

**Audio Duration:** {len(audio)/16000:.2f}s

## Model 1: {Path(model1_path).name}
- Prediction: **{pred1.upper()}**
- Confidence: **{conf1*100:.2f}%**

## Model 2: {Path(model2_path).name}
- Prediction: **{pred2.upper()}**
- Confidence: **{conf2*100:.2f}%**

---
"""

        return (comparison_summary, df, fig, consensus_text)

    except Exception as e:
        error_msg = f"⚠️ **Error during comparison:**\n\n```\n{str(e)}\n```"
        return (error_msg, None, None, "")


# ============================================================================
# TAB 4: TRAINING MONITOR
# ============================================================================

def parse_training_log(log_path: str) -> Dict:
    """Parse training log and extract metrics."""
    if not Path(log_path).exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract progress information
    progress_lines = re.findall(r'(\d+)%\|.*?\| (\d+)/(\d+) \[.*?\<(.*?),\s*([\d.]+)it/s\]', content)

    if not progress_lines:
        return {"status": "starting", "message": "Training starting..."}

    latest = progress_lines[-1]
    percent = int(latest[0])
    current = int(latest[1])
    total = int(latest[2])
    eta = latest[3]
    speed = float(latest[4])

    # Check if training completed
    if "TRAINING COMPLETE" in content:
        status = "completed"
    elif "Early stopping" in content:
        status = "early_stopped"
    else:
        status = "running"

    return {
        "status": status,
        "percent": percent,
        "current_batch": current,
        "total_batches": total,
        "eta": eta,
        "speed": speed
    }


def monitor_training(log_path_input, auto_refresh=False):
    """
    Monitor training progress from log file.

    Returns:
        (status_text, progress_plot)
    """
    if not log_path_input or not Path(log_path_input).exists():
        return ("⚠️ Training log not found. Please provide a valid log file path.", None)

    try:
        info = parse_training_log(log_path_input)

        if info is None:
            return ("⚠️ Could not parse training log.", None)

        if info.get('status') == 'starting':
            return (info['message'], None)

        # Create status text
        status_icon = {
            'running': '🔄',
            'completed': '✅',
            'early_stopped': '⏹️'
        }.get(info['status'], '❓')

        status_text = f"""
# {status_icon} Training Status: {info['status'].upper().replace('_', ' ')}

## Progress:
- **Batch:** {info['current_batch']}/{info['total_batches']} ({info['percent']}%)
- **Speed:** {info['speed']:.2f} iterations/second
- **ETA:** {info['eta']}

---
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # Create progress bar visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh([0], [info['percent']], height=0.5, color='steelblue', edgecolor='black', linewidth=2)
        ax.set_xlim([0, 100])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel('Progress (%)', fontsize=12, fontweight='bold')
        ax.set_title(f"Training Progress: {info['percent']}%", fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.text(info['percent']/2, 0, f"{info['percent']}%",
               ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()

        return (status_text, fig)

    except Exception as e:
        return (f"⚠️ Error monitoring training: {str(e)}", None)


# ============================================================================
# TAB 5: DATASET EXPLORER
# ============================================================================

def explore_dataset(dataset_path, attack_type_filter="All", label_filter="All"):
    """
    Browse and explore ASVspoof datasets.

    Returns:
        (statistics_text, sample_table, distribution_plot)
    """
    if not dataset_path or not Path(dataset_path).exists():
        return ("⚠️ Dataset path not found.", None, None)

    try:
        # Find protocol file
        protocol_files = list(Path(dataset_path).parent.glob("**/ASVspoof2019.LA.cm.*.txt"))

        if not protocol_files:
            return ("⚠️ No protocol file found in dataset directory.", None, None)

        protocol_file = protocol_files[0]

        # Parse protocol
        samples = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    samples.append({
                        "utt_id": parts[1],
                        "attack_type": parts[3],
                        "label": parts[4]
                    })

        # Apply filters
        filtered = samples
        if attack_type_filter != "All":
            filtered = [s for s in filtered if s['attack_type'] == attack_type_filter]
        if label_filter != "All":
            filtered = [s for s in filtered if s['label'] == label_filter]

        # Statistics
        total_samples = len(filtered)
        bonafide_count = sum(1 for s in filtered if s['label'] == 'bonafide')
        spoof_count = sum(1 for s in filtered if s['label'] == 'spoof')

        # Attack type distribution
        attack_counts = {}
        for s in filtered:
            attack_counts[s['attack_type']] = attack_counts.get(s['attack_type'], 0) + 1

        stats_text = f"""
# 🗂️ Dataset Explorer

## Statistics:
- **Total Samples:** {total_samples:,}
- **Bonafide:** {bonafide_count:,} ({bonafide_count/max(total_samples,1)*100:.1f}%)
- **Spoof:** {spoof_count:,} ({spoof_count/max(total_samples,1)*100:.1f}%)

## Attack Type Distribution:
"""
        for attack, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
            stats_text += f"- **{attack}**: {count:,} ({count/max(total_samples,1)*100:.1f}%)\n"

        # Create sample table (first 100)
        df = pd.DataFrame(filtered[:100])

        # Create distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Label distribution
        labels = ['Bonafide', 'Spoof']
        sizes = [bonafide_count, spoof_count]
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax1.set_title('Label Distribution', fontsize=14, fontweight='bold')

        # Attack type distribution
        attack_types = list(attack_counts.keys())[:10]  # Top 10
        attack_vals = [attack_counts[a] for a in attack_types]
        ax2.barh(attack_types, attack_vals, color='steelblue', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('Top Attack Types', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        return (stats_text, df, fig)

    except Exception as e:
        return (f"⚠️ Error exploring dataset: {str(e)}", None, None)


# ============================================================================
# TAB 6: SCIENTIFIC EVALUATION
# ============================================================================

def run_scientific_evaluation(model_path, dataset_protocol_path, device_selection="cuda"):
    """
    Run comprehensive scientific evaluation.

    Returns:
        (evaluation_report, metrics_table, det_curve_plot)
    """
    return (
        "⚠️ Scientific evaluation requires full comprehensive_eval.py integration. Coming soon!",
        None,
        None
    )


# ============================================================================
# MAIN INTERFACE CREATION
# ============================================================================

def create_interface():
    """Create comprehensive multi-tab Gradio interface."""

    # Find available models
    model_dir = Path("exp_result")
    model_files = []

    if model_dir.exists():
        for model_path in model_dir.rglob("*.pth"):
            model_files.append(str(model_path))

    pretrained_dir = Path("models/weights")
    if pretrained_dir.exists():
        for model_path in pretrained_dir.glob("*.pth"):
            model_files.append(str(model_path))

    # Add XLS-R+SLS models if available
    xls_r_dir = Path("../xls_r_sls/SLSforASVspoof-2021-DF/models")
    if xls_r_dir.exists():
        for model_path in xls_r_dir.rglob("*.pth"):
            model_files.append(str(model_path))

    if not model_files:
        model_files = ["models/weights/AASIST.pth"]

    model_files = sorted(set(model_files))
    device_options = ["cuda"] if torch.cuda.is_available() else ["cpu"]

    # Create interface with tabs
    with gr.Blocks(title="AASIST Deepfake Detection System", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🎙️ Comprehensive Deepfake Audio Detection System

        **Multi-Model Framework: AASIST + XLS-R+SLS**

        Six integrated modules for complete audio anti-spoofing analysis and research.

        ---
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: SINGLE AUDIO DETECTION
            # ================================================================
            with gr.Tab("🎙️ Single Detection"):
                gr.Markdown("### Upload or record audio to detect if it's genuine or spoofed")

                with gr.Row():
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Upload Audio or Record",
                            type="numpy",
                            sources=["upload", "microphone"]
                        )

                        model_dropdown = gr.Dropdown(
                            choices=model_files,
                            value=model_files[0] if model_files else None,
                            label="🤖 Select Model"
                        )

                        device_dropdown = gr.Dropdown(
                            choices=device_options,
                            value=device_options[0],
                            label="⚙️ Device"
                        )

                        analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        result_output = gr.Markdown(value="*Upload audio and click 'Analyze'*")

                        with gr.Tab("Prediction"):
                            prediction_plot = gr.Plot(label="Detection Scores")

                        with gr.Tab("Waveform"):
                            waveform_plot = gr.Plot(label="Audio Waveform")

                        with gr.Tab("Spectrogram"):
                            spectrogram_plot = gr.Plot(label="Frequency Spectrum")

                        with gr.Tab("Chunk Analysis"):
                            chunk_plot = gr.Plot(label="Per-Chunk Scores")

                analyze_btn.click(
                    fn=analyze_audio,
                    inputs=[audio_input, model_dropdown, device_dropdown],
                    outputs=[result_output, waveform_plot, spectrogram_plot, prediction_plot, chunk_plot]
                )

            # ================================================================
            # TAB 2: BATCH PROCESSING
            # ================================================================
            with gr.Tab("📦 Batch Processing"):
                gr.Markdown("### Process multiple audio files simultaneously")

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="Upload Multiple Audio Files",
                            file_count="multiple",
                            type="file"
                        )

                        batch_model = gr.Dropdown(
                            choices=model_files,
                            value=model_files[0] if model_files else None,
                            label="🤖 Select Model"
                        )

                        batch_device = gr.Dropdown(
                            choices=device_options,
                            value=device_options[0],
                            label="⚙️ Device"
                        )

                        batch_process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        batch_summary = gr.Markdown(value="*Upload files and click 'Process Batch'*")
                        batch_table = gr.Dataframe(label="Detailed Results")
                        batch_plot = gr.Plot(label="Distribution Chart")
                        batch_csv = gr.Textbox(label="Results saved to", interactive=False)

                batch_process_btn.click(
                    fn=process_batch,
                    inputs=[batch_files, batch_model, batch_device],
                    outputs=[batch_summary, batch_table, batch_plot, batch_csv]
                )

            # ================================================================
            # TAB 3: MODEL COMPARISON
            # ================================================================
            with gr.Tab("⚖️ Model Comparison"):
                gr.Markdown("### Compare predictions from different models side-by-side")

                with gr.Row():
                    with gr.Column(scale=1):
                        compare_audio = gr.Audio(
                            label="Upload Audio",
                            type="numpy",
                            sources=["upload", "microphone"]
                        )

                        compare_model1 = gr.Dropdown(
                            choices=model_files,
                            value=model_files[0] if model_files else None,
                            label="🤖 Model 1"
                        )

                        compare_model2 = gr.Dropdown(
                            choices=model_files,
                            value=model_files[1] if len(model_files) > 1 else model_files[0],
                            label="🤖 Model 2"
                        )

                        compare_device = gr.Dropdown(
                            choices=device_options,
                            value=device_options[0],
                            label="⚙️ Device"
                        )

                        compare_btn = gr.Button("⚖️ Compare Models", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        compare_summary = gr.Markdown(value="*Select models and audio to compare*")
                        compare_table = gr.Dataframe(label="Comparison Table")
                        compare_chart = gr.Plot(label="Score Comparison")
                        consensus_output = gr.Markdown(value="")

                compare_btn.click(
                    fn=compare_models,
                    inputs=[compare_audio, compare_model1, compare_model2, compare_device],
                    outputs=[compare_summary, compare_table, compare_chart, consensus_output]
                )

            # ================================================================
            # TAB 4: TRAINING MONITOR
            # ================================================================
            with gr.Tab("📈 Training Monitor"):
                gr.Markdown("### Monitor ongoing training progress")

                with gr.Row():
                    with gr.Column(scale=1):
                        log_path_input = gr.Textbox(
                            label="Training Log Path",
                            placeholder="/path/to/training_output.log",
                            value="/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF/training_output_LA.log"
                        )

                        monitor_btn = gr.Button("🔍 Check Status", variant="primary", size="lg")
                        refresh_checkbox = gr.Checkbox(label="Auto-refresh every 30s", value=False)

                    with gr.Column(scale=2):
                        training_status = gr.Markdown(value="*Enter log path and click 'Check Status'*")
                        training_progress = gr.Plot(label="Progress Bar")

                monitor_btn.click(
                    fn=monitor_training,
                    inputs=[log_path_input, refresh_checkbox],
                    outputs=[training_status, training_progress]
                )

            # ================================================================
            # TAB 5: DATASET EXPLORER
            # ================================================================
            with gr.Tab("🗂️ Dataset Explorer"):
                gr.Markdown("### Browse and analyze ASVspoof datasets")

                with gr.Row():
                    with gr.Column(scale=1):
                        dataset_path_input = gr.Textbox(
                            label="Dataset Directory Path",
                            placeholder="/path/to/ASVspoof2019_LA_train",
                            value="/home/lab2208/Documents/deepfake_models/data/asvspoof/asvspoof2019/LA/ASVspoof2019_LA_train"
                        )

                        attack_filter = gr.Dropdown(
                            choices=["All", "-", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19"],
                            value="All",
                            label="Attack Type Filter"
                        )

                        label_filter = gr.Dropdown(
                            choices=["All", "bonafide", "spoof"],
                            value="All",
                            label="Label Filter"
                        )

                        explore_btn = gr.Button("🔍 Explore Dataset", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        dataset_stats = gr.Markdown(value="*Enter dataset path and click 'Explore'*")
                        dataset_table = gr.Dataframe(label="Sample Data (first 100)")
                        dataset_plot = gr.Plot(label="Distribution Analysis")

                explore_btn.click(
                    fn=explore_dataset,
                    inputs=[dataset_path_input, attack_filter, label_filter],
                    outputs=[dataset_stats, dataset_table, dataset_plot]
                )

            # ================================================================
            # TAB 6: SCIENTIFIC EVALUATION
            # ================================================================
            with gr.Tab("🔬 Scientific Evaluation"):
                gr.Markdown("### Run comprehensive benchmark evaluation (EER, t-DCF, DET curves)")

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_model = gr.Dropdown(
                            choices=model_files,
                            value=model_files[0] if model_files else None,
                            label="🤖 Select Model"
                        )

                        eval_protocol = gr.Textbox(
                            label="Protocol File Path",
                            placeholder="/path/to/ASVspoof2021.LA.cm.eval.trl.txt"
                        )

                        eval_device = gr.Dropdown(
                            choices=device_options,
                            value=device_options[0],
                            label="⚙️ Device"
                        )

                        eval_btn = gr.Button("🚀 Run Evaluation", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        eval_report = gr.Markdown(value="*Configure and run evaluation*")
                        eval_metrics = gr.Dataframe(label="Evaluation Metrics")
                        eval_plot = gr.Plot(label="DET Curve")

                eval_btn.click(
                    fn=run_scientific_evaluation,
                    inputs=[eval_model, eval_protocol, eval_device],
                    outputs=[eval_report, eval_metrics, eval_plot]
                )

        gr.Markdown("""
        ---
        ### ℹ️ About This System

        **AASIST** (Jung et al., 2022) uses Graph Attention Networks for audio anti-spoofing.
        **XLS-R + SLS** (Zhang et al., 2024) uses self-supervised learning with layer selection.

        **Target Benchmarks:**
        - AASIST on ASVspoof2019 LA: 0.83% EER, 0.0275 min-tDCF
        - XLS-R+SLS on ASVspoof2021 LA: 2.87% EER

        ---
        🔬 **Research-grade detection** | 🚀 **Real-time inference** | 🎯 **State-of-the-art accuracy**
        """)

    return interface


def main():
    parser = argparse.ArgumentParser(description="Comprehensive AASIST Multi-Tab Interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server on")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server name (use 0.0.0.0 for external access)")
    args = parser.parse_args()

    print("="*70)
    print("Comprehensive Deepfake Detection System - Multi-Tab Interface")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    interface = create_interface()

    print(f"\n🚀 Launching Gradio interface...")
    print(f"📍 Server: http://{args.server_name}:{args.port}")

    if args.share:
        print("🌐 Creating public share link...")

    interface.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
