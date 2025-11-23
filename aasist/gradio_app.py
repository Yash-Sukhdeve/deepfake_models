#!/usr/bin/env python3
"""
AASIST Gradio Web Interface for Deepfake Audio Detection
Interactive GUI with visualizations, model selector, and example audio.

Usage:
    python gradio_app.py
    python gradio_app.py --share  # Create public link
    python gradio_app.py --port 7860
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Import inference functions
from inference import load_model, load_audio, predict

warnings.filterwarnings("ignore")

# Global model cache
MODEL_CACHE = {}


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

    # Add value labels on bars
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


def analyze_audio(audio_input, model_selection, device_selection="cuda"):
    """
    Main analysis function called by Gradio interface.

    Args:
        audio_input: Tuple of (sample_rate, audio_array) from gr.Audio
        model_selection: Selected model path
        device_selection: Device to use (cuda/cpu)

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

            # Normalize if needed
            if audio.max() > 1.0:
                audio = audio / 32768.0

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

        elif isinstance(audio_input, str):
            # File path provided
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


def create_interface():
    """Create and configure Gradio interface."""

    # Find available models
    model_dir = Path("exp_result")
    model_files = []

    # Check for trained models
    if model_dir.exists():
        for model_path in model_dir.rglob("*.pth"):
            model_files.append(str(model_path))

    # Check for pretrained models
    pretrained_dir = Path("models/weights")
    if pretrained_dir.exists():
        for model_path in pretrained_dir.glob("*.pth"):
            model_files.append(str(model_path))

    if not model_files:
        model_files = ["models/weights/AASIST.pth"]  # Default

    model_files = sorted(set(model_files))

    # Device selection
    device_options = ["cuda"] if torch.cuda.is_available() else ["cpu"]

    # Create interface
    with gr.Blocks(title="AASIST Deepfake Audio Detection", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🎙️ AASIST: Deepfake Audio Detection System

        **Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks**

        Upload an audio file or record your voice to detect if it's genuine (bonafide) or spoofed (fake).
        Supports all common audio formats and any duration.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")

                audio_input = gr.Audio(
                    label="Upload Audio or Record",
                    type="numpy",
                    sources=["upload", "microphone"]
                )

                model_dropdown = gr.Dropdown(
                    choices=model_files,
                    value=model_files[0] if model_files else None,
                    label="🤖 Select Model",
                    info="Choose trained or pretrained model"
                )

                device_dropdown = gr.Dropdown(
                    choices=device_options,
                    value=device_options[0],
                    label="⚙️ Device",
                    info="Processing device"
                )

                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")

                gr.Markdown("""
                ---
                ### 📊 Example Audio Files
                Try the examples below:
                """)

                # Add examples if available
                example_dir = Path("examples")
                if example_dir.exists():
                    examples = [[str(f)] for f in example_dir.glob("*.wav")]
                    if examples:
                        gr.Examples(
                            examples=examples,
                            inputs=[audio_input],
                            label="Example Audio"
                        )

            with gr.Column(scale=2):
                gr.Markdown("### 📋 Results")

                result_output = gr.Markdown(value="*Upload audio and click 'Analyze' to see results.*")

                gr.Markdown("### 📈 Visualizations")

                with gr.Tab("Prediction"):
                    prediction_plot = gr.Plot(label="Detection Scores")

                with gr.Tab("Waveform"):
                    waveform_plot = gr.Plot(label="Audio Waveform")

                with gr.Tab("Spectrogram"):
                    spectrogram_plot = gr.Plot(label="Frequency Spectrum")

                with gr.Tab("Chunk Analysis"):
                    chunk_plot = gr.Plot(label="Per-Chunk Scores (for long audio)")

        # Connect button to analysis function
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[audio_input, model_dropdown, device_dropdown],
            outputs=[result_output, waveform_plot, spectrogram_plot, prediction_plot, chunk_plot]
        )

        gr.Markdown("""
        ---
        ### ℹ️ About

        **AASIST** (Jung et al., 2022) uses Graph Attention Networks to detect spoofed audio from:
        - Text-to-Speech (TTS) synthesis
        - Voice Conversion (VC)
        - Replay attacks

        **Scientific Reference:**
        > Jung, J. W., Heo, H. S., Tak, H., Shim, H. J., Chung, J. S., Lee, B. J., ... & Evans, N. (2022).
        > AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.
        > *IEEE/ACM Transactions on Audio, Speech, and Language Processing*.

        **Benchmark Performance (ASVspoof2019 LA):**
        - EER: 0.83%
        - min t-DCF: 0.0275

        ---
        🔬 **Research-grade detection** | 🚀 **Real-time inference** | 🎯 **State-of-the-art accuracy**
        """)

    return interface


def main():
    parser = argparse.ArgumentParser(description="AASIST Gradio Web Interface")
    parser.add_argument("--share", action="store_true",
                       help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run server on")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server name (use 0.0.0.0 for external access)")
    args = parser.parse_args()

    print("="*70)
    print("AASIST Deepfake Audio Detection - Gradio Interface")
    print("="*70)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create and launch interface
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
