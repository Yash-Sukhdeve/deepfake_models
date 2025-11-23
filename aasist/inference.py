#!/usr/bin/env python3
"""
AASIST Inference Script for Custom Audio Files
Supports any audio format, any length (with chunking), batch processing.

Usage:
    python inference.py --model weights/best.pth --audio sample.wav
    python inference.py --model weights/best.pth --audio_dir samples/ --batch
    python inference.py --model weights/best.pth --audio sample.mp3 --visualize
"""
import argparse
import json
import sys
import warnings
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


def load_model(model_path: str, device: str = "cuda") -> nn.Module:
    """
    Load AASIST model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pth file)
        device: 'cuda' or 'cpu'

    Returns:
        Loaded model in eval mode
    """
    # Default AASIST architecture config
    model_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    # Load model architecture
    module = import_module(f"models.{model_config['architecture']}")
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f"✓ Model loaded: {nb_params:,} parameters")

    return model


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file (supports WAV, FLAC, MP3, OGG, M4A, etc.)
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        Audio waveform as numpy array
    """
    try:
        audio, sr = sf.read(audio_path, dtype='float32')

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            try:
                import resampy
                audio = resampy.resample(audio, sr, target_sr)
            except ImportError:
                # Fallback: simple linear interpolation
                from scipy import signal
                num_samples = int(len(audio) * target_sr / sr)
                audio = signal.resample(audio, num_samples)

        return audio

    except Exception as e:
        print(f"✗ Error loading {audio_path}: {e}")
        return None


def pad_audio(audio: np.ndarray, target_length: int = 64600) -> np.ndarray:
    """
    Pad or crop audio to target length.

    Args:
        audio: Audio waveform
        target_length: Target length in samples (default: 64600 = 4.0375 seconds at 16kHz)

    Returns:
        Padded/cropped audio
    """
    if len(audio) >= target_length:
        # Crop from center
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    else:
        # Pad with zeros
        pad_width = target_length - len(audio)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')


def chunk_audio(audio: np.ndarray, chunk_length: int = 64600, overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split long audio into overlapping chunks.

    Args:
        audio: Audio waveform
        chunk_length: Length of each chunk (default: 64600 samples)
        overlap: Overlap ratio between chunks (default: 0.5 = 50%)

    Returns:
        List of audio chunks
    """
    if len(audio) <= chunk_length:
        return [pad_audio(audio, chunk_length)]

    chunks = []
    step_size = int(chunk_length * (1 - overlap))

    for start in range(0, len(audio) - chunk_length + 1, step_size):
        chunk = audio[start:start + chunk_length]
        chunks.append(chunk)

    # Handle last chunk if there's remaining audio
    if len(audio) - (start + chunk_length) > chunk_length * 0.2:  # If >20% remains
        last_chunk = audio[-chunk_length:]
        chunks.append(last_chunk)

    return chunks


def predict(model: nn.Module, audio: np.ndarray, device: str = "cuda") -> Tuple[str, float, Dict]:
    """
    Predict if audio is bonafide or spoof.

    Args:
        model: Loaded AASIST model
        audio: Audio waveform (can be any length)
        device: 'cuda' or 'cpu'

    Returns:
        (prediction_label, confidence_score, detailed_scores)
    """
    # Handle long audio with chunking
    chunks = chunk_audio(audio)

    chunk_predictions = []
    chunk_scores = []

    with torch.no_grad():
        for chunk in chunks:
            # Prepare input
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)  # (1, 64600)

            # Forward pass
            _, output = model(chunk_tensor)  # output shape: (1, 2)

            # Get probabilities
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # [bonafide_prob, spoof_prob]
            score = probs[1]  # Spoof score

            chunk_scores.append(score)
            chunk_predictions.append("spoof" if score > 0.5 else "bonafide")

    # Aggregate predictions
    avg_score = np.mean(chunk_scores)
    std_score = np.std(chunk_scores) if len(chunk_scores) > 1 else 0.0
    final_prediction = "spoof" if avg_score > 0.5 else "bonafide"
    confidence = max(avg_score, 1 - avg_score)

    detailed = {
        "prediction": final_prediction,
        "spoof_score": float(avg_score),
        "bonafide_score": float(1 - avg_score),
        "confidence": float(confidence),
        "num_chunks": len(chunks),
        "chunk_scores": [float(s) for s in chunk_scores],
        "score_std": float(std_score)
    }

    return final_prediction, confidence, detailed


def predict_batch(model: nn.Module, audio_paths: List[str], device: str = "cuda") -> List[Dict]:
    """
    Batch prediction for multiple audio files.

    Args:
        model: Loaded AASIST model
        audio_paths: List of audio file paths
        device: 'cuda' or 'cpu'

    Returns:
        List of prediction results
    """
    results = []

    print(f"\nProcessing {len(audio_paths)} audio files...\n")

    for i, audio_path in enumerate(audio_paths, 1):
        print(f"[{i}/{len(audio_paths)}] {Path(audio_path).name}...", end=" ")

        audio = load_audio(audio_path)
        if audio is None:
            results.append({
                "file": str(audio_path),
                "error": "Failed to load audio",
                "prediction": None
            })
            print("✗ FAILED")
            continue

        prediction, confidence, detailed = predict(model, audio, device)

        results.append({
            "file": str(audio_path),
            "prediction": prediction,
            "confidence": confidence,
            **detailed
        })

        print(f"✓ {prediction.upper()} ({confidence*100:.1f}%)")

    return results


def visualize_prediction(audio_path: str, audio: np.ndarray, prediction: str,
                        detailed: Dict, output_dir: str = "inference_results"):
    """
    Create visualization of audio and prediction results.

    Args:
        audio_path: Path to audio file
        audio: Audio waveform
        prediction: Prediction label
        detailed: Detailed prediction scores
        output_dir: Output directory for visualizations
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # Audio waveform
        ax1 = fig.add_subplot(gs[0, :])
        time = np.arange(len(audio)) / 16000
        ax1.plot(time, audio, linewidth=0.5, color='steelblue')
        ax1.set_title(f"Audio Waveform: {Path(audio_path).name}", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # Spectrogram
        ax2 = fig.add_subplot(gs[1, :])
        from scipy import signal as scipy_signal
        f, t, Sxx = scipy_signal.spectrogram(audio, fs=16000, nperseg=512)
        ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax2.set_title("Spectrogram", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylim([0, 8000])

        # Prediction bar chart
        ax3 = fig.add_subplot(gs[2, 0])
        labels = ['Bonafide', 'Spoof']
        scores = [detailed['bonafide_score'], detailed['spoof_score']]
        colors = ['#2ecc71' if prediction == 'bonafide' else '#95a5a6',
                  '#e74c3c' if prediction == 'spoof' else '#95a5a6']
        bars = ax3.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_ylim([0, 1])
        ax3.set_ylabel("Score")
        ax3.set_title("Prediction Scores", fontsize=12, fontweight='bold')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Chunk scores (if multiple chunks)
        ax4 = fig.add_subplot(gs[2, 1])
        if len(detailed['chunk_scores']) > 1:
            chunks_idx = np.arange(1, len(detailed['chunk_scores']) + 1)
            ax4.plot(chunks_idx, detailed['chunk_scores'], marker='o', linestyle='-',
                    linewidth=2, markersize=8, color='steelblue')
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax4.set_xlabel("Chunk Index")
            ax4.set_ylabel("Spoof Score")
            ax4.set_title(f"Per-Chunk Scores (n={len(detailed['chunk_scores'])})",
                         fontsize=12, fontweight='bold')
            ax4.set_ylim([0, 1])
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, f"Single chunk audio\n({len(audio)/16000:.2f}s)",
                    ha='center', va='center', fontsize=14, transform=ax4.transAxes)
            ax4.axis('off')

        # Overall result text
        result_text = f"""
PREDICTION: {prediction.upper()}
Confidence: {detailed['confidence']*100:.2f}%
Spoof Score: {detailed['spoof_score']:.4f}
Bonafide Score: {detailed['bonafide_score']:.4f}
Duration: {len(audio)/16000:.2f}s
        """
        fig.text(0.99, 0.01, result_text.strip(),
                fontsize=11, fontfamily='monospace',
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_file = output_path / f"{Path(audio_path).stem}_prediction.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved: {output_file}")

    except ImportError as e:
        print(f"✗ Visualization requires matplotlib and scipy: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AASIST Inference for Deepfake Audio Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type=str, required=True,
                       help="Path to model checkpoint (.pth file)")
    parser.add_argument("--audio", type=str,
                       help="Path to single audio file")
    parser.add_argument("--audio_dir", type=str,
                       help="Path to directory containing audio files")
    parser.add_argument("--batch", action="store_true",
                       help="Enable batch processing for directory")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--output", type=str, default="inference_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use (default: cuda)")
    parser.add_argument("--save_json", action="store_true",
                       help="Save results to JSON file")

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = "cpu"

    print("="*70)
    print("AASIST Deepfake Audio Detection - Inference")
    print("="*70)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model, args.device)

    # Collect audio files
    audio_files = []
    if args.audio:
        audio_files = [args.audio]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a', '.opus']
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))
            audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
        audio_files = sorted(set(audio_files))
    else:
        print("✗ Error: Provide either --audio or --audio_dir")
        sys.exit(1)

    if not audio_files:
        print(f"✗ No audio files found!")
        sys.exit(1)

    # Run inference
    if len(audio_files) == 1 and not args.batch:
        # Single file prediction with detailed output
        audio_path = audio_files[0]
        print(f"\nProcessing: {audio_path}")

        audio = load_audio(audio_path)
        if audio is None:
            sys.exit(1)

        print(f"✓ Audio loaded: {len(audio)/16000:.2f}s, {len(audio)} samples")

        prediction, confidence, detailed = predict(model, audio, args.device)

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Prediction:      {prediction.upper()}")
        print(f"Confidence:      {confidence*100:.2f}%")
        print(f"Spoof Score:     {detailed['spoof_score']:.4f}")
        print(f"Bonafide Score:  {detailed['bonafide_score']:.4f}")
        print(f"Num Chunks:      {detailed['num_chunks']}")
        if detailed['num_chunks'] > 1:
            print(f"Score StdDev:    {detailed['score_std']:.4f}")
        print("="*70)

        # Visualization
        if args.visualize:
            print("\nGenerating visualization...")
            visualize_prediction(audio_path, audio, prediction, detailed, args.output)

        # Save JSON
        if args.save_json:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            json_file = output_path / f"{Path(audio_path).stem}_result.json"
            with open(json_file, 'w') as f:
                json.dump({"file": str(audio_path), **detailed}, f, indent=2)
            print(f"✓ Results saved: {json_file}")

    else:
        # Batch prediction
        results = predict_batch(model, audio_files, args.device)

        # Print summary
        print("\n" + "="*70)
        print("BATCH RESULTS SUMMARY")
        print("="*70)
        bonafide_count = sum(1 for r in results if r.get('prediction') == 'bonafide')
        spoof_count = sum(1 for r in results if r.get('prediction') == 'spoof')
        error_count = sum(1 for r in results if r.get('error'))

        print(f"Total files:   {len(results)}")
        print(f"Bonafide:      {bonafide_count}")
        print(f"Spoof:         {spoof_count}")
        print(f"Errors:        {error_count}")
        print("="*70)

        # Save results
        if args.save_json:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            json_file = output_path / "batch_results.json"
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved: {json_file}")

            # Save CSV
            csv_file = output_path / "batch_results.csv"
            with open(csv_file, 'w') as f:
                f.write("filename,prediction,confidence,spoof_score,bonafide_score,num_chunks\n")
                for r in results:
                    if r.get('error'):
                        f.write(f"{Path(r['file']).name},ERROR,0,0,0,0\n")
                    else:
                        f.write(f"{Path(r['file']).name},{r['prediction']},{r['confidence']:.4f},"
                               f"{r['spoof_score']:.4f},{r['bonafide_score']:.4f},{r['num_chunks']}\n")
            print(f"✓ CSV saved: {csv_file}")


if __name__ == "__main__":
    main()
