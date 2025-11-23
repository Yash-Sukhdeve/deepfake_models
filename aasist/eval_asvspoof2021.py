#!/usr/bin/env python
"""
Evaluate AASIST models on ASVspoof 2021 LA dataset.
This script adapts AASIST to evaluate on the ASVspoof 2021 dataset.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import numpy as np
from tqdm import tqdm
from importlib import import_module

# Add path for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Dataset_ASVspoof2021_eval(Dataset):
    """Dataset class for ASVspoof 2021 LA evaluation."""

    def __init__(self, database_path, protocol_path, cut=64600):
        self.database_path = database_path
        self.cut = cut
        self.file_list = []

        # Parse protocol file
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # ASVspoof 2021 format: speaker_id file_id ... attack_type key
                    file_id = parts[1]
                    self.file_list.append(file_id)

        print(f"Loaded {len(self.file_list)} files for evaluation")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]

        # Try to load audio file
        audio_path = os.path.join(self.database_path, 'flac', f'{file_id}.flac')

        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            # Return zeros if file can't be read
            audio = np.zeros(self.cut)

        # Pad or cut to fixed length
        if len(audio) < self.cut:
            audio = np.pad(audio, (0, self.cut - len(audio)), 'constant')
        else:
            audio = audio[:self.cut]

        audio = torch.tensor(audio, dtype=torch.float32)

        return audio, file_id


def load_model(model_path, device):
    """Load AASIST model."""
    # Import AASIST model
    model_module = import_module("models.AASIST")

    # Model config from AASIST.conf
    model_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }

    # AASIST Model takes only d_args dict
    model = model_module.Model(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def evaluate(model, dataloader, device, output_file):
    """Run evaluation and save scores."""

    model.eval()
    scores = []
    file_ids = []

    with torch.no_grad():
        for batch_x, batch_file_ids in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)

            # Get model output
            _, batch_out = model(batch_x)

            # Get score (probability of being bonafide)
            # AASIST outputs [batch, 2] where index 1 is bonafide
            batch_score = batch_out[:, 1].cpu().numpy()

            scores.extend(batch_score.tolist())
            file_ids.extend(batch_file_ids)

    # Save scores
    with open(output_file, 'w') as f:
        for file_id, score in zip(file_ids, scores):
            f.write(f"{file_id} {score}\n")

    print(f"Scores saved to {output_file}")
    print(f"Total: {len(scores)} utterances")

    return scores, file_ids


def main():
    parser = argparse.ArgumentParser(description="Evaluate AASIST on ASVspoof 2021")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--database_path', type=str,
                        default='/home/lab2208/Documents/deepfake_models/data/asvspoof/ASVspoof2021_LA_eval/',
                        help='Path to ASVspoof 2021 LA eval data')
    parser.add_argument('--protocol_path', type=str,
                        default='/home/lab2208/Documents/deepfake_models/data/asvspoof/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt',
                        help='Path to protocol file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output score file')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.database_path}")
    dataset = Dataset_ASVspoof2021_eval(
        args.database_path,
        args.protocol_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)

    # Evaluate
    scores, file_ids = evaluate(model, dataloader, device, args.output)


if __name__ == "__main__":
    main()
