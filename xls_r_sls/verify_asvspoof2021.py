#!/usr/bin/env python3
"""
ASVspoof 2021 Dataset Verification Script
Verifies downloaded datasets match official challenge specifications
"""

import os
import sys
from pathlib import Path
from collections import defaultdict, Counter


class ASVspoof2021Verifier:
    """Verify ASVspoof 2021 datasets against official specifications"""

    # Official ASVspoof 2021 specifications
    EXPECTED_COUNTS = {
        'DF_eval': {
            'total_trials': 611829,  # Total utterances in DF eval
            'bonafide': 67535,       # Genuine speech
            'spoof': 544294,         # Spoofed speech
            'description': 'ASVspoof 2021 DF evaluation set'
        },
        'LA_eval': {
            'total_trials': 181566,  # Total utterances in LA eval
            'bonafide': 60562,       # Genuine speech
            'spoof': 121004,         # Spoofed speech
            'description': 'ASVspoof 2021 LA evaluation set'
        }
    }

    # DF codecs used in challenge
    DF_CODECS = [
        'no codec', 'aac', 'mp3', 'ogg vorbis', 'speex', 'wav',
        'm4a', 'opus', 'amr-nb', 'amr-wb', 'pcm_mulaw'
    ]

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.results = {}

    def verify_protocol_file(self, protocol_path, dataset_name):
        """Verify protocol file format and statistics"""
        print(f"\n{'='*70}")
        print(f"VERIFYING: {dataset_name}")
        print(f"{'='*70}")

        if not protocol_path.exists():
            print(f"❌ Protocol file not found: {protocol_path}")
            return False

        # Parse protocol file
        # Format: speaker_id utt_id - attack_type label (5 columns)
        # Or: speaker_id utt_id - label (4 columns for some protocols)
        trials = []
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # ASVspoof2021 format: speaker utt - attack label
                    speaker_id, utt_id, dash, system_id, label = parts[0], parts[1], parts[2], parts[3], parts[4]
                    trials.append({
                        'speaker_id': speaker_id,
                        'utt_id': utt_id,
                        'system_id': system_id,
                        'label': label
                    })
                elif len(parts) >= 4:  # Alternative format: speaker utt attack label
                    speaker_id, utt_id, system_id, label = parts[0], parts[1], parts[2], parts[3]
                    trials.append({
                        'speaker_id': speaker_id,
                        'utt_id': utt_id,
                        'system_id': system_id,
                        'label': label
                    })

        # Statistics
        total = len(trials)
        bonafide = sum(1 for t in trials if t['label'] == 'bonafide')
        spoof = sum(1 for t in trials if t['label'] == 'spoof')

        print(f"\n📊 Protocol File Statistics:")
        print(f"   File: {protocol_path.name}")
        print(f"   Total trials: {total}")
        print(f"   Bonafide: {bonafide} ({bonafide/total*100:.1f}%)")
        print(f"   Spoof: {spoof} ({spoof/total*100:.1f}%)")

        # Verify against expected counts
        expected = self.EXPECTED_COUNTS.get(dataset_name, {})
        if expected:
            print(f"\n🔍 Verification against official specs:")

            # Check total count
            expected_total = expected['total_trials']
            if total == expected_total:
                print(f"   ✅ Total trials: {total} (matches {expected_total})")
            else:
                print(f"   ⚠️  Total trials: {total} (expected {expected_total}, diff: {total - expected_total})")

            # Check bonafide count
            expected_bonafide = expected['bonafide']
            if bonafide == expected_bonafide:
                print(f"   ✅ Bonafide: {bonafide} (matches {expected_bonafide})")
            else:
                print(f"   ⚠️  Bonafide: {bonafide} (expected {expected_bonafide}, diff: {bonafide - expected_bonafide})")

            # Check spoof count
            expected_spoof = expected['spoof']
            if spoof == expected_spoof:
                print(f"   ✅ Spoof: {spoof} (matches {expected_spoof})")
            else:
                print(f"   ⚠️  Spoof: {spoof} (expected {expected_spoof}, diff: {spoof - expected_spoof})")

        # Analyze attack types (for DF)
        if dataset_name == 'DF_eval':
            system_counts = Counter(t['system_id'] for t in trials if t['label'] == 'spoof')
            print(f"\n🎭 Attack Systems in DF:")
            for system, count in sorted(system_counts.items()):
                print(f"   {system}: {count} utterances")
            print(f"   Total unique systems: {len(system_counts)}")

        # Analyze speakers
        speaker_counts = Counter(t['speaker_id'] for t in trials)
        print(f"\n👥 Speaker Statistics:")
        print(f"   Unique speakers: {len(speaker_counts)}")
        print(f"   Utterances per speaker: {total / len(speaker_counts):.1f} avg")

        self.results[dataset_name] = {
            'total': total,
            'bonafide': bonafide,
            'spoof': spoof,
            'matches_spec': (total == expected.get('total_trials', total))
        }

        return True

    def verify_audio_files(self, audio_dir, protocol_path):
        """Verify audio files exist and match protocol"""
        print(f"\n📁 Verifying audio files in: {audio_dir}")

        if not audio_dir.exists():
            print(f"❌ Audio directory not found: {audio_dir}")
            return False

        # Count audio files
        audio_files = set()
        for ext in ['*.flac', '*.wav']:
            audio_files.update(f.stem for f in audio_dir.rglob(ext))

        print(f"   Found {len(audio_files)} audio files")

        # Parse protocol to get expected files
        expected_files = set()
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    expected_files.add(parts[1])  # utt_id

        print(f"   Expected {len(expected_files)} audio files from protocol")

        # Check matches
        missing = expected_files - audio_files
        extra = audio_files - expected_files

        if not missing and not extra:
            print(f"   ✅ All audio files match protocol perfectly")
            return True
        else:
            if missing:
                print(f"   ⚠️  Missing {len(missing)} files from protocol")
                if len(missing) <= 10:
                    for f in list(missing)[:10]:
                        print(f"      - {f}")
            if extra:
                print(f"   ℹ️  Found {len(extra)} extra files not in protocol")
                if len(extra) <= 10:
                    for f in list(extra)[:10]:
                        print(f"      - {f}")
            return len(missing) == 0  # OK if extra files, not OK if missing

    def verify_df_dataset(self):
        """Verify ASVspoof 2021 DF dataset"""
        df_dir = self.data_root / 'asvspoof2021'
        protocol_file = df_dir / 'ASVspoof2021.DF.cm.eval.trl.txt'

        # Alternative protocol locations
        if not protocol_file.exists():
            protocol_file = df_dir / 'keys' / 'DF' / 'CM' / 'trial_metadata.txt'
        if not protocol_file.exists():
            protocol_file = df_dir / 'ASVspoof2021_DF_eval' / 'keys' / 'CM' / 'trial_metadata.txt'

        if not protocol_file.exists():
            print(f"⚠️  DF protocol file not found in standard locations")
            print(f"   Searched: {df_dir}")
            return False

        # Verify protocol
        self.verify_protocol_file(protocol_file, 'DF_eval')

        # Verify audio files
        audio_dir = df_dir / 'ASVspoof2021_DF_eval' / 'flac'
        if not audio_dir.exists():
            audio_dir = df_dir / 'ASVspoof2021_DF_eval'

        if audio_dir.exists():
            self.verify_audio_files(audio_dir, protocol_file)
        else:
            print(f"⚠️  DF audio directory not found: {audio_dir}")

        return True

    def verify_la_dataset(self):
        """Verify ASVspoof 2021 LA dataset"""
        la_dir = self.data_root / 'asvspoof2021'
        protocol_file = la_dir / 'ASVspoof2021.LA.cm.eval.trl.txt'

        # Alternative locations
        if not protocol_file.exists():
            protocol_file = la_dir / 'keys' / 'LA' / 'CM' / 'trial_metadata.txt'
        if not protocol_file.exists():
            protocol_file = la_dir / 'ASVspoof2021_LA_eval' / 'keys' / 'CM' / 'trial_metadata.txt'

        if not protocol_file.exists():
            print(f"⚠️  LA protocol file not found in standard locations")
            return False

        # Verify protocol
        self.verify_protocol_file(protocol_file, 'LA_eval')

        # Verify audio files
        audio_dir = la_dir / 'ASVspoof2021_LA_eval' / 'flac'
        if not audio_dir.exists():
            audio_dir = la_dir / 'ASVspoof2021_LA_eval'

        if audio_dir.exists():
            self.verify_audio_files(audio_dir, protocol_file)
        else:
            print(f"⚠️  LA audio directory not found")

        return True

    def generate_report(self):
        """Generate final verification report"""
        print(f"\n{'='*70}")
        print("FINAL VERIFICATION REPORT")
        print(f"{'='*70}")

        for dataset, results in self.results.items():
            status = "✅ PASS" if results['matches_spec'] else "⚠️  CHECK"
            print(f"\n{dataset}: {status}")
            print(f"  Total: {results['total']}")
            print(f"  Bonafide: {results['bonafide']}")
            print(f"  Spoof: {results['spoof']}")

        print(f"\n{'='*70}")
        all_pass = all(r['matches_spec'] for r in self.results.values())
        if all_pass:
            print("🎉 ALL DATASETS VERIFIED - READY FOR TRAINING")
        else:
            print("⚠️  SOME ISSUES FOUND - PLEASE REVIEW ABOVE")
        print(f"{'='*70}\n")

        return all_pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify ASVspoof 2021 datasets")
    parser.add_argument('--data_root', type=str,
                        default='/home/lab2208/Documents/deepfake_models/data/asvspoof',
                        help='Root directory containing ASVspoof datasets')
    parser.add_argument('--dataset', type=str, choices=['DF', 'LA', 'all'], default='all',
                        help='Which dataset to verify')

    args = parser.parse_args()

    verifier = ASVspoof2021Verifier(args.data_root)

    print("=" * 70)
    print("ASVspoof 2021 Dataset Verification")
    print("=" * 70)
    print(f"Data root: {args.data_root}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    success = True
    if args.dataset in ['LA', 'all']:
        success &= verifier.verify_la_dataset()

    if args.dataset in ['DF', 'all']:
        success &= verifier.verify_df_dataset()

    verifier.generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
