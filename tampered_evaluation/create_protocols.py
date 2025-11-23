#!/usr/bin/env python3
"""Create protocol files for tampered audio evaluation."""
import os
import json
from pathlib import Path

# Trans-Splicing protocol
trans_splicing_dir = Path("/home/lab2208/Documents/deepfake_models/tampered_evaluation/trans_splicing")
trans_splicing_protocol = []

for category in ["xtts-clean", "xtts-unclean", "yourtts-clean", "yourtts-unclean"]:
    cat_dir = trans_splicing_dir / category
    if not cat_dir.exists():
        continue
    for wav_file in cat_dir.rglob("*_tampered.wav"):
        # Get relative path from trans_splicing_dir
        rel_path = wav_file.relative_to(trans_splicing_dir)
        # Extract subject ID from path
        subject_id = wav_file.parent.name.replace(".wav", "")
        # All are tampered (spoof)
        trans_splicing_protocol.append({
            "file_path": str(wav_file),
            "rel_path": str(rel_path),
            "category": category,
            "subject_id": subject_id,
            "label": "spoof",
            "tts_system": category.split("-")[0],  # xtts or yourtts
            "processing": category.split("-")[1]   # clean or unclean
        })

# Save Trans-Splicing protocol
with open(trans_splicing_dir / "protocol.json", "w") as f:
    json.dump(trans_splicing_protocol, f, indent=2)

print(f"Trans-Splicing Dataset: {len(trans_splicing_protocol)} files")
for cat in ["xtts-clean", "xtts-unclean", "yourtts-clean", "yourtts-unclean"]:
    count = sum(1 for p in trans_splicing_protocol if p["category"] == cat)
    print(f"  {cat}: {count}")

# Semantic Tampering protocol  
semantic_dir = Path("/home/lab2208/Documents/deepfake_models/tampered_evaluation/semantic")
semantic_protocol = []

# Original files (bonafide) - only use the _result_ files that have tampered versions
original_dir = semantic_dir / "original"
tampered_dir = semantic_dir / "tampered"
metadata_dir = semantic_dir / "metadata"

# Get list of tampered files and their sources
tampered_sources = set()
if tampered_dir.exists():
    for wav_file in tampered_dir.glob("*.wav"):
        # Parse filename: df_sub096_LP1_result_1_T009_M_DEL_back.wav
        parts = wav_file.stem.split("_T")
        if len(parts) >= 1:
            source_name = parts[0]  # e.g., df_sub096_LP1_result_1
            tampered_sources.add(source_name)
            
            # Load metadata if exists
            meta_file = metadata_dir / f"{wav_file.stem}.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            
            semantic_protocol.append({
                "file_path": str(wav_file),
                "source_audio": source_name,
                "label": "spoof",
                "tamper_type": meta.get("candidate_details", {}).get("tamper_type", "unknown"),
                "difficulty": meta.get("candidate_details", {}).get("difficulty", {}).get("level", "unknown"),
                "word": meta.get("candidate_details", {}).get("word", "unknown")
            })

# Add original files that have tampered versions
if original_dir.exists():
    for wav_file in original_dir.glob("*_result_*.wav"):
        source_name = wav_file.stem
        if source_name in tampered_sources:
            semantic_protocol.append({
                "file_path": str(wav_file),
                "source_audio": source_name,
                "label": "bonafide",
                "tamper_type": "none",
                "difficulty": "none",
                "word": "none"
            })

# Save Semantic protocol
with open(semantic_dir / "protocol.json", "w") as f:
    json.dump(semantic_protocol, f, indent=2)

bonafide_count = sum(1 for p in semantic_protocol if p["label"] == "bonafide")
spoof_count = sum(1 for p in semantic_protocol if p["label"] == "spoof")
print(f"\nSemantic Tampering: {len(semantic_protocol)} files")
print(f"  Bonafide: {bonafide_count}")
print(f"  Spoof: {spoof_count}")
print(f"  By type:")
for t in ["deletion", "insertion", "substitution", "unknown"]:
    count = sum(1 for p in semantic_protocol if p["tamper_type"] == t)
    if count > 0:
        print(f"    {t}: {count}")
