#!/usr/bin/env python3
"""Pre-flight test: Verify data loading before starting full training"""

import sys
sys.path.insert(0, '/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF')

import torch
from data_utils_SSL import Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from torch.utils.data import DataLoader

print("=" * 70)
print("PRE-FLIGHT DATA LOADING TEST")
print("=" * 70)

# Paths
database_path = "/home/lab2208/Documents/deepfake_models/data/asvspoof"
protocols_path = "/home/lab2208/Documents/deepfake_models/data/asvspoof"

try:
    # Test training dataset
    print("\n1. Testing ASVspoof 2019 LA training dataset...")
    train_protocol = f"{protocols_path}/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    train_dir = f"{database_path}/asvspoof2019/LA/ASVspoof2019_LA_train/flac"

    train_dataset = Dataset_ASVspoof2019_train(
        list_IDs=[],  # Will be populated by genSpoof_list
        base_dir=train_dir
    )
    print(f"   ✅ Training dataset initialized")

    # Test loading one batch
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    batch_x, batch_y = next(iter(train_loader))
    print(f"   ✅ Sample batch loaded: x.shape={batch_x.shape}, y.shape={batch_y.shape}")

except Exception as e:
    print(f"   ❌ Training dataset FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Test evaluation dataset
    print("\n2. Testing ASVspoof 2021 LA evaluation dataset...")
    eval_protocol = f"{protocols_path}/asvspoof2021/ASVspoof2021.LA.cm.eval.trl.txt"
    eval_dir = f"{database_path}/asvspoof2021/ASVspoof2021_LA_eval/flac"

    eval_dataset = Dataset_ASVspoof2021_eval(
        list_IDs=[],  # Will be populated
        base_dir=eval_dir
    )
    print(f"   ✅ Evaluation dataset initialized")

    eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
    batch_x, batch_utt = next(iter(eval_loader))
    print(f"   ✅ Sample eval batch loaded: x.shape={batch_x.shape}, utt_ids={batch_utt}")

except Exception as e:
    print(f"   ❌ Evaluation dataset FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("PRE-FLIGHT CHECK: ALL SYSTEMS GO ✅")
print("=" * 70)
print("\nReady to start training!")
print("Estimated time: 5-6 days (50 epochs)")
print("=" * 70)
