#!/usr/bin/env python3
"""Verify all required paths and files exist before training"""

import os
from pathlib import Path

print("=" * 70)
print("PRE-FLIGHT PATH VERIFICATION")
print("=" * 70)

base_path = "/home/lab2208/Documents/deepfake_models/data/asvspoof"
checks_passed = 0
checks_total = 0

def check_path(path, description):
    global checks_passed, checks_total
    checks_total += 1
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}")
    if exists:
        checks_passed += 1
        if Path(path).is_dir():
            file_count = sum(1 for _ in Path(path).rglob('*.flac'))
            if file_count > 0:
                print(f"   └─ {file_count} FLAC files found")
    else:
        print(f"   └─ NOT FOUND: {path}")
    return exists

print("\n📁 Training Data (ASVspoof 2019 LA):")
check_path(f"{base_path}/asvspoof2019/LA/ASVspoof2019_LA_train/flac", "Training audio directory")
check_path(f"{base_path}/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "Training protocol")

print("\n📁 Development Data (ASVspoof 2019 LA):")
check_path(f"{base_path}/asvspoof2019/LA/ASVspoof2019_LA_dev/flac", "Dev audio directory")
check_path(f"{base_path}/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt", "Dev protocol")

print("\n📁 Evaluation Data (ASVspoof 2021 LA):")
check_path(f"{base_path}/asvspoof2021/ASVspoof2021_LA_eval/flac", "Eval audio directory")
check_path(f"{base_path}/asvspoof2021/ASVspoof2021.LA.cm.eval.trl.txt", "Eval protocol")

print("\n🤖 Model:")
check_path("/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF/xlsr2_300m.pt", "XLS-R checkpoint")

print("\n" + "=" * 70)
if checks_passed == checks_total:
    print(f"RESULT: ALL CHECKS PASSED ✅ ({checks_passed}/{checks_total})")
    print("=" * 70)
    print("🚀 READY TO START TRAINING!")
    print("=" * 70)
    exit(0)
else:
    print(f"RESULT: SOME CHECKS FAILED ⚠️  ({checks_passed}/{checks_total})")
    print("=" * 70)
    exit(1)
