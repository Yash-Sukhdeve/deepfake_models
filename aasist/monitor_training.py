#!/usr/bin/env python3
"""
Monitor AASIST training progress in real-time.
Displays metrics, checkpoint information, and GPU usage.
"""
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

def get_gpu_usage():
    """Get GPU utilization and memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_util, mem_util, mem_used, mem_total = result.stdout.strip().split(', ')
            return f"GPU: {gpu_util}, Memory: {mem_used}/{mem_total} ({mem_util})"
    except:
        pass
    return "GPU: N/A"

def monitor_training(exp_dir="exp_result/LA_AASIST_ep100_bs24", interval=30):
    """Monitor training progress."""
    exp_path = Path(exp_dir)
    metric_log = exp_path / "metric_log.txt"
    weights_dir = exp_path / "weights"

    print("=" * 80)
    print("AASIST Training Monitor")
    print("=" * 80)
    print(f"Monitoring: {exp_path}")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")

    last_size = 0
    last_checkpoint_count = 0
    epoch_count = 0

    try:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] " + "=" * 60)

            # GPU status
            print(f"  {get_gpu_usage()}")

            # Check metric log
            if metric_log.exists():
                current_size = metric_log.stat().st_size
                if current_size > last_size:
                    print(f"\n  📊 Training Metrics (last 10 lines):")
                    with open(metric_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            if line.strip():
                                print(f"     {line.strip()}")
                    last_size = current_size

                    # Count epochs
                    epoch_count = sum(1 for line in lines if line.startswith('epoch'))
                    print(f"\n  ✓ Completed epochs: {epoch_count}/100")
                else:
                    print(f"\n  ⏳ Training in progress (epoch {epoch_count})...")
            else:
                print("  ⏳ Waiting for training to start...")

            # Check checkpoints
            if weights_dir.exists():
                checkpoints = sorted(weights_dir.glob("epoch_*.pth"))
                if len(checkpoints) > last_checkpoint_count:
                    print(f"\n  💾 Saved Checkpoints ({len(checkpoints)}):")
                    for cp in checkpoints[-5:]:  # Show last 5
                        size_mb = cp.stat().st_size / (1024**2)
                        print(f"     - {cp.name} ({size_mb:.1f} MB)")
                    last_checkpoint_count = len(checkpoints)

                # Check for special checkpoints
                if (weights_dir / "best.pth").exists():
                    print("  ✓ best.pth (best eval t-DCF) saved")
                if (weights_dir / "swa.pth").exists():
                    print("  ✓ swa.pth (final SWA model) saved")

            print("=" * 70)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print(f"Final status: {epoch_count} epochs completed")
        print(f"Checkpoints saved: {last_checkpoint_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor AASIST training")
    parser.add_argument("--dir", default="exp_result/LA_AASIST_ep100_bs24",
                        help="Training output directory")
    parser.add_argument("--interval", type=int, default=30,
                        help="Update interval in seconds")
    args = parser.parse_args()

    monitor_training(args.dir, args.interval)
