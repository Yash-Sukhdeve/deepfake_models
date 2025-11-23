#!/usr/bin/env python3
"""Monitor XLS-R + SLS Training Progress"""

import os
import time
import re
from datetime import datetime, timedelta

LOG_FILE = "/home/lab2208/Documents/deepfake_models/xls_r_sls/SLSforASVspoof-2021-DF/training_output_LA.log"

def parse_progress(log_file):
    """Extract training progress from log"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Find progress lines
    progress_lines = re.findall(r'(\d+)%\|.*?\| (\d+)/(\d+) \[.*?\<(.*?),\s*([\d.]+)it/s\]', content)

    if not progress_lines:
        return None

    # Get latest progress
    latest = progress_lines[-1]
    percent = int(latest[0])
    current = int(latest[1])
    total = int(latest[2])
    eta = latest[3]
    speed = float(latest[4])

    return {
        'percent': percent,
        'current': current,
        'total': total,
        'eta': eta,
        'speed': speed
    }

def estimate_completion():
    """Estimate training completion time"""
    info = parse_progress(LOG_FILE)
    if not info:
        return "Training starting..."

    # Calculate time per epoch
    iterations_per_epoch = info['total']
    speed = info['speed']  # iterations/second
    seconds_per_epoch = iterations_per_epoch / speed

    # Estimate total time for 50 epochs
    total_seconds = seconds_per_epoch * 50
    completion_time = datetime.now() + timedelta(seconds=total_seconds)

    return f"""
Training Progress:
- Batch: {info['current']}/{info['total']} ({info['percent']}%)
- Speed: {info['speed']:.2f} it/s
- ETA this epoch: {info['eta']}
- Time per epoch: {seconds_per_epoch/60:.1f} minutes
- Estimated completion (50 epochs): {completion_time.strftime('%Y-%m-%d %H:%M:%S')}
- Total estimated time: {total_seconds/3600:.1f} hours
"""

if __name__ == "__main__":
    print("=" * 70)
    print("XLS-R + SLS TRAINING MONITOR")
    print("=" * 70)
    print(estimate_completion())
    print("=" * 70)
