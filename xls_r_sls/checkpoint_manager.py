#!/usr/bin/env python3
"""
Checkpoint Rotation Manager for XLS-R + SLS Training
Automatically keeps only the best N checkpoints to prevent disk space exhaustion
"""

import os
import glob
import re
from pathlib import Path


class CheckpointManager:
    """Manages checkpoint rotation during training"""

    def __init__(self, checkpoint_dir, keep_best_n=5, keep_every_n_epochs=10):
        """
        Args:
            checkpoint_dir: Directory where checkpoints are saved
            keep_best_n: Keep top N checkpoints by performance (default: 5)
            keep_every_n_epochs: Keep checkpoints every N epochs regardless of performance
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best_n = keep_best_n
        self.keep_every_n_epochs = keep_every_n_epochs
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_info(self, checkpoint_path):
        """Extract epoch number and EER from checkpoint filename"""
        filename = os.path.basename(checkpoint_path)

        # Match pattern: epoch_XX_Y.YYY.pth where Y.YYY is EER
        match = re.match(r'epoch_(\d+)_(\d+\.\d+)\.pth', filename)
        if match:
            epoch = int(match.group(1))
            eer = float(match.group(2))
            return epoch, eer
        return None, None

    def clean_old_checkpoints(self):
        """Remove old checkpoints, keeping only best N and milestone epochs"""
        checkpoints = list(self.checkpoint_dir.glob('epoch_*.pth'))

        if len(checkpoints) <= self.keep_best_n:
            return  # Not enough checkpoints to clean

        # Parse checkpoint info
        checkpoint_data = []
        for ckpt in checkpoints:
            epoch, eer = self.get_checkpoint_info(ckpt)
            if epoch is not None and eer is not None:
                checkpoint_data.append({
                    'path': ckpt,
                    'epoch': epoch,
                    'eer': eer
                })

        if not checkpoint_data:
            return

        # Sort by EER (lower is better)
        checkpoint_data.sort(key=lambda x: x['eer'])

        # Identify checkpoints to keep
        keep_paths = set()

        # 1. Keep best N by EER
        for item in checkpoint_data[:self.keep_best_n]:
            keep_paths.add(str(item['path']))

        # 2. Keep milestone epochs (every N epochs)
        for item in checkpoint_data:
            if item['epoch'] % self.keep_every_n_epochs == 0:
                keep_paths.add(str(item['path']))

        # 3. Always keep best.pth and swa.pth if they exist
        special_files = ['best.pth', 'swa.pth', 'final.pth']
        for special in special_files:
            special_path = self.checkpoint_dir / special
            if special_path.exists():
                keep_paths.add(str(special_path))

        # Delete checkpoints not in keep list
        deleted_count = 0
        freed_space = 0
        for item in checkpoint_data:
            path_str = str(item['path'])
            if path_str not in keep_paths:
                file_size = item['path'].stat().st_size
                item['path'].unlink()
                deleted_count += 1
                freed_space += file_size

        if deleted_count > 0:
            freed_mb = freed_space / (1024 * 1024)
            print(f"🗑️  Cleaned {deleted_count} old checkpoints, freed {freed_mb:.1f} MB")
            print(f"📊 Keeping {len(keep_paths)} checkpoints (best {self.keep_best_n} + milestones)")

    def get_disk_usage(self):
        """Get current disk usage of checkpoint directory"""
        total_size = sum(f.stat().st_size for f in self.checkpoint_dir.glob('*.pth'))
        return total_size / (1024 ** 3)  # Return in GB

    def report_status(self):
        """Print current checkpoint status"""
        checkpoints = list(self.checkpoint_dir.glob('epoch_*.pth'))
        total_size_gb = self.get_disk_usage()

        print("=" * 60)
        print("CHECKPOINT STATUS")
        print("=" * 60)
        print(f"Total checkpoints: {len(checkpoints)}")
        print(f"Disk usage: {total_size_gb:.2f} GB")
        print(f"Policy: Keep best {self.keep_best_n}, milestone every {self.keep_every_n_epochs} epochs")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage XLS-R training checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoints')
    parser.add_argument('--keep_best', type=int, default=5,
                        help='Number of best checkpoints to keep (default: 5)')
    parser.add_argument('--keep_milestone', type=int, default=10,
                        help='Keep checkpoint every N epochs (default: 10)')
    parser.add_argument('--clean', action='store_true',
                        help='Perform cleanup')
    parser.add_argument('--status', action='store_true',
                        help='Show status only')

    args = parser.parse_args()

    manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        keep_best_n=args.keep_best,
        keep_every_n_epochs=args.keep_milestone
    )

    if args.status:
        manager.report_status()
    elif args.clean:
        print("Running checkpoint cleanup...")
        manager.clean_old_checkpoints()
        manager.report_status()
    else:
        print("Use --status to show checkpoint info or --clean to remove old checkpoints")
        manager.report_status()
