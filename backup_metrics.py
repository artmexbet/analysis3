"""
Backup current metrics before running new experiments.
"""
import json
import shutil
from datetime import datetime

# Create backup directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_suffix = "no_augmentation"

metrics_files = ["cnn.json", "vgg.json", "efficientnet.json"]

for metric_file in metrics_files:
    try:
        # Read current metrics
        with open(metric_file, "r") as f:
            data = json.load(f)

        # Save backup
        backup_name = metric_file.replace(".json", f".{backup_suffix}.json")
        with open(backup_name, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Backed up {metric_file} -> {backup_name}")
    except FileNotFoundError:
        print(f"Skipping {metric_file} (not found)")

print(f"\nBackup complete: {backup_suffix}")

