import re
import json
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(log_path):
    epochs = []
    val_f1 = []
    
    # Pattern to match: val: {'epoch': 0, ..., 'f1': 0.9635}
    pattern = re.compile(r"val: (\{.*\})")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    data_str = match.group(1).replace("'", '"').replace("None", "null")
                    data = json.loads(data_str)
                    epochs.append(data['epoch'])
                    val_f1.append(data['f1'])
                except Exception:
                    continue
    return epochs, val_f1

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    if not points:
        return []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

results_dir = "/home/lisihang/GT_PE/Benchmarking-PEs/results"
variants = {
    "IMDB-noPE": "imdb-GRITSparse-noPE",
    "IMDB-LapPE": "imdb-GRITSparse-LapPE",
    "IMDB-GPSE": "imdb-GRITSparse-GPSE",
    "IMDB-RWSE": "imdb-GRITSparse-RWSE",
}

results_summary = []

plt.figure(figsize=(10, 6))
plt.title("IMDB Dataset: Validation F1 Comparison (Smoothed)")

for name, folder in variants.items():
    log_path = os.path.join(results_dir, folder, "0", "logging.log")
    if os.path.exists(log_path):
        epochs, f1 = parse_log(log_path)
        if f1:
            best_f1 = max(f1)
            best_epoch = f1.index(best_f1)
            results_summary.append({"Variant": name, "Best Val F1": best_f1, "Best Epoch": best_epoch, "Final Epoch": epochs[-1]})
            
            smoothed_f1 = smooth_curve(f1)
            plt.plot(epochs[:len(smoothed_f1)], smoothed_f1, label=f"{name} (Smoothed)")
            plt.plot(epochs[:len(f1)], f1, alpha=0.2, color=plt.gca().lines[-1].get_color())
            print(f"Processed {name}: Best F1 {best_f1:.4f} at epoch {best_epoch}")

plt.xlabel("Epoch")
plt.ylabel("Validation F1 (Higher is better)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()

output_path = "/home/lisihang/GT_PE/Benchmarking-PEs/imdb_comparison_chart.png"
plt.savefig(output_path)
print(f"Chart saved to {output_path}")

# Table
print("\n### IMDB Performance Summary")
print("| Variant | Best Val F1 | Best Epoch | Total Epochs |")
print("| :--- | :--- | :--- | :--- |")
for res in sorted(results_summary, key=lambda x: x["Best Val F1"], reverse=True):
    print(f"| {res['Variant']} | {res['Best Val F1']:.4f} | {res['Best Epoch']} | {res['Final Epoch']} |")
