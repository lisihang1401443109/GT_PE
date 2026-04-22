import re
import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def parse_log(log_path):
    epochs = []
    train_mae = []
    val_mae = []
    test_mae = []
    
    # Pattern to match the JSON-like dict in the log
    pattern = re.compile(r"(train|val|test): (\{.*\})")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                mode = match.group(1)
                try:
                    data_str = match.group(2).replace("'", '"').replace("None", "null")
                    data = json.loads(data_str)
                    
                    if mode == 'train':
                        epochs.append(data['epoch'])
                        train_mae.append(data['mae'])
                    elif mode == 'val':
                        val_mae.append(data['mae'])
                    elif mode == 'test':
                        test_mae.append(data['mae'])
                except Exception as e:
                    continue
    
    return epochs, train_mae, val_mae, test_mae

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

# Define the variants to plot
results_dir = "/home/lisihang/GT_PE/Benchmarking-PEs/results"
variants = {
    "GRIT-noPE": "zinc-GRITSparse-noPE",
    "GRIT-LapPE": "zinc-GRITSparse-LapPE",
    "GRIT-RWSE": "zinc-GRITSparse-RWSE",
    "GRIT-GPSE": "zinc-GRITSparse-GPSE",
}

results_summary = []

# First plot: Original Curves
plt.figure(figsize=(10, 6))
for name, folder in variants.items():
    log_path = os.path.join(results_dir, folder, "0", "logging.log")
    if os.path.exists(log_path):
        epochs, train, val, test = parse_log(log_path)
        if val:
            plt.plot(epochs[:len(val)], val, label=name, alpha=0.3)
            
            best_val = min(val)
            best_epoch = val.index(best_val)
            results_summary.append({"Variant": name, "Best Val MAE": best_val, "Best Epoch": best_epoch, "Final Epoch": epochs[-1]})
            
            smoothed_val = smooth_curve(val)
            plt.plot(epochs[:len(smoothed_val)], smoothed_val, label=f"{name} (Smoothed)")
            print(f"Processed {name}: Best MAE {best_val:.4f} at epoch {best_epoch}")

plt.title("ZINC Dataset: Validation MAE comparison (Smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Validation MAE")
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()

output_path = "/home/lisihang/GT_PE/Benchmarking-PEs/zinc_comparison_chart.png"
plt.savefig(output_path)
print(f"Chart saved to {output_path}")

# Generate Markdown Table
print("\n### ZINC Transformation Performance Summary")
print("| Variant | Best Val MAE | Best Epoch | Total Epochs |")
print("| :--- | :--- | :--- | :--- |")
for res in sorted(results_summary, key=lambda x: x["Best Val MAE"]):
    print(f"| {res['Variant']} | {res['Best Val MAE']:.4f} | {res['Best Epoch']} | {res['Final Epoch']} |")
