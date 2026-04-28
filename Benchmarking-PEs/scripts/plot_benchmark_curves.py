"""
Plot training and validation MAE curves for the main PE × dataset × model benchmark.
Reads local stats.json files from results/.

Usage:
    python scripts/plot_benchmark_curves.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUT_FILE = Path(__file__).parent / "benchmark_training_curves.png"

# ── Discover all experiments ─────────────────────────────────────────────────
# Folder structure: results/{exp_name}/0/train/stats.json
#                                      /0/val/stats.json
def load_stats(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def parse_name(name):
    """Extract dataset and model+PE tag from folder name."""
    # e.g. zinc-GRIT-GPSE, imdb-GRITSparse-LapPE, voc-GRITSparse-noPE
    parts = name.split("-", 1)
    dataset = parts[0].upper()  # ZINC, IMDB, VOC
    model_pe = parts[1] if len(parts) > 1 else name
    return dataset, model_pe

experiments = {}
for exp_dir in sorted(RESULTS_DIR.iterdir()):
    if not exp_dir.is_dir():
        continue
    train_file = exp_dir / "0" / "train" / "stats.json"
    val_file   = exp_dir / "0" / "val"   / "stats.json"
    if train_file.exists() and val_file.exists():
        dataset, model_pe = parse_name(exp_dir.name)
        experiments[exp_dir.name] = {
            "dataset":  dataset,
            "model_pe": model_pe,
            "train":    load_stats(train_file),
            "val":      load_stats(val_file),
        }

print(f"Loaded {len(experiments)} experiments: {list(experiments.keys())}")

# ── Group by dataset ─────────────────────────────────────────────────────────
datasets = sorted(set(v["dataset"] for v in experiments.values()))

# Color by PE, style by model type
PE_COLORS = {
    "noPE":  "#607D8B",
    "LapPE": "#2196F3",
    "RWSE":  "#4CAF50",
    "GPSE":  "#FF5722",
    "RWDIFF":"#9C27B0",
}
MODEL_STYLES = {
    "GRIT":       "-",
    "GRITSparse": "--",
    "OriginGT":   ":",
}

def get_pe(model_pe):
    for pe in PE_COLORS:
        if model_pe.endswith(pe):
            return pe
    return "noPE"

def get_model(model_pe):
    for m in MODEL_STYLES:
        if model_pe.startswith(m):
            return m
    return "GRIT"

# ── Plot: one row per dataset, train vs val side by side ─────────────────────
nrows = len(datasets)
fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
if nrows == 1:
    axes = [axes]

metric = "mae"

for row, dataset in enumerate(datasets):
    ax_train = axes[row][0]
    ax_val   = axes[row][1]
    ax_train.set_title(f"{dataset} — Train MAE", fontsize=12, fontweight="bold")
    ax_val.set_title(f"{dataset} — Val MAE",   fontsize=12, fontweight="bold")

    for ax in [ax_train, ax_val]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)

    for name, exp in experiments.items():
        if exp["dataset"] != dataset:
            continue
        pe    = get_pe(exp["model_pe"])
        model = get_model(exp["model_pe"])
        color = PE_COLORS.get(pe, "gray")
        ls    = MODEL_STYLES.get(model, "-")
        label = exp["model_pe"]

        train_vals = [e[metric] for e in exp["train"] if metric in e]
        val_vals   = [e[metric] for e in exp["val"]   if metric in e]
        epochs_t   = range(len(train_vals))
        epochs_v   = range(len(val_vals))

        ax_train.plot(epochs_t, train_vals, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax_val.plot(epochs_v,   val_vals,   color=color, linestyle=ls, linewidth=1.5, label=label)

    ax_val.legend(fontsize=7, loc="upper right", framealpha=0.7)

# ── Shared legend for PE colors & model styles ────────────────────────────────
pe_handles   = [mlines.Line2D([], [], color=c, linewidth=2, label=pe)
                for pe, c in PE_COLORS.items()]
model_handles= [mlines.Line2D([], [], color="black", linestyle=ls, linewidth=2, label=m)
                for m, ls in MODEL_STYLES.items()]
fig.legend(handles=pe_handles + model_handles,
           loc="lower center", ncol=8, fontsize=9,
           title="PE (color)  |  Model (linestyle)", title_fontsize=9,
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("GRIT Benchmark — Training & Validation MAE Curves", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
