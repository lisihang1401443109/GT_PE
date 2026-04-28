"""
Plot training and validation MAE curves for the Peptides-Struct LR sensitivity sweep.
Fetches data directly from WandB API.

Usage:
    pip install wandb matplotlib
    python scripts/plot_lr_sweep_curves.py
"""

import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
ENTITY  = "sihang-personal"
PROJECT = "PEGT"

# Runs are named: Peptides-Struct-GRIT-{PE}-LR-{LR}-Sparse
PE_TYPES = ["noPE", "LapPE", "RWSE", "GPSE"]
LR_TYPES = ["0.0001", "0.001", "0.003"]   # update if you changed the LR set

# ── Fetch runs ───────────────────────────────────────────────────────────────
api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={
    "display_name": {"$regex": "Peptides-Struct-GRIT-.*-Sparse"}
})

# Group by PE
run_map = {}  # key: (PE, LR) -> run
for run in runs:
    name = run.name  # e.g. Peptides-Struct-GRIT-noPE-LR-0.0001-Sparse
    for pe in PE_TYPES:
        for lr in LR_TYPES:
            tag = f"Peptides-Struct-GRIT-{pe}-LR-{lr}-Sparse"
            if tag == name:
                run_map[(pe, lr)] = run

print(f"Found {len(run_map)} matching runs: {list(run_map.keys())}")

# ── Plot ─────────────────────────────────────────────────────────────────────
LR_COLORS = {
    "0.0001": "#2196F3",   # blue
    "0.001":  "#4CAF50",   # green
    "0.003":  "#FF9800",   # orange
    "0.005":  "#F44336",   # red
}
PE_STYLES = {
    "noPE":  "-",
    "LapPE": "--",
    "RWSE":  "-.",
    "GPSE":  ":",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

for ax_idx, pe in enumerate(PE_TYPES):
    ax = axes[ax_idx]
    ax.set_title(f"PE: {pe}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)

    for lr in LR_TYPES:
        if (pe, lr) not in run_map:
            continue
        run = run_map[(pe, lr)]
        try:
            h = run.history(pandas=True)
        except Exception as e:
            print(f"  [{pe} LR={lr}] error: {e}")
            continue
            
        if h is None or h.empty:
            print(f"  [{pe} LR={lr}] No history yet")
            continue

        epochs = range(len(h))
        color  = LR_COLORS.get(lr, "gray")

        # Find train and val columns
        tr_col = next((c for c in ["train/mae", "train_mae"] if c in h.columns and h[c].notna().any()), None)
        vl_col = next((c for c in ["val/mae", "val_mae"] if c in h.columns and h[c].notna().any()), None)

        if tr_col:
            ax.plot(epochs, h[tr_col], color=color,
                    linestyle="-", alpha=0.7, label=f"LR={lr} train")
        if vl_col:
            ax.plot(epochs, h[vl_col], color=color,
                    linestyle="--", linewidth=2, label=f"LR={lr} val")

    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Peptides-Struct GRIT Sparse — LR Sensitivity (Train/Val MAE)", fontsize=15)
plt.tight_layout()
out = "scripts/lr_sweep_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved plot → {out}")
plt.show()
