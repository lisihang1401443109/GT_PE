"""
Fetch and plot training/val loss curves from WandB for main benchmark runs.
Covers: ZINC, Peptides-Struct, Peptides-Func, IMDB.
Plots separate figures for Sparse, Dense, and GAT models.

Usage:
    WANDB_API_KEY=<key> python scripts/plot_benchmark_curves_wandb.py
"""

import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ENTITY  = "sihang-personal"
PROJECT = "PEGT"

PE_COLORS = {
    "noPE":  "#607D8B",
    "LapPE": "#2196F3",
    "RWSE":  "#4CAF50",
    "GPSE":  "#FF5722",
    "RRWP":  "#9C27B0",
}
MODEL_STYLES = {
    "Dense":  "-",
    "Sparse": "--",
    "GAT":    ":",
}
DATASETS = ["ZINC", "Peptides-Struct", "Peptides-Func", "IMDB"]

# ── Name → metadata helper ────────────────────────────────────────────────────
def parse_run(name):
    """Return (dataset, model, pe) or None if not parseable."""
    if not name.startswith("Verify-"):
        return None
    
    parts = name.split("-")
    if len(parts) < 4:
        return None
    
    pe = parts[-1]
    if pe not in PE_COLORS:
        return None
        
    model = parts[-2]
    if model not in MODEL_STYLES:
        if "Sparse" in name:
            model = "Sparse"
        elif "Dense" in name:
            model = "Dense"
        elif "GAT" in name:
            model = "GAT"
        else:
            return None

    dataset_raw = "-".join(parts[1:-2]).lower()
    
    if "peptides-struct" in dataset_raw or "peptides_struct" in dataset_raw:
        dataset = "Peptides-Struct"
    elif "peptides-func" in dataset_raw or "peptides_func" in dataset_raw:
        dataset = "Peptides-Func"
    elif "imdb" in dataset_raw:
        dataset = "IMDB"
    elif "zinc" in dataset_raw:
        dataset = "ZINC"
    else:
        return None

    return dataset, model, pe

# ── Fetch finished runs ───────────────────────────────────────────────────────
api  = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"})

grouped = {ds: {} for ds in DATASETS}  # ds -> {(model,pe): run}

for run in all_runs:
    parsed = parse_run(run.name)
    if parsed is None:
        continue
    dataset, model, pe = parsed
    if dataset not in grouped:
        continue
    key = (model, pe)
    if key not in grouped[dataset]:
        grouped[dataset][key] = run

print("Runs found per dataset:")
for ds, runs in grouped.items():
    print(f"  {ds}: {list(runs.keys())}")

# ── Plot ─────────────────────────────────────────────────────────────────────
nrows = len(DATASETS)

for target_model in ["Sparse", "Dense", "GAT"]:
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.5 * nrows))

    for row, dataset in enumerate(DATASETS):
        ax_tr = axes[row][0]
        ax_vl = axes[row][1]
        ax_tr.set_title(f"{dataset} — Train (GRIT {target_model})", fontsize=12, fontweight="bold")
        ax_vl.set_title(f"{dataset} — Val (GRIT {target_model})",   fontsize=12, fontweight="bold")
        for ax in [ax_tr, ax_vl]:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss / MAE (log scale)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        for (model, pe), run in grouped[dataset].items():
            if model != target_model:
                continue
                
            color = PE_COLORS.get(pe, "gray")
            ls    = MODEL_STYLES.get(model, "-")
            label = f"{pe}"

            try:
                h = run.history(pandas=True)
            except Exception as e:
                print(f"  [{dataset}/{label}] error: {e}")
                continue
                
            if h is None or h.empty:
                print(f"  [{dataset}/{label}] no history")
                continue

            tr_col = None
            vl_col = None
            
            for cand in ["train/mae", "train/ap", "train/loss", "train_mae", "train_loss"]:
                if cand in h.columns and h[cand].notna().any():
                    tr_col = cand
                    break
                    
            for cand in ["val/mae", "val/ap", "val/loss", "val_mae", "val_loss"]:
                if cand in h.columns and h[cand].notna().any():
                    vl_col = cand
                    break

            if not tr_col and not vl_col:
                print(f"  [{dataset}/{label}] missing expected metric columns. Available: {list(h.columns)}")
                continue

            ep = range(len(h))
            if tr_col in h:
                ax_tr.plot(ep, h[tr_col], color=color, linestyle=ls, linewidth=1.5, label=label)
            if vl_col in h:
                ax_vl.plot(ep, h[vl_col], color=color, linestyle=ls, linewidth=1.5, label=label)

        ax_vl.legend(fontsize=7, loc="upper right", framealpha=0.8)

    # shared legend
    pe_h  = [mlines.Line2D([], [], color=c, linewidth=2, label=pe)   for pe, c in PE_COLORS.items()]
    mod_h = [mlines.Line2D([], [], color="k", linestyle=ls, linewidth=2, label=m) for m, ls in MODEL_STYLES.items()]
    fig.legend(handles=pe_h + mod_h, loc="lower center", ncol=8, fontsize=9,
               title="PE (color)  |  Model (linestyle)", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Benchmark — Training & Validation Curves ({target_model}) [Log Scale]", fontsize=15, y=1.01)
    plt.tight_layout()
    out = f"scripts/benchmark_wandb_curves_{target_model.lower()}_log.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)
