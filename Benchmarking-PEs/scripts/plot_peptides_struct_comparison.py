"""
Specific plot for Peptides-Struct comparing all three architectures (Sparse, Dense, GAT).
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
}
MODEL_STYLES = {
    "Dense":  "-",
    "Sparse": "--",
    "GAT":    ":",
}

def parse_run(name):
    if name.startswith("Verify-"):
        parts = name.split("-")
        if len(parts) < 4: return None
        pe = parts[-1]
        model = parts[-2]
        dataset_raw = "-".join(parts[1:-2]).lower()
        
        if "peptides-struct" in dataset_raw or "peptides_struct" in dataset_raw:
            if pe in PE_COLORS and model in MODEL_STYLES:
                return model, pe
    return None

api = wandb.Api()
all_runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"})

grouped = {} # (model, pe) -> run
for run in all_runs:
    parsed = parse_run(run.name)
    if parsed:
        if parsed not in grouped:
            grouped[parsed] = run

print(f"Found {len(grouped)} runs for Peptides-Struct")

fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharey=True)
models = ["Sparse", "Dense", "GAT"]

for i, model in enumerate(models):
    ax_tr = axes[i, 0]
    ax_vl = axes[i, 1]
    
    for ax, title_suffix in [(ax_tr, "Train"), (ax_vl, "Val")]:
        ax.set_title(f"GRIT {model} — {title_suffix}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    
    ax_tr.set_ylabel("MAE (log scale)")
    
    for pe, color in PE_COLORS.items():
        run = grouped.get((model, pe))
        if not run: continue
        
        try:
            h = run.history(pandas=True)
            if h is not None and not h.empty:
                # Train metric
                tr_col = None
                for cand in ["train/mae", "train_mae", "train/loss", "train_loss"]:
                    if cand in h.columns:
                        tr_col = cand
                        break
                if tr_col:
                    ax_tr.plot(range(len(h)), h[tr_col], color=color, label=pe, linewidth=1.5)
                
                # Val metric
                vl_col = None
                for cand in ["val/mae", "val_mae"]:
                    if cand in h.columns:
                        vl_col = cand
                        break
                if vl_col:
                    ax_vl.plot(range(len(h)), h[vl_col], color=color, label=pe, linewidth=1.5)
        except Exception as e:
            print(f"Error plotting {model}-{pe}: {e}")

# Global legend
pe_handles = [mlines.Line2D([], [], color=c, linewidth=2, label=pe) for pe, c in PE_COLORS.items()]
fig.legend(handles=pe_handles, loc="lower center", ncol=4, fontsize=12, 
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Peptides-Struct: Train vs Val MAE Comparison", fontsize=18, y=1.02)
plt.tight_layout()

out = "final_results/peptides_struct_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
