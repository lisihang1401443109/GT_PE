import pandas as pd
import numpy as np

# Data setup
data = [
    # Dataset, Model, Metric, noPE, LapPE, RWSE, GPSE, Direction (1 for higher is better, -1 for lower)
    ["ZINC", "Dense", "MAE", 0.1171, 0.1167, 0.1126, 0.1326, -1],
    ["ZINC", "GAT", "MAE", 0.3341, 0.2598, 0.2359, 0.2404, -1],
    ["ZINC", "Sparse", "MAE", 0.1639, 0.1236, 0.0871, 0.0675, -1],
    ["IMDB", "Dense", "F1", 0.9872, 0.9741, 0.9865, 0.9775, 1],
    ["IMDB", "GAT", "F1", 0.9760, 0.9762, 0.9972, 0.9634, 1],
    ["IMDB", "Sparse", "F1", 0.9848, 0.9818, 0.9879, 0.9756, 1],
    ["Peptides-Struct", "Dense", "MAE", 0.2524, 0.2444, 0.2556, 0.2680, -1],
    ["Peptides-Struct", "GAT", "MAE", 0.2541, 0.2483, 0.2508, 0.2568, -1],
    ["Peptides-Struct", "Sparse", "MAE", 0.2659, 0.2519, 0.2665, 0.3481, -1],
    ["Peptides-Func", "Dense", "AP", 0.6214, 0.6489, 0.6456, 0.6321, 1],
    ["Peptides-Func", "GAT", "AP", 0.6058, 0.5856, 0.6012, 0.6401, 1],
    ["Peptides-Func", "Sparse", "AP", 0.4560, 0.5355, 0.4998, 0.5835, 1],
    ["MolPCBA", "Sparse", "AP", 0.1359, 0.1384, 0.1658, 0.1554, 1],
]

df = pd.DataFrame(data, columns=["Dataset", "Model", "Metric", "noPE", "LapPE", "RWSE", "GPSE", "Direction"])

# 1. Percentage increase against noPE for each cell
def calc_pct_inc(row, pe_col):
    nope = row["noPE"]
    pe = row[pe_col]
    if row["Direction"] == 1:
        return (pe - nope) / nope * 100
    else:
        # For lower is better, improvement is (nope - pe) / nope
        return (nope - pe) / nope * 100

for pe in ["LapPE", "RWSE", "GPSE"]:
    df[f"{pe}_pct_inc"] = df.apply(lambda r: calc_pct_inc(r, pe), axis=1)

# 2. Winrate against noPE
winrates = {}
for pe in ["LapPE", "RWSE", "GPSE"]:
    wins = (df[f"{pe}_pct_inc"] > 0).sum()
    total = len(df)
    winrates[pe] = f"{wins}/{total} ({wins/total:.1%})"

# 3. Best PE for each Task + Model
def get_best_pe(row):
    nope = row["noPE"]
    lappe = row["LapPE"]
    rwse = row["RWSE"]
    gpse = row["GPSE"]
    
    pes = {"noPE": nope, "LapPE": lappe, "RWSE": rwse, "GPSE": gpse}
    if row["Direction"] == 1:
        best = max(pes, key=pes.get)
    else:
        best = min(pes, key=pes.get)
    return best

df["Best_PE"] = df.apply(get_best_pe, axis=1)

# 4. Average increase of PE in Model
model_avg = df.groupby("Model")[[f"{pe}_pct_inc" for pe in ["LapPE", "RWSE", "GPSE"]]].mean()

# Output generating markdown
output = []
output.append("# Comparative Analysis: PE Benchmarks")
output.append("\n## 1. Winrate against noPE")
output.append("| PE Variant | Winrate |")
output.append("| :--- | :--- |")
for pe, wr in winrates.items():
    output.append(f"| {pe} | {wr} |")

output.append("\n## 2. Percentage Improvement Over noPE")
cols = ["Dataset", "Model", "LapPE_pct_inc", "RWSE_pct_inc", "GPSE_pct_inc"]
output.append("| Dataset | Model | LapPE | RWSE | GPSE |")
output.append("| :--- | :--- | :---: | :---: | :---: |")
for _, row in df.iterrows():
    output.append(f"| {row['Dataset']} | {row['Model']} | {row['LapPE_pct_inc']:.2f}% | {row['RWSE_pct_inc']:.2f}% | {row['GPSE_pct_inc']:.2f}% |")

output.append("\n## 3. Best PE for each Task + Model")
output.append("| Dataset | Model | Best PE |")
output.append("| :--- | :--- | :--- |")
for _, row in df.iterrows():
    output.append(f"| {row['Dataset']} | {row['Model']} | **{row['Best_PE']}** |")

output.append("\n## 4. Average Improvement by Architecture")
output.append("| Model Architecture | LapPE Avg | RWSE Avg | GPSE Avg |")
output.append("| :--- | :---: | :---: | :---: |")
for model, row in model_avg.iterrows():
    output.append(f"| {model} | {row['LapPE_pct_inc']:.2f}% | {row['RWSE_pct_inc']:.2f}% | {row['GPSE_pct_inc']:.2f}% |")

with open("final_results/comparative_analysis.md", "w") as f:
    f.write("\n".join(output))

print("Analysis note generated at final_results/comparative_analysis.md")
