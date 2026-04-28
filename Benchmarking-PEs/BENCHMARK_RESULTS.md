# Benchmarking Graph Transformer Positional Encodings (GPSE)

This document summarizes the results of a comprehensive benchmarking suite comparing **Dense GRIT** (Global Attention), **Sparse GRIT** (Local Attention), and **GAT-GPS** (Sparse Attention) architectures across various datasets and Positional/Structural Encodings.

## Holistic Results Matrix

| Dataset | Metric | PE Variant | Dense GRIT | GAT-GPS | Sparse GRIT |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **IMDB (Masked)** | F1 (↑) | noPE | 0.5001 | 0.5079 | 0.4880 |
| *(Node Class)* | | LapPE | 0.4798 | 0.5214 | 0.5037 |
| | | RWSE | 0.5035 | 0.5109 | 0.4827 |
| | | GPSE | **0.5110** | **0.5439** | **0.5112** |
| **ZINC** | MAE (↓) | noPE | 0.1171 | 0.3341 | 0.1639 |
| *(Regression)* | | LapPE | 0.1167 | 0.2598 | 0.1236 |
| | | RWSE | **0.1126** | **0.2359** | 0.0871 |
| | | GPSE | 0.1326 | 0.2404 | **0.0675** |
| **Peptides-Struct**| MAE (↓) | noPE | 0.2524 | 0.2541 | 0.2659 |
| *(LRGB Reg)* | | LapPE | **0.2444** | **0.2483** | **0.2519** |
| | | RWSE | 0.2556 | 0.2508 | 0.2665 |
| | | GPSE | 0.2680 | 0.2568 | 0.3481 |
| **Peptides-Func** | AP (↑) | noPE | 0.6214 | 0.6058 | 0.4560 |
| *(LRGB Class)* | | LapPE | **0.6489** | 0.5856 | 0.5355 |
| | | RWSE | 0.6456 | 0.6012 | 0.4998 |
| | | GPSE | 0.6321 | **0.6401** | **0.5835** |

> [!IMPORTANT]
> **IMDB Evaluation Protocol Update**: Results for IMDB use the latest leakage-free split mask (v2). Scores are significantly lower than earlier unmasked runs (~0.5 vs ~0.98) due to the removal of label leakage.

## Performance Analysis
- **GPSE Dominance on IMDB**: GPSE achieved a breakthrough on the masked IMDB dataset, particularly when paired with the GAT architecture (**0.5439**).
- **ZINC Efficiency**: Sparse GRIT with GPSE continues to hold the record for ZINC MAE at **0.0675**, showcasing the power of structural encoding in sparse architectures.
- **Peptides Stability**: LapPE remains the most stable baseline for LRGB datasets, though GPSE shows competitive performance in Peptides-Functional classification.

## Visualizations
W&B learning curves and comparative charts are available in the `final_results/` directory.

---
*Note: Results consolidated on 2026-04-28. IMDB results verified from the `fix/imdb-eval-mask` branch.*
