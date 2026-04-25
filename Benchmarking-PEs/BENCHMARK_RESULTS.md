# Benchmarking Graph Transformer Positional Encodings

This document summarizes the results of a comprehensive benchmarking suite comparing **Dense GRIT** (Global Attention) and **GAT-GPS** (Sparse Local Attention) architectures using various Positional and Structural Encodings.

## Summary of Findings
- **Architectural Efficiency**: Dense models benefit marginally from PE (+1-2%), while Sparse (Local) models like GAT-GPS see dramatic boosts (up to +30%) when equipped with global structural context.
- **Top PE Variant**: **RWSE** (Random Walk Structural Encoding) emerged as the most robust baseline, providing the highest average performance improvement across both local and global architectures.
- **GPSE Sensitivity**: GPSE excels in complex classification (Peptides-Functional) but can introduce noise in Dense regression tasks (ZINC) due to pre-training distribution shifts.

## Holistic Results Matrix

| Dataset | Metric | PE Variant | Dense GRIT | GAT-GPS |
| :--- | :--- | :--- | :--- | :--- |
| **ZINC** | MAE (↓) | noPE | 0.1171 | 0.3341 |
| *(Regression)* | | LapPE | 0.1167 | 0.2598 |
| | | RWSE | **0.1126** | **0.2359** |
| | | GPSE | 0.1326 | 0.2404 |
| **IMDB** | F1 (↑) | noPE | **0.9872** | 0.9760 |
| *(Node Class)* | | LapPE | 0.9741 | 0.9762 |
| | | RWSE | 0.9865 | **0.9972** |
| | | GPSE | 0.9775 | 0.9634 |
| **Peptides-Struct**| MAE (↓) | noPE | 0.2524 | 0.2541 |
| *(LRGB Reg)* | | LapPE | **0.2444** | **0.2483** |
| | | RWSE | 0.2556 | 0.2508 |
| | | GPSE | 0.2680 | 0.2568 |
| **Peptides-Func** | AP (↑) | noPE | 0.6214 | 0.6058 |
| *(LRGB Class)* | | LapPE | **0.6489** | 0.5856 |
| | | RWSE | 0.6456 | 0.6012 |
| | | GPSE | 0.6321 | **0.6401** |

## Performance Visualizations

### Dense GRIT
![Dense GRIT Benchmarks](results/dense_grit_benchmarks.png)

### GAT-GPS
![GAT-GPS Benchmarks](results/gat_gps_benchmarks.png)

## Archive Details
- **Total Experiments**: 32
- **Logs**: Archived raw logs are available in the repository's `plotting_data/` directory.
