# Comparative Analysis: PE Benchmarks

## 1. Winrate against noPE
| PE Variant | Winrate |
| :--- | :--- |
| LapPE | 10/13 (76.9%) |
| RWSE | 9/13 (69.2%) |
| GPSE | 6/13 (46.2%) |

## 2. Percentage Improvement Over noPE
| Dataset | Model | LapPE | RWSE | GPSE |
| :--- | :--- | :---: | :---: | :---: |
| ZINC | Dense | 0.34% | 3.84% | -13.24% |
| ZINC | GAT | 22.24% | 29.39% | 28.05% |
| ZINC | Sparse | 24.59% | 46.86% | 58.82% |
| IMDB | Dense | -1.33% | -0.07% | -0.98% |
| IMDB | GAT | 0.02% | 2.17% | -1.29% |
| IMDB | Sparse | -0.30% | 0.31% | -0.93% |
| Peptides-Struct | Dense | 3.17% | -1.27% | -6.18% |
| Peptides-Struct | GAT | 2.28% | 1.30% | -1.06% |
| Peptides-Struct | Sparse | 5.27% | -0.23% | -30.91% |
| Peptides-Func | Dense | 4.43% | 3.89% | 1.72% |
| Peptides-Func | GAT | -3.33% | -0.76% | 5.66% |
| Peptides-Func | Sparse | 17.43% | 9.61% | 27.96% |
| MolPCBA | Sparse | 1.84% | 22.00% | 14.35% |

## 3. Best PE for each Task + Model
| Dataset | Model | Best PE |
| :--- | :--- | :--- |
| ZINC | Dense | **RWSE** |
| ZINC | GAT | **RWSE** |
| ZINC | Sparse | **GPSE** |
| IMDB | Dense | **noPE** |
| IMDB | GAT | **RWSE** |
| IMDB | Sparse | **RWSE** |
| Peptides-Struct | Dense | **LapPE** |
| Peptides-Struct | GAT | **LapPE** |
| Peptides-Struct | Sparse | **LapPE** |
| Peptides-Func | Dense | **LapPE** |
| Peptides-Func | GAT | **GPSE** |
| Peptides-Func | Sparse | **GPSE** |
| MolPCBA | Sparse | **RWSE** |

## 4. Average Improvement by Architecture
| Model Architecture | LapPE Avg | RWSE Avg | GPSE Avg |
| :--- | :---: | :---: | :---: |
| Dense | 1.65% | 1.60% | -4.67% |
| GAT | 5.30% | 8.03% | 7.84% |
| Sparse | 9.76% | 15.71% | 13.86% |