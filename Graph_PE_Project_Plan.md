# Project Plan: Graph Positional Encoding (PE) Analysis

## 1. Project Overview
This project investigates the impact of different Positional and Structural Encodings (PE/SE) on Graph Transformer and GAT architectures. The goal is to benchmark classic spectral methods (LapPE) against diffusion-based methods (RWSE) and foundation-based encoders (GPSE).

## 2. Infrastructure & Environment
* **Primary Framework:** [ETH-DISCO Benchmarking-PEs](https://github.com/ETH-DISCO/Benchmarking-PEs)
* **Backbone Architectures:** * **Vanilla GT:** Standard Multi-Head Self-Attention.
    * **GAT:** Graph Attention Network baseline.
* **Compute:** Duke Compute Cluster (DCC) / Local GPU.

## 3. Data Modalities
| Dataset | Task | Graph Nature | Metric |
| :--- | :--- | :--- | :--- |
| **ZINC** | Regression | Homogeneous | MAE |
| **Pascal-VOC** | Node Classification | Homogeneous | F1-score |
| **IMDB** | Node Classification | Heterogeneous | F1-score |

## 4. Encoding Evaluation Matrix
1.  **Laplacian PE (LapPE):** Spectral eigenvectors (global geometry).
2.  **Random Walk SE (RWSE):** Return probabilities $P^k_{ii}$ (local topology).
3.  **GPSE:** Pre-trained Foundation Structural Encodings.

## 5. Implementation Roadmap

### Step 1: Preprocessing
* Generate Laplacian Eigenvectors for ZINC, Pascal-VOC, and IMDB.
* Compute Random Walk transition matrices ($P = D^{-1}A$) and calculate diagonals for $k=1...20$.

### Step 2: Configuration & Integration
* Configure `Benchmarking-PEs` to run in "Vanilla" mode (standard Transformer blocks).
* Add MLP projection layers to map PE/SE dimensions to the hidden model dimension ($d$).
* Integrate GPSE embeddings as additional node features.

### Step 3: Execution & Analysis
* Perform hyperparameter grid search (LR, Dropout, Attention Heads).
* Log results to compare performance variance across encoding strategies.
* Analyze the effectiveness of RWSE on the heterogeneous IMDB graph vs. homogeneous baselines.
