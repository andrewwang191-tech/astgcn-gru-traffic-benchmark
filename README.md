# Spatiotemporal Traffic Flow Forecasting: ASTGCN vs. GRU Architecture Benchmark

An empirical performance evaluation comparing Attention-based Spatial-Temporal Graph Convolutional Networks (**ASTGCN**) against a hybrid **GRU-RNN** temporal architecture on spatiotemporal traffic sensor networks.

---

## 📌 Project Overview

Accurate urban traffic flow prediction requires simultaneous modeling of complex spatial dependencies across road networks and dynamic temporal trends over time. This research evaluates two distinct temporal modeling approaches:
1. **ASTGCN Baseline**: Standard architecture employing spatial and temporal attention mechanisms combined with Spatial Graph Convolutions and Temporal Convolutional Networks (TCN).
2. **ASTGCN-GRU Variant**: A modified architecture replacing dilated temporal convolutions with a Gated Recurrent Unit (GRU) to evaluate recurrent step-by-step state updates versus non-recurrent convolution gates.

Both architectures were implemented in PyTorch and trained over 80 epochs using GPU acceleration on Google Colab, evaluated across 307 traffic sensor nodes from the California Department of Transportation PeMS dataset.

---

## 🔑 Key Empirical Findings

* **Horizon Decay Rate**: The GRU-augmented model exhibited an **8.2% higher Mean Absolute Error (MAE) decay** over longer prediction horizons (e.g., 60-minute predictions) compared to the standard TCN baseline.
* **Vanishing Gradient Analysis**: Recurrent hidden state updates suffered from gradient degradation across extended temporal sequences, demonstrating why non-recurrent dilated convolutions perform superiorly in capturing long-term dependencies across spatial graph nodes.

---

## 📊 Performance & Visualizations

*(Upload your generated result images into the `fig/` folder and link them here)*

| Architecture | 15-min MAE | 30-min MAE | 60-min MAE | Temporal Layer Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **ASTGCN (Baseline)** | Standard | Baseline | Baseline | Dilated Temporal Convolutions (TCN) |
| **ASTGCN-GRU (Ours)** | Comparative | Higher | **+8.2% Decay** | Multi-Layer Gated Recurrent Unit (GRU) |

![Architecture & Findings](fig/ASTGCN%20architecture.png)

---

## 🛠️ Tools & Frameworks

* **Language**: Python 3.x
* **Deep Learning Framework**: PyTorch
* **Data Processing**: Pandas, NumPy
* **Compute Environment**: Google Colab (GPU)
* **Dataset**: Caltrans Performance Measurement System (PeMS) — 307 spatial sensor nodes

---

## 📁 Repository Structure

```text
astgcn-gru-traffic-benchmark/
├── configurations/          # Network configuration parameters (.conf)
│   ├── METR_LA_astgcn.conf
│   ├── PEMS04_astgcn.conf
│   └── PEMS08_astgcn.conf
├── fig/                     # Architecture diagrams and evaluation plots
│   └── ASTGCN architecture.png
├── lib/                     # Data loaders and metric computation utilities
│   ├── metrics.py
│   └── utils.py
├── logs/                    # Training output logs
│   ├── NewModelRNN_Training.txt
│   └── OldModelTCN_Training.txt
├── model/                   # PyTorch neural network class definitions
│   ├── ASTGCN_r.py          # Baseline ASTGCN model
│   ├── ASTGCN_r_new.py      # Custom GRU-augmented ASTGCN variant
│   └── MSTGCN_r.py          # Multi-component Spatial-Temporal Graph Conv
├── prepareData.py           # Dataset preprocessing and matrix construction
├── train_ASTGCN_r.py        # Baseline training pipeline
├── train_ASTGCN_r_new.py    # GRU variant training pipeline
├── AAAI-GuoS.2690.pdf       # Reference paper (Guo et al., 2019)
└── README.md                # Project documentation


## 🚀 Methodology Summary
Preprocessing: Normalized traffic flow/speed arrays across 307 nodes; constructed spatial adjacency matrices derived from node distance topologies.

Model Training:

Optimizer: Adam

Epochs: 80

Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

Ablation Experiment: Replaced temporal convolution blocks with multi-layer GRU units while keeping spatial attention components identical to isolate temporal layer impact.

## 📜 Attribution & Acknowledgments
This codebase is adapted from the official PyTorch implementation of ASTGCN:

Paper: Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting (Guo et al., AAAI 2019).

Original Repository: Guoziwei/ASTGCN-2019-pytorch

Modifications Made in This Repository:
Developed custom ASTGCN_r_new.py incorporating GRU/RNN temporal layers in place of standard dilated temporal convolutions.

Built benchmarking scripts (train_ASTGCN_r_new.py) to evaluate horizon decay and gradient stability.

##👤 Author
Andrew Wang

Research Conducted during High School Summer Internship

Department of Systems Engineering | City University of Hong Kong
