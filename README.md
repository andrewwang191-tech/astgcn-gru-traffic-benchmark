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

## 🛠️ Tools & Frameworks

* **Language**: Python 3.x
* **Deep Learning Framework**: PyTorch
* **Data Processing**: Pandas, NumPy
* **Compute Environment**: Google Colab (GPU)
* **Dataset**: Caltrans Performance Measurement System (PeMS) — 307 spatial sensor nodes

---

## 📁 Repository Structure
├── ASTGCN_vs_GRU_Benchmark.ipynb   # Complete PyTorch training pipeline & loss curve visualization
├── README.md                       # Project documentation & summary of findings


---

## 🚀 Methodology Summary

1. **Preprocessing**: Normalized traffic flow/speed arrays across 307 nodes; constructed spatial adjacency matrices derived from node distance topologies.
2. **Model Training**: 
   * **Optimizer**: Adam
   * **Epochs**: 80
   * **Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
3. **Ablation Experiment**: Replaced temporal convolution blocks with multi-layer GRU units while keeping spatial attention components identical to isolate temporal layer impact.

---

## 📜 Attribution & Acknowledgments

This codebase is adapted from the official PyTorch implementation of **ASTGCN**:
* Paper: *Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting* (Guo et al., AAAI 2019).
* Original Repository: [Guoziwei/ASTGCN-2019-pytorch](https://github.com/Guoziwei/ASTGCN-2019-pytorch)

### Modifications Made in This Repository:
* Added custom `ASTGCN_r_new.py` implementing GRU/RNN temporal layers in place of standard dilated temporal convolutions.
* Created benchmarking scripts (`train_ASTGCN_r_new.py`) to evaluate horizon decay and gradient stability.

---

## 👤 Author

**Andrew Wang**  
*Research Conducted during High School Summer Internship*  
Department of Systems Engineering | City University of Hong Kong

## 👤 Author

**Andrew [Your Last Name]**  
*Research Conducted during High School Summer Internship*  
Department of Systems Engineering | City University of Hong Kong
