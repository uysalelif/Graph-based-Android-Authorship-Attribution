# Graph-based Android Authorship Attribution

Graph-based Android authorship attribution using call graphs, multi-level (node-level and graph-level) feature representation, and Graph Neural Networks (GNNs).

This repository contains the code accompanying for identifying the author(s) of Android applications from their call-graph structure and code-derived features, using both GNN-based classifiers and classical machine learning baselines.

## Overview

Android applications are represented as **call graphs**, from which we extract:

- **Node-level features**: structure-based, opcode-based, and method-type-based
- **Graph-level features**: handcrafted (independent of the graph) and motif-based (graph-dependent)

These representations are used to train and evaluate:

- **GNN models**: GraphSAGE, GIN, GCN, GAT
- **Classical ML baselines**: Random Forest, SVM, Logistic Regression, XGBoost, 
- **Graph kernel baseline**: Weisfeiler-Lehman (WL) graph kernel + SVM

The repository also includes an **obfuscation robustness analysis**, evaluating how different obfuscation techniques affect call-graph structure and classification performance.

## Repository Structure

```
.
├── data/               # Raw and processed datasets (call graphs, extracted features)
├── extraction/         # Call graph extraction and feature extraction scripts
├── models/             # GNN model architectures (GraphSAGE, GIN, GCN, GAT)
├── baselines/          # Classical ML baselines (RF, SVM, LR, XGBoost, WL kernel)
├── obfuscation/        # Obfuscation analysis scripts and results
├── utils/              # Shared utilities (data loading, evaluation, etc.)
└── README.md
```


## Installation

```bash
git clone https://github.com/<your-username>/Graph-based-Android-Authorship-Attribution.git
cd Graph-based-Android-Authorship-Attribution
pip install -r requirements.txt
```

## Usage

```bash
# Example: extract call graphs and features
python3 extraction/extract_call_graphs.py

# Example: train a GNN model
python3 models/train_gnn.py

# Example: run a classical ML baseline
python3 baselines/train_rf.py 
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_citation_key,
  title   = {Your Paper Title},
  author  = {Your Name and Co-authors},
  journal = {Applied Sciences},
  year    = {2026},
  doi     = {your-doi-here}
}
```

