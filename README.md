# Graph-based Android Authorship Attribution


Android applications are represented as call graphs (and, as an alternative,
control flow graphs). Node-level features (opcode, structural and centrality
features) and graph-level features (AppAuth, source code-based, permission and
library features, 3-node motifs) are extracted, and four GNN architectures
(GCN, GraphSAGE, GIN, GAT) are compared with conventional machine learning
models (Random Forest, XGBoost, Logistic Regression, SVM, MLP) using the same
features, and with a structure-only Weisfeiler–Lehman (WL) graph kernel.

## Repository layout

| Folder | Content  |
|---|---|
| `data/` | How to obtain the datasets; fold generation scripts | 
| `feature_extraction/call_graph/` | Call graph and node-level feature extraction with Androguard | 
| `feature_extraction/graph_features/` | Source code-based, permission and library features; AppAuth (external code, see its README) | 
| `feature_extraction/motifs/` | 3-node motif features |
| `feature_extraction/cfg/` | Control flow graph extraction and statistics | 
| `models/gnn/` | GCN, GraphSAGE, GIN, GAT (NF only and NF + GF), 10-fold CV | 
| `models/ml/` | RF, XGBoost, LR, SVM, MLP|
| `models/wl_kernel/` | WL graph kernel + SVM | 
| `obfuscation/` | Obfuscation with Obfuscapk, Orig→Obf and Obf→Obf experiments | 

## Setup

The main experiments were run with Python 3.6.8. The versions in
`requirements.txt` are those of this main environment (for the scripts run on
TRUBA, see *TRUBA environment* below).

1. Install PyTorch 1.10.1 for your CUDA version (the experiments used CUDA 11.1),
   following https://pytorch.org/get-started/previous-versions/ , e.g.

   ```bash
   pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
   ```

2. Install PyTorch Geometric 2.0.3 and its extensions (torch-scatter 2.0.9,
   torch-sparse 0.6.12, torch-cluster 1.5.9, torch-spline-conv 1.2.1) for this
   PyTorch / CUDA version, following
   https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html

3. Install the remaining packages (the PyTorch and PyG entries in
   `requirements.txt` are then already satisfied):

   ```bash
   pip install -r requirements.txt
   ```

The original experiments were run on a Linux server (GPU: Quadro GV100) and
on the TRUBA HPC cluster (`*.sh` files are SLURM job scripts).

### TRUBA environment

The following scripts were run on TRUBA with Python 3.9.16 and the package
versions in `requirements-truba.txt` (PyTorch 2.2.2 with CUDA 11.8, PyTorch
Geometric 2.6.1):

- `obfuscation/` (Orig→Obf, Obf→Obf) for testing

All other scripts were run in the main environment described above.

### Paths

The scripts originally used absolute paths on the authors' machines. These
have been replaced by three environment variables; the folder structure below
them is the same as in the original setup (for example
`$AA_DATA_ROOT/{dataset}_AppAuth_SPL_Motif_node_features_ALL.pt`).

| Variable | Default | Used for |
|---|---|---|
| `AA_DATA_ROOT` | `data` | Main data and result folders (server) |
| `AA_SCRATCH_ROOT` | `data` | Data and results of the TRUBA runs |
| `AA_LOCAL_ROOT` | `data` | A few local (Windows) paths |

```bash
export AA_DATA_ROOT=/path/to/your/data
```

## Typical order

1. Obtain the datasets (`data/README.md`).
2. Extract call graphs and node-level features (`feature_extraction/call_graph/`),
   graph-level features (`feature_extraction/graph_features/`) and motifs
   (`feature_extraction/motifs/`), and save them as `.pt` files.
3. Create the fold files with `data/scripts/`.
4. Run the main experiments in `models/`.

The main results are averages over seeds 34, 61 and 70. 
`models/gnn/run_all_seeds.sh` creates the folds and runs all main
GNN experiments for the three seeds:

```bash
bash models/gnn/run_all_seeds.sh genome
bash models/gnn/run_all_seeds.sh malware
```

## Notes

- APK files are not distributed (see `data/README.md`).

## License

This code is released under the MIT License (see `LICENSE`). The AppAuth-based
scripts in `feature_extraction/graph_features/` are adapted from AppAuth
(https://github.com/buptkick/AppAuth); the AppAuth, LiteRadar / LibRadar and
`dex_parser.py` code needed to run them is not included and keeps its own terms.
The source code-based, permission and library features follow Aydogan and Sen (2024);
their library and permission lists are taken from
https://github.com/emreaydogan/SourceCodeBasedAAA and are not included here.

## Citation

If you use this code, please cite:

> E. Uysal and S. Sen, "Graph-Based Learning for Android Authorship Attribution:
> A Comparative Analysis of GNN Models," *Applied Sciences*, vol. 16, no. 17,
> p. 8759, 2026. doi: [10.3390/app16178759](https://doi.org/10.3390/app16178759)


```bibtex
@article{uysal2026graph,
  author  = {Uysal, Elif and Sen, Sevil},
  title   = {Graph-Based Learning for Android Authorship Attribution: A Comparative Analysis of GNN Models},
  journal = {Applied Sciences},
  volume  = {16},
  number  = {17},
  pages   = {8759},
  year    = {2026},
  doi     = {10.3390/app16178759}
}

```
