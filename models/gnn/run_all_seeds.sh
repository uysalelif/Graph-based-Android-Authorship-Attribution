#!/bin/bash
# Runs the main GNN experiments (GAT, GCN, GIN, GraphSAGE; NF only and NF + GF)
# for the three seeds used in the thesis (34, 61, 70).
#
# For each seed, the fold file ({dataset}_fold_indices.pkl) is created first in
# the current directory and then read by the GNN scripts. Results are written to
# ./<model>_results/ and ./<model>_NFonly_results/ with the seed in the file name.
#
# Usage (from the folder where results should be written):
#   export AA_DATA_ROOT=/path/to/data
#   bash /path/to/repo/models/gnn/run_all_seeds.sh genome
#   bash /path/to/repo/models/gnn/run_all_seeds.sh malware
set -e
DS=${1:?usage: run_all_seeds.sh genome|malware}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SEEDS="34 61 70"

mkdir -p gat_results gcn_results gin_results sage_results \
         gat_NFonly_results gcn_NFonly_results gin_NFonly_results sage_NFonly_results

for SEED in $SEEDS; do
  echo "=== $DS, seed $SEED ==="
  if [ "$DS" = "genome" ]; then
    python "$REPO/data/scripts/genome_save_fold_indices.py" "$SEED"
    for M in GAT GCN GIN Sage; do
      python "$REPO/models/gnn/genome/genome_test_CV_${M}Conv.py" "$SEED"
      python "$REPO/models/gnn/genome/genome_test_${M}Conv_NodeFeatures_CV.py" "$SEED"
    done
  elif [ "$DS" = "malware" ]; then
    python "$REPO/data/scripts/save_fold_indices.py" "$SEED"
    for M in GAT GCN GIN Sage; do
      python "$REPO/models/gnn/malware/test_CV_${M}Conv.py" "$SEED"
      python "$REPO/models/gnn/malware/test_${M}Conv_NodeFeatures_CV.py" "$SEED"
    done
  else
    echo "dataset must be genome or malware"; exit 1
  fi
done
