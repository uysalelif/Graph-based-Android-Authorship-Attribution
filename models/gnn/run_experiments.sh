#!/bin/bash
# Runs the main experiments of the study on one dataset:
#   gnn : GAT, GCN, GIN, GraphSAGE (NF only and NF + GF), seeds 34, 61, 70
#   ml  : RF, XGBoost, LR, SVM, MLP (NF only, GF only, NF + GF)
#   wl  : Weisfeiler-Lehman graph kernel + SVM
# The fold files ({dataset}_fold_indices_seed{seed}.pkl) for all three seeds are
# created first. All outputs are written to the current directory.
#
# Usage (from the folder where results should be written):
#   export AA_DATA_ROOT=/path/to/data
#   bash /path/to/repo/run_experiments.sh genome            # all experiments
#   bash /path/to/repo/run_experiments.sh malware gnn ml    # only GNN and ML
set -e
DS=${1:?usage: run_experiments.sh genome|malware [gnn] [ml] [wl]}
shift
PARTS=${*:-gnn ml wl}
REPO=$(cd "$(dirname "$0")" && pwd)
SEEDS="34 61 70"

if [ "$DS" != "genome" ] && [ "$DS" != "malware" ]; then
  echo "dataset must be genome or malware"; exit 1
fi

mkdir -p gat_results gcn_results gin_results sage_results \
         gat_NFonly_results gcn_NFonly_results gin_NFonly_results sage_NFonly_results \
         nf_only_results gf_only_results nf_gf_results \
         ${DS}_nf_only_results ${DS}_gf_only_results ${DS}_nf_gf_results \
         nf_only_fold_results gf_only_fold_results nf_gf_fold_results \
         final_wl_kernel_results

echo "=== fold files ($DS) ==="
python "$REPO/data/scripts/${DS}_save_fold_indices.py"

for PART in $PARTS; do
  case $PART in
    gnn)
      for SEED in $SEEDS; do
        echo "=== GNN, $DS, seed $SEED ==="
        for M in GAT GCN GIN Sage; do
          python "$REPO/models/gnn/${DS}/${DS}_test_CV_${M}Conv.py" "$SEED"
          python "$REPO/models/gnn/${DS}/${DS}_test_${M}Conv_NodeFeatures_CV.py" "$SEED"
        done
      done ;;
    ml)
      for F in NF_only GF_only NF_GF; do
        echo "=== ML, $DS, $F ==="
        python "$REPO/models/ml/${DS}_test_ML_models_${F}.py"
      done ;;
    wl)
      echo "=== WL kernel, $DS ==="
      python "$REPO/models/wl_kernel/test_WL_kernel_${DS}.py" ;;
    *)
      echo "unknown part: $PART (use gnn, ml, wl)"; exit 1 ;;
  esac
done