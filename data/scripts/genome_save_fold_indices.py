import os as _os  # repo: paths come from environment variables (see README)

DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import pickle

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric import seed_everything
from torch_geometric.data import Dataset


class AndroidGraphDataset(Dataset):
    def __init__(self):
        super(AndroidGraphDataset, self).__init__()
        self.custom_data = get_data_objects()
        self.graphs = self.custom_data[0]
        self.labels = self.custom_data[1]

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        return data


X = torch.load(f=f"{DATA_ROOT}/genome_AppAuth_SPL_Motif_node_features_ALL.pt")

seeds = [34, 61, 70]
for seed in seeds:
    seed_everything(seed)

    print("Number of data:", len(X))
    labels = [int(data.y) for data in X]
    print(labels)
    print(f"Number of classes: {len(set(labels))}")

    y = np.array(labels)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    skf.get_n_splits(X, y)
    print(f"skf: {skf}")

    fold_indices = {}
    for fold_index, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        fold_indices[fold_index] = {"train_index": train_index, "test_index": test_index}
        print(f"Train index: {train_index}")
        print(f"Test index: {test_index}")

    with open(f"genome_fold_indices_seed{seed}.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print("Saved fold indices!")
