"""
TEST
WL Kernel
"""
import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")

import copy
import pickle
import time

import grakel
import numpy as np
import torch
from grakel.kernels import WeisfeilerLehman
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch_geometric import seed_everything
from torch_geometric.data import Dataset

start_time = time.time()

dataset_type = "genome"

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


X = torch.load(
    f=f"{DATA_ROOT}/{dataset_type}_AppAuth_SPL_Motif_node_features_ALL.pt",
    map_location='cpu')

WL_graphs = []
for data in X:
    node_labels={i:0 for i in range(data.num_nodes)}
    WL_graphs.append(grakel.Graph(list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())), node_labels=node_labels))


print(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation")
print("Number of data:", len(X))
labels = [int(data.y) for data in X]
print(labels)
print(f"Number of classes: {len(set(labels))}")

y = np.array(labels)

all_accuracy = []
all_f1_weighted = []
all_f1_macro = []
all_precision = []
all_recall = []

final_report = open(f"./final_wl_kernel_results/{dataset_type}_WL_kernel_output.txt", "w")
final_report.write(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation\n")

seeds = [34, 61, 70]
for seed in seeds:
    seed_everything(seed)

    print(f"seed={seed}")
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    with open(f"{dataset_type}_fold_indices_seed{seed}.pkl", "rb") as f:
        fold_indices = pickle.load(f)

    start_index = 1
    end_index = 11

    fold_accuracy = []
    fold_f1_weighted = []
    fold_f1_macro = []
    fold_precision = []
    fold_recall = []
    for fold_index in range(start_index, end_index):
        myout = open(f"./final_wl_kernel_results/{dataset_type}_WL_kernel_result_{fold_index}_fold_cv_seed{seed}.txt", "w")
        print(f"Fold {fold_index}:")
        print(f"  Train: index={fold_indices[fold_index]['train_index']}")
        print(f"  Test: index={fold_indices[fold_index]['test_index']}")

        train_graphs = [copy.deepcopy(WL_graphs[index]) for index in fold_indices[fold_index]['train_index']]
        test_dataset = [copy.deepcopy(WL_graphs[index]) for index in fold_indices[fold_index]['test_index']]

        train_labels = [labels[idx] for idx in fold_indices[fold_index]['train_index']]
        test_y = [labels[idx] for idx in fold_indices[fold_index]['test_index']]

        train_idx2, val_idx = train_test_split(list(range(len(train_graphs))),
                                               test_size=0.1,
                                               stratify=train_labels,
                                               random_state=seed)

        train_dataset = [train_graphs[i] for i in train_idx2]
        train_y = [train_labels[i] for i in train_idx2]

        validation_dataset = [train_graphs[i] for i in val_idx]

        # define WL kernel model
        # Uses the Weisfeiler-Lehman subtree kernel to generate the kernel matrices
        gk = WeisfeilerLehman()
        K_train = gk.fit_transform(train_dataset)
        K_test = gk.transform(test_dataset)

        # Uses the SVM classifier to perform classification
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, train_y)
        test_pred = clf.predict(K_test)

        # calculate test accuracy
        test_accuracy = accuracy_score(test_y, test_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        myout.write(f"Test accuracy: {test_accuracy:.4f}\n")

        test_f1score = f1_score(test_y, test_pred, average='weighted')
        print(f"Test f1 score: {test_f1score:.4f}")
        myout.write(f"Test f1 score: {test_f1score:.4f}\n")

        test_f1score_macro = f1_score(test_y, test_pred, average='macro')
        print(f"Test f1 score macro: {test_f1score_macro:.4f}")
        myout.write(f"Test f1 score macro: {test_f1score_macro:.4f}\n")

        test_precision_score = precision_score(test_y, test_pred, average='weighted')
        print(f"Test precision score: {test_precision_score:.4f}")
        myout.write(f"Test precision score: {test_precision_score:.4f}\n")

        test_recall_score = recall_score(test_y, test_pred, average='weighted')
        print(f"Test recall score: {test_recall_score:.4f}")
        myout.write(f"Test recall score: {test_recall_score:.4f}\n")

        fold_accuracy.append(test_accuracy)
        fold_f1_weighted.append(test_f1score)
        fold_f1_macro.append(test_f1score_macro)
        fold_precision.append(test_precision_score)
        fold_recall.append(test_recall_score)

        final_report.write(f"Fold: {fold_index}, Accuracy: {test_accuracy:.4f}, F1 weighted score: {test_f1score:.4f}, "
                           f"F1 macro score: {test_f1score_macro:.4f}, Precision score: {test_precision_score:.4f}, "
                           f"Recall score: {test_recall_score:.4f}\n")

        myout.close()

    all_accuracy.append(sum(fold_accuracy)/len(fold_accuracy))
    all_f1_weighted.append(sum(fold_f1_weighted)/len(fold_f1_weighted))
    all_f1_macro.append(sum(fold_f1_macro)/len(fold_f1_macro))
    all_precision.append(sum(fold_precision)/len(fold_precision))
    all_recall.append(sum(fold_recall)/len(fold_recall))

final_report.write(f"Accuracy: {np.mean(all_accuracy)*100:.2f} $\pm$ {np.std(all_accuracy)*100:.2f}\n")
final_report.write(f"f1 weighted: {np.mean(all_f1_weighted)*100:.2f} $\pm$ {np.std(all_f1_weighted)*100:.2f}\n")
final_report.write(f"f1 macro: {np.mean(all_f1_macro)*100:.2f} $\pm$ {np.std(all_f1_macro)*100:.2f}\n")
final_report.write(f"precision: {np.mean(all_precision)*100:.2f} $\pm$ {np.std(all_precision)*100:.2f}\n")
final_report.write(f"recall: {np.mean(all_recall)*100:.2f} $\pm$ {np.std(all_recall)*100:.2f}\n")

end_time = time.time()
execution_time = end_time - start_time
final_report.write(f"Execution time: {int(execution_time/60)} minutes.\n")
final_report.close()
