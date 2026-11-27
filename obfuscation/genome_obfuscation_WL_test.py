"""
TEST
WL Kernel
"""
import os as _os  # repo: paths come from environment variables (see README)
SCRATCH_ROOT = _os.environ.get("AA_SCRATCH_ROOT", "data")
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
from original_train_obfuscated_test_split import build_fixed_folds, build_condition_data


start_time = time.time()

dataset_type = "genome"
TECHNIQUES = ["ClassRename", "FieldRename", "MethodRename",
              "ConstStringEncryption", "LibEncryption",
              "ResStringEncryption"]


def to_grakel_list(pyg_data_list):
    WL_graphs = []
    for data in pyg_data_list:
        node_labels={i:0 for i in range(data.num_nodes)}
        WL_graphs.append(grakel.Graph(list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())),
                                      node_labels=node_labels))
    return WL_graphs


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


INTERSECTION_JSON = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/GENOME/genome_intersection_apk_ids.json"
ORIGINAL_PT_FILE = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/GENOME/{dataset_type}_original_node_only.pt"


for technique in TECHNIQUES:
    TECHNIQUE_PT_FILE = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/GENOME/{dataset_type}_{technique}_node_only.pt"

    original_graphs = torch.load(ORIGINAL_PT_FILE, weights_only=False)
    technique_graphs = torch.load(TECHNIQUE_PT_FILE, weights_only=False)

    print(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation")
    print("Number of data:", len(original_graphs))
    labels = [int(data.y) for data in original_graphs]
    # print(labels)
    # print(f"Number of classes: {len(set(labels))}")

    y = np.array(labels)

    all_accuracy = []
    all_f1_weighted = []
    all_f1_macro = []
    all_precision = []
    all_recall = []

    final_report = open(f"./{dataset_type}_{technique}_wl_kernel_results/{dataset_type}_{technique}_WL_kernel_output.txt", "w")
    # final_report.write(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation\n")
    final_report.write(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation \n"
                       f"(train=original, test={technique})\n")

    seeds = [34, 61, 70]
    for seed in seeds:
        seed_everything(seed)

        print(f"seed={seed}")
        torch.manual_seed(seed)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")

        fold_ids, label_map, dropped_authors = build_fixed_folds(
            INTERSECTION_JSON, n_splits=10, random_state=seed
        )
        total_final_apks = len(fold_ids[0][0]) + len(fold_ids[0][1])
        final_report.write(f"Number of data (final ortak APK): {total_final_apks}\n")
        final_report.write(f"Number of classes: {len(label_map)}\n")
        final_report.write(f"Dropped authors (min_samples altinda): {len(dropped_authors)}\n")

        start_index = 1
        end_index = 11

        fold_accuracy = []
        fold_f1_weighted = []
        fold_f1_macro = []
        fold_precision = []
        fold_recall = []
        for fold_index, (train_ids, test_ids) in enumerate(fold_ids, start=1):
            myout = open(f"./{dataset_type}_{technique}_wl_kernel_results/{dataset_type}_{technique}_WL_kernel_result_{fold_index}_fold_cv_seed{seed}.txt", "w")
            print(f"Fold {fold_index}: train={len(train_ids)} APK (original), test={len(test_ids)} APK ({technique})")
            myout.write(f"Fold {fold_index}: train={len(train_ids)} APK (original), test={len(test_ids)} APK ({technique})\n")

            train_graphs = [copy.deepcopy(d) for d in build_condition_data(original_graphs, train_ids, label_map)]
            test_dataset_pyg = [copy.deepcopy(d) for d in build_condition_data(technique_graphs, test_ids, label_map)]

            train_labels = [int(d.y.item()) for d in train_graphs]
            test_y = [int(d.y.item()) for d in test_dataset_pyg]

            train_idx2, val_idx = train_test_split(list(range(len(train_graphs))),
                                                   test_size=0.1,
                                                   stratify=train_labels,
                                                   random_state=seed)

            train_dataset_pyg = [train_graphs[i] for i in train_idx2]
            train_y = [train_labels[i] for i in train_idx2]
            validation_dataset_pyg = [train_graphs[i] for i in val_idx]

            train_dataset = to_grakel_list(train_dataset_pyg)
            test_dataset = to_grakel_list(test_dataset_pyg)

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

    final_report.write(f"Accuracy: {np.mean(all_accuracy)*100:.2f} $\pm$ {np.std(all_accuracy, ddof=1)*100:.2f}\n")
    final_report.write(f"f1 weighted: {np.mean(all_f1_weighted)*100:.2f} $\pm$ {np.std(all_f1_weighted, ddof=1)*100:.2f}\n")
    final_report.write(f"f1 macro: {np.mean(all_f1_macro)*100:.2f} $\pm$ {np.std(all_f1_macro, ddof=1)*100:.2f}\n")
    final_report.write(f"precision: {np.mean(all_precision)*100:.2f} $\pm$ {np.std(all_precision, ddof=1)*100:.2f}\n")
    final_report.write(f"recall: {np.mean(all_recall)*100:.2f} $\pm$ {np.std(all_recall, ddof=1)*100:.2f}\n")

    end_time = time.time()
    execution_time = end_time - start_time
    final_report.write(f"Execution time: {int(execution_time/60)} minutes.\n")
    final_report.close()
