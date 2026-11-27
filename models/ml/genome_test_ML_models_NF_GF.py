"""
TEST
Machine Learning Algorithms
Node-level + Graph-level features
"""
import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")

import copy
import pickle
import time

import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier

start_time = time.time()

names = ["RF", "logisticRegression", "svc", "xgboost", "mlp"]


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
    f=f"{DATA_ROOT}/genome_AppAuth_SPL_Motif_node_features_ALL.pt",
    map_location='cpu')

log_file = open("log_nf_gf.txt", "w")
for name in names:
    try:
        all_accuracy = []
        all_f1_weighted = []
        all_f1_macro = []
        all_precision = []
        all_recall = []

        final_report = open(f"./genome_nf_gf_results/{name}_NF_GF_output.txt", "w")
        final_report.write(f"{name}\n")
        final_report.write("Genome dataset - 10-fold stratified cross-validation\n")
        final_report.write("Node-level and graph-level features...\n")

        seeds = [34, 61, 70]
        for seed in seeds:
            seed_everything(seed)

            print(f"seed={seed}")
            torch.manual_seed(seed)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"device: {device}")

            if name == "RF":
                ml_model = RandomForestClassifier(random_state=seed, n_jobs=40)
            elif name == "logisticRegression":
                ml_model = LogisticRegression(random_state=seed, n_jobs=40)
            elif name == "svc":
                ml_model = SVC(random_state=seed)
            elif name == "xgboost":
                ml_model = XGBClassifier(random_state=seed, n_jobs=40)
            elif name == "mlp":
                ml_model = MLPClassifier(random_state=seed)

            final_report.write(f"Model: {ml_model}\n")
            final_report.write(f"Seed: {seed}\n")
            print("Genome dataset - 10-fold stratified cross-validation")
            print("Number of data:", len(X))
            labels = [int(data.y) for data in X]
            print(labels)
            print(f"Number of classes: {len(set(labels))}")

            y = np.array(labels)

            with open(f"genome_fold_indices_seed{seed}.pkl", "rb") as f:
                fold_indices = pickle.load(f)

            start_index = 1
            end_index = 11

            fold_accuracy = []
            fold_f1_weighted = []
            fold_f1_macro = []
            fold_precision = []
            fold_recall = []
            for fold_index in range(start_index, end_index):
                myout = open(f"./nf_gf_fold_results/{name}_genome_NF_GF_result_{fold_index}_fold_cv_seed{seed}.txt", "w")
                print(f"Fold {fold_index}:")
                print(f"  Train: index={fold_indices[fold_index]['train_index']}")
                print(f"  Test: index={fold_indices[fold_index]['test_index']}")

                train_graphs = [copy.deepcopy(X[index]) for index in fold_indices[fold_index]['train_index']]
                test_dataset = [copy.deepcopy(X[index]) for index in fold_indices[fold_index]['test_index']]

                train_labels = [labels[idx] for idx in fold_indices[fold_index]['train_index']]

                train_idx2, val_idx = train_test_split(list(range(len(train_graphs))),
                                                       test_size=0.1,
                                                       stratify=train_labels,
                                                       random_state=seed)

                train_dataset = [train_graphs[i] for i in train_idx2]
                validation_dataset = [train_graphs[i] for i in val_idx]

                # imputer
                # SimpleImputer...
                impute = SimpleImputer(missing_values=np.nan, strategy='mean')
                train_temp = []
                for d in train_dataset:
                    train_temp.append(np.array(d.graph_feats, dtype=float))

                impute.fit(train_temp)
                train_imp_values = impute.transform(train_temp)
                for train_imp_value, graph in zip(train_imp_values, train_dataset):
                    graph.graph_feats = torch.tensor(train_imp_value, dtype=torch.float32)

                val_temp = []
                for d in validation_dataset:
                    val_temp.append(np.array(d.graph_feats, dtype=float))

                validation_imp_values = impute.transform(val_temp)
                for val_imp_value, graph in zip(validation_imp_values, validation_dataset):
                    graph.graph_feats = torch.tensor(val_imp_value, dtype=torch.float32)

                test_temp = []
                for d in test_dataset:
                    test_temp.append(np.array(d.graph_feats, dtype=float))

                test_imp_values = impute.transform(test_temp)
                for test_imp_value, graph in zip(test_imp_values, test_dataset):
                    graph.graph_feats = torch.tensor(test_imp_value, dtype=torch.float32)

                # Variance Threshold
                train_features = [data.graph_feats.tolist() for data in train_dataset]
                selector = VarianceThreshold()
                selector.fit(train_features)
                train_sel_values = selector.transform(train_features)
                for train_sel_value, graph in zip(train_sel_values, train_dataset):
                    graph.graph_feats = torch.tensor(train_sel_value, dtype=torch.float32)

                val_features = [data.graph_feats.tolist() for data in validation_dataset]
                validation_sel_values = selector.transform(val_features)
                for val_sel_value, graph in zip(validation_sel_values, validation_dataset):
                    graph.graph_feats = torch.tensor(val_sel_value, dtype=torch.float32)

                test_features = [data.graph_feats.tolist() for data in test_dataset]

                test_sel_values = selector.transform(test_features)
                for test_sel_value, graph in zip(test_sel_values, test_dataset):
                    graph.graph_feats = torch.tensor(test_sel_value, dtype=torch.float32)

                # Standard Scaler
                scaler_node = StandardScaler()
                scaler_graph = StandardScaler()

                # scaler fit only train dataset
                train_node_features = torch.cat([graph.x for graph in train_dataset], dim=0).numpy()
                train_graph_features = torch.stack([graph.graph_feats for graph in train_dataset], dim=0).numpy()
                scaler_node.fit(train_node_features)
                scaler_graph.fit(train_graph_features)

                # scaler transform train/val/test dataset
                for graph in train_dataset:
                    normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
                    graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)
                    normalized_graph_feats = scaler_graph.transform(graph.graph_feats.reshape(1, -1))
                    graph.graph_feats = torch.tensor(normalized_graph_feats.reshape(-1), dtype=torch.float32)

                for graph in validation_dataset:
                    normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
                    graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)
                    normalized_graph_feats = scaler_graph.transform(graph.graph_feats.reshape(1, -1))
                    graph.graph_feats = torch.tensor(normalized_graph_feats.reshape(-1), dtype=torch.float32)

                for graph in test_dataset:
                    normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
                    graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)
                    normalized_graph_feats = scaler_graph.transform(graph.graph_feats.reshape(1, -1))
                    graph.graph_feats = torch.tensor(normalized_graph_feats.reshape(-1), dtype=torch.float32)

                # start training algorithm
                clf = ml_model

                # train
                train_data = []
                train_y = []
                with torch.no_grad():
                    for data in train_dataset:
                        node_feats = data.x.numpy()
                        node_mean = node_feats.mean(axis=0)
                        node_max = node_feats.max(axis=0)
                        node_std = node_feats.std(axis=0)
                        node_aggr = np.concatenate([node_mean, node_max, node_std])
                        graph_node_aggr = np.concatenate([node_aggr, data.graph_feats])
                        train_data.append(graph_node_aggr)
                        train_y.append(data.y.item())

                print(f"length of the train data: {len(train_data)}")
                myout.write(f"length of the train data: {len(train_data)}\n")
                print(f"length of the train label: {len(train_y)}")
                myout.write(f"length of the train label: {len(train_y)}\n")

                clf.fit(train_data, train_y)

                test_data = []
                test_y = []

                with torch.no_grad():
                    for data in test_dataset:
                        node_feats = data.x.numpy()
                        node_mean = node_feats.mean(axis=0)
                        node_max = node_feats.max(axis=0)
                        node_std = node_feats.std(axis=0)
                        node_aggr = np.concatenate([node_mean, node_max, node_std])
                        graph_node_aggr = np.concatenate([node_aggr, data.graph_feats])
                        test_data.append(graph_node_aggr)
                        test_y.append(data.y.item())

                print(f"length of the test data: {len(test_data)}")
                myout.write(f"length of the test data: {len(test_data)}\n")
                print(f"length of the test label: {len(test_y)}")
                myout.write(f"length of the test label: {len(test_y)}\n")

                test_pred = clf.predict(test_data)

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
    except:
        log_file.write("Error.\n")
        log_file.write(f"{name} algorithm gets error...\n")

log_file.close()
