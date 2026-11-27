"""
GraphSAGE (SAGEConv) obfuscation test:
train=original, test=obfuscated.
"""
import os as _os  # repo: paths come from environment variables (see README)
SCRATCH_ROOT = _os.environ.get("AA_SCRATCH_ROOT", "data")
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear
from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from original_train_obfuscated_test_split import build_fixed_folds, build_condition_data

start_time = time.time()

dataset_type = "malware"  # malware or genome
technique = "ClassRename"

INTERSECTION_JSON = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/intersection_apk_ids.json"
ORIGINAL_PT_FILE = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/{dataset_type}_original_node_only.pt"
TECHNIQUE_PT_FILE = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/{dataset_type}_{technique}_node_only.pt"


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


original_graphs = torch.load(ORIGINAL_PT_FILE, weights_only=False)
technique_graphs = torch.load(TECHNIQUE_PT_FILE, weights_only=False)

all_accuracy = []
all_f1_weighted = []
all_f1_macro = []
all_precision = []
all_recall = []

final_report = open(f"./sage_results/{dataset_type}_train_original_test_obfuscated_{technique}_sageconv_output.txt", "w")
final_report.write(f"Train (original): {ORIGINAL_PT_FILE}\n")
final_report.write(f"Test ({technique} obfuscated): {TECHNIQUE_PT_FILE}\n")
final_report.write("SAGEConv model\n")
final_report.write(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation\n")
final_report.write(f"{technique} obfuscation - CROSS-DOMAIN (train=original, test={technique})\n")
final_report.write("Only node-level features...\n")

seeds = [34, 61, 70]
for seed in seeds:
    final_report.write(f"\nThe results for seed {seed}:\n")
    seed_everything(seed)

    print(f"seed={seed}")
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    print(f"{dataset_type.upper()} dataset - 10-fold stratified cross-validation "
          f"(train=original, test={technique})")

    fold_ids, label_map, dropped_authors = build_fixed_folds(
        INTERSECTION_JSON, n_splits=10, random_state=seed
    )
    total_final_apks = len(fold_ids[0][0]) + len(fold_ids[0][1])
    final_report.write(f"Number of data: {total_final_apks}\n")
    final_report.write(f"Number of classes: {len(label_map)}\n")
    final_report.write(f"Dropped authors: {len(dropped_authors)}\n")


    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels * 2, out_channels)

        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Readout layer
            x_mean = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            x_max = global_max_pool(x, batch)  # [batch_size, hidden_channels]
            x = torch.cat([x_mean, x_max], dim=1)

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)

            x = self.lin(x)

            return x


    def train(train_data_loader):
        model.train()

        for data in train_data_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = 0
        predictions = []
        ground_truths = []
        total_loss = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            total_correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            predictions.extend(pred.tolist())
            ground_truths.extend(data.y.tolist())
            loss = criterion(out, data.y)
            total_loss += float(loss) * data.y.size(0)
        acc = total_correct / len(loader.dataset)  # Derive ratio of correct predictions.
        loss = total_loss / len(loader.dataset)
        return acc, predictions, ground_truths, loss


    fold_accuracy = []
    fold_f1_weighted = []
    fold_f1_macro = []
    fold_precision = []
    fold_recall = []
    for fold_index, (train_ids, test_ids) in enumerate(fold_ids, start=1):
        myout = open(f"./sage_results/{dataset_type}_{technique}_result_{fold_index}_fold_cv_seed{seed}.txt", "w")
        print(f"Fold {fold_index}: train={len(train_ids)} APK (original), test={len(test_ids)} APK ({technique})")

        train_graphs = [copy.deepcopy(d) for d in build_condition_data(original_graphs, train_ids, label_map)]
        test_dataset = [copy.deepcopy(d) for d in build_condition_data(technique_graphs, test_ids, label_map)]

        train_labels = [int(d.y.item()) for d in train_graphs]

        train_idx2, val_idx = train_test_split(list(range(len(train_graphs))),
                                               test_size=0.1,
                                               stratify=train_labels,
                                               random_state=seed)

        train_dataset = [train_graphs[i] for i in train_idx2]
        validation_dataset = [train_graphs[i] for i in val_idx]

        # Standard Scaler
        scaler_node = StandardScaler()

        # scaler fit only train dataset
        train_node_features = torch.cat([graph.x for graph in train_dataset], dim=0).numpy()
        scaler_node.fit(train_node_features)

        # scaler transform train/val/test dataset
        for graph in train_dataset:
            normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
            graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)

        for graph in validation_dataset:
            normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
            graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)

        for graph in test_dataset:
            normalized_node_feats = scaler_node.transform(graph.x)  # normalize node features
            graph.x = torch.tensor(normalized_node_feats, dtype=torch.float32)

        # get data loader
        train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_data_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        node_feats_size = train_dataset[0].x.shape[1]
        print(f"node features size: {node_feats_size}")

        # model
        model = GNN(in_channels=node_feats_size,
                    hidden_channels=64,
                    out_channels=len(label_map))

        model = model.to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_accuracy = 0.0
        best_model = None
        early_stopping = 0
        patience = 50
        best_val_loss = float('inf')
        for epoch in range(1, 501):
            # TRAIN
            train(train_data_loader)
            val_acc, val_predictions, val_ground_truths, val_loss = test(val_data_loader)

            print(f"Fold: {fold_index}, Epoch {epoch:03d}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            myout.write(f"Fold: {fold_index}, Epoch {epoch:03d}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}\n")

            if val_acc > best_val_accuracy:
                print("best model is saving...")
                best_val_accuracy = val_acc
                # save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }, f"./sage_results/{dataset_type}_{technique}_best_model_fold_{fold_index}_seed{seed}.pth")

            if val_loss >= best_val_loss:
                early_stopping += 1
                if early_stopping >= patience:
                    print(f'Pretrain Early stopped by Epoch: {epoch}')
                    myout.write(f'Pretrain Early stopped by Epoch: {epoch}\n')
                    break
            else:
                best_val_loss = val_loss
                early_stopping = 0

        # load the best model
        model = GNN(in_channels=node_feats_size,
                    hidden_channels=64,
                    out_channels=len(label_map))
        model.load_state_dict(
            torch.load(f"./sage_results/{dataset_type}_{technique}_best_model_fold_{fold_index}_seed{seed}.pth",
                       map_location=device)[
                'model_state_dict'])
        model.eval()
        model = model.to(device)

        # calculate test accuracy
        test_accuracy, test_preds, test_labels, test_loss = test(test_data_loader)
        print(f"Test accuracy: {test_accuracy:.4f}")
        myout.write(f"Test accuracy: {test_accuracy:.4f}\n")
        print(f"Test Loss: {test_loss:.4f}")
        myout.write(f"Test Loss: {test_loss:.4f}\n")

        test_f1score = f1_score(test_labels, test_preds, average='weighted')
        print(f"Test f1 score: {test_f1score:.4f}")
        myout.write(f"Test f1 score: {test_f1score:.4f}\n")

        test_f1score_macro = f1_score(test_labels, test_preds, average='macro')
        print(f"Test f1 score macro: {test_f1score_macro:.4f}")
        myout.write(f"Test f1 score macro: {test_f1score_macro:.4f}\n")

        test_precision_score = precision_score(test_labels, test_preds, average='weighted')
        print(f"Test precision score: {test_precision_score:.4f}")
        myout.write(f"Test precision score: {test_precision_score:.4f}\n")

        test_recall_score = recall_score(test_labels, test_preds, average='weighted')
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

    all_accuracy.append(sum(fold_accuracy) / len(fold_accuracy))
    all_f1_weighted.append(sum(fold_f1_weighted) / len(fold_f1_weighted))
    all_f1_macro.append(sum(fold_f1_macro) / len(fold_f1_macro))
    all_precision.append(sum(fold_precision) / len(fold_precision))
    all_recall.append(sum(fold_recall) / len(fold_recall))

final_report.write(f"Accuracy: {np.mean(all_accuracy) * 100:.2f} $\\pm$ {np.std(all_accuracy, ddof=1) * 100:.2f}\n")
final_report.write(
    f"f1 weighted: {np.mean(all_f1_weighted) * 100:.2f} $\\pm$ {np.std(all_f1_weighted, ddof=1) * 100:.2f}\n")
final_report.write(f"f1 macro: {np.mean(all_f1_macro) * 100:.2f} $\\pm$ {np.std(all_f1_macro, ddof=1) * 100:.2f}\n")
final_report.write(f"precision: {np.mean(all_precision) * 100:.2f} $\\pm$ {np.std(all_precision, ddof=1) * 100:.2f}\n")
final_report.write(f"recall: {np.mean(all_recall) * 100:.2f} $\\pm$ {np.std(all_recall, ddof=1) * 100:.2f}\n")

end_time = time.time()
execution_time = end_time - start_time
final_report.write(f"Execution time: {int(execution_time / 60)} minutes.\n")
final_report.close()
