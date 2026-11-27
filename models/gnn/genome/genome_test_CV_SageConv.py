import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import copy
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear
from torch_geometric import seed_everything
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
import time

start_time = time.time()
import sys
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 34  # usage: python genome_test_CV_SageConv.py [seed]
seed_everything(seed)

print(f"seed={seed}")
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

final_report = open(f"./sage_results/genome_SAGEConv_output_seed{seed}.txt", "w")
final_report.write("SAGEConv model\n")
final_report.write("Genome dataset - 10-fold stratified cross-validation\n")
final_report.write("All node features, and Appauth and spl graph features...\n")


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

# only get AppAuth + SPL
for data in X:
    data.graph_feats = data.graph_feats[:710]

print(X[0])
print(X[0].x[0, :])
print()

print("Genome dataset - 10-fold stratified cross-validation")
print("Number of data:", len(X))
labels = [int(data.y) for data in X]
print(labels)
print(f"Number of classes: {len(set(labels))}")

y = np.array(labels)


# GNN model (SAGEConv)
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, graph_features):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels*2+graph_features, out_channels)

    def forward(self, x, edge_index, batch, graph_feats):
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

        # combine graph features
        graph_feats = graph_feats.view(x.shape[0], -1)
        x = torch.cat([x, graph_feats], dim=1)

        x = self.lin(x)

        return x


# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# skf.get_n_splits(X, y)
# print(f"skf: {skf}")

with open(f"genome_fold_indices_seed{seed}.pkl", "rb") as f:
    fold_indices = pickle.load(f)

start_index = 1
end_index = 11


def train(train_data_loader):
    model.train()

    # total_loss = 0
    for data in train_data_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.batch, data.graph_feats)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # total_loss += float(loss) * data.y.size(0)
    # loss = total_loss / len(train_data_loader.dataset)
    # return loss


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    predictions = []
    ground_truths = []
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.graph_feats)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        total_correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        predictions.extend(pred.tolist())
        ground_truths.extend(data.y.tolist())
        loss = criterion(out, data.y)
        total_loss += float(loss) * data.y.size(0)
    acc = total_correct / len(loader.dataset)  # Derive ratio of correct predictions.
    loss = total_loss / len(loader.dataset)
    return acc, predictions, ground_truths, loss


for fold_index in range(start_index, end_index):
    myout = open(f"./sage_results/genome_result_{fold_index}_fold_cv_seed{seed}.txt", "w")
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

    # get data loader
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_data_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    graph_feats_size = len(train_dataset[0].graph_feats)
    print(f"graph features size: {graph_feats_size}")

    node_feats_size = train_dataset[0].x.shape[1]
    print(f"node features size: {node_feats_size}")

    # model
    model = GNN(in_channels=node_feats_size,
                hidden_channels=64,
                out_channels=len(set(labels)),
                graph_features=graph_feats_size)

    model=model.to(device)
    print(model)

    # model.double()
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
        train_acc, _, _, train_loss = test(train_data_loader)
        val_acc, val_predictions, val_ground_truths, val_loss = test(val_data_loader)

        print(f"Fold: {fold_index}, Epoch {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        myout.write(f"Fold: {fold_index}, Epoch {epoch:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}\n")

        if val_acc > best_val_accuracy:
            print("best model is saving...")
            best_val_accuracy = val_acc
            # save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, f"./sage_results/genome_best_model_fold_{fold_index}_seed{seed}.pth")

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
                out_channels=len(set(labels)),
                graph_features=graph_feats_size)
    # model.double()
    model.load_state_dict(torch.load(f"./sage_results/genome_best_model_fold_{fold_index}_seed{seed}.pth", map_location=device)[
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

    final_report.write(f"Fold: {fold_index}, Accuracy: {test_accuracy:.4f}, F1 weighted score: {test_f1score:.4f}, "
                       f"F1 macro score: {test_f1score_macro:.4f}, Precision score: {test_precision_score:.4f}, "
                       f"Recall score: {test_recall_score:.4f}\n")

    myout.close()

end_time = time.time()
execution_time = end_time - start_time
final_report.write(f"Execution time: {int(execution_time/60)} minutes.\n")
final_report.close()
