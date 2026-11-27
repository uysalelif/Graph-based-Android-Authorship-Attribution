import igraph as ig
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_networkx
import torch_geometric

torch.manual_seed(12345)


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


# single .pt file
torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])
pt_filename = "<PT_FILE_PATH>"
pyg_graphs = torch.load(f=pt_filename)

for graph in pyg_graphs:
    converted_graph = ig.Graph.from_networkx(to_networkx(graph, to_multi=True))
    # size=3 gets 16 motifs
    # size=4 gets 218 motifs
    motif_Gfeats = converted_graph.motifs_randesu(size=3)
    motif_Gfeats = torch.tensor(motif_Gfeats)
    graph.graph_feats = torch.hstack((graph.graph_feats, motif_Gfeats))

# save to file
new_pt_filename = pt_filename.split("\\")[-1].split(".pt")[0] + "_motifFeats.pt"
torch.save(pyg_graphs, new_pt_filename)
