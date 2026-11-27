import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import json
import os

import networkx as nx
import numpy as np
import torch
from androguard.misc import AnalyzeAPK
from torch_geometric.data import Data, Dataset


# Get appAuth+SPL features
with open("genome_appauth_spl_features.json") as f:
    graphLevelFeatures = json.load(f)


def get_node_features(graph):
    # GRAPH STRUCTURAL FEATURES
    # closeness centrality
    closeness_centrality_feats = nx.closeness_centrality(graph)

    # degree centrality
    degree_centrality_feats = nx.degree_centrality(graph)

    # CONVERT SIMPLE GRAPH to get other graph structural features
    simple_graph = nx.Graph(graph)

    # betweenness centrality
    betweenness_centrality_feats = nx.betweenness_centrality(simple_graph, k=5)

    # clustering coefficient
    clustering_coeff_feats = nx.clustering(simple_graph)

    # eigenvector centrality
    eigenvector_centrality_feats = nx.eigenvector_centrality(simple_graph, tol=1e-03)

    # pagerank
    pagerank_feats = nx.pagerank(simple_graph)

    opcodes = {}

    for node in graph.nodes:
        result = [0] * 27
        total_number_of_instructions = 0

        if node.is_external():
            result[0] = 1
        else:
            for instr in node.get_method().get_instructions():
                value = instr.get_op_value()
                if value == 0x00:  # nop
                    result[1] += 1
                elif value >= 0x01 and value <= 0x0D:  # move
                    result[2] += 1
                elif value >= 0x0E and value <= 0x11:  # return
                    result[3] += 1
                elif (value >= 0x12 and value <= 0x1C) or (value >= 0xFE and value <= 0xFF):  # const
                    result[4] += 1
                elif value == 0x1D or value == 0x1E:  # monitor
                    result[5] += 1
                elif value == 0x1F:  # check-cast
                    result[6] += 1
                elif value == 0x20:  # instance-of
                    result[7] += 1
                elif value == 0x22 or value == 0x23:  # new
                    result[8] += 1
                elif value >= 0x24 and value <= 0x26:  # fill
                    result[9] += 1
                elif value == 0x27:  # throw
                    result[10] += 1
                elif value >= 0x28 and value <= 0x2C:  # goto/switch
                    result[11] += 1
                elif value >= 0x2D and value <= 0x31:  # cmp
                    result[12] += 1
                elif value >= 0x32 and value <= 0x3D:  # if
                    result[13] += 1
                elif (value >= 0x44 and value <= 0x51) or value == 0x21:  # array
                    result[14] += 1
                elif value >= 0x52 and value <= 0x5F:  # instance-op
                    result[15] += 1
                elif value >= 0x60 and value <= 0x6D:  # static-op
                    result[16] += 1
                elif (value >= 0x6E and value <= 0x72) or (value >= 0x74 and value <= 0x78) or (
                        value >= 0xFA and value <= 0xFD):  # invoke
                    result[17] += 1
                elif value >= 0x7B and value <= 0x8F:  # unary-op
                    result[18] += 1
                elif value >= 0x90 and value <= 0xE2:  # binary-op
                    result[19] += 1
                else:  # unused/remaining
                    result[20] += 1
                    # print(instr, " --> opcode: ", value)
                total_number_of_instructions += 1

        #result[21] = total_number_of_instructions

        # ADD GRAPH STRUCTURAL FEATURES
        result[21] = closeness_centrality_feats.get(node)
        result[22] = degree_centrality_feats.get(node)
        result[23] = betweenness_centrality_feats.get(node)
        result[24] = clustering_coeff_feats.get(node)
        result[25] = eigenvector_centrality_feats.get(node)
        result[26] = pagerank_feats.get(node)

        #print(f"Node feats: {result}")
        opcodes[node] = {"opcode_features": result}

    nx.set_node_attributes(graph, opcodes)
    labels = {x: {"name": x.full_name} for x in graph.nodes}
    nx.set_node_attributes(graph, labels)
    graph = nx.convert_node_labels_to_integers(graph)

    # node features
    nodes = np.zeros((len(graph.in_degree), 33))
    node_in_degree_feats = list(graph.in_degree)
    node_out_degree_feats = list(graph.out_degree)
    for i, my_nodes in graph.nodes._nodes.items():
        # node in_degree and out_degree features
        nodes[i, 0] = node_in_degree_feats[i][1]
        nodes[i, 1] = node_out_degree_feats[i][1]

        # add method attributes
        method_attributes = list()
        #method_attributes.append(int(my_nodes["entrypoint"]))
        method_attributes.append(int(my_nodes["native"]))
        method_attributes.append(int(my_nodes["public"]))
        method_attributes.append(int(my_nodes["static"]))
        method_attributes.append(my_nodes["codesize"])
        nodes[i, 2:6] = method_attributes

        # add opcode features
        nodes[i, 6:33] = my_nodes["opcode_features"]

    node_features = np.array(nodes)
    node_features = torch.tensor(node_features)

    # get edges
    edges = []
    src_node_lst = []
    dest_node_lst = []
    for index, edge in enumerate(graph.edges):
        src_node = int(edge[0])
        dest_node = int(edge[1])
        src_node_lst.append(src_node)
        dest_node_lst.append(dest_node)

    edges.append(src_node_lst)
    edges.append(dest_node_lst)

    edge_indices = torch.tensor(edges, dtype=torch.long)

    return node_features, edge_indices


def get_graph(apk_file):
    a, d, dx = AnalyzeAPK(apk_file)
    call_graph = dx.get_call_graph()

    node_feats, edge_index = get_node_features(call_graph)

    # get appAuth features
    apk_feature = graphLevelFeatures[apk_file]
    graph_feats = []
    if len(apk_feature) != 710:
        for feature in apk_feature:
            if type(feature) == list:
                graph_feats.extend(feature)
            else:
                graph_feats.append(feature)
    else:
        graph_feats = apk_feature

    # print(f"apk features: {graph_feats}")
    graph_feats = torch.tensor(graph_feats)

    return node_feats, edge_index, graph_feats


# prepare Data object
def get_data(graph_folder, target_value):
    graphs = list()
    labels = list()

    log_file = open("./log_genome_appauth_spl_node_features_with_apkids.txt", "a")
    data_size = 0
    for graph_file_path_ in os.listdir(graph_out_path + graph_folder + "/"):
        # print("-- ", graph_out_path+graph_folder+"/"+graph_out + "/" + graph_file_path_)
        try:
            node_features, edge_indices, graph_features = get_graph(
                graph_out_path + graph_folder + "/" + graph_file_path_)

            label = torch.tensor([target_value])  # based on folder names (0, 1, 2, ...)
            # for example, folder "0" belongs to author1, folder "1" belongs to author2
            # add Data object into data_obj_lst list
            data = Data(x=node_features, edge_index=edge_indices, y=label, graph_feats=graph_features,
                        apkid=graph_folder+"/"+graph_file_path_)
            # data.validate(raise_on_error=True)
            if data.num_nodes != 0:
                graphs.append(data)
                labels.append(label)
                data_size = data_size + 1
            else:
                print(data)
                # print(data,"is created!!!")
        except Exception as e:
            print("\nError:", e, "\n")
            log_file.write("\nError: "+str(e) + "\n")
    print(graph_folder, "-> Done!", target_value)
    print(graph_folder, "-> data size:", data_size)
    log_file.write(str(graph_folder) + " -> Done! " + str(target_value) + "\n")
    log_file.write(str(graph_folder) + " -> data size: " + str(data_size) + "\n")
    log_file.close()

    return graphs, labels


# create custom Graph Dataset
class AndroidGraphDataset(Dataset):
    def __init__(self, apk_folder, label):
        super(AndroidGraphDataset, self).__init__()
        self.custom_data = get_data(apk_folder, label)
        self.graphs = self.custom_data[0]
        self.labels = self.custom_data[1]

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        return data


graph_out_path = f"{DATA_ROOT}/TEST/Genome/"
# graph_out_path = "<AA_LOCAL_ROOT>/PycharmProjects/AndroidAA/data/"
# graph_out_path = "<AA_DATA_ROOT>/TEST/malware/malware_data/"

graph_folders = {}
x = 0 # for folder label
for folder_name in os.listdir(graph_out_path):
    graph_folders[folder_name] = x
    x += 1

log=open("./log_genome_appauth_spl_node_features_with_apkids.txt", "a")
log.write("GENOME dataset\n")
log.write("node-level feats: indegree/outdegree, method types, opcode features, graph structural features\n")
log.write("graph-level feats: smali, permission, library and appauth\n")
log.write(str(graph_folders))
log.write("\n\n")
log.close()

for graph_folder, target_value in graph_folders.items():
    # dataset = AndroidGraphDataset(graph_folder, target_value)
    if (str(target_value)+".pt") in os.listdir("./genome_appauth_spl_node_feats_with_apkids/"):
        continue
    else:
        dataset = AndroidGraphDataset(graph_folder, target_value)
        # save dataset
        torch.save(dataset, "./genome_appauth_spl_node_feats_with_apkids/"+str(target_value)+".pt")
