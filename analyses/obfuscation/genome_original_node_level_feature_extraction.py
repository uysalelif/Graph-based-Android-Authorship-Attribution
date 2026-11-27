"""
Original genome APKs
NODE-LEVEL-ONLY feature extraction.
"""
import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import json
import os

import networkx as nx
import numpy as np
import torch
from androguard.misc import AnalyzeAPK
from torch_geometric.data import Data

ORIGINAL_RAW_BASE = f"{DATA_ROOT}/TEST/Genome"
INTERSECTION_IDS_JSON = f"{DATA_ROOT}/genome_obfuscation/genome_intersection_apk_ids.json"  # 6 teknigin kesisimi (author_hash/apk_hash listesi)
OUT_PT_FILE = "./genome_original_node_only.pt"
OUT_SUCCESS_IDS_JSON = "./original_success_ids.json"
OUT_FAILED_IDS_JSON = "./original_failed_ids.json"
LOG_FILE = "./log_genome_original_node_only.txt"


def get_node_features(graph):
    # GRAPH STRUCTURAL FEATURES
    closeness_centrality_feats = nx.closeness_centrality(graph)
    degree_centrality_feats = nx.degree_centrality(graph)

    simple_graph = nx.Graph(graph)
    betweenness_centrality_feats = nx.betweenness_centrality(simple_graph, k=5)
    clustering_coeff_feats = nx.clustering(simple_graph)
    eigenvector_centrality_feats = nx.eigenvector_centrality(simple_graph, tol=1e-03)
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
                total_number_of_instructions += 1

        # ADD GRAPH STRUCTURAL FEATURES
        result[21] = closeness_centrality_feats.get(node)
        result[22] = degree_centrality_feats.get(node)
        result[23] = betweenness_centrality_feats.get(node)
        result[24] = clustering_coeff_feats.get(node)
        result[25] = eigenvector_centrality_feats.get(node)
        result[26] = pagerank_feats.get(node)

        opcodes[node] = {"opcode_features": result}

    nx.set_node_attributes(graph, opcodes)
    labels = {x: {"name": x.full_name} for x in graph.nodes}
    nx.set_node_attributes(graph, labels)
    graph = nx.convert_node_labels_to_integers(graph)

    nodes = np.zeros((len(graph.in_degree), 33))
    node_in_degree_feats = list(graph.in_degree)
    node_out_degree_feats = list(graph.out_degree)
    for i, my_nodes in graph.nodes._nodes.items():
        nodes[i, 0] = node_in_degree_feats[i][1]
        nodes[i, 1] = node_out_degree_feats[i][1]

        method_attributes = list()
        method_attributes.append(int(my_nodes["native"]))
        method_attributes.append(int(my_nodes["public"]))
        method_attributes.append(int(my_nodes["static"]))
        method_attributes.append(my_nodes["codesize"])
        nodes[i, 2:6] = method_attributes

        nodes[i, 6:33] = my_nodes["opcode_features"]

    node_features = np.array(nodes)
    node_features = torch.tensor(node_features)

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
    return node_feats, edge_index


def main():
    with open(INTERSECTION_IDS_JSON) as f:
        candidate_ids = json.load(f)
    print(f"There are {len(candidate_ids)} apks in this intersection.")

    by_author = {}
    for cid in candidate_ids:
        author_hash, apk_hash = cid.split("/")
        by_author.setdefault(author_hash, []).append(apk_hash)

    author_folders = sorted(by_author.keys())
    label_map = {author: idx for idx, author in enumerate(author_folders)}
    print(f"{len(author_folders)} different authors (in this intersection).")

    log_file = open(LOG_FILE, "a")
    graphs = []
    failed = []

    processed = 0
    total = len(candidate_ids)
    for author_hash in author_folders:
        target_value = label_map[author_hash]
        for apk_hash in by_author[author_hash]:
            canonical_id = f"{author_hash}/{apk_hash}"
            apk_path = os.path.join(ORIGINAL_RAW_BASE, author_hash, f"{apk_hash}.apk")
            processed += 1

            try:
                node_features, edge_indices = get_graph(apk_path)
                label = torch.tensor([target_value])
                data = Data(
                    x=node_features,
                    edge_index=edge_indices,
                    y=label,
                    apkname=canonical_id,
                )
                if data.num_nodes != 0:
                    graphs.append(data)
                    log_file.write(f"[OK {processed}/{total}] {canonical_id}: n_nodes={data.num_nodes}\n")
                else:
                    failed.append({"apk_id": canonical_id, "error": "empty graph (n_nodes=0)"})
                    log_file.write(f"[EMPTY {processed}/{total}] {canonical_id}: n_nodes=0\n")
            except Exception as e:
                failed.append({"apk_id": canonical_id, "error": str(e)})
                log_file.write(f"[ERROR {processed}/{total}] {canonical_id}: {e}\n")

    log_file.write(f"\nTOTAL: {len(graphs)} success, {len(failed)} failed (tried: {total})\n")
    log_file.close()

    torch.save(graphs, OUT_PT_FILE)
    print(f"{len(graphs)} data objects are saved into '{OUT_PT_FILE}'.")

    with open(OUT_SUCCESS_IDS_JSON, "w") as f:
        json.dump([d.apkname for d in graphs], f, indent=2)
    with open(OUT_FAILED_IDS_JSON, "w") as f:
        json.dump(failed, f, indent=2)

    print(f"success: {len(graphs)}, failed: {len(failed)} (detail '{OUT_FAILED_IDS_JSON}')")


if __name__ == "__main__":
    main()
