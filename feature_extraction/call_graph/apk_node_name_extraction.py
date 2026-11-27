import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import os
import pickle

from androguard.misc import AnalyzeAPK

dataset_name = "malware"


def get_node_names(graph):
    node_names = []
    for node in graph.nodes:

        node_names.append(str(node.full_name))

    return node_names


def get_graph(apk_file):
    a, d, dx = AnalyzeAPK(apk_file)
    call_graph = dx.get_call_graph()

    node_names = get_node_names(call_graph)

    return node_names


if dataset_name == "genome":
    graph_out_path = f"{DATA_ROOT}/TEST/Genome/"
elif dataset_name == "malware":
    graph_out_path = f"{DATA_ROOT}/TEST/malware/malware_data/"

# for test
# graph_out_path = "<AA_LOCAL_ROOT>/PycharmProjects/AndroidAA/data/"

graph_folders = {}
x = 0 # for folder label
for folder_name in os.listdir(graph_out_path):
    graph_folders[folder_name] = x
    x += 1

log=open(f"./log_{dataset_name}_node_names.txt", "a")
log.write(f"{dataset_name.upper()} dataset\n")
log.write(f"using folder: {graph_out_path}\n")
log.write(f"{str(graph_folders)}\n")
log.write("\n\n")

print(f"node names extraction started for {dataset_name} dataset...")


node_names_map = {}
processed_count = 0
for graph_folder, target_value in graph_folders.items():
    for graph_file_path_ in os.listdir(graph_out_path + graph_folder + "/"):
        # print("-- ", graph_out_path+graph_folder+"/"+graph_out + "/" + graph_file_path_)
        try:
            node_names = get_graph(graph_out_path + graph_folder + "/" + graph_file_path_)

            node_names_map[graph_folder+"/"+graph_file_path_] = node_names
            log.write(f"{processed_count} --> {graph_file_path_} islendi...\n")
            print(f"{processed_count} --> {graph_file_path_} islendi...")

            processed_count += 1

        except Exception as e:
            print("\nError:", e, "\n")
            log.write("\nError: "+str(e) + "\n")

log.close()


with open(f"{dataset_name}_node_names_map.pkl", "wb") as f:
    pickle.dump(node_names_map, f)

print(f"Done! for {dataset_name}")
