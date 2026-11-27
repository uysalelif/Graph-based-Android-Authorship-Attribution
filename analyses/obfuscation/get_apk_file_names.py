import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import os

dataset = "malware" # genome

# graph_out_path = "<AA_DATA_ROOT>/TEST/Genome/"
# graph_out_path = "<AA_LOCAL_ROOT>/PycharmProjects/AndroidAA/data/"
# graph_out_path = "<AA_DATA_ROOT>/TEST/malware/malware_data/"

TECHNIQUES = ["ClassRename", "FieldRename", "MethodRename",
              "ConstStringEncryption", "LibEncryption",
              "ResStringEncryption"]

for technique in TECHNIQUES:
    graph_out_path = f"{DATA_ROOT}/{dataset}_obfuscation/{dataset}_obfuscated_apks_{technique}/"

    graph_folders = {}
    x = 0   # for folder label
    for folder_name in os.listdir(graph_out_path):
        graph_folders[folder_name] = x
        x += 1

    apk_files = open(f"./{dataset}_obfuscated_apks/apkfile_names_{dataset}_obfuscated_{technique}.txt", "w")
    for graph_folder, target_value in graph_folders.items():
        for graph_file_path_ in os.listdir(graph_out_path + graph_folder + "/"):
            if graph_file_path_.endswith(".apk"):
                # print(graph_out_path + graph_folder + "/" + graph_file_path_)
                apk_files.write(graph_out_path + graph_folder + "/" + graph_file_path_ + "\n")

    apk_files.close()
    print(f"{technique} Done!")
