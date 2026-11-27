import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import json
import os
import shutil
import subprocess

dataset = "malware" # genome

if dataset == "malware":
    data_path = f"{DATA_ROOT}/TEST/malware/malware_data/"
    output_base = f"{DATA_ROOT}/malware_obfuscation/"
elif dataset == "genome":
    data_path = "<AA_DATA_ROOT>/TEST/Genome/"
    output_base = f"{DATA_ROOT}/genome_obfuscation/"

TECHNIQUES = ["ClassRename", "FieldRename", "MethodRename",
              "ConstStringEncryption", "LibEncryption",
              "ResStringEncryption"]

graph_folders = {}
x = 0
for folder_name in os.listdir(data_path):
    graph_folders[folder_name] = x
    x += 1

for technique in TECHNIQUES:

    out_dir = output_base + f"{dataset}_obfuscated_apks_{technique}/"
    log_path = output_base + f"obfuscation_log_{dataset}_{technique}.json"

    os.makedirs(out_dir, exist_ok=True)

    results = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            results = json.load(f)

    for graph_folder, target_value in graph_folders.items():
        for graph_file_path_ in os.listdir(data_path + graph_folder + "/"):
            apk_file_path = data_path + graph_folder + "/" + graph_file_path_

            if apk_file_path in results:
                continue

            command = f"python3 -m obfuscapk.cli -o {technique}" \
                      f" -o Rebuild -o NewAlignment -o NewSignature -w {str(out_dir)}{str(graph_folder)} {str(apk_file_path)} -p"

            get_apk_name = graph_file_path_.split(".apk")[0]
            try:
                subprocess.run(command, shell=True, check=True)
                results[apk_file_path] = True
                try:
                    shutil.rmtree(out_dir + graph_folder + "/" + get_apk_name)
                except:
                    pass
            except Exception as e:
                print(f"ERROR! [{technique}] {apk_file_path}: {e}")
                results[apk_file_path] = False
                try:
                    shutil.rmtree(out_dir + graph_folder + "/" + get_apk_name)
                except Exception:
                    pass

            # save for log
            with open(log_path, "w") as f:
                json.dump(results, f, indent=2)

    n_ok = sum(results.values())
    print(f"{technique} done! success={n_ok}, fail={len(results) - n_ok}")

print("Done!")

