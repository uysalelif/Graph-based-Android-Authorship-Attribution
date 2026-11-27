import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import json

dataset = "malware"  # genome

TECHNIQUES = ["ClassRename", "FieldRename", "MethodRename",
              "ConstStringEncryption", "LibEncryption", "ResStringEncryption"]
output_base = f"{DATA_ROOT}/{dataset}_obfuscation/"


def canonical_id(apk_file_path):
    parts = apk_file_path.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1].split('.apk')[0]}"


technique_success_sets = {}
for technique in TECHNIQUES:
    with open(output_base + f"obfuscation_log_{dataset}_{technique}.json") as f:
        results = json.load(f)
    technique_success_sets[technique] = {canonical_id(p) for p, ok in results.items() if ok}
    print(f"{technique}: {len(technique_success_sets[technique])} successful")

intersection = set.intersection(*technique_success_sets.values())
print(f"\nIntersection: {len(intersection)} APK")

with open(output_base + "{dataset}_intersection_apk_ids.json", "w") as f:
    json.dump(sorted(intersection), f, indent=2)
