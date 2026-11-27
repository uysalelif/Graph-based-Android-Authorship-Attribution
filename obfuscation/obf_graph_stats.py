"""
Obfuscation tekniklerinin cagri grafi boyutuna etkisi (dugum / kenar sayisi)

Yalnizca OKUR, hicbir dosyayi degistirmez; sonucu ekrana ve ./obf_graph_stats_{ds}.txt dosyasina yazar.
Obfuscation deneylerindeki APK kumesi kullanilir: intersection json + build_fixed_folds (yazar basina en az
10 APK esigi) -> Genome 1341 / malware 2461 APK. Orijinal ve 6 teknik ayni APK'ler uzerinden karsilastirilir.

Raporlanan (her kosul icin, orijinale gore):
  - ortalama dugum ve kenar sayisi; ortalamalarin yuzde degisimi        (makaledeki ifade buna karsilik gelir)
  - APK bazinda eslestirilmis yuzde degisimin ortalamasi ve medyani      (kontrol icin)
  - degisen APK orani (dugum / kenar sayisi orijinalden farkli olan)

Calistirma (original_train_obfuscated_test_split.py ile ayni klasorden; yollar WL script'leriyle ayni):
  python obf_graph_stats.py genome
  python obf_graph_stats.py malware
"""
import os as _os  # repo: paths come from environment variables (see README)
SCRATCH_ROOT = _os.environ.get("AA_SCRATCH_ROOT", "data")
import sys

import numpy as np
import torch

from original_train_obfuscated_test_split import build_fixed_folds

TECHNIQUES = ["ClassRename", "FieldRename", "MethodRename",
              "ConstStringEncryption", "LibEncryption", "ResStringEncryption"]

dataset_type = sys.argv[1] if len(sys.argv) > 1 else "genome"
if dataset_type == "genome":
    INTERSECTION_JSON = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/GENOME/genome_intersection_apk_ids.json"
    PT_DIR = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/GENOME"
elif dataset_type == "malware":
    INTERSECTION_JSON = f"{SCRATCH_ROOT}/mytest/obfuscation_tests/intersection_apk_ids.json"
    PT_DIR = f"{SCRATCH_ROOT}/mytest/obfuscation_tests"
else:
    sys.exit(f"Bilinmeyen veri seti: {dataset_type} (genome | malware)")

# Deneylerdeki APK kumesi (fold atamasi seed'e bagli ama kume ayni; seed 34 kullanilir)
fold_ids, label_map, dropped = build_fixed_folds(INTERSECTION_JSON, n_splits=10, random_state=34, verbose=False)
final_ids = set(fold_ids[0][0]) | set(fold_ids[0][1])

lines = []
def out(s=""):
    print(s)
    lines.append(s)

out(f"=== {dataset_type.upper()} | deneylerdeki APK sayisi: {len(final_ids)} | yazar: {len(label_map)} "
    f"(elenen yazar: {len(dropped)}) ===")


def sizes(condition):
    path = f"{PT_DIR}/{dataset_type}_{condition}_node_only.pt"
    graphs = torch.load(path, weights_only=False)
    res = {}
    for d in graphs:
        if d.apkname in final_ids:
            res[d.apkname] = (int(d.num_nodes), int(d.edge_index.shape[1]))
    del graphs
    return path, res


orig_path, orig = sizes("original")
out(f"original: {orig_path} | bulunan APK: {len(orig)}")
o_nodes = np.mean([v[0] for v in orig.values()])
o_edges = np.mean([v[1] for v in orig.values()])
out(f"  ortalama dugum {o_nodes:.1f} | ortalama kenar {o_edges:.1f}\n")

out(f"{'teknik':24s} {'APK':>5s} {'ort.dugum':>10s} {'ort.kenar':>10s} {'dugum %':>9s} {'kenar %':>9s} "
    f"{'eslesik dugum % (ort/med)':>26s} {'eslesik kenar % (ort/med)':>26s} {'degisen APK (dugum/kenar)':>26s}")
for t in TECHNIQUES:
    path, cur = sizes(t)
    common = sorted(set(cur) & set(orig))
    n_mean = np.mean([cur[a][0] for a in common])
    e_mean = np.mean([cur[a][1] for a in common])
    on = np.mean([orig[a][0] for a in common])     # ayni APK'ler uzerinden orijinal ortalama
    oe = np.mean([orig[a][1] for a in common])
    dn = [(cur[a][0] - orig[a][0]) / orig[a][0] * 100 for a in common if orig[a][0] > 0]
    de = [(cur[a][1] - orig[a][1]) / orig[a][1] * 100 for a in common if orig[a][1] > 0]
    ch_n = np.mean([cur[a][0] != orig[a][0] for a in common]) * 100
    ch_e = np.mean([cur[a][1] != orig[a][1] for a in common]) * 100
    out(f"{t:24s} {len(common):5d} {n_mean:10.1f} {e_mean:10.1f} {(n_mean/on-1)*100:+8.2f}% {(e_mean/oe-1)*100:+8.2f}% "
        f"{np.mean(dn):+11.2f}% / {np.median(dn):+7.2f}% {np.mean(de):+11.2f}% / {np.median(de):+7.2f}% "
        f"{ch_n:11.1f}% / {ch_e:5.1f}%")

with open(f"./obf_graph_stats_{dataset_type}.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n[yazildi] ./obf_graph_stats_{dataset_type}.txt")
