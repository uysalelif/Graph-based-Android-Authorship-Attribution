"""
CFG extraction for genome
"""
import os as _os  # repo: paths come from environment variables (see README)
DATA_ROOT = _os.environ.get("AA_DATA_ROOT", "data")
import json
import logging
import os
import time

import numpy as np
import torch
from androguard.misc import AnalyzeAPK
from torch_geometric.data import Data

# ==================== SETTINGS ====================
LIBRARY_LIST_PATH = f"{DATA_ROOT}/FEATURE_EXTRACTION/appAuth/Data/library/library_list.txt"
GRAPH_FEATS_JSON = "genome_appauth_spl_features.json"
GRAPH_OUT_PATH = f"{DATA_ROOT}/TEST/Genome/"
OUT_DIR = "./genome_cfg/"
LOG_PATH = "./log_genome_cfg.txt"

MIN_BLOCKS = 3       # eliminate blocks
MAX_METHODS = None   # None = use all developer methods; if the number is given, use first N methods.
MAX_BLOCKS_TOTAL = None   # None = no restriction
# ================================================

logger = logging.getLogger("cfg_extraction")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.FileHandler(LOG_PATH, mode="a")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False


with open(LIBRARY_LIST_PATH) as f:
    LIBRARY_LIST = ["L" + line.strip() for line in f]

FRAMEWORK_PREFIXES = ["Landroid/", "Landroidx/", "Ljava/", "Ljavax/", "Lkotlin/", "Lkotlinx/",
                      "Ldalvik/", "Lsun/", "Lorg/apache/", "Lorg/json/", "Lorg/intellij/",
                      "Lorg/jetbrains/"]
# bilinen kütüphane aileleri de developer sayılmasın
LIBRARY_FAMILIES = ["Lcom/google", "Lokhttp3", "Lokio", "Lretrofit2", "Lcom/squareup",
                    "Lcom/bumptech", "Lcom/facebook", "Lio/reactivex", "Ldagger",
                    "Lorg/koin", "Lcom/airbnb"]
ALL_EXTERNAL = tuple(set(LIBRARY_LIST + FRAMEWORK_PREFIXES + [f + "/" for f in LIBRARY_FAMILIES]))

with open(GRAPH_FEATS_JSON) as f:
    graphLevelFeatures = json.load(f)

N_OPCODE_BINS = 20
BLOCK_FEAT_DIM = N_OPCODE_BINS + 10   # + [total instructions, n_succ, n_pred, is_entry, is_exit]


def is_developer(ma):
    return (not ma.is_external()) and (not str(ma.class_name).startswith(ALL_EXTERNAL))


def opcode_bin(v):
    if v == 0x00: return 0                                   # nop
    if 0x01 <= v <= 0x0D: return 1                           # move
    if 0x0E <= v <= 0x11: return 2                           # return
    if (0x12 <= v <= 0x1C) or (0xFE <= v <= 0xFF): return 3  # const
    if v in (0x1D, 0x1E): return 4                           # monitor
    if v == 0x1F: return 5                                   # check-cast
    if v == 0x20: return 6                                   # instance-of
    if v in (0x22, 0x23): return 7                           # new
    if 0x24 <= v <= 0x26: return 8                           # fill
    if v == 0x27: return 9                                   # throw
    if 0x28 <= v <= 0x2C: return 10                          # goto/switch
    if 0x2D <= v <= 0x31: return 11                          # cmp
    if 0x32 <= v <= 0x3D: return 12                          # if
    if (0x44 <= v <= 0x51) or v == 0x21: return 13           # array
    if 0x52 <= v <= 0x5F: return 14                          # instance-op
    if 0x60 <= v <= 0x6D: return 15                          # static-op
    if (0x6E <= v <= 0x72) or (0x74 <= v <= 0x78) or (0xFA <= v <= 0xFD): return 16  # invoke
    if 0x7B <= v <= 0x8F: return 17                          # unary-op
    if 0x90 <= v <= 0xE2: return 18                          # binary-op
    return 19                                                # unused/remaining


def block_features(bb, is_entry, n_succ, n_pred):
    f = [0.0] * BLOCK_FEAT_DIM
    total = 0
    invoke = 0
    branch = 0
    arith = 0

    try:
        instrs = list(bb.get_instructions())
        for instr in instrs:
            op = instr.get_op_value()
            bin_id = opcode_bin(op)
            f[bin_id] += 1
            total += 1
            if bin_id == 16:
                invoke += 1
            elif bin_id in [10, 12]:
                branch += 1
            elif bin_id == 18:
                arith += 1
    except Exception:
        instrs = []
    f[N_OPCODE_BINS + 0] = total
    f[N_OPCODE_BINS + 1] = n_succ
    f[N_OPCODE_BINS + 2] = n_pred
    f[N_OPCODE_BINS + 3] = float(is_entry)
    f[N_OPCODE_BINS + 4] = float(n_succ == 0)   # is_exit

    invoke_ratio = invoke / max(total, 1)
    branch_ratio = branch / max(total, 1)
    arith_ratio = arith / max(total, 1)

    f[N_OPCODE_BINS + 5] = invoke_ratio
    f[N_OPCODE_BINS + 6] = branch_ratio
    f[N_OPCODE_BINS + 7] = arith_ratio

    is_branch_node = float(n_succ>1)
    is_merge_node = float(n_pred>1)
    f[N_OPCODE_BINS + 8] = is_branch_node
    f[N_OPCODE_BINS + 9] = is_merge_node

    term = [0, 0, 0, 0, 0]  # return, throw, goto, switch, if
    if instrs:
        last = instrs[-1]
        op = last.get_op_value()

        if 0x0E<=op<=0x11:  # return
            term[0] = 1
        elif op == 0x27:    # throw
            term[1] = 1
        elif 0x28<=op<=0x2A:    # goto
            term[2] = 1
        elif 0x2B<=op<=0x2C:    # switch
            term[3] = 1
        elif 0x32<=op<=0x3D:    # if
            term[4] = 1

    f.extend(term)

    return f


def method_cfg_blocks(ma):
    try:
        bbs = list(ma.get_basic_blocks().get())
    except Exception:
        return None
    if len(bbs) < MIN_BLOCKS:
        return None
    idx = {bb: i for i, bb in enumerate(bbs)}

    edges = set()
    succ = {i: 0 for i in range(len(bbs))}
    pred = {i: 0 for i in range(len(bbs))}
    for bb in bbs:
        u = idx[bb]
        try:
            childs = list(bb.childs)
        except Exception:
            childs = []
        for child in childs:
            target = child[-1]              # (last_offset, next_offset, target_block)
            if target in idx:
                v = idx[target]
                if (u, v) not in edges:
                    edges.add((u, v))
                    succ[u] += 1
                    pred[v] += 1

        try:
            exc = bb.get_exception_analysis()
        except Exception:
            exc = None
        if exc is not None:
            for handler in exc.exceptions:      # [type_name, handler_offset, handler_block]
                target = handler[2]
                if target is not None and target in idx:
                    v = idx[target]
                    if (u, v) not in edges:
                        edges.add((u, v))
                        succ[u] += 1
                        pred[v] += 1

    entry = min(bbs, key=lambda b: b.start)
    feats = [block_features(bb, is_entry=(bb is entry),
                            n_succ=succ[idx[bb]], n_pred=pred[idx[bb]]) for bb in bbs]
    return feats, edges


def build_apk_data(apk_file, label):
    a, d, dx = AnalyzeAPK(apk_file)

    methods = []
    for ma in dx.get_methods():
        if not is_developer(ma):
            continue
        res = method_cfg_blocks(ma)
        if res is None:
            continue
        methods.append(res)          # (feats, edges)

    if not methods:
        return None

    methods.sort(key=lambda r: len(r[0]), reverse=True)
    if MAX_METHODS:
        methods = methods[:MAX_METHODS]

    if MAX_BLOCKS_TOTAL:
        total = 0
        kept = []
        for m in methods:
            if total + len(m[0]) > MAX_BLOCKS_TOTAL:
                break
            kept.append(m); total += len(m[0])
        methods = kept if kept else methods[:1]

    X, EI_src, EI_dst, MID = [], [], [], []
    offset = 0
    for mi, (feats, edges) in enumerate(methods):
        X.extend(feats)
        MID.extend([mi] * len(feats))
        for (u, v) in edges:
            EI_src.append(u + offset); EI_dst.append(v + offset)
        offset += len(feats)

    apk_feature = graphLevelFeatures[apk_file]
    if len(apk_feature) != 710:
        gf = []
        for feature in apk_feature:
            gf.extend(feature) if isinstance(feature, list) else gf.append(feature)
    else:
        gf = apk_feature

    assert len(gf) == 710, f"{apk_file}: expected 710, incoming {len(gf)}"

    return Data(x=torch.tensor(np.array(X), dtype=torch.float32),
                edge_index=torch.tensor([EI_src, EI_dst], dtype=torch.long),
                method_id=torch.tensor(MID, dtype=torch.long),
                num_methods=len(methods),
                y=torch.tensor([label]),
                graph_feats=torch.tensor(gf, dtype=torch.float32))


def get_data(graph_folder, target_value):
    graphs = []
    folder_path = GRAPH_OUT_PATH + graph_folder + "/"
    apk_list = os.listdir(folder_path)
    logger.info(f"[{graph_folder}] started: {len(apk_list)} apk find, "
                f"MAX_METHODS={MAX_METHODS}, MAX_BLOCKS_TOTAL={MAX_BLOCKS_TOTAL}")

    for i, fn in enumerate(apk_list, 1):
        t0 = time.time()
        try:
            data = build_apk_data(folder_path + fn, target_value)
            if data is not None and data.num_nodes > 0:
                graphs.append(data)
                elapsed = time.time() - t0
                logger.info(
                    f"[{graph_folder} {i}/{len(apk_list)}] {fn}: {elapsed:.1f}s, "
                    f"method={data.num_methods}, block={data.x.shape[0]}, edge={data.edge_index.shape[1]}"
                )
            else:
                logger.warning(f"[{graph_folder} {i}/{len(apk_list)}] {fn}: empty data, skipped")
                logger.warning(f"[{graph_folder} {i}/{len(apk_list)}] {fn}: empty data, skipped")
        except Exception as e:
            logger.error(f"[{graph_folder} {i}/{len(apk_list)}] {fn}: ERROR - {e}")

    n_meth = [int(g.num_methods) for g in graphs]
    n_blk = [int(g.x.shape[0]) for g in graphs]
    msg = (f"{graph_folder} -> Done! {target_value}  apk={len(graphs)}  "
           f"method/apk mean={np.mean(n_meth) if n_meth else 0:.1f}  "
           f"block/apk mean={np.mean(n_blk) if n_blk else 0:.1f}")
    logger.info(msg)
    return graphs


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    graph_folders = {name: i for i, name in enumerate(os.listdir(GRAPH_OUT_PATH))}
    logger.info(f"CFG extraction  MIN_BLOCKS={MIN_BLOCKS} MAX_METHODS={MAX_METHODS} "
                f"MAX_BLOCKS_TOTAL={MAX_BLOCKS_TOTAL}")
    logger.info(str(graph_folders))

    for folder, target in graph_folders.items():
        if f"{target}.pt" in os.listdir(OUT_DIR):
            continue
        torch.save(get_data(folder, target), OUT_DIR + f"{target}.pt")
    print(f"Finished")
