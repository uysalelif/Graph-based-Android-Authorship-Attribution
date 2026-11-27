"""
Statistical information for CFGs
Min / Max / Mean / Median

For each application (Data object):
  - number of methods        : Data.num_methods (
  - number of basic blocks   : x.shape[0]
  - number of edges          : edge_index.shape[1]
  - blocks per method        : block / method

"""
import statistics
import sys

import torch


def load_graphs(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], (list, tuple)):
        return obj[0]
    return obj


def n_methods(g):
    nm = getattr(g, "num_methods", None)
    if nm is not None:
        return int(nm)
    mid = getattr(g, "method_id", None)
    return int(torch.unique(mid).numel()) if mid is not None else None


def describe(values):
    return (min(values), max(values), statistics.mean(values), statistics.median(values))


def main(paths):
    for path in paths:
        graphs = load_graphs(path)
        methods, blocks, edges, bpm, labels = [], [], [], [], set()
        for g in graphs:
            m = n_methods(g)
            b = int(g.x.shape[0])
            e = int(g.edge_index.shape[1])
            blocks.append(b)
            edges.append(e)
            if m:
                methods.append(m)
                bpm.append(b / m)
            labels.add(int(g.y.view(-1)[0]))

        print(f"=== {path} ===")
        print(f"Application size : {len(graphs)}")
        print(f"Author size    : {len(labels)}")
        print(f"{'':22}{'Min':>10}{'Max':>12}{'Mean':>12}{'Median':>12}")
        for name, vals in [("Number of methods", methods), ("Number of basic blocks", blocks),
                           ("Number of edges", edges), ("Blocks per method", bpm)]:
            if not vals:
                print(f"{name:22}  (not calculated)")
                continue
            mn, mx, mean, med = describe(vals)
            print(f"{name:22}{mn:>10,.0f}{mx:>12,.0f}{mean:>12,.2f}{med:>12,.1f}")
        print()


if __name__ == "__main__":
    # USAGE: python cfg_dataset_stats.py /path/genome_cfg.pt [/path/malware_cfg.pt]
    main(sys.argv[1:])
