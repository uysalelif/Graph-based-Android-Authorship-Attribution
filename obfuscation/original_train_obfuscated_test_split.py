"""
"train=original, test=obfuscated"
"""
import json
from collections import defaultdict

import torch
from sklearn.model_selection import StratifiedKFold


def build_fixed_folds(intersection_json_path, n_splits=10, min_samples_per_author=None,
                       random_state=42, verbose=True):
    with open(intersection_json_path) as f:
        candidate_ids = json.load(f)

    by_author = defaultdict(list)
    for cid in candidate_ids:
        author_hash = cid.split("/")[0]
        by_author[author_hash].append(cid)

    if min_samples_per_author is None:
        min_samples_per_author = n_splits

    kept_authors = sorted(a for a, ids in by_author.items() if len(ids) >= min_samples_per_author)
    dropped_authors = sorted(
        ({"author": a, "n_samples": len(ids)} for a, ids in by_author.items() if len(ids) < min_samples_per_author),
        key=lambda d: -d["n_samples"],
    )

    if verbose:
        print(f"candidate list: {len(candidate_ids)} APK, {len(by_author)} author.")
        print(f"min_samples_per_author={min_samples_per_author}: "
              f"{len(kept_authors)} author left, {len(dropped_authors)} author dropped.")
        if dropped_authors:
            print("Dropped authors:")
            for d in dropped_authors:
                print(f"  {d['author']}: {d['n_samples']}")

    label_map = {author: idx for idx, author in enumerate(kept_authors)}

    final_ids = [cid for a in kept_authors for cid in by_author[a]]
    y_for_split = [label_map[cid.split("/")[0]] for cid in final_ids]

    if verbose:
        print(f"final apk ids: {len(final_ids)}\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_ids = []
    for train_idx, test_idx in skf.split(final_ids, y_for_split):
        train_ids = {final_ids[i] for i in train_idx}
        test_ids = {final_ids[i] for i in test_idx}
        assert train_ids.isdisjoint(test_ids)
        fold_ids.append((train_ids, test_ids))

    return fold_ids, label_map, dropped_authors


def build_condition_data(graphs, target_ids, label_map, verbose=True):
    by_id = {d.apkname: d for d in graphs}
    missing = target_ids - set(by_id)
    if verbose and missing:
        print(f"  WARNING: {len(missing)} skipping.")

    result = []
    for cid in target_ids:
        if cid not in by_id:
            continue
        d = by_id[cid]
        d.y = torch.tensor([label_map[cid.split("/")[0]]])
        result.append(d)
    return result
