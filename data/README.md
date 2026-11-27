# Data

The APK files are not included in this repository. Both datasets were obtained
from Aydogan and Sen (2024); please refer to that study to request access:

> E. Aydogan and S. Sen, "Android Authorship Attribution Using Source Code-Based
> Features," *IEEE Access*, vol. 12, pp. 6569–6589, 2024.
> doi: [10.1109/ACCESS.2024.3351945](https://doi.org/10.1109/ACCESS.2024.3351945)
> Code: https://github.com/emreaydogan/SourceCodeBasedAAA

| Dataset | Applications | Authors |
|---|---|---|
| Genome  | 1,530 | 39 |
| Malware | 3,268 (3,266 used) | 153 |

Two malware samples are excluded because call graph extraction failed.

## Folds

All experiments use stratified 10-fold cross-validation with random seeds.
The fold files for random seeds are created with `scripts/malware_save_fold_indices.py` (malware)
or `scripts/genome_save_fold_indices.py` (Genome), e.g.

```bash
python data/scripts/malware_save_fold_indices.py
```

The files are written for each seed as `{dataset}_fold_indices_seed{seed}.pkl` in the current directory.
