# Graph-level features (AppAuth and source code-based features)

`spl_feature_extraction.py` extracts the source code-based (smali),
permission and library features. The feature definitions follow
Aydogan and Sen (2024); the script reads the library and permission lists
from a `Data/` folder in the working directory:

- `Data/library/library_list.txt` (500 third-party libraries)
- `Data/permission/permission_list_158.txt` (158 permissions)

Both lists are available in the SourceCodeBasedAAA repository
(`Data/library/library_list`, `Data/permission/permission_list_158`):
https://github.com/emreaydogan/SourceCodeBasedAAA

`genome_appauth_spl_extraction.py` and `malware_appauth_spl_extraction.py`
extract the AppAuth features together with the features above. They are adapted
from `authfeature.py` of AppAuth:

- AppAuth repository: https://github.com/buptkick/AppAuth
- M. Xu et al., "AppAuth: Authorship Attribution for Android App Clones,"
  *IEEE Access*, 2019. doi: [10.1109/ACCESS.2019.2944684](https://doi.org/10.1109/ACCESS.2019.2944684)

They need `_settings.py`, `literadar.py`, `litedextree.py` (LiteRadar / LibRadar),
`dex_parser.py` and the `Data/` files from the AppAuth repository; these are not
redistributed here. Place them in the same folder as the extraction scripts.

The resulting 710 graph-level features (34 AppAuth + 18 source code-based +
158 permission + 500 library) are stored in the `.pt` files together with the
13 motif features (723 in total).
