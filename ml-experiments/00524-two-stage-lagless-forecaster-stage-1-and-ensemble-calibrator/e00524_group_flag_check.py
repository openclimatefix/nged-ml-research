"""Check that the metering-group input of experiment 524's stage 2 was always 0 (read-only).

`e00524_stage2.group_flag` compares each Primary's group to the string "signed_six", but the
label in the data is "signed-MW six". This script prints the real group labels, the flag the code
builds from them, and the distinct values of the group input (the last column of `aux_train` and
`aux_test`) in every stored stage-2 cache. It changes nothing.

```bash
cd <worktree>; source ml-experiments/00524-*/env.sh
uv run --no-sync python ml-experiments/00524-*/e00524_group_flag_check.py
```
"""

import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00524_stage2 as s2  # ty: ignore[unresolved-import]
import numpy as np


def main() -> None:
    """Print the group labels, the flag, and the distinct group-input values per cache file."""
    table = refs448.sites_table().sort("primary")
    print("group labels in the data:", sorted(table["group"].unique().to_list()))
    print("flag built by group_flag():", s2.group_flag().tolist())
    for path in sorted(s2.CACHE.glob("*_s?.npz")):
        cache = np.load(path)
        print(
            f"{path.name}: group input, train {np.unique(cache['aux_train'][:, -1]).tolist()}, "
            f"test {np.unique(cache['aux_test'][:, -1]).tolist()}"
        )
        assert not cache["aux_train"][:, -1].any()
        assert not cache["aux_test"][:, -1].any()


if __name__ == "__main__":
    main()
