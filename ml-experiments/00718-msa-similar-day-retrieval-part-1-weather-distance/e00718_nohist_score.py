# ruff: noqa: E501
"""Experiments 718 and 719: score the masked-history forecasts of `R` and `U` (report-only, from stored forecasts).

Usage: e00718_nohist_score.py

`<arm>-nohist` is the arm's stage 3 network forecasting with all history power masked (`e00718_predict.py --mask-history`).
Prints, at 95% on the headline band over all 8 folds and seeds 0 and 1, the CRPS contrast of the masked forecast minus the
unmasked forecast of the same arm (how much the network relies on the history rows), and minus `P`'s unmasked forecast, in points
of capacity, at the six, the eight, and all 16 Primaries. A positive first contrast says the history rows help.
"""

import json
import sys

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_score as s713  # ty: ignore[unresolved-import]
import e00718_contrasts as con
import e00718_score as sc


def main() -> int:
    """Print the masked-history contrasts."""
    names = ["R", "R-nohist", "U", "U-nohist", a713.m_star(data_dir=sc.ML_EXPERIMENTS_DATA_DIR)]
    per_seed, averaged, groups = sc.load_all(names=names, months=a713.ALL_FOLDS, seeds=(0, 1))
    wide = con.with_all_group(groups=groups)
    out = {}
    for arm in ("R", "U"):
        for ref, label in (
            (arm, "masked minus the same arm unmasked"),
            (names[-1], "masked minus M* unmasked"),
        ):
            out[f"{arm}-nohist vs {ref}: {label}"] = {
                g: {
                    k: round(v, 3)
                    for k, v in s713.contrast(
                        per_seed=per_seed,
                        averaged=averaged,
                        treatment=f"{arm}-nohist",
                        reference=ref,
                        group=g,
                        groups=wide,
                        level=95.0,
                    ).items()
                    if k in {"point", "lower", "upper"}
                }
                for g in ("six", "eight", "all")
            }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
