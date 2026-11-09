# ruff: noqa: E501
"""Experiment 674, step 1: build the day tables for the six (levels 0 to 3) and the eight (level 0).

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-*:ml-experiments/00674-* \
        OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00674-*/e00674_build.py

Writes `day_tables_six.parquet` and `day_tables_eight.parquet` under the experiment's work folder.
The eight are built by a separate call (`--eight`) so that the thresholds are frozen first.
"""

import argparse

import e00506_data as data  # ty: ignore[unresolved-import]
import e00674_features as feat  # ty: ignore[unresolved-import]
import polars as pl


def build(*, sites: tuple[str, ...], levels: tuple[float, ...]) -> pl.DataFrame:
    """Return the concatenated day tables of the sites at the injection levels."""
    gate = feat.pv_gate_table()
    parts = []
    for site in sites:
        mats = feat.load_mats(site=site)
        for level in levels:
            table = feat.day_table(site=site, mats=mats, level=level, gate_mw=gate)
            parts.append(feat.add_w(table=table, site=site, mats=mats, level=level))
            print(site, level, parts[-1].height, flush=True)
    return pl.concat(parts)


def main() -> None:
    """Build the six, or the eight with `--eight`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eight", action="store_true")
    args = parser.parse_args()
    feat.WORK.mkdir(parents=True, exist_ok=True)
    if args.eight:
        build(sites=data.MVA_ONLY, levels=(0.0,)).write_parquet(
            feat.WORK / "day_tables_eight.parquet"
        )
    else:
        build(sites=data.SIGNED_SIX, levels=feat.LEVELS).write_parquet(
            feat.WORK / "day_tables_six.parquet"
        )


if __name__ == "__main__":
    main()
