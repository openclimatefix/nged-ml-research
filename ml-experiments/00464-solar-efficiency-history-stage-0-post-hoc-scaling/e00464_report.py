# ruff: noqa: E501
"""Contrasts of every scaled arm against its unscaled network, from the per-arm aggregates.

Per farm and band set (all bands, band d1), the percentage change of the mean quantile score, with
a month-block bootstrap interval at the Bonferroni level of the primary family (108 contrasts).
Group means (Solar A, B, D and Solar C, E, F) are descriptive.

    PYTHONPATH=ml-experiments/00464-* uv run --no-sync python ml-experiments/00464-*/e00464_report.py
"""

import e00464_common as common
import numpy as np
import polars as pl

BAND_SETS: dict[str, list[str] | None] = {"all": None, "d1": ["d1"]}
SUMMED: list[str] = [
    "score",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cover01",
    "cover99",
    "median_minus_mean",
    "ae_mean",
    "missing",
    "clipped",
]


def monthly(*, path_name: str, bands: list[str] | None) -> pl.DataFrame:
    """Return per month and farm the row-weighted mean of each metric over the bands."""
    agg = pl.read_parquet(common.SCORE_DIR / "agg" / path_name)
    if bands is not None:
        agg = agg.filter(pl.col("band").is_in(bands))
    return agg.group_by("month", "time_series_id").agg(
        n=pl.col("n").sum(),
        **{m: (pl.col(m) * pl.col("n")).sum() / pl.col("n").sum() for m in SUMMED},
    )


def group_values(*, table: pl.DataFrame, farms: tuple[str, ...], metric: str) -> np.ndarray:
    """Return the farm-equal mean of the metric in each month over the farms."""
    sub = (
        table.filter(pl.col("time_series_id").is_in(farms))
        .group_by("month")
        .agg(pl.col(metric).mean())
    )
    return sub.sort("month")[metric].to_numpy()


def bootstrap(*, arm: np.ndarray, ref: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Return the percentage change of arm against ref and its Bonferroni-level interval."""
    rng = np.random.default_rng(seed)
    n = len(arm)
    point = 100.0 * (arm.mean() / ref.mean() - 1.0)
    index = rng.integers(0, n, size=(common.DRAWS, n))
    draws = 100.0 * (arm[index].mean(axis=1) / ref[index].mean(axis=1) - 1.0)
    half = 100.0 * common.BONFERRONI_ALPHA / 2
    return float(point), float(np.percentile(draws, half)), float(np.percentile(draws, 100 - half))


def main() -> None:
    """Write the contrast table and print the primary arms."""
    rows = []
    for path in sorted((common.SCORE_DIR / "agg").glob("*.parquet")):
        if "unscaled" in path.name:
            continue
        arm = path.stem.replace("__", "|")
        network = arm.split("|")[0]
        ref_name = f"{network}__unscaled.parquet"
        for band_name, bands in BAND_SETS.items():
            a = monthly(path_name=path.name, bands=bands)
            r = monthly(path_name=ref_name, bands=bands)
            groups = {f: (f,) for f in common.FARMS} | {
                "ABD": common.GROUP_DRIFT,
                "CEF": common.GROUP_OTHER,
            }
            for name, farms in groups.items():
                va = group_values(table=a, farms=farms, metric="score")
                vr = group_values(table=r, farms=farms, metric="score")
                point, lo, hi = bootstrap(arm=va, ref=vr)
                rows.append(
                    {
                        "arm": arm,
                        "band": band_name,
                        "group": name,
                        "score_arm": va.mean(),
                        "score_ref": vr.mean(),
                        "pct": point,
                        "lo": lo,
                        "hi": hi,
                        "half_width": (hi - lo) / 2,
                        "pin01_diff": group_values(table=a, farms=farms, metric="pin01").mean()
                        - group_values(table=r, farms=farms, metric="pin01").mean(),
                        "pin99_diff": group_values(table=a, farms=farms, metric="pin99").mean()
                        - group_values(table=r, farms=farms, metric="pin99").mean(),
                        "cover99_arm": group_values(table=a, farms=farms, metric="cover99").mean(),
                        "cover99_ref": group_values(table=r, farms=farms, metric="cover99").mean(),
                        "mean_err_arm": group_values(table=a, farms=farms, metric="ae_mean").mean(),
                        "mean_err_ref": group_values(table=r, farms=farms, metric="ae_mean").mean(),
                        "missing": group_values(table=a, farms=farms, metric="missing").mean(),
                        "clipped": group_values(table=a, farms=farms, metric="clipped").mean(),
                    }
                )
    table = pl.DataFrame(rows)
    table.write_csv(common.SCORE_DIR / "contrasts.csv")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(60)
    primary = table.filter(
        pl.col("arm").str.starts_with("a2_member|")
        & pl.col("arm").str.contains(r"\|(b|c)\|")
        & pl.col("arm").str.ends_with("|norm|lag2")
        & (pl.col("band") == "all")
        & (pl.col("group").str.len_chars() == 1)
    ).sort("pct")
    print(primary.select("arm", "group", "score_ref", "pct", "lo", "hi", "pin99_diff", "missing"))


if __name__ == "__main__":
    main()
