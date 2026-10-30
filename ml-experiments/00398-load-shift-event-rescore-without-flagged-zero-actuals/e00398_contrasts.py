"""Paired month-bootstrap contrasts of stored arms, with and without the flagged half-hours.

DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00398-*/e00398_contrasts.py NAME
"""

import sys
from time import perf_counter
from typing import Final

import polars as pl
from e00398_flagged_keys import OUT, ROW_FILES
from studies.eval_protocol.metrics import month_bootstrap_contrast
from studies.sources import ML_EXPERIMENTS_DATA_DIR

HEADLINE: Final[pl.Expr] = pl.col("lead_days_1_10")
GROUPS: Final[dict[str, pl.Expr]] = {
    "signed-MW six": pl.col("group") == "signed-MW",
    "MVA-only eight": pl.col("group") == "MVA-only",
}
# name -> (file key, reference arm, treatment arms, extra slice conditions)
PLANS: Final[dict[str, tuple[str, str, list[str], dict[str, pl.Expr]]]] = {
    "379": (
        "379_components",
        "bg2",
        [
            "drop_W",
            "drop_M",
            "drop_AN",
            "drop_PT",
            "drop_ST",
            "only_W",
            "only_M",
            "only_AN",
            "only_PT",
            "only_ST",
            "a5_no_spread",
            "s4_weekly",
            "manual_heuristic",
            "climatology",
        ],
        {"": pl.lit(value=True)},
    ),
    "391": (
        "391",
        "b0",
        ["n0", "n1", "n1_era5", "o", "d"],
        {
            "all rows": pl.lit(value=True),
            "lag in event": pl.col("lag_in_event"),
            "lag outside event": ~pl.col("lag_in_event"),
        },
    ),
    "395": (
        "395_control",
        "m_control",
        ["m_e7", "m_hl025", "m_hl1", "m_leadw", "m_solar", "m_sunny", "m_win12"],
        {"": pl.lit(value=True)},
    ),
}


def main() -> None:
    """Run the contrasts for the experiment named on the command line."""
    key = sys.argv[1]
    file_key, ref, arms, extras = PLANS[key]
    n_resamples = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    flagged = pl.read_parquet(OUT / "flagged_keys.parquet").select(
        "time_series_id", "valid_time", flagged=pl.lit(value=True)
    )
    if key == "395":
        base = ML_EXPERIMENTS_DATA_DIR / ROW_FILES["395_control"].rsplit("/", 1)[0]
        paths = {a: base / f"b_rows_{a}.parquet" for a in [ref, *arms]}
    else:
        paths = dict.fromkeys([ref, *arms], ML_EXPERIMENTS_DATA_DIR / ROW_FILES[file_key])
    wanted = [
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        "label",
        "group",
        "crps",
        "mae_median",
        "lead_days_1_10",
    ]
    if key == "391":
        wanted.append("lag_in_event")
    frames: dict[str, pl.DataFrame] = {}
    for arm, path in paths.items():
        lf = pl.scan_parquet(path)
        if key != "395":
            lf = lf.filter(pl.col("arm") == arm)
        frames[arm] = (
            lf.filter(pl.col("lead_days_1_10"))
            .select(wanted)
            .join(flagged.lazy(), on=["time_series_id", "valid_time"], how="left")
            .with_columns(flagged=pl.col("flagged").fill_null(value=False))
            .collect(engine="streaming")
        )
    out = []
    for arm in arms:
        for gname, gexpr in GROUPS.items():
            for ename, eexpr in extras.items():
                for metric in ("crps", "mae_median") if key == "395" else ("crps",):
                    row: dict[str, object] = {
                        "experiment": key,
                        "arm": arm,
                        "reference": ref,
                        "group": gname,
                        "rows": ename,
                        "metric": metric,
                    }
                    for version, drop in (("old", False), ("new", True)):
                        t0 = perf_counter()
                        sel = gexpr & eexpr
                        if drop:
                            sel = sel & ~pl.col("flagged")
                        try:
                            r = month_bootstrap_contrast(
                                treatment=frames[arm],
                                reference=frames[ref],
                                metric=metric,
                                where=sel,
                                level_percent=99.0,
                                n_resamples=n_resamples,
                            )
                        except ValueError:  # no shared half-hour in this row set
                            r = {
                                "difference": None,
                                "lower": None,
                                "upper": None,
                                "n_half_hours": 0,
                            }
                        row |= {
                            f"{version}_diff": r["difference"],
                            f"{version}_lower": r["lower"],
                            f"{version}_upper": r["upper"],
                            f"{version}_n": r["n_half_hours"],
                        }
                    out.append(row)
                    print(
                        row["arm"],
                        gname,
                        ename,
                        metric,
                        f"old {row['old_diff']} [{row['old_lower']},{row['old_upper']}]"
                        f" new {row['new_diff']} [{row['new_lower']},{row['new_upper']}]",
                        f"{perf_counter() - t0:.1f}s",
                        flush=True,
                    )
    pl.DataFrame(out).write_parquet(OUT / f"contrasts_{key}.parquet")


if __name__ == "__main__":
    main()
