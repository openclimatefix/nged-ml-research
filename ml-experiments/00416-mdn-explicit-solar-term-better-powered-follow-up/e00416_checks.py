# ruff: noqa: E501
"""The write-up's "Checks": median versus mean forecast, median error, and leave-one-Primary-out.

For each arm it prints, per signed-direction Primary, the mean of the median minus the mean of the
11 quantiles (a spread that shifts the forecast shows up here); the mean absolute error of the
median forecast on sunny middays and on all half-hours, beside the CRPS contrasts; and the
sunny-midday CRPS contrast C3 minus A0-doy with each Primary left out.

```bash
PYTHONPATH=<as in e00416_score.py> uv run --no-sync python ml-experiments/00416-*/e00416_checks.py
```
"""

import json

import e00087_common as common  # ty: ignore[unresolved-import]
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00416_score as sc  # ty: ignore[unresolved-import]
import polars as pl

MONTHS_12 = [
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06",
]  # fmt: skip


def main() -> None:
    """Print the checks."""
    sc.MONTHS[:] = MONTHS_12
    arms = (*sc.ARMS, *sc.LOG_PRIOR_ARMS)
    series, ctx, table, _, _ = sc.build(arms=(sc.REFERENCE,), seeds=sc.SEEDS)
    label = dict(zip(series["time_series_id"].to_list(), series["label"].to_list(), strict=True))
    flags = table.select(*["time_series_id", "power_fcst_init_time", "valid_time", "sunny_midday"])
    print(
        "== median minus mean of the 11 quantiles, mean over half-hours (fraction of capacity) =="
    )
    mae: dict[str, dict[str, float]] = {}
    for arm in arms:
        frame = adapter.forecast_frame(
            stored=sc.arm_frame(arm=arm, seeds=sc.SEEDS), series=series, unit="fraction"
        )
        frame = (
            frame.join(
                flags, on=["time_series_id", "power_fcst_init_time", "valid_time"], how="inner"
            )
            .join(ctx["actuals"], on=["time_series_id", "valid_time"], how="left")
            .with_columns(site=pl.col("time_series_id").replace_strict(label, default=None))
            .filter(pl.col("site").is_in(list(sc.SIGNED)))
        )
        mean_q = frame.select(common.LEVEL_COLUMNS).to_numpy().mean(axis=1)
        frame = frame.with_columns(
            gap=pl.col("q50") - pl.Series(mean_q), abs_err=(pl.col("q50") - pl.col("actual")).abs()
        )
        per_site = frame.group_by("site").agg(pl.col("gap").mean()).sort("site")
        print(arm, {r[0]: round(r[1], 4) for r in per_site.iter_rows()})
        mae[arm] = {
            "all": float(frame["abs_err"].mean()),
            "sunny_midday": float(frame.filter(pl.col("sunny_midday"))["abs_err"].mean()),
        }
    print(
        "\n== mean absolute error of the median forecast, signed six (all half-hours incl. beyond day 10; MW) =="
    )
    for arm, v in mae.items():
        print(f"{arm:14s} all {v['all']:.3f}  sunny-midday {v['sunny_midday']:.3f}")
    stored = json.loads(sc.RESULTS_PATH.read_text())
    per_primary = pl.DataFrame(stored["per_primary_sunny_midday"]).filter(
        pl.col("label").is_in(list(sc.SIGNED))
    )
    print(
        "\n== leave-one-Primary-out: sunny-midday CRPS, C3 minus A0-doy (mean of Primary means) =="
    )
    wide = per_primary.pivot(on="arm", index="label", values="crps")
    full = float((wide["C3"] - wide["A0-doy"]).mean())  # ty: ignore[invalid-argument-type]
    print(f"all six: {full:+.3f}")
    for site in sc.SIGNED:
        kept = wide.filter(pl.col("label") != site)
        gap = float((kept["C3"] - kept["A0-doy"]).mean())  # ty: ignore[invalid-argument-type]
        print(f"without {site}: {gap:+.3f}")
    print(
        "per Primary C3 minus A0-doy:",
        {r[0]: round(r[1] - r[2], 3) for r in wide.select("label", "C3", "A0-doy").iter_rows()},
    )


if __name__ == "__main__":
    main()
