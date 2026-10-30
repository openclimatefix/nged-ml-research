"""Experiment 401, Job 1, report step: real against synthetic median share, and why 0 lags reads 0.52.

Real arm: experiment 395 Part F's stored rows (experiment 381's monthly control and weekly bundle,
experiment 383's raw lagless arm from experiment 400's tables) with Part F's unpadded tags. Synthetic
arm: `work/sweep/*.parquet` from `e00401_sweep.py` (b0 boosters, corrected depth of each event
injected at its Primary). Shares, per row:

- *Observed share* (Part H): `1 - (actual - q50) / signed depth`. A forecast that ignores the event
  reads `1 - measured step / labelled step`, which is 0 only when the labelled step is the step the
  telemetry shows.
- *Response share* (Part E): `(q50 with the step - q50 without) / signed depth`, synthetic arm only.

Each is a median over rows of an event, then a median over events with at least 10 rows in the lag
status; intervals resample events.

```bash
PYTHONPATH=<as in e00401_score.py> DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00401-*/e00401_share.py
```
"""
# ruff: noqa: E501, PLR0912, PLR0915

import sys
from pathlib import Path
from typing import Final

import e00400_blend as blend
import e00400_partf as pf  # ty: ignore[unresolved-import]
import e00401_arms as arms
import e00401_events as ev
import numpy as np
import polars as pl

KEYS: Final[list[str]] = arms.KEYS
STATUS: Final[tuple[str, ...]] = ("0 lags", "1 or 2 lags", "3 lags")
MIN_ROWS: Final[int] = 10
N_RESAMPLES: Final[int] = 4000


def status_expr() -> pl.Expr:
    """Return the lag-status label of a row from its lag count."""
    return (
        pl.when(pl.col("lag_count") == 0)
        .then(pl.lit(STATUS[0]))
        .when(pl.col("lag_count") == 3)
        .then(pl.lit(STATUS[2]))
        .otherwise(pl.lit(STATUS[1]))
    )


def median_ci(*, values: np.ndarray) -> str:
    """Return the median over events and a 95% interval from resampling events."""
    if len(values) < 2:
        return f"{np.median(values):.3f} (n/a, {len(values)} events)"
    rng = np.random.default_rng(0)
    draws = np.median(values[rng.integers(0, len(values), size=(N_RESAMPLES, len(values)))], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return f"{np.median(values):.3f} ({lo:.3f} to {hi:.3f}) [{len(values)} events]"


def real_rows(*, name: str, tags: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Return the arm's rows with target inside an event, lead under 240 h, with residuals."""
    path = (
        blend.WORK / f"rows_{name}.parquet"
        if name == "lagless"
        else pf.WORK / f"rows_{name}.parquet"
    )
    return (
        pl.read_parquet(path, columns=[*KEYS, "lead_hours", "flagged", "q50_mw", "actual_mw"])
        .filter(pl.col("lead_hours") < 240, ~pl.col("flagged"))
        .join(tags, on=KEYS)
        .filter(pl.col("target_event_id") >= 0)
        .join(
            events.select(
                "event_id", "corrected_depth_mw", "measured_depth_mw", "length_class", "label"
            ),
            left_on="target_event_id",
            right_on="event_id",
        )
        .with_columns(status=status_expr(), res=pl.col("actual_mw") - pl.col("q50_mw"))
        .with_columns(
            share=1 - pl.col("res") / pl.col("corrected_depth_mw"),
            share_measured=pl.when(pl.col("measured_depth_mw") * pl.col("corrected_depth_mw") > 0)
            .then(1 - pl.col("res") / pl.col("measured_depth_mw"))
            .otherwise(None),
        )
    )


def per_event(*, rows: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return the per-event median of a column by lag status, for events with enough rows."""
    return (
        rows.group_by("status", "target_event_id")
        .agg(v=pl.col(column).median(), n=pl.col(column).drop_nulls().len())
        .filter(pl.col("n") >= MIN_ROWS)
        .sort("status", "target_event_id")
    )


def synthetic_table() -> pl.DataFrame:
    """Return the per-(event, status) medians of the synthetic arm."""
    sweep = pl.concat([pl.read_parquet(p) for p in sorted((arms.WORK / "sweep").glob("*.parquet"))])
    sweep = sweep.with_columns(
        status=pl.when(pl.col("k") == 0)
        .then(pl.lit(STATUS[0]))
        .when(pl.col("k") == 3)
        .then(pl.lit(STATUS[2]))
        .otherwise(pl.lit(STATUS[1])),
        response=(pl.col("q50_shifted") - pl.col("q50_base")) / pl.col("depth"),
        observed=(pl.col("q50_shifted") - pl.col("actual")) / pl.col("depth"),
    )
    return (
        sweep.group_by("status", "event_id")
        .agg(
            response=pl.col("response").median(),
            observed=pl.col("observed").median(),
            observed_q25=pl.col("observed").quantile(0.25),
            observed_q75=pl.col("observed").quantile(0.75),
            n=pl.len(),
        )
        .sort("status", "event_id")
    )


def main() -> None:
    """Print the tables the write-up quotes."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(220)
    events = ev.events_with_depths()
    tags = pl.read_parquet(pf.WORK / "tags.parquet")
    real = {
        n: real_rows(name=n, tags=tags, events=events)
        for n in ("monthly_control", "bundle", "lagless")
    }
    synth = synthetic_table()
    print(
        "events with a corrected depth:", events.height, "| Primaries in the sweep:", synth.height
    )

    print("\n== R1: real observed share, labelled depth (Part H convention), median over events ==")
    pairs: dict[str, dict[str, set[int]]] = {}
    for name, rows in real.items():
        pe = per_event(rows=rows, column="share")
        pairs[name] = {s: set(pe.filter(pl.col("status") == s)["target_event_id"]) for s in STATUS}
        for s in STATUS:
            v = pe.filter(pl.col("status") == s)["v"].to_numpy()
            print(f"{name:16s} {s:12s} {median_ci(values=v)}")

    print("\n== S1: matched events (those in R1's monthly-control list), real against synthetic ==")
    mc = real["monthly_control"]
    for s in STATUS:
        ids = pairs["monthly_control"][s]
        r = per_event(rows=mc, column="share").filter(pl.col("status") == s)
        sy = synth.filter(pl.col("status") == s, pl.col("event_id").is_in(list(ids)))
        both = r.join(sy, left_on="target_event_id", right_on="event_id")
        print(
            f"{s:12s} events with both arms {both.height} of {len(ids)}\n"
            f"   real observed share (monthly control): {median_ci(values=both['v'].to_numpy())}\n"
            f"   synthetic observed share (b0):         {median_ci(values=both['observed'].to_numpy())}\n"
            f"   synthetic response share (b0):         {median_ci(values=both['response'].to_numpy())}\n"
            f"   real minus synthetic observed, per event: {median_ci(values=(both['v'] - both['observed']).to_numpy())}"
        )
        both.write_parquet(arms.WORK / f"matched_{s.replace(' ', '_')}.parquet")

    print(
        "\n== B1: 0 lags, labelled against measured depth (events with a same-sign measured step) =="
    )
    for name, rows in real.items():
        for col in ("share", "share_measured"):
            pe = per_event(rows=rows.filter(pl.col("status") == STATUS[0]), column=col)
            print(f"{name:16s} {col:15s} {median_ci(values=pe['v'].to_numpy())}")
    ratio = events.filter(pl.col("measured_depth_mw").is_not_null()).with_columns(
        r=pl.col("measured_depth_mw") / pl.col("corrected_depth_mw")
    )
    print(
        "measured / labelled step, median over events:",
        round(ratio.select(pl.col("r").median()).item(), 3),
        "| events with opposite sign:",
        int((ratio["r"] < 0).sum()),
        "of",
        ratio.height,
    )

    print(
        "\n== B2: the blind reference: 1 - median(res)/labelled depth for the lagless forecast, 0 lags =="
    )
    ids0 = pairs["monthly_control"][STATUS[0]]
    for name in ("lagless", "monthly_control", "bundle"):
        sub = real[name].filter(
            pl.col("status") == STATUS[0], pl.col("target_event_id").is_in(list(ids0))
        )
        print(name, median_ci(values=per_event(rows=sub, column="share")["v"].to_numpy()))

    print(
        "\n== B3: forecast noise: synthetic k = 0, observed share of a zero-information forecast =="
    )
    k0 = synth.filter(pl.col("status") == STATUS[0], pl.col("event_id").is_in(list(ids0)))
    print("per-event median:", median_ci(values=k0["observed"].to_numpy()))
    print(
        "per-event interquartile range of rows, median over events:",
        round(k0.select((pl.col("observed_q75") - pl.col("observed_q25")).median()).item(), 3),
    )

    print(
        "\n== B5: response to the lags in real events: forecast with lags minus lagless, on the same rows =="
    )
    lagless = real["lagless"].select(*KEYS, share_lagless="share")
    for name in ("bundle", "monthly_control"):
        both = (
            real[name]
            .join(lagless, on=KEYS)
            .with_columns(response=pl.col("share") - pl.col("share_lagless"))
        )
        pe = per_event(rows=both, column="response")
        for s in STATUS:
            v = pe.filter(pl.col("status") == s)["v"].to_numpy()
            print(f"{name:16s} {s:12s} {median_ci(values=v)}")
    print("   synthetic response share (b0), all events in the status:")
    for s in STATUS:
        v = synth.filter(pl.col("status") == s)["response"].to_numpy()
        print(f"   {s:12s} {median_ci(values=v)}")

    print(
        "\n== B4: dilution and shape: real observed share by hours since onset and by length class =="
    )
    for name in ("monthly_control", "lagless"):
        rows = real[name].with_columns(
            band=pl.when(pl.col("hours_since_onset") < 6).then(pl.lit("0-6 h"))
            .when(pl.col("hours_since_onset") < 48).then(pl.lit("6-48 h"))
            .otherwise(pl.lit("over 48 h"))
        )  # fmt: skip
        for col in ("band", "length_class"):
            out = []
            for s in STATUS:
                for g in sorted(rows[col].unique().to_list()):
                    pe = (
                        rows.filter(pl.col("status") == s, pl.col(col) == g)
                        .group_by("target_event_id")
                        .agg(v=pl.col("share").median(), n=pl.len())
                        .filter(pl.col("n") >= MIN_ROWS)
                    )
                    if pe.height:
                        out.append(
                            (s, g, pe.height, round(pe.select(pl.col("v").median()).item(), 3))
                        )
            print(name, col, out)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
