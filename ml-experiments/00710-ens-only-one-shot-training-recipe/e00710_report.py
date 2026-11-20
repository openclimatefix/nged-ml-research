# ruff: noqa: E501
"""Experiment 710: print every number that `result.md` quotes, and draw its four figures.

The final scorer output (`work/logs_run/score_full_4.log`, written by `e00710_score.py`) holds the
contrasts, the metric tables, the slump tables, and the convergence table. This script parses that
file and prints the numbers back, then adds five readings that the scorer does not print, each
computed from the stored aggregates (`work/score_cache/`), the stored forecasts (`work/seed_preds/`
and `work/bank/`), or both: C1 on seeds 0 to 2 and on seeds 3 and 4 separately, C1 without fold
2026-01, the CRPS and MAE of the median on rows outside events and at least 7 days after one, the
slump share with a month-block interval for each recipe and injection state, and the median
forecast minus the mean forecast per Primary. It then draws four figures into the experiment's
docs folder.

```bash
ml-experiments/00710-*/run_phase.sh e00710_report.py > report.log
ml-experiments/00710-*/run_phase.sh e00710_report.py --skip-outside --skip-figures
```

The first run scores the outside-events rows of 11 variants one fold at a time, which takes about
10 minutes, and caches the sums under `work/report_cache/`.
"""

import argparse
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00377_adapter as adapter  # ty: ignore[unresolved-import]
import e00581_bank as bank581  # ty: ignore[unresolved-import]
import e00707_common as common  # ty: ignore[unresolved-import]
import e00707_prep as prep  # ty: ignore[unresolved-import]
import e00707_score as s707  # ty: ignore[unresolved-import]
import e00710_arms as arms  # ty: ignore[unresolved-import]
import e00710_inject as inject  # ty: ignore[unresolved-import]
import e00710_score as score  # ty: ignore[unresolved-import]
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

LOG: Final[Path] = score.OUT_DIR / "logs_run" / "score_full_4.log"
"""The final scorer output that the page quotes."""
CACHE_DIR: Final[Path] = score.OUT_DIR / "report_cache"
FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00710-ens-only-one-shot-training-recipe"
)
SIX: Final[str] = score.SIX
EIGHT: Final[str] = score.EIGHT
GROUP_PATTERN: Final[str] = rf"({re.escape(SIX)}|{re.escape(EIGHT)})"
NUMBER: Final[str] = r"([+-]?\d+\.\d+)"
WIDTH_PX: Final[int] = 680
AFTER_DAYS: Final[int] = 7
SEEDS_PAIRED: Final[tuple[int, ...]] = (0, 1, 2)
ORIGIN_HOUR: Final[int] = 9
DAY_AHEAD_LEADS: Final[tuple[float, float]] = (15.0, 39.0)
"""Valid times from the next midnight to the one after, for an origin at 09:00 UTC."""
EXAMPLE_FOLD: Final[str] = "2026-01"
EXAMPLE_ORIGIN: Final[str] = "2026-01-12 09:00"
EXAMPLE_SITE: Final[str] = "D10"
"""The Primary and the origin of experiment 425's winter example, so that the two pages compare."""
EVENT_LENGTHS: Final[tuple[int, ...]] = (3, 7, 15)
EVENT_DEPTH: Final[float] = 0.5
REAL_SITES: Final[tuple[str, ...]] = ("D5", "D16")
BLACK: Final[str] = "#222222"
GREY: Final[str] = "#8A8A8A"
PNG_DIR: Path | None = None
"""Where `save_svg` also writes a PNG, set by `--png-dir`; `None` writes no PNG."""

CONTRAST_ROWS: Final[tuple[tuple[str, str], ...]] = (
    ("C1", "C1 FTh90d-i minus M1-i (headline)"),
    ("C2", "C2 M2-i minus M1-i"),
    ("C3", "C3 PP-i minus M2-i"),
    ("C4", "C4 W-h1-i minus M3-i"),
    ("C5", "C5 W-h1-i minus P-i (placebo)"),
    ("E1", "E1 M1h90-i minus FTh90d-i"),
    ("S2", "S2 FTh90d-i minus M2-i"),
    ("I1", "I1 M1-i minus M1-n"),
    ("I2", "I2 M2-i minus M2-n"),
    ("I3", "I3 FTh90d-i minus FTh90d-n"),
)


# --- parsing the stored scorer output ----------------------------------------------------------


def parse_log(*, path: Path) -> dict[str, list[dict]]:
    """Parse the tables of the scorer output into lists of records.

    Args:
        path: The scorer output file.

    Returns:
        Records by table: `contrasts`, `absolute`, `coverage`, `slump`, `slump_contrasts`,
        `real`, `convergence`, `t_tests`, `verdicts`, and `facts` (single lines).
    """
    out: dict[str, list[dict]] = {
        k: []
        for k in (
            "contrasts",
            "absolute",
            "coverage",
            "slump",
            "slump_contrasts",
            "real",
            "convergence",
            "t_tests",
            "verdicts",
            "facts",
        )
    }
    plain = re.compile(
        rf"^(C1|E1|I\d|S\d|C\d) (\S+ minus \S+) +{GROUP_PATTERN} +(\w+) +{NUMBER} "
        rf"\({NUMBER} to {NUMBER}\) level ([\d.]+)% n=(\d+) months=(\d+)"
    )
    guard = re.compile(
        rf"^(C1|E1) (without D8|tail q\d+|cold nights|lead_\w+) +{GROUP_PATTERN} +(\w+) +{NUMBER} "
        rf"\({NUMBER} to {NUMBER}\) level ([\d.]+)% n=(\d+) months=(\d+)"
    )
    absolute = re.compile(
        rf"^(all rows|\w+) +{GROUP_PATTERN} +(\S+@\d+) +(crps|mae_median) +(\d+\.\d+) "
        rf"\((\d+\.\d+) to (\d+\.\d+)\) n=(\d+) months=(\d+)"
    )
    coverage = re.compile(rf"^{GROUP_PATTERN} +(\S+@\d+) +q05 (\d\.\d+) q95 (\d\.\d+)")
    slump = re.compile(
        rf"^(\S+@\d+) +{GROUP_PATTERN} +(total|first_4_days|day_4_to_end|after_7_days) +"
        rf"events= *(\d+) half-hours=(\d+) share {NUMBER} NRA error {NUMBER} pts "
        rf"CRPS change {NUMBER} pts"
    )
    slump_c = re.compile(
        rf"^(I[456]) (\S+) minus (\S+) +{GROUP_PATTERN} +{NUMBER} \({NUMBER} to {NUMBER}\) "
        r"level ([\d.]+)% events=(\d+)"
    )
    real = re.compile(
        rf"^(\S+@\d+) +(D\d+) transfer (\d+) onset (\S+) depth {NUMBER} half-hours (\d+) share "
        rf"{NUMBER} NRA error {NUMBER} pts, shifted-minus-corrected error {NUMBER} pts \[(.*)\]"
    )
    convergence = re.compile(
        r"^(\S+) fits +(\d+) mean steps +(\d+) seconds +(\d+) final stage at a limit ([\d.]+) "
        r"effective rows per batch +(\d+)"
    )
    t_test = re.compile(
        rf"^(C\d|E1) +{GROUP_PATTERN} +mean {NUMBER} t {NUMBER} p ([\d.]+) months (\d+)"
    )
    for line in path.read_text().splitlines():
        if m := guard.match(line) or plain.match(line):
            g = m.groups()
            out["contrasts"].append(
                {
                    "name": g[0],
                    "what": g[1],
                    "group": g[2],
                    "metric": g[3],
                    "estimate": float(g[4]),
                    "lower": float(g[5]),
                    "upper": float(g[6]),
                    "level": float(g[7]),
                    "n": int(g[8]),
                    "months": int(g[9]),
                }
            )
        elif m := absolute.match(line):
            g = m.groups()
            out["absolute"].append(
                {
                    "slice": g[0],
                    "group": g[1],
                    "variant": g[2],
                    "metric": g[3],
                    "mean": float(g[4]),
                    "lower": float(g[5]),
                    "upper": float(g[6]),
                    "n": int(g[7]),
                    "months": int(g[8]),
                }
            )
        elif m := coverage.match(line):
            g = m.groups()
            out["coverage"].append(
                {"group": g[0], "variant": g[1], "q05": float(g[2]), "q95": float(g[3])}
            )
        elif m := slump.match(line):
            g = m.groups()
            out["slump"].append(
                {
                    "variant": g[0],
                    "group": g[1],
                    "part": g[2],
                    "events": int(g[3]),
                    "n": int(g[4]),
                    "share": float(g[5]),
                    "nra_error": float(g[6]),
                    "crps_change": float(g[7]),
                }
            )
        elif m := slump_c.match(line):
            g = m.groups()
            out["slump_contrasts"].append(
                {
                    "name": g[0],
                    "treatment": g[1],
                    "reference": g[2],
                    "group": g[3],
                    "estimate": float(g[4]),
                    "lower": float(g[5]),
                    "upper": float(g[6]),
                    "events": int(g[8]),
                }
            )
        elif m := real.match(line):
            g = m.groups()
            out["real"].append(
                {
                    "variant": g[0],
                    "site": g[1],
                    "transfer": int(g[2]),
                    "onset": g[3],
                    "depth": float(g[4]),
                    "n": int(g[5]),
                    "share": float(g[6]),
                    "nra_error": float(g[7]),
                    "shift_error": float(g[8]),
                    "flag": g[9],
                }
            )
        elif m := convergence.match(line):
            g = m.groups()
            out["convergence"].append(
                {
                    "variant": g[0],
                    "fits": int(g[1]),
                    "steps": int(g[2]),
                    "seconds": int(g[3]),
                    "at_limit": float(g[4]),
                    "rows": int(g[5]),
                }
            )
        elif m := t_test.match(line):
            g = m.groups()
            out["t_tests"].append(
                {
                    "name": g[0],
                    "group": g[1],
                    "mean": float(g[2]),
                    "t": float(g[3]),
                    "p": float(g[4]),
                }
            )
        elif re.search(r"verdict|^choices|shared half-hours|dropped for|SD of the seed", line):
            out["verdicts"].append({"line": line.strip()})
    for table, keys in (
        ("absolute", ("slice", "group", "variant", "metric")),
        ("slump", ("variant", "group", "part")),
    ):
        unique: dict[tuple, dict] = {}
        for row in out[table]:
            unique.setdefault(tuple(row[k] for k in keys), row)
        out[table] = list(unique.values())
    return out


def pick(*, rows: list[dict], **where: object) -> dict:
    """Return the one record whose fields equal `where`.

    Args:
        rows: The records.
        **where: Field values to match.

    Returns:
        The record.

    Raises:
        KeyError: If no record or more than one record matches.
    """
    found = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if len(found) != 1:
        msg = f"{len(found)} records match {where}"
        raise KeyError(msg)
    return found[0]


def interval_text(*, row: dict) -> str:
    """Return `+0.012 (-0.013 to +0.038)` for a record with an estimate and limits."""
    est = row.get("estimate", row.get("mean"))
    return f"{est:+.3f} ({row['lower']:+.3f} to {row['upper']:+.3f})"


def print_parsed(*, parsed: dict[str, list[dict]]) -> None:
    """Print the parsed tables, so that every quoted number appears in this script's output.

    Args:
        parsed: The output of `parse_log`.
    """
    print(f"== numbers parsed from {LOG.name} ==")
    for row in parsed["verdicts"]:
        print(row["line"])
    print("\n-- contrasts, treatment minus reference, CRPS in points of capacity")
    for r in parsed["contrasts"]:
        print(
            f"{r['name']:3s} {r['what']:28s} {r['group']:15s} {r['metric']:12s} "
            f"{interval_text(row=r)} level {r['level']:.2f}% n={r['n']} months={r['months']}"
        )
    print("\n-- CRPS and MAE of the median (95% month-block intervals)")
    for r in parsed["absolute"]:
        print(
            f"{r['slice']:24s} {r['group']:15s} {r['variant']:16s} {r['metric']:10s} "
            f"{r['mean']:.3f} ({r['lower']:.3f} to {r['upper']:.3f}) n={r['n']} months={r['months']}"
        )
    print("\n-- coverage of q05 and q95")
    for r in parsed["coverage"]:
        print(f"{r['group']:15s} {r['variant']:16s} q05 {r['q05']:.3f} q95 {r['q95']:.3f}")
    print("\n-- slump share and NRA error in injected events (mean over events)")
    for r in parsed["slump"]:
        print(
            f"{r['variant']:16s} {r['group']:15s} {r['part']:13s} events={r['events']} "
            f"half-hours={r['n']} share {r['share']:+.3f} NRA error {r['nra_error']:+.3f} "
            f"CRPS change {r['crps_change']:+.3f}"
        )
    print("\n-- I4 to I6")
    for r in parsed["slump_contrasts"]:
        print(
            f"{r['name']} {r['treatment']} minus {r['reference']} {r['group']:15s} "
            f"{interval_text(row=r)} events={r['events']}"
        )
    print("\n-- real events")
    for r in parsed["real"]:
        print(
            f"{r['variant']:16s} {r['site']} transfer {r['transfer']} onset {r['onset']} "
            f"depth {r['depth']:+.3f} half-hours {r['n']} share {r['share']:+.3f} NRA error "
            f"{r['nra_error']:+.3f} shifted-minus-corrected {r['shift_error']:+.3f} [{r['flag']}]"
        )
    print("\n-- steps, seconds, final-stage limit share, effective rows per batch")
    for r in parsed["convergence"]:
        print(
            f"{r['variant']:10s} fits={r['fits']} steps={r['steps']} seconds={r['seconds']} "
            f"at_limit={r['at_limit']:.2f} rows={r['rows']}"
        )
    print("\n-- t tests on the 12 monthly differences")
    for r in parsed["t_tests"]:
        print(
            f"{r['name']:3s} {r['group']:15s} mean {r['mean']:+.3f} t {r['t']:+.2f} p {r['p']:.3f}"
        )


# --- readings computed from the stored aggregates ----------------------------------------------


def latest_cache(*, name: str) -> pl.DataFrame:
    """Return the newest cached aggregates of a variant (the scorer's last run).

    Args:
        name: The variant name, such as `M1-i@01234`.

    Returns:
        The aggregates (slice, month, label, n, and the summed losses).

    Raises:
        FileNotFoundError: If the scorer cached nothing for the variant.
    """
    files = sorted(
        score.SCORE_CACHE_DIR.glob(f"{name}__*.parquet"), key=lambda p: p.stat().st_mtime
    )
    if not files:
        msg = f"no cached aggregates for {name}"
        raise FileNotFoundError(msg)
    return pl.read_parquet(files[-1])


def extra_contrasts() -> list[dict]:
    """Print the contrast readings that the scorer does not print, from the cached aggregates.

    The readings are C1 on seeds 0 to 2 and on seeds 3 and 4, C1 without fold 2026-01, C1 without
    each Primary in turn, C1 in each fold, the effect of the 90-day half-life on the reference
    (`M1h90-i` minus `M1-i`), and the placebo against equal weights (`P-i` minus `M3-i`).

    Returns:
        One record per reading and group (point, 95% limits).
    """
    print("\n== readings the scorer does not print (95% intervals, 100,000 resamples) ==")
    out = []
    pairs = {
        "C1 seeds 0 to 2": ("FTh90d-i@012", "M1-i@012", None),
        "C1 seeds 3 and 4": ("FTh90d-i@34", "M1-i@34", None),
        "C1 without fold 2026-01": ("FTh90d-i@01234", "M1-i@01234", "2026-01"),
        "M1h90-i minus M1-i (half-life on the reference)": ("M1h90-i@012", "M1-i@012", None),
        "P-i minus M3-i (placebo against equal weights)": ("P-i@012", "M3-i@012", None),
        "M2-n minus M1-n (no injection)": ("M2-n@01234", "M1-n@01234", None),
        "FTh90d-i minus M1-n": ("FTh90d-i@01234", "M1-n@01234", None),
        "FTh90d-i minus M2-n": ("FTh90d-i@01234", "M2-n@01234", None),
        "M3-i minus M2-i (the fine-tune)": ("M3-i@012", "M2-i@012", None),
    }
    for label, (t_name, r_name, drop) in pairs.items():
        t, r = latest_cache(name=t_name), latest_cache(name=r_name)
        if drop is not None:
            t, r = t.filter(pl.col("month") != drop), r.filter(pl.col("month") != drop)
        for group in score.GROUPS:
            found = score.contrast(label=label, group=group, treatment=t, reference=r)
            if found is not None:
                out.append({"label": label, "group": group, **found})
    t, r = latest_cache(name="FTh90d-i@01234"), latest_cache(name="M1-i@01234")
    for group in score.GROUPS:
        for site in score.SUBSETS[group]:
            found = score.contrast(
                label=f"C1 without {site}",
                group=group,
                treatment=t,
                reference=r,
                without_label=site,
            )
            if found is not None:
                out.append({"label": f"C1 without {site}", "group": group, **found})
    print("\n-- C1 in each fold (equal weight over half-hours of the group)")
    for group in score.GROUPS:
        months, t_sums, counts = score.month_totals(aggregate=t, group=group, metric="crps")
        _m, r_sums, _c = score.month_totals(aggregate=r, group=group, metric="crps")
        for month, value in zip(months, (t_sums - r_sums) / counts, strict=True):
            print(f"C1 fold {month} {group:15s} {value:+.3f}")
            out.append({"label": f"C1 fold {month}", "group": group, "difference": float(value)})
    return out


MAE_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("FTh90d-i@01234", "M1-i@01234"),
    "C2": ("M2-i@01234", "M1-i@01234"),
    "C3": ("PP-i@012", "M2-i@012"),
    "C4": ("W-h1-i@012", "M3-i@012"),
    "C5": ("W-h1-i@012", "P-i@012"),
    "E1": ("M1h90-i@012", "FTh90d-i@012"),
    "S2": ("FTh90d-i@01234", "M2-i@01234"),
    "I1": ("M1-i@01234", "M1-n@01234"),
    "I2": ("M2-i@01234", "M2-n@01234"),
    "I3": ("FTh90d-i@012", "FTh90d-n@012"),
}
"""The treatment and reference of each contrast, with the seeds the scorer paired."""


def mae_contrasts() -> list[dict]:
    """Print each contrast on the MAE of the median forecast, at 95%, beside the CRPS contrast.

    Returns:
        One record per contrast and group.
    """
    print("\n== the same contrasts on the MAE of the median (95% intervals, descriptive) ==")
    out = []
    for key, (t_name, r_name) in MAE_PAIRS.items():
        t, r = latest_cache(name=t_name), latest_cache(name=r_name)
        for group in score.GROUPS:
            found = score.contrast(
                label=f"{key} {t_name} minus {r_name}",
                group=group,
                treatment=t,
                reference=r,
                metric="mae_median",
            )
            if found is not None:
                out.append({"label": key, "group": group, **found})
    return out


class OutsideScorer(score.Scorer):
    """A scorer whose aggregates hold only the half-hours outside events and 7 days after them."""

    def month_sums(self, *, rows: pl.DataFrame, variant: score.Variant) -> pl.DataFrame:
        """Return the sums of the row losses outside events, by month and Primary.

        Args:
            rows: The scored rows of one fold with the slice columns.
            variant: The variant, for the message of an error.

        Returns:
            The sums, with the slice named `all rows` so that the scorer's helpers read them.

        Raises:
            ValueError: If a headline-band row loss is null.
        """
        headline = rows.filter(pl.col(score.HEADLINE)).with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
        )
        if headline.select(score.SUMMED_METRICS).null_count().sum_horizontal().item():
            msg = f"{variant.name}: a headline-band row loss is null"
            raise ValueError(msg)
        outside = headline.filter(~(pl.col("event_rows") | pl.col("event_following_7_days")))
        return (
            outside.group_by("month", "label")
            .agg(
                pl.len().cast(pl.Int64).alias("n"),
                *[pl.col(m).sum() for m in score.SUMMED_METRICS],
            )
            .with_columns(slice=pl.lit(score.ALL_ROWS))
        )


def outside_aggregates(*, names: list[str]) -> dict[str, pl.DataFrame]:
    """Return the outside-events aggregates of the variants, scoring those not cached.

    Args:
        names: The variant names the scorer's full run printed.

    Returns:
        The aggregates by name.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {n: CACHE_DIR / f"outside_{n}.parquet" for n in names}
    if all(p.exists() for p in wanted.values()):
        return {n: pl.read_parquet(p) for n, p in wanted.items()}
    choices = arms.read_choices()
    args = argparse.Namespace(
        b_star=choices["b_star"], o_star=choices["o_star"], h_t=choices["h_t"]
    )
    months = tuple(arms.SCORED_MONTHS)
    series = adapter.series_table()
    ctx = s707.cleaned_scoring_context(series=series)
    variants, _have = score.needed_variants(months=months, args=args)
    scorer = OutsideScorer(
        series=series,
        ctx=ctx,
        variants=variants,
        months=months,
        reference_keys=score.reference_scored_keys(series=series, ctx=ctx),
    )
    print(f"outside scorer: shared half-hours {scorer.table.height}")
    out = {}
    for name in names:
        if wanted[name].exists():
            out[name] = pl.read_parquet(wanted[name])
            continue
        out[name] = scorer.aggregate(variant=scorer.variants[name])
        common.atomic_write(frame=out[name], target=wanted[name])
        print(f"scored outside events: {name}")
    return out


def outside_table(*, names: list[str]) -> list[dict]:
    """Print the CRPS and MAE of the median outside events, and the overlap of the event slices.

    Args:
        names: The variant names to score.

    Returns:
        One record per variant, group, and metric.
    """
    print("\n== rows outside events and at least 7 days after one (95% month-block intervals) ==")
    aggregates = outside_aggregates(names=names)
    out = []
    for group in score.GROUPS:
        for name, aggregate in aggregates.items():
            for metric in score.SCALE_METRICS:
                mean, low, high, n, months = score.absolute_interval(
                    aggregate=aggregate, group=group, metric=metric, slice_name=score.ALL_ROWS
                )
                print(
                    f"outside events           {group:16s} {name:18s} {metric:9s} {mean:.3f} "
                    f"({low:.3f} to {high:.3f}) n={n} months={months}"
                )
                out.append(
                    {
                        "group": group,
                        "variant": name,
                        "metric": metric,
                        "mean": mean,
                        "lower": low,
                        "upper": high,
                        "n": n,
                    }
                )
    reference = latest_cache(name=names[0])
    for group in score.GROUPS:
        counts = {
            s: score.month_totals(aggregate=reference, group=group, metric="crps", slice_name=s)[2]
            for s in (score.ALL_ROWS, "event_rows", "event_following_7_days")
        }
        outside = score.month_totals(aggregate=aggregates[names[0]], group=group, metric="crps")[2]
        total, inside, after = (counts[k].sum() for k in counts)
        print(
            f"{group}: all {total:.0f}, inside events {inside:.0f}, 7 days after {after:.0f}, "
            f"outside {outside.sum():.0f}, rows in both event slices "
            f"{inside + after - (total - outside.sum()):.0f}"
        )
    for ta, ra in (("FTh90d-i@01234", "M1-i@01234"),):
        for group in score.GROUPS:
            score.contrast(
                label=f"C1 outside events ({ta} minus {ra})",
                group=group,
                treatment=aggregates[ta],
                reference=aggregates[ra],
            )
    return out


# --- slump share with intervals ----------------------------------------------------------------


def month_block_interval(
    *, values: np.ndarray, months: np.ndarray, level: float = 95.0, n_resamples: int = 20_000
) -> tuple[float, float, float]:
    """Return the mean of per-event values and a month-block bootstrap interval.

    Args:
        values: One value per event.
        months: The fold month of each event.
        level: The interval's coverage in percent.
        n_resamples: How many resamples to draw.

    Returns:
        The mean, the lower limit, and the upper limit.
    """
    unique = np.unique(months)
    sums = np.array([values[months == m].sum() for m in unique])
    counts = np.array([float((months == m).sum()) for m in unique])
    rng = np.random.default_rng(score.ABSOLUTE_SEED)
    draws = rng.integers(0, len(unique), size=(n_resamples, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(values.mean()), float(low), float(high)


def slump_with_intervals() -> list[dict]:
    """Print the slump share of each recipe and state with a month-block interval.

    Returns:
        One record per recipe, state, and group (seeds 0 to 2, events over both bank passes).
    """
    print("\n== slump share per recipe and state, seeds 0 to 2 (95% month-block intervals) ==")
    months = tuple(arms.SCORED_MONTHS)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / "slump_with_intervals.json"
    if cached.exists():
        out = json.loads(cached.read_text())
        for r in out:
            print(
                f"{r['arm']}-{r['state']}@012 {r['group']:15s} share {r['mean']:+.3f} "
                f"({r['lower']:+.3f} to {r['upper']:+.3f}) events={r['events']}"
            )
        return out
    out = []
    for arm in ("M1", "M2", "FTh90d"):
        for state in ("i", "n"):
            variant = score.Variant(arm=arm, state=state, seeds=SEEDS_PAIRED)
            events = score.event_means(bank=score.bank_table(variant=variant, months=months))
            for group in score.GROUPS:
                e = events.filter(
                    pl.col("label").is_in(list(score.SUBSETS[group])) & (pl.col("part") == "total")
                )
                mean, low, high = month_block_interval(
                    values=e["share"].to_numpy(), months=e["month"].to_numpy()
                )
                print(
                    f"{arm}-{state}@012 {group:15s} share {mean:+.3f} ({low:+.3f} to {high:+.3f}) "
                    f"events={e.height}"
                )
                out.append(
                    {
                        "arm": arm,
                        "state": state,
                        "group": group,
                        "mean": mean,
                        "lower": low,
                        "upper": high,
                        "events": e.height,
                    }
                )
    cached.write_text(json.dumps(out))
    return out


def fit_log_summary() -> list[dict]:
    """Print, from the fit logs of the scored folds, how the fine-tune stage was selected.

    For each arm and state with a fine-tune (`FT`) stage: the number of fits, the number whose
    selected epoch is the first (index 0), the number whose selected validation loss is worse than
    the loss of the state it started from, and the mean share of training rows with a shifted lag.

    Returns:
        One record per arm and state.
    """
    print("\n== fine-tune selection and shifted-lag share, scored folds, from the fit logs ==")
    out = []
    for folder in sorted((score.OUT_DIR / "logs").iterdir()):
        records = []
        for path in folder.glob("*.json"):
            log = json.loads(path.read_text())
            if log["month"] not in arms.SCORED_MONTHS or "FT" not in log["stages"]:
                continue
            records.append(log["stages"]["FT"])
        if not records:
            continue
        first = sum(r["best_epoch"] == 0 for r in records)
        worse = sum(r["best_val"] > r["initial_val"] for r in records)
        share = [r["shifted_share"]["unweighted"] for r in records if r.get("shifted_share")]
        mean_share = float(np.mean(share)) if share else float("nan")
        print(
            f"{folder.name:10s} fits={len(records)} selected epoch 0: {first} "
            f"worse than the starting state: {worse} mean shifted-lag share {mean_share:.3f}"
        )
        out.append({"arm": folder.name, "fits": len(records), "first": first, "worse": worse})
    return out


# --- median minus mean -------------------------------------------------------------------------


def median_minus_mean(*, arm_states: tuple[tuple[str, str], ...]) -> dict[str, dict[str, float]]:
    """Print, per Primary, the median forecast minus the mean forecast in points of capacity.

    The stored forecasts hold 11 quantiles and no mean. The mean forecast is the mass-weighted
    mean of the 11 quantiles (`common.ATOM_MASSES`), the same stand-in that experiment 707 uses.
    Rows are those with a lead of at most 240 hours and a valid time before the holdout.

    Args:
        arm_states: The arms and states to read, seeds 0 to 2 and all 12 folds.

    Returns:
        The mean gap by arm-state and Primary.
    """
    print("\n== median forecast minus mean forecast, points of capacity, per Primary ==")
    sites = prep.primaries()
    labels = dict(zip(sites["time_series_id"].to_list(), sites["site"].to_list(), strict=True))
    quantiles = list(common.LEVEL_COLUMNS)
    result: dict[str, dict[str, float]] = {}
    for arm, state in arm_states:
        folder = score.forecast_folder(arm=arm, state=state)
        pieces = []
        for month in arms.SCORED_MONTHS:
            for seed in SEEDS_PAIRED:
                frame = pl.read_parquet(folder / f"{month}_s{seed}.parquet").filter(
                    (pl.col("valid_time") < score.HOLDOUT)
                    & (
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
                        <= score.MAX_LEAD_HOURS
                    )
                )
                mean = sum(
                    pl.col(q.lower()) * float(m)
                    for q, m in zip(quantiles, common.ATOM_MASSES, strict=True)
                )
                pieces.append(
                    frame.with_columns(gap=100.0 * (pl.col("q50") - mean))
                    .group_by("time_series_id")
                    .agg(pl.col("gap").sum(), pl.len().alias("n"))
                )
        total = (
            pl.concat(pieces).group_by("time_series_id").agg(pl.col("gap").sum(), pl.col("n").sum())
        )
        values = {
            labels[int(i)]: round(g / n, 3)
            for i, g, n in total.sort("time_series_id").iter_rows()
            if int(i) in labels
        }
        print(f"{arm}-{state}@012: {values}")
        result[f"{arm}-{state}"] = values
    return result


# --- figures -----------------------------------------------------------------------------------


def save_svg(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG into the docs folder and optimise it with svgo.

    Args:
        chart: The chart.
        name: The file name without the extension.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    target = FIGURE_DIR / f"{name}.svg"
    chart.save(str(target))
    if PNG_DIR is not None:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        chart.save(str(PNG_DIR / f"{name}.png"), scale_factor=1.5)
    npx = shutil.which("npx")
    if npx is not None:
        subprocess.run(
            [
                npx,
                "--yes",
                "svgo@4",
                "--multipass",
                "--precision=1",
                "--final-newline",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {target.name}: {target.stat().st_size} bytes")


def figure_1(*, parsed: dict[str, list[dict]]) -> None:
    """Draw the forest plot of the pre-registered contrasts and the secondary readings.

    Args:
        parsed: The output of `parse_log`.
    """
    rows = []
    for key, label in CONTRAST_ROWS:
        for group in score.GROUPS:
            r = pick(
                rows=parsed["contrasts"],
                name=key,
                group=group,
                what=next(
                    c["what"]
                    for c in parsed["contrasts"]
                    if c["name"] == key and c["what"].count("minus") == 1
                ),
                metric="crps",
            )
            rows.append(
                {
                    "label": f"{label} ({r['level']:g}%)",
                    "group": group,
                    "estimate": r["estimate"],
                    "lower": r["lower"],
                    "upper": r["upper"],
                    "headline": key == "C1",
                }
            )
    frame = pl.DataFrame(rows)
    order = [f"{label} ({pick_level(parsed=parsed, key=key)})" for key, label in CONTRAST_ROWS]
    panels = []
    for group, title in (
        (SIX, "The six signed-MW Primaries"),
        (EIGHT, "The eight MVA-only Primaries"),
    ):
        data = frame.filter(pl.col("group") == group)
        y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=340))
        colour = alt.Color(
            "headline:N",
            scale=alt.Scale(domain=[True, False], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=None,
        )
        x_title = "Treatment minus reference, CRPS (points of capacity; negative: treatment better)"
        x_scale = alt.Scale(domain=[-0.17, 0.17])
        x = alt.X("estimate:Q", title=x_title, scale=x_scale)
        base = alt.Chart(data)
        panel = (
            base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lower:Q", title=x_title, scale=x_scale), x2="upper:Q", y=y, color=colour
            )
            + base.mark_point(filled=True, size=70).encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
            + alt.Chart(pl.DataFrame({"v": [0.0]})).mark_rule(color=BLACK).encode(x="v:Q")  # ty: ignore[unresolved-attribute]
            + alt.Chart(pl.DataFrame({"v": [0.02]}))
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(x="v:Q")  # ty: ignore[unresolved-attribute]
        ).properties(width=WIDTH_PX - 280, height=250, title=title)
        panels.append(panel)
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Contrasts of experiment 710, all with injection in training unless the row says -n",
            subtitle=[
                "Dot: estimate. Bar: interval from resampling whole months, at the level in brackets.",
                "Solid line: no difference. Dashed line: the +0.02 non-inferiority margin of C1.",
                "Data: 12 monthly refits, 2025-07 to 2026-06, lead days 1 to 10.",
            ],
        )
    )
    save_svg(chart=chart, name="contrasts")


def pick_level(*, parsed: dict[str, list[dict]], key: str) -> str:
    """Return the Bonferroni level of a contrast as printed in the row label.

    Args:
        parsed: The output of `parse_log`.
        key: The contrast name.

    Returns:
        The level, such as `99.17%`.
    """
    row = next(c for c in parsed["contrasts"] if c["name"] == key and c["metric"] == "crps")
    return f"{row['level']:g}%"


def figure_2(*, slump: list[dict]) -> None:
    """Draw the slump share with and without injection for each recipe and group.

    Args:
        slump: The output of `slump_with_intervals`.
    """
    frame = pl.DataFrame(slump).with_columns(
        state=pl.when(pl.col("state") == "i")
        .then(pl.lit("Injection in training"))
        .otherwise(pl.lit("No injection")),
        arm=pl.col("arm").replace(
            {
                "M1": "M1 (3 stages)",
                "M2": "M2 (ENS, then fine-tune)",
                "FTh90d": "FTh90d (90-day fine-tune)",
            }
        ),
    )
    panels = []
    for group, title in (
        (SIX, "The six signed-MW Primaries"),
        (EIGHT, "The eight MVA-only Primaries"),
    ):
        data = frame.filter(pl.col("group") == group)
        colour = alt.Color(
            "state:N",
            scale=alt.Scale(
                domain=["Injection in training", "No injection"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(orient="top", title=None),
        )
        y = alt.Y("arm:N", title=None, axis=alt.Axis(labelLimit=340))
        base = alt.Chart(data)
        scale = alt.Scale(domain=[0, 0.4])
        panel = (
            base.mark_rule(strokeWidth=2.5).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lower:Q", scale=scale, title="Slump share"),
                x2="upper:Q",
                y=y,
                yOffset=alt.YOffset("state:N"),
                color=colour,
            )
            + base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("mean:Q", scale=scale),
                y=y,
                yOffset=alt.YOffset("state:N"),
                color=colour,
            )
        ).properties(width=WIDTH_PX - 280, height=130, title=title)
        panels.append(panel)
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Share of an injected load shift that the median forecast follows",
            subtitle=[
                "0: the forecast keeps to the normal running arrangement. 1: it takes up the whole shift.",
                "Lower is better. Dot: mean over injected events. Bar: 95% interval from resampling months.",
                "Seeds 0 to 2; events from both bank passes of each of the 12 folds.",
            ],
        )
    )
    save_svg(chart=chart, name="slump_share")


def week_frame(
    *,
    folder: Path,
    pattern: str,
    months: list[str],
    site: str,
    window: tuple[datetime, datetime],
    leads: tuple[float, float] | None,
    value: str = "q50",
) -> pl.DataFrame:
    """Return one Primary's seed-averaged forecast over a window, as percent of capacity.

    Args:
        folder: The forecast folder.
        pattern: The file name with `{month}` and `{seed}` fields.
        months: The fold months to read.
        site: The Primary's label.
        window: The first and last valid time to keep.
        leads: The range of lead hours to keep, or `None` for all leads.
        value: The quantile column to return.

    Returns:
        Columns `valid_time` and `value`; one row per valid time (day-ahead) or per origin.
    """
    sites = prep.primaries()
    ts_id = sites.filter(pl.col("site") == site)["time_series_id"][0]
    pieces = []
    for month in months:
        paths = [folder / pattern.format(month=month, seed=s) for s in SEEDS_PAIRED]
        if not all(p.exists() for p in paths):
            continue
        frame = score.average_files(paths=paths).filter(pl.col("time_series_id") == ts_id)
        lead = (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours()
        if leads is not None:
            frame = frame.filter((lead >= leads[0]) & (lead < leads[1]))
        pieces.append(frame)
    joined = pl.concat(pieces).filter(
        (pl.col("valid_time") >= window[0]) & (pl.col("valid_time") <= window[1])
    )
    return joined.select(
        "valid_time", (100.0 * pl.col(value).cast(pl.Float64)).alias("value")
    ).sort("valid_time")


def observed(*, site: str, window: tuple[datetime, datetime]) -> pl.DataFrame:
    """Return a Primary's observed power as percent of capacity over a window.

    Args:
        site: The Primary's label.
        window: The first and last time to keep (UTC datetimes).

    Returns:
        Columns `valid_time` and `value`.
    """
    first = int(common.grid_index(times=window[0]))  # ty: ignore[invalid-argument-type]
    last = int(common.grid_index(times=window[1]))  # ty: ignore[invalid-argument-type]
    index = np.arange(first, last + 1)
    values = 100.0 * common.take(values=inject.observed_power()[site], index=index)
    times = [common.GRID_START + int(i) * common.STEP for i in index]
    return pl.DataFrame({"valid_time": times, "value": values.astype(np.float64)})


def figure_3() -> None:
    """Draw a day-ahead forecast and a 14-day forecast of one Primary for M1-i and FTh90d-i."""
    site = EXAMPLE_SITE
    origin = datetime.fromisoformat(EXAMPLE_ORIGIN).replace(tzinfo=UTC)
    print(f"\n== Figure 3: Primary {site}, fold {EXAMPLE_FOLD}, origin {origin}, seeds 0 to 2 ==")
    panels = []
    for title, window in (
        (
            "Day ahead: the next day from the 09:00 UTC origin",
            (origin + timedelta(hours=15), origin + timedelta(hours=39)),
        ),
        (
            "Full horizon: 14 days from the same origin",
            (origin + timedelta(hours=1), origin + timedelta(days=14)),
        ),
    ):
        pieces = [
            observed(site=site, window=window).with_columns(
                series=pl.lit("Observed power"), band=pl.lit(0)
            )
        ]
        for arm, name in (("M1", "M1-i (3 stages)"), ("FTh90d", "FTh90d-i (ENS only)")):
            folder = score.forecast_folder(arm=arm, state="i")
            sites = prep.primaries()
            ts_id = sites.filter(pl.col("site") == site)["time_series_id"][0]
            full = score.average_files(
                paths=[folder / f"{EXAMPLE_FOLD}_s{s}.parquet" for s in SEEDS_PAIRED]
            ).filter(
                (pl.col("time_series_id") == ts_id) & (pl.col("power_fcst_init_time") == origin)
            )
            full = full.filter(
                (pl.col("valid_time") >= window[0]) & (pl.col("valid_time") <= window[1])
            )
            last_valid = full.get_column("valid_time").max()
            max_lead_hours = (
                (last_valid - origin).total_seconds() / 3600
                if isinstance(last_valid, datetime)
                else float("nan")
            )
            print(
                f"{name}: {full.height} half-hours in the panel; max lead hours {max_lead_hours:.0f}"
            )
            pieces.append(
                full.select(
                    "valid_time",
                    (100.0 * pl.col("q50").cast(pl.Float64)).alias("value"),
                    (100.0 * pl.col("q10").cast(pl.Float64)).alias("low"),
                    (100.0 * pl.col("q90").cast(pl.Float64)).alias("high"),
                ).with_columns(series=pl.lit(name))
            )
        data = pl.concat(pieces, how="diagonal")
        lines = (
            alt.Chart(data)
            .mark_line(clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("valid_time:T", title=None),
                y=alt.Y("value:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=["Observed power", "M1-i (3 stages)", "FTh90d-i (ENS only)"],
                        range=[BLACK, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        band = (
            alt.Chart(data.filter(pl.col("low").is_not_null()))
            .mark_area(opacity=0.18, clip=True)
            .encode(x="valid_time:T", y="low:Q", y2="high:Q", color="series:N")  # ty: ignore[unresolved-attribute]
        )
        panels.append((band + lines).properties(width=WIDTH_PX - 90, height=190, title=title))
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            f"One forecast of the Primary {site}, fold {EXAMPLE_FOLD}, origin {EXAMPLE_ORIGIN} UTC",
            subtitle=[
                "Line: median forecast. Shade: 10% to 90% band. Both recipes trained with injection;",
                "weather is the 00Z ENS run issued before the origin. The Primary and origin are those of experiment 425's winter example.",
            ],
        )
    )
    save_svg(chart=chart, name="example_forecast")


def synthetic_events() -> list[dict]:
    """Return the injected events plotted in Figure 4, chosen by a rule fixed before looking.

    For each group, the first event in fold order (then pass, then Primary number) of depth 0.5
    for each length in `EVENT_LENGTHS`.

    Returns:
        Records with `month`, `pass`, `primary`, `site`, `group`, `depth`, and `length`.
    """
    sites = prep.primaries()["site"].to_list()
    chosen = []
    for group in score.GROUPS:
        for length in EVENT_LENGTHS:
            done = False
            for k, month in enumerate(arms.SCORED_MONTHS):
                for pass_id in (k, k + len(bank581.COMBOS)):
                    for p, site in enumerate(sites):
                        if site not in score.SUBSETS[group] or done:
                            continue
                        signed, n_days, _ = bank581.event(primary=p, r=pass_id)
                        if n_days == length and abs(signed) == EVENT_DEPTH:
                            chosen.append(
                                {
                                    "month": month,
                                    "pass": pass_id,
                                    "primary": p,
                                    "site": site,
                                    "group": group,
                                    "depth": signed,
                                    "length": n_days,
                                }
                            )
                            done = True
    return chosen


def event_panel(
    *,
    title: str,
    site: str,
    months: list[str],
    window: tuple[datetime, datetime],
    onset: datetime,
    end: datetime,
    shift: float,
    bank_pass: int,
    arms_states: tuple[tuple[str, str, str, str], ...],
    corrected: bool,
) -> alt.LayerChart:
    """Return one event panel: observed power, the lags' reading, and the day-ahead medians.

    Args:
        title: The panel title.
        site: The Primary's label.
        months: The fold months that the window spans.
        window: The first and last valid time shown.
        onset: The event's first half-hour.
        end: The event's end (exclusive).
        shift: The event's signed depth as a share of capacity.
        bank_pass: The bank pass whose forecast is drawn.
        arms_states: Tuples of arm, state, label, and colour.
        corrected: Whether the pass holds corrected lags (real events) or shifted lags.

    Returns:
        The layered chart.
    """
    pieces = [observed(site=site, window=window).with_columns(series=pl.lit("Observed power"))]
    lag = observed(site=site, window=window)
    if not corrected:
        lag = lag.with_columns(
            value=pl.when((pl.col("valid_time") >= onset) & (pl.col("valid_time") < end))
            .then(pl.col("value") + 100.0 * shift)
            .otherwise(pl.col("value"))
        )
        pieces.append(lag.with_columns(series=pl.lit("Power as the lags read it")))
    for arm, state, label, _colour in arms_states:
        pattern = "{month}_s{seed}_p" + str(bank_pass) + ".parquet"
        folder = score.OUT_DIR / "bank" / f"{arm}-{state}"
        if corrected:
            plain = week_frame(
                folder=score.forecast_folder(arm=arm, state=state),
                pattern="{month}_s{seed}.parquet",
                months=months,
                site=site,
                window=window,
                leads=DAY_AHEAD_LEADS,
            )
            pieces.append(plain.with_columns(series=pl.lit(f"{label}, lags as observed")))
        frame = week_frame(
            folder=folder,
            pattern=pattern,
            months=months,
            site=site,
            window=window,
            leads=DAY_AHEAD_LEADS,
        )
        suffix = ", lags corrected" if corrected else ", shifted lags"
        pieces.append(frame.with_columns(series=pl.lit(label + suffix)))
    data = pl.concat(pieces)
    domain = list(dict.fromkeys(data["series"].to_list()))
    palette = {"Observed power": BLACK, "Power as the lags read it": GREY}
    for _arm, _state, label, colour in arms_states:
        palette[label + (", lags corrected" if corrected else ", shifted lags")] = colour
        palette[f"{label}, lags as observed"] = colour
    lines = (
        alt.Chart(data)
        .mark_line(clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("valid_time:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("value:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=[palette[d] for d in domain]),
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(
                    domain=domain,
                    range=[
                        [3, 3] if ("lags as observed" in d or "read it" in d) else [1, 0]
                        for d in domain
                    ],
                ),
                legend=None,
            ),
        )
    )
    event = (
        alt.Chart(pl.DataFrame({"a": [onset], "b": [end]}))
        .mark_rect(opacity=0.12, color=ocf.BRAND_ORANGE)
        .encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
    )
    after_end = min(end + timedelta(days=AFTER_DAYS), window[1])
    after = (
        alt.Chart(pl.DataFrame({"a": [end], "b": [after_end]}))
        .mark_rect(opacity=0.12, color=ocf.DATA_BLUE)
        .encode(x="a:T", x2="b:T")  # ty: ignore[unresolved-attribute]
    )
    marks = (
        alt.Chart(pl.DataFrame({"t": [onset, end], "what": ["start", "end"]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=1.5)
        .encode(x="t:T")  # ty: ignore[unresolved-attribute]
    )
    return (event + after + marks + lines).properties(width=WIDTH_PX - 90, height=170, title=title)


def figure_4() -> None:
    """Draw eight event panels: six injected events (three lengths, two groups) and two real."""
    print("\n== Figure 4: event panels ==")
    panels = []
    synthetic_arms = (
        ("M1", "i", "M1-i", ocf.DATA_BLUE),
        ("FTh90d", "i", "FTh90d-i", ocf.BRAND_ORANGE),
        ("FTh90d", "n", "FTh90d-n", ocf.DATA_PURPLE),
    )
    for e in synthetic_events():
        refit = datetime.fromisoformat(f"{e['month']}-01").replace(tzinfo=UTC)
        end = refit + timedelta(days=bank581.END_DAY)
        onset = end - timedelta(days=e["length"])
        window = (refit, end + timedelta(days=7))
        group = "six signed-MW" if e["group"] == SIX else "eight MVA-only"
        title = (
            f"Injected, {group} Primary {e['site']}: {e['length']} days, "
            f"{e['depth']:+.2f} of capacity, fold {e['month']}"
        )
        print(f"{title} (pass {e['pass']})")
        panels.append(
            event_panel(
                title=title,
                site=e["site"],
                months=[e["month"]],
                window=window,
                onset=onset,
                end=end,
                shift=e["depth"],
                bank_pass=e["pass"],
                arms_states=synthetic_arms,
                corrected=False,
            )
        )
    sites = prep.primaries()["site"].to_list()
    real_arms = (
        ("M1", "i", "M1-i", ocf.DATA_BLUE),
        ("FTh90d", "i", "FTh90d-i", ocf.BRAND_ORANGE),
    )
    for event in inject.real_events_table():
        site = sites[event["primary"]]
        onset = common.GRID_START + event["onset_idx"] * common.STEP
        end = common.GRID_START + event["end_idx"] * common.STEP
        if site not in REAL_SITES or onset.strftime("%Y-%m") != "2025-12":
            continue
        months = sorted({onset.strftime("%Y-%m"), (end + timedelta(days=7)).strftime("%Y-%m")})
        window = (
            datetime.fromisoformat(f"{months[0]}-01").replace(tzinfo=UTC),
            end + timedelta(days=7),
        )
        title = f"Real, six signed-MW Primary {site}: transfer {event['transfer_id']}, step {event['depth']:+.2f} of capacity"
        print(f"{title}: onset {onset}, end {end}")
        panels.append(
            event_panel(
                title=title,
                site=site,
                months=months,
                window=window,
                onset=onset,
                end=end,
                shift=event["depth"],
                bank_pass=bank581.REAL_PASS,
                arms_states=real_arms,
                corrected=True,
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Day-ahead median forecasts around injected and real load shifts",
            subtitle=[
                "Orange shade: event, between the two orange rules. Blue shade: the 7 days after the event ends.",
                "Injected events change the lag inputs only; the forecast should stay on the observed power.",
                "Real events: dashed lines use the lags as observed, solid lines the lags corrected for the event.",
                "NGED's event log may miss events, so some clean-looking rows may hold an unlogged event.",
            ],
        )
    )
    save_svg(chart=chart, name="load_shift_events")


# --- markdown tables for the page --------------------------------------------------------------

ARM_ORDER: Final[tuple[str, ...]] = (
    "M1-i",
    "M1-n",
    "M2-i",
    "M2-n",
    "M3-i",
    "W-h1-i",
    "PP-i",
    "P-i",
    "FTh90d-i",
    "FTh90d-n",
    "M1h90-i",
)
GROUP_SHORT: Final[dict[str, str]] = {SIX: "The six", EIGHT: "The eight"}


def full_name(*, parsed: dict[str, list[dict]], arm: str) -> str:
    """Return the longest-seed variant name of an arm, such as `M1-i@01234`."""
    names = {r["variant"] for r in parsed["absolute"] if r["variant"].split("@")[0] == arm}
    return max(names, key=lambda n: len(n.split("@")[1]))


def cell(*, parsed: dict[str, list[dict]], slice_name: str, group: str, variant: str) -> str:
    """Return `CRPS (MAE)` of a variant on a slice and group, as printed by the scorer."""
    crps = pick(
        rows=parsed["absolute"], slice=slice_name, group=group, variant=variant, metric="crps"
    )
    mae = pick(
        rows=parsed["absolute"], slice=slice_name, group=group, variant=variant, metric="mae_median"
    )
    return f"{crps['mean']:.3f} ({mae['mean']:.3f})"


def write_tables(  # noqa: PLR0912, PLR0915
    *,
    parsed: dict[str, list[dict]],
    outside: list[dict],
    mae: list[dict],
    gaps: dict[str, dict[str, float]],
    slump: list[dict],
) -> None:
    """Write the page's tables as markdown to `report_tables.md`, one named block per table.

    Args:
        parsed: The output of `parse_log`.
        outside: The output of `outside_table`.
        mae: The output of `mae_contrasts`.
        gaps: The output of `median_minus_mean`.
        slump: The output of `slump_with_intervals`.
    """
    blocks: dict[str, list[str]] = {}
    names = {arm: full_name(parsed=parsed, arm=arm) for arm in ARM_ORDER}

    def seeds(arm: str) -> str:
        return names[arm].split("@")[1]

    head = ["| Arm | Seeds | The six | The eight |", "|---|---|---|---|"]
    blocks["headline"] = [
        *head,
        *[
            f"| `{arm}` | {seeds(arm)} | "
            + " | ".join(
                cell(parsed=parsed, slice_name=score.ALL_ROWS, group=g, variant=names[arm])
                for g in score.GROUPS
            )
            + " |"
            for arm in ARM_ORDER
        ],
    ]
    outside_by = {(r["variant"], r["group"], r["metric"]): r for r in outside}
    for group in score.GROUPS:
        rows = [
            "| Arm | All rows | Outside events and 7 days after | Inside events | The 7 days after |",
            "|---|---|---|---|---|",
        ]
        for arm in ARM_ORDER:
            v = names[arm]
            o_c = outside_by[(v, group, "crps")]["mean"]
            o_m = outside_by[(v, group, "mae_median")]["mean"]
            rows.append(
                f"| `{arm}` | "
                + cell(parsed=parsed, slice_name=score.ALL_ROWS, group=group, variant=v)
                + f" | {o_c:.3f} ({o_m:.3f}) | "
                + cell(parsed=parsed, slice_name="event_rows", group=group, variant=v)
                + " | "
                + cell(parsed=parsed, slice_name="event_following_7_days", group=group, variant=v)
                + " |"
            )
        blocks[f"scenarios_{'six' if group == SIX else 'eight'}"] = rows
    for group in score.GROUPS:
        rows = ["| Arm | Sunny half-hours | Cold nights |", "|---|---|---|"]
        for arm in ARM_ORDER:
            v = names[arm]
            rows.append(
                f"| `{arm}` | "
                + cell(parsed=parsed, slice_name="sunny_unit_shape", group=group, variant=v)
                + " | "
                + cell(parsed=parsed, slice_name="cold_nights", group=group, variant=v)
                + " |"
            )
        blocks[f"sunny_cold_{'six' if group == SIX else 'eight'}"] = rows
    counts = ["| Subset | The six | The eight |", "|---|---|---|"]
    for label, key in (
        ("All rows", score.ALL_ROWS),
        ("Sunny half-hours (months)", "sunny_unit_shape"),
        ("Cold nights", "cold_nights"),
        ("Inside events", "event_rows"),
        ("The 7 days after", "event_following_7_days"),
    ):
        cells = []
        for g in score.GROUPS:
            r = pick(
                rows=parsed["absolute"],
                slice=key,
                group=g,
                variant=names["M2-i"],
                metric="crps",
            )
            cells.append(f"{r['n']:,} ({r['months']})")
        counts.append(f"| {label} | " + " | ".join(cells) + " |")
    out_n = {g: next(r["n"] for r in outside if r["group"] == g) for g in score.GROUPS}
    counts.append(
        "| Outside events and 7 days after | "
        + " | ".join(f"{out_n[g]:,} (12)" for g in score.GROUPS)
        + " |"
    )
    blocks["counts"] = counts
    mae_by = {(r["label"], r["group"]): r for r in mae}
    rows = [
        "| Contrast | Group | CRPS (level) | MAE of the median (95%) |",
        "|---|---|---|---|",
    ]
    for key, label in CONTRAST_ROWS:
        for group in score.GROUPS:
            c = next(
                r
                for r in parsed["contrasts"]
                if r["name"] == key
                and r["group"] == group
                and r["metric"] == "crps"
                and r["what"].count("minus") == 1
            )
            m = mae_by[(key, group)]
            rows.append(
                f"| {label} | {GROUP_SHORT[group]} | {interval_text(row=c)} ({c['level']:g}%) | "
                f"{m['difference']:+.3f} ({m['lower']:+.3f} to {m['upper']:+.3f}) |"
            )
    blocks["contrasts"] = rows
    rows = [
        (
            "| Arm | The six: all events | First 4 days | Day 4 to end | 7 days after | "
            "The eight: all events | First 4 days | Day 4 to end | 7 days after |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in ("M1-i", "M2-i", "M3-i", "W-h1-i", "PP-i", "P-i", "FTh90d-i", "FTh90d-n", "M1h90-i"):
        v = names[arm]
        if arm in ("M3-i", "W-h1-i", "PP-i", "P-i", "FTh90d-n", "M1h90-i"):
            v = v.split("@")[0] + "@012"
        cells = []
        for g in score.GROUPS:
            for part in ("total", "first_4_days", "day_4_to_end", "after_7_days"):
                r = pick(rows=parsed["slump"], variant=v, group=g, part=part)
                cells.append(f"{r['share']:.3f}")
        rows.append(f"| `{arm}` ({v.split('@')[1]}) | " + " | ".join(cells) + " |")
    blocks["slump"] = rows
    rows = [
        "| Arm | The six: share (95% interval) | The eight: share (95% interval) |",
        "|---|---|---|",
    ]
    for arm in ("M1", "M2", "FTh90d"):
        for state in ("i", "n"):
            cells = []
            for g in score.GROUPS:
                r = next(
                    x for x in slump if x["arm"] == arm and x["state"] == state and x["group"] == g
                )
                cells.append(f"{r['mean']:.3f} ({r['lower']:.3f} to {r['upper']:.3f})")
            rows.append(f"| `{arm}-{state}` | " + " | ".join(cells) + " |")
    blocks["slump_intervals"] = rows
    rows = ["| Arm | The six: q05 / q95 | The eight: q05 / q95 |", "|---|---|---|"]
    for arm in ARM_ORDER:
        cells = []
        for g in score.GROUPS:
            r = pick(rows=parsed["coverage"], group=g, variant=names[arm])
            cells.append(f"{r['q05']:.3f} / {r['q95']:.3f}")
        rows.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    blocks["coverage"] = rows
    rows = [
        (
            "| Arm | Fits | Mean steps of the final stage | Mean seconds per fit | "
            "Share at the limit | Effective rows per batch |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for r in parsed["convergence"]:
        rows.append(
            f"| `{r['variant']}` | {r['fits']} | {r['steps']:,} | {r['seconds']} | "
            f"{r['at_limit']:.2f} | {r['rows']:,} |"
        )
    blocks["convergence"] = rows
    rows = [
        (
            "| Primary | Transfer | Step (share of capacity) | Arm | Half-hours | Slump share | "
            "Error against the NRA (points) |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in parsed["real"]:
        if "real-event" in r["flag"] and r["variant"] == names["FTh90d-i"]:
            for v in (names["M1-i"], names["M2-i"], names["FTh90d-i"]):
                for x in parsed["real"]:
                    if (
                        x["variant"] == v
                        and x["site"] == r["site"]
                        and x["n"] == r["n"]
                        and "real-event" in x["flag"]
                    ):
                        rows.append(
                            f"| {x['site']} | {x['transfer']} | {x['depth']:+.3f} | "
                            f"`{x['variant'].split('@')[0]}` | {x['n']:,} | {x['share']:+.3f} | "
                            f"{x['nra_error']:+.2f} |"
                        )
    blocks["real"] = rows
    sites_six = sorted(score.SUBSETS[SIX], key=lambda s: int(s[1:]))
    sites_eight = sorted(score.SUBSETS[EIGHT], key=lambda s: int(s[1:]))
    rows = [
        "| Arm | " + " | ".join(sites_six + sites_eight) + " |",
        "|---|" + "---|" * (len(sites_six) + len(sites_eight)),
    ]
    for arm, values in gaps.items():
        rows.append(
            f"| `{arm}` | "
            + " | ".join(f"{values[s]:+.2f}" for s in sites_six + sites_eight)
            + " |"
        )
    blocks["gaps"] = rows
    tuning = LOG.parent / "tuning_rerun.log"
    rows = ["| Arm | Mean CRPS on tuning folds 2025-03 and 2025-06 |", "|---|---|"]
    on = False
    for line in tuning.read_text().splitlines():
        if line.startswith("== ranking"):
            on = True
            continue
        if on and (m := re.match(r"^(\S+) +(\d+\.\d+)$", line)):
            rows.append(f"| `{m.group(1)}` | {m.group(2)} |")
        elif on and line.strip() == "":
            break
    blocks["tuning"] = rows
    target = score.OUT_DIR / "report_tables.md"
    target.write_text(
        "\n\n".join(f"<!-- {name} -->\n" + "\n".join(lines) for name, lines in blocks.items())
        + "\n"
    )
    print(f"\nwrote {target}")


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    """Print every number the page quotes, write the tables, and draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="*",
        type=int,
        default=[1, 2, 3, 4],
        help="the figures to draw (default: all four); an empty list draws none",
    )
    parser.add_argument(
        "--png-dir", type=Path, default=None, help="also write a PNG of each figure"
    )
    parser.add_argument(
        "--figures-only", action="store_true", help="skip the readings and the tables"
    )
    args = parser.parse_args()
    global PNG_DIR  # noqa: PLW0603
    PNG_DIR = args.png_dir
    parsed = parse_log(path=LOG)
    print_parsed(parsed=parsed)
    slump = slump_with_intervals()
    if not args.figures_only:
        extra_contrasts()
        mae = mae_contrasts()
        fit_log_summary()
        gaps = median_minus_mean(arm_states=(("M1", "i"), ("M2", "i"), ("FTh90d", "i")))
        names = sorted({r["variant"] for r in parsed["absolute"] if r["variant"].count("@") == 1})
        full = [n for n in names if n.split("@")[0] in ARM_ORDER]
        outside = outside_table(names=full)
        write_tables(parsed=parsed, outside=outside, mae=mae, gaps=gaps, slump=slump)
    if 1 in args.figures:
        figure_1(parsed=parsed)
    if 2 in args.figures:
        figure_2(slump=slump)
    if 3 in args.figures:
        figure_3()
    if 4 in args.figures:
        figure_4()


if __name__ == "__main__":
    main()
