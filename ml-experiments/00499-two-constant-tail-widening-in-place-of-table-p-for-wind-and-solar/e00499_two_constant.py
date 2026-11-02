"""Experiment 499: two constants per farm in place of table P, on experiment 461's wind and solar.

Reads experiment 461's stored quantile rows in MW (wind `member_plain`, solar `a2_member`, both at
the p99.9 capacity, seeds already averaged) and re-scores six rules, each fitted on earlier months
only: `raw`, `w1.15` (symmetric widening), `P` (table P, copied from experiment 450's
`fit_factors` and experiment 461's `apply_table_p`), `d` (two constants per family), `e` (two
constants per farm), and `f` (two constants per farm and lead-band group). Rules `w1.15`, `P`, `d`,
`e`, and `f` use experiment 451's clip: the upper bound is the larger of the capacity in use and the
series' largest actual in the earlier months, the lower bound of every tail is 0. `raw` is
unclipped, as experiment 461 scores it. The first month of each family is only ever fitted on.

The script prints the minimum detectable effect before it computes any contrast, then writes
`work/rows_scored.parquet`, `work/contrasts.csv`, `work/scores.csv`, `work/constants.csv`, and
`work/mde.csv`.

```bash
export DATA_PATH_INTERNAL=<the private data folder>
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00499-*/e00499_two_constant.py
```
"""

import os
from pathlib import Path
from statistics import NormalDist
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00499-two-constant-tail-widening-in-place-of-table-p-for-wind-and-solar"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
ROWS_461: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00461-wind-and-solar-retrain-with-p99-9-effective-capacity" / "work"
)
FAMILIES: Final[dict[str, str]] = {"wind": "member_plain", "solar": "a2_member"}
LAST_MONTH: Final[str] = "2026-06"  # July 2026 stays unused
LEVELS: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVEL_TAU: Final[dict[str, float]] = dict(
    zip(LEVELS, (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99), strict=True)
)
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
LOWER: Final[tuple[str, ...]] = ("q01", "q05")
UPPER: Final[tuple[str, ...]] = ("q95", "q99")
GRID: Final[np.ndarray] = np.round(np.arange(0.50, 1.801, 0.01), 2)
WIDENING: Final[float] = 1.15
MIN_FIT_ROWS: Final[int] = 200
RULES: Final[tuple[str, ...]] = ("raw", "w1.15", "P", "d", "e", "f")
CANDIDATES: Final[tuple[str, ...]] = ("d", "e", "f")
GROUP_OF_BAND: Final[dict[str, str]] = {
    "12-36h": "first", "36-72h": "36-72h", "days4-5": "72-240h", "days6-10": "72-240h",
}  # fmt: skip
CRITERIA: Final[tuple[str, ...]] = (
    "cov_q01", "cov_q05", "cov_q95", "cov_q99", "share_above_q99", "pinball_tails",
)  # fmt: skip
LIMITS: Final[dict[str, float]] = {
    "cov_q01": 0.01, "cov_q05": 0.01, "cov_q95": 0.01, "cov_q99": 0.01,
    "share_above_q99": 0.002, "pinball_tails": 0.05,
}  # fmt: skip
BONFERRONI_COUNT: Final[int] = 36  # 3 rules x 2 families x 6 criteria
RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 499
# "block" is a run for wind and an initialisation week for solar; "init" (a run for both) is the
# sensitivity check, selected by setting BLOCK_COLUMN=init, which also writes `*_init_blocks` files.
BLOCK: Final[str] = os.environ.get("BLOCK_COLUMN", "block")
SUFFIX: Final[str] = "" if BLOCK == "block" else "_init_blocks"
KEYS: Final[list[str]] = ["family", "site", "init", "lead_hours"]


def load(*, family: str) -> pl.DataFrame:
    """Return the stored rows of `family` up to LAST_MONTH, with the lead-band group."""
    frame = pl.read_parquet(ROWS_461 / f"rows_{FAMILIES[family]}_p999.parquet")
    return (
        frame.filter(pl.col("month") <= LAST_MONTH)
        .with_columns(
            family=pl.lit(family),
            group=pl.col("lead_band").replace_strict(GROUP_OF_BAND),
        )
        .select("family", "site", "month", "init", "lead_hours", "lead_band", "group", "cap_mw",
                "actual", *LEVELS)
        .sort(KEYS)
    )  # fmt: skip


def pinball(*, y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Return the per-row pinball loss of `q` at level `tau`."""
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)


def fit_table_p(*, fit: pl.DataFrame) -> dict[tuple[str, str, str], float]:
    """Return table P's factor per (farm, lead band, tail), as experiment 450's `fit_factors`."""
    out: dict[tuple[str, str, str], float] = {}
    for (site, band), part in fit.group_by("site", "lead_band"):
        for name, tau in TAILS.items():
            spread = (pl.col(name) - pl.col("q50")).abs()
            ratio = (
                (pl.col("actual") - pl.col("q50")) / spread
                if name in UPPER
                else (pl.col("q50") - pl.col("actual")) / spread
            )
            level = tau if name in UPPER else 1.0 - tau
            use = part.filter(spread > 1e-6)
            if use.height >= MIN_FIT_ROWS:
                out[site, band, name] = float(use.select(ratio.quantile(level)).item())
    return out


def fit_constant(*, fit: pl.DataFrame, names: tuple[str, ...]) -> float:
    """Return the grid factor minimising the pinball loss of `names` (in fractions of capacity)."""
    cap = fit["cap_mw"].to_numpy()
    y = fit["actual"].to_numpy() / cap
    q50 = fit["q50"].to_numpy() / cap
    loss = np.zeros(len(GRID))
    for name in names:
        d = fit[name].to_numpy() / cap - q50
        for i, f in enumerate(GRID):
            loss[i] += pinball(y=y, q=q50 + f * d, tau=TAILS[name]).mean()
    return float(GRID[int(np.argmin(loss))])


def fit_pair(*, fit: pl.DataFrame) -> tuple[float, float]:
    """Return the (lower, upper) constants fitted on `fit`."""
    return fit_constant(fit=fit, names=LOWER), fit_constant(fit=fit, names=UPPER)


def fit_constants(*, fit: pl.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    """Return the (lower, upper) pair per (rule, key): d by family, e by farm, f by farm and group.

    A farm-and-group cell with fewer than MIN_FIT_ROWS fitting rows takes its farm's pair.
    """
    out: dict[tuple[str, str], tuple[float, float]] = {("d", "all"): fit_pair(fit=fit)}
    for (site,), part in fit.group_by("site"):
        out["e", site] = fit_pair(fit=part)
        for group in sorted(set(GROUP_OF_BAND.values())):
            cell = part.filter(pl.col("group") == group)
            out["f", f"{site}|{group}"] = (
                fit_pair(fit=cell) if cell.height >= MIN_FIT_ROWS else out["e", site]
            )
    return out


def scale_tails(*, held: pl.DataFrame, lower: np.ndarray, upper: np.ndarray) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its row's lower or upper factor."""
    return held.with_columns(
        *[(pl.col("q50") + pl.Series(lower) * (pl.col(q) - pl.col("q50"))).alias(q) for q in LOWER],
        *[(pl.col("q50") + pl.Series(upper) * (pl.col(q) - pl.col("q50"))).alias(q) for q in UPPER],
    )


def clip_rows(*, rows: pl.DataFrame, upper: dict[str, float]) -> pl.DataFrame:
    """Apply experiment 451's clip: tails at least 0, every level at most the upper bound."""
    bound = pl.col("site").replace_strict(upper, return_dtype=pl.Float64)
    return rows.with_columns(
        *[pl.col(q).clip(lower_bound=0.0) for q in TAILS],
    ).with_columns(*[pl.col(q).clip(upper_bound=bound) for q in LEVELS])


def apply_rules(
    *, held: pl.DataFrame, fit: pl.DataFrame
) -> tuple[dict[str, pl.DataFrame], list[dict]]:
    """Return the held rows under each rule, and the fitted constants and fallback counts."""
    cap_now = {
        r["site"]: r["c"]
        for r in held.group_by("site").agg(pl.col("cap_mw").max().alias("c")).iter_rows(named=True)
    }
    earlier_max = {
        r["site"]: r["m"]
        for r in fit.group_by("site").agg(pl.col("actual").max().alias("m")).iter_rows(named=True)
    }
    upper = {s: max(cap_now[s], earlier_max.get(s, 0.0)) for s in cap_now}
    out: dict[str, pl.DataFrame] = {"raw": held}
    ones = np.ones(held.height)
    out["w1.15"] = clip_rows(
        rows=scale_tails(held=held, lower=ones * WIDENING, upper=ones * WIDENING), upper=upper
    )
    # Table P.
    factors = fit_table_p(fit=fit)
    table = held
    for name in TAILS:
        lookup = pl.DataFrame(
            [(s, b, f) for (s, b, n), f in factors.items() if n == name],
            schema={"site": pl.String, "lead_band": pl.String, "factor": pl.Float64},
            orient="row",
        )
        table = (
            table.join(lookup, on=["site", "lead_band"], how="left", maintain_order="left")
            .with_columns(
                (
                    pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50"))
                ).alias(name)
            )
            .drop("factor")
        )
    out["P"] = clip_rows(rows=table, upper=upper)
    # Two-constant rules.
    constants = fit_constants(fit=fit)
    site = held["site"].to_list()
    group = held["group"].to_list()
    keyed = {
        "d": ["all"] * held.height,
        "e": site,
        "f": [f"{s}|{g}" for s, g in zip(site, group, strict=True)],
    }
    records: list[dict] = []
    for rule in CANDIDATES:
        pairs = np.array([constants[rule, k] for k in keyed[rule]])
        out[rule] = clip_rows(
            rows=scale_tails(held=held, lower=pairs[:, 0], upper=pairs[:, 1]), upper=upper
        )
    for (rule, key), (lo, up) in constants.items():
        records.append({"rule": rule, "key": key, "lower": lo, "upper": up})
    return out, records


def row_scores(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return per-row hit indicators and pinball losses (MW) for the tails and all 11 levels."""
    y = rows["actual"].to_numpy()
    q = np.sort(rows.select(list(LEVELS)).to_numpy(), axis=1)
    cols: dict[str, np.ndarray] = {}
    tail_pin = np.zeros(rows.height)
    for name, tau in TAILS.items():
        i = LEVELS.index(name)
        cols[f"cov_{name}"] = (y <= q[:, i]).astype(np.float64)
        p = pinball(y=y, q=q[:, i], tau=tau)
        cols[f"pin_{name}"] = p
        tail_pin += p / len(TAILS)
    cols["share_above_q99"] = 1.0 - cols["cov_q99"]
    cols["pinball_tails"] = tail_pin
    levels_pin = np.mean(
        [pinball(y=y, q=q[:, i], tau=LEVEL_TAU[n]) for i, n in enumerate(LEVELS)], axis=0
    )
    cols["quantile_score_mw"] = levels_pin
    return pl.DataFrame(cols)


def score_all(*, strict: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit and apply every rule on every scored month; return scored rows and the constants.

    With `strict`, a fitting row must have its valid time (`init` plus `lead_hours`) before the
    first initialisation time of the scored month, so no actual that was unknown at the first
    scored run is fitted. Without it, fitting rows are all earlier months, as in experiment 461.
    """
    parts: list[pl.DataFrame] = []
    constants: list[dict] = []
    for family in FAMILIES:
        frame = load(family=family)
        months = sorted(frame["month"].unique())
        for i, month in enumerate(months[1:], start=1):
            fit = frame.filter(pl.col("month").is_in(months[:i]))
            held = frame.filter(pl.col("month") == month)
            if strict:
                fit = fit.filter(
                    pl.col("init") + pl.duration(hours=pl.col("lead_hours")) < held["init"].min()
                )
            by_rule, records = apply_rules(held=held, fit=fit)
            constants += [{"family": family, "month": month, **r} for r in records]
            n_cells = fit.group_by("site", "lead_band").len()
            constants.append(
                {
                    "family": family,
                    "month": month,
                    "rule": "fit_rows",
                    "key": "n_fit_rows",
                    "lower": float(fit.height),
                    "upper": float(
                        n_cells.filter(pl.col("len") < MIN_FIT_ROWS).height
                    ),  # table P cells below the minimum (factor 1)
                }
            )
            for rule, rows in by_rule.items():
                assert rows.select(KEYS).equals(held.select(KEYS)), "row mismatch"
                parts.append(
                    pl.concat(
                        [
                            rows.select("family", "site", "month", "init", "lead_hours",
                                        "lead_band", "group", "cap_mw", "actual").with_columns(
                                rule=pl.lit(rule)),
                            row_scores(rows=rows),
                        ],
                        how="horizontal",
                    )
                )  # fmt: skip
            print("scored", family, month, flush=True)
    scored = pl.concat(parts).with_columns(
        block=pl.when(pl.col("family") == "solar")
        .then(pl.col("init").dt.truncate("1w"))
        .otherwise(pl.col("init"))
    )
    return scored, pl.DataFrame(constants)


def run_blocks(
    *, rows: pl.DataFrame, blocks: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return counts and criterion sums per block id in `blocks` (zero for an absent block)."""
    grouped = rows.group_by(BLOCK).agg(
        pl.len().alias("n"), *[pl.col(c).sum() for c in CRITERIA]
    )  # fmt: skip
    index = {t: i for i, t in enumerate(blocks)}
    n = np.zeros(len(blocks))
    sums = {c: np.zeros(len(blocks)) for c in CRITERIA}
    for r in grouped.iter_rows(named=True):
        i = index[r[BLOCK]]
        n[i] = r["n"]
        for c in CRITERIA:
            sums[c][i] = r[c]
    return n, sums


def paired_draws(
    *, a: pl.DataFrame, b: pl.DataFrame, rng: np.random.Generator
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return point differences (a minus b) and their run-block bootstrap draws per criterion."""
    blocks = np.array(sorted(a[BLOCK].unique().to_list()))
    n_a, s_a = run_blocks(rows=a, blocks=blocks)
    n_b, s_b = run_blocks(rows=b, blocks=blocks)
    assert np.array_equal(n_a, n_b)
    pick = rng.integers(0, len(blocks), size=(RESAMPLES, len(blocks)))
    point: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    n_all = n_a.sum()
    n_pick = n_a[pick].sum(axis=1)
    for c in CRITERIA:
        if c == "pinball_tails":
            point[c] = s_a[c].sum() / s_b[c].sum() - 1.0
            draws[c] = s_a[c][pick].sum(axis=1) / s_b[c][pick].sum(axis=1) - 1.0
        else:
            point[c] = (s_a[c].sum() - s_b[c].sum()) / n_all
            draws[c] = (s_a[c][pick].sum(axis=1) - s_b[c][pick].sum(axis=1)) / n_pick
    return point, draws


def minimum_detectable_effect(*, table_p_rows: pl.DataFrame, rng: np.random.Generator) -> dict:
    """Return the minimum detectable effect of each criterion from table P's own run-block spread.

    The standard error of one rule's criterion is taken from the run-block bootstrap of table P's
    rows alone; the paired difference has at most `sqrt(2)` times that, so this is an upper bound.
    """
    blocks = np.array(sorted(table_p_rows[BLOCK].unique().to_list()))
    n, sums = run_blocks(rows=table_p_rows, blocks=blocks)
    pick = rng.integers(0, len(blocks), size=(RESAMPLES, len(blocks)))
    z_alpha = NormalDist().inv_cdf(1.0 - 0.05 / BONFERRONI_COUNT / 2.0)
    z_power = NormalDist().inv_cdf(0.8)
    out: dict[str, float] = {}
    for c in CRITERIA:
        if c == "pinball_tails":
            mean_loss = sums[c].sum() / n.sum()
            se = float(np.std(sums[c][pick].sum(axis=1) / n[pick].sum(axis=1))) / mean_loss
        else:
            se = float(np.std(sums[c][pick].sum(axis=1) / n[pick].sum(axis=1)))
        out[c] = (z_alpha + z_power) * np.sqrt(2.0) * se
    return out


def verdict(*, criterion: str, low: float, high: float) -> str:
    """Return 'equivalent', 'not equivalent', or 'inconclusive' against the limit."""
    limit = LIMITS[criterion]
    if criterion in ("share_above_q99", "pinball_tails"):
        if high <= limit:
            return "equivalent"
        return "not equivalent" if low > limit else "inconclusive"
    if low >= -limit and high <= limit:
        return "equivalent"
    return "not equivalent" if (low > limit or high < -limit) else "inconclusive"


def absolute_scores(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Return coverage, pinball, and quantile score per family, rule, farm, and slice."""
    slices = {
        "all": scored,
        "first_band": scored.filter(pl.col("lead_band") == "12-36h"),
    }
    metrics = [pl.col(c).mean() for c in CRITERIA[:4] + CRITERIA[4:5]]
    out = [
        frame.group_by(by)
        .agg(
            pl.len().alias("n"),
            *metrics,
            pl.col("pinball_tails").mean().alias("pinball_tails_mw"),
            pl.col("quantile_score_mw").mean(),
        )
        .with_columns(slice=pl.lit(name), by=pl.lit("+".join(by[2:]) or "family"))
        for name, frame in slices.items()
        for by in (["family", "rule"], ["family", "rule", "site"])
    ]
    return pl.concat(out, how="diagonal")


def main() -> None:
    """Score every rule, print the minimum detectable effect, then the contrasts."""
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scored, constants = score_all()
    loose = score_all(strict=False)[0].filter(pl.col("rule") == "P")
    print("GATE: table P share above q99 with experiment 461's fit window (wind 0.88% expected):")
    print(loose.group_by("family").agg(pl.col("share_above_q99").mean()), flush=True)
    scored.write_parquet(WORK / "rows_scored.parquet")
    constants.write_csv(WORK / "constants.csv")
    level = 100.0 * (1.0 - 0.05 / BONFERRONI_COUNT)
    tail = (100.0 - level) / 2.0
    # The minimum detectable effect comes first.
    mde_rows = []
    for family in FAMILIES:
        base = scored.filter((pl.col("family") == family) & (pl.col("rule") == "P"))
        for slice_name, frame in (
            ("all", base), ("first_band", base.filter(pl.col("lead_band") == "12-36h")),
        ):  # fmt: skip
            for site in [None, *sorted(frame["site"].unique())]:
                part = frame if site is None else frame.filter(pl.col("site") == site)
                mde = minimum_detectable_effect(table_p_rows=part, rng=rng)
                mde_rows += [
                    {
                        "family": family,
                        "slice": slice_name,
                        "site": site or "pooled",
                        "criterion": c,
                        "mde": v,
                        "limit": LIMITS[c],
                        "mde_above_limit": v > LIMITS[c],
                    }
                    for c, v in mde.items()
                ]
    mde_table = pl.DataFrame(mde_rows)
    mde_table.write_csv(WORK / f"mde{SUFFIX}.csv")
    print("MINIMUM DETECTABLE EFFECT (printed before any contrast)", flush=True)
    print(mde_table.filter(pl.col("site") == "pooled"), flush=True)
    # Contrasts.
    contrasts: list[dict] = []
    for family in FAMILIES:
        fam = scored.filter(pl.col("family") == family)
        for slice_name, frame in (
            ("all", fam), ("first_band", fam.filter(pl.col("lead_band") == "12-36h")),
        ):  # fmt: skip
            for site in [None, *sorted(frame["site"].unique())]:
                part = frame if site is None else frame.filter(pl.col("site") == site)
                base = part.filter(pl.col("rule") == "P")
                for rule in ("raw", "w1.15", *CANDIDATES):
                    point, draws = paired_draws(
                        a=part.filter(pl.col("rule") == rule), b=base, rng=rng
                    )
                    for c in CRITERIA:
                        low = float(np.percentile(draws[c], tail))
                        high = float(np.percentile(draws[c], 100.0 - tail))
                        contrasts.append(
                            {
                                "family": family, "slice": slice_name, "site": site or "pooled",
                                "rule": rule, "criterion": c, "diff": float(point[c]),
                                "low": low, "high": high, "limit": LIMITS[c],
                                "verdict": verdict(criterion=c, low=low, high=high),
                                "counted": site is None and slice_name == "all"
                                and rule in CANDIDATES,
                            }
                        )  # fmt: skip
        print("contrasts", family, flush=True)
    pl.DataFrame(contrasts).write_csv(WORK / f"contrasts{SUFFIX}.csv")
    absolute_scores(scored=scored).write_csv(WORK / "scores.csv")


if __name__ == "__main__":
    main()
