# ruff: noqa: E501
"""Experiment 674, step 2: thresholds, labels, known-answer scores, and the eight.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-*:ml-experiments/00508-*:ml-experiments/00674-* \
        OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00674-*/e00674_score.py {known|freeze|eight}

`known` scores the six leave-one-Primary-out; `freeze` writes thresholds from all six to
`thresholds_frozen.json` (run before `e00674_build.py --eight`); `eight` applies them to the eight.
"""

import json
import sys
from typing import Final

import e00674_features as feat  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

SEED: Final[int] = 674
N_RESAMPLES: Final[int] = 5000
MIN_CELL: Final[int] = 30
RATIO_EDGES: Final[tuple[float, float]] = (1.5, 3.0)
SAME_FARM: Final[dict[str, str]] = {
    "D5": "D11",
    "D11": "D5",
    "D9": "D14",
    "D14": "D9",
    "D10": "D16",
    "D16": "D10",
}
# alpha per stratum; each rule alone flags alpha / 2 of non-reverse training days.
ALPHAS: Final[dict[str, dict[str, float]]] = {
    "C": {"all": 0.05},
    "B": {"high": 0.05},
    "A": {"high": 0.05, "mixed": 0.025, "cloudy": 0.005},
    "A_flat": {"high": 0.05, "mixed": 0.05, "cloudy": 0.05},
    "A_steep": {"high": 0.05, "mixed": 0.0125, "cloudy": 0.0025},
}
POST_HOC: Final[tuple[str, ...]] = ("A_flat", "A_steep")
CLASSES: Final[tuple[str, ...]] = ("reverse", "no_reverse", "unknown", "not_scorable")


def prepare(
    *, table: pl.DataFrame, use_gate: bool = True, months: tuple[int, ...] = feat.SUMMER_MONTHS
) -> pl.DataFrame:
    """Add the stratum and ratio bin; keep days in `months` with a ratio and a stratum."""
    gate = pl.col("gate") if use_gate else pl.lit(True)
    return (
        table.filter(pl.col("month").is_in(months), pl.col("k_mid").is_not_null())
        .with_columns(
            stratum=pl.when((pl.col("k_mid") >= feat.SUNNY_K) & gate)
            .then(pl.lit("high"))
            .when(pl.col("k_mid") < feat.CLOUDY_K)
            .then(pl.lit("cloudy"))
            .otherwise(pl.lit("mixed")),
            rbin=pl.when(pl.col("r").is_null())
            .then(pl.lit("none"))
            .when(pl.col("r") >= RATIO_EDGES[1])
            .then(pl.lit("unknown"))
            .when(pl.col("r") >= RATIO_EDGES[0])
            .then(pl.lit("1.5-3"))
            .otherwise(pl.lit("<1.5")),
        )
        .filter(pl.col("rbin") != "none")
    )


def fit_thresholds(*, train: pl.DataFrame) -> dict:
    """Return thresholds[arm][stratum][rbin] = (tau_s, tau_w, n_days, fallback)."""
    base = train.filter(~pl.col("crossing"), pl.col("rbin").is_in(["<1.5", "1.5-3"]))
    out: dict = {}
    for arm, alphas in ALPHAS.items():
        out[arm] = {}
        for stratum, alpha in alphas.items():
            sub = base if stratum == "all" else base.filter(pl.col("stratum") == stratum)
            for rbin in ("<1.5", "1.5-3"):
                cell = sub.filter(pl.col("rbin") == rbin)
                fallback = 0
                if cell.height < MIN_CELL:
                    cell, fallback = sub, 1
                if cell.height < MIN_CELL:
                    cell, fallback = base, 2
                s = cell["s"].drop_nulls().to_numpy()
                w = cell["w"].drop_nulls().to_numpy()
                tau_s = float(np.quantile(s, alpha / 2)) if len(s) else -np.inf
                tau_w = float(np.quantile(w, 1 - alpha / 2)) if len(w) else np.inf
                out[arm].setdefault(stratum, {})[rbin] = (tau_s, tau_w, cell.height, fallback)
    return out


def label(*, days: pl.DataFrame, thresholds: dict) -> pl.DataFrame:
    """Add one class column per arm."""
    rows = days.select("s", "w", "stratum", "rbin").to_dicts()
    cols: dict[str, list[str]] = {arm: [] for arm in ALPHAS}
    for row in rows:
        for arm in ALPHAS:
            cols[arm].append(_label_day(row=row, cells=thresholds[arm], arm=arm))
    return days.with_columns(**{f"cls_{arm}": pl.Series(v) for arm, v in cols.items()})


def _label_day(*, row: dict, cells: dict, arm: str) -> str:
    if row["rbin"] == "unknown":
        return "unknown"
    s, w = row["s"], row["w"]
    if s is None and w is None:
        return "not_scorable"
    stratum = "all" if arm == "C" else row["stratum"]
    if stratum not in cells:  # arm B: days outside the PV-high stratum cannot be flagged
        return "no_reverse"
    tau_s, tau_w, _, _ = cells[stratum][row["rbin"]]
    flag = (s is not None and s < tau_s) or (w is not None and w > tau_w)
    return "reverse" if flag else "no_reverse"


def known_answer() -> pl.DataFrame:
    """Return the six's days with labels from leave-one-Primary-out thresholds."""
    six = prepare(table=pl.read_parquet(feat.WORK / "day_tables_six.parquet"))
    parts, fits = [], {}
    for site in sorted(SAME_FARM):
        train = six.filter(~pl.col("site").is_in([site, SAME_FARM[site]]))
        th = fit_thresholds(train=train)
        fits[site] = th
        parts.append(label(days=six.filter(pl.col("site") == site), thresholds=th))
    (feat.WORK / "thresholds_leave_one_out.json").write_text(json.dumps(fits, indent=1))
    return pl.concat(parts)


def block_counts(*, days: pl.DataFrame, arm: str) -> tuple[np.ndarray, list[str]]:
    """Return per-month-block counts: crossing, hit, noncross_known, false, unknown, notscor, n."""
    cls = days[f"cls_{arm}"]
    cross = days["crossing"]
    frame = days.select(
        block=pl.col("site") + "_" + pl.col("block").cast(pl.Utf8),
        crossing=cross.cast(pl.Float64),
        hit=(cross & (cls == "reverse")).cast(pl.Float64),
        noncross_known=((~cross) & cls.is_in(["reverse", "no_reverse"])).cast(pl.Float64),
        false=((~cross) & (cls == "reverse")).cast(pl.Float64),
        labelled_cross=(cross & cls.is_in(["reverse", "no_reverse"])).cast(pl.Float64),
        unknown=(cls == "unknown").cast(pl.Float64),
        not_scorable=(cls == "not_scorable").cast(pl.Float64),
        n=pl.lit(1.0),
    )
    g = frame.group_by("block").sum()
    names = [
        "crossing",
        "hit",
        "noncross_known",
        "false",
        "labelled_cross",
        "unknown",
        "not_scorable",
        "n",
    ]
    return g.select(names).to_numpy(), names


def metrics(*, days: pl.DataFrame, arm: str, rng: np.random.Generator) -> dict | None:
    """Return recall, false-reverse rate, unknown and not-scorable shares with block bootstrap."""
    if days.height == 0:
        return None
    counts, names = block_counts(days=days, arm=arm)
    ix = {n: i for i, n in enumerate(names)}
    nb = len(counts)
    w = rng.multinomial(nb, np.full(nb, 1 / nb), size=N_RESAMPLES)
    tot = w @ counts
    point = counts.sum(axis=0)

    def ratio(num: str, den: str) -> dict:
        d0 = point[ix[den]]
        est = float(point[ix[num]] / d0) if d0 > 0 else None
        with np.errstate(all="ignore"):
            r = tot[:, ix[num]] / tot[:, ix[den]]
        r = r[np.isfinite(r)]
        lo, hi = (
            (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if len(r) else (None, None)
        )
        return {"est": est, "lo": lo, "hi": hi, "den": float(d0)}

    return {
        "recall": ratio("hit", "crossing"),
        "recall_labelled": ratio("hit", "labelled_cross"),
        "false_reverse": ratio("false", "noncross_known"),
        "unknown": ratio("unknown", "n"),
        "not_scorable": ratio("not_scorable", "n"),
        "n_days": float(point[ix["n"]]),
        "n_blocks": nb,
    }


def score_known() -> None:
    """Write the known-answer metric table."""
    days = known_answer()
    days.write_parquet(feat.WORK / "known_answer_days.parquet")
    rng = np.random.default_rng(SEED)
    rows = []
    subsets = {
        "all": pl.lit(True),
        "high": pl.col("stratum") == "high",
        "mixed": pl.col("stratum") == "mixed",
        "cloudy": pl.col("stratum") == "cloudy",
        "mixed_cloudy": pl.col("stratum") != "high",
        "low_demand": pl.col("low_demand"),
        "other_demand": ~pl.col("low_demand"),
    }
    for arm in ALPHAS:
        for rbin in ("<1.5", "1.5-3", "pooled_below_3", "all_with_unknown"):
            base = {
                "<1.5": days.filter(pl.col("rbin") == "<1.5"),
                "1.5-3": days.filter(pl.col("rbin") == "1.5-3"),
                "pooled_below_3": days.filter(pl.col("rbin").is_in(["<1.5", "1.5-3"])),
                "all_with_unknown": days,
            }[rbin]
            for name, expr in subsets.items():
                for level in ("all", 0.0, 1.0, 2.0, 3.0):
                    sub = base.filter(expr)
                    if level != "all":
                        sub = sub.filter(pl.col("level") == level)
                    m = metrics(days=sub, arm=arm, rng=rng)
                    if m is not None:
                        rows.append(
                            {
                                "arm": arm,
                                "rbin": rbin,
                                "subset": name,
                                "level": str(level),
                                **{
                                    f"{k}_{f}": v[f]
                                    for k, v in m.items()
                                    if isinstance(v, dict)
                                    for f in ("est", "lo", "hi", "den")
                                },
                                "n_days": m["n_days"],
                                "n_blocks": m["n_blocks"],
                            }
                        )
    pl.DataFrame(rows).write_parquet(feat.WORK / "known_answer_metrics.parquet")
    print(
        pl.DataFrame(rows)
        .filter(pl.col("subset") == "all", pl.col("level") == "all")
        .select(
            "arm",
            "rbin",
            "recall_est",
            "recall_lo",
            "recall_hi",
            "false_reverse_est",
            "false_reverse_lo",
            "false_reverse_hi",
            "unknown_est",
            "not_scorable_est",
            "n_days",
        )
    )


def freeze() -> None:
    """Fit thresholds on all six and write them before the eight are built."""
    six = prepare(table=pl.read_parquet(feat.WORK / "day_tables_six.parquet"))
    (feat.WORK / "thresholds_frozen.json").write_text(
        json.dumps(fit_thresholds(train=six), indent=1)
    )
    print("frozen")


def eight() -> None:
    """Apply the frozen thresholds to the eight (April to September; all months for D12 winter)."""
    th = json.loads((feat.WORK / "thresholds_frozen.json").read_text())
    raw = pl.read_parquet(feat.WORK / "day_tables_eight.parquet")
    for use_gate, tag in ((True, "gate"), (False, "nogate")):
        days = label(days=prepare(table=raw, use_gate=use_gate), thresholds=th)
        days.write_parquet(feat.WORK / f"eight_days_{tag}.parquet")
    # All months, for D12's winter dips: thresholds are April to September (outside their domain).
    winter = prepare(table=raw, months=tuple(range(1, 13)))
    label(days=winter, thresholds=th).write_parquet(feat.WORK / "eight_days_allmonths.parquet")
    print("eight done")


if __name__ == "__main__":
    {"known": score_known, "freeze": freeze, "eight": eight}[sys.argv[1]]()
