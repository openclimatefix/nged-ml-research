"""Score experiment 461: old versus p99.9 effective capacity, in MW, with table P and tails.

Every arm's forecasts are converted to MW (quantiles and actuals times the capacity in use: the
stored old capacity for the old arm, the fit's p99.9 capacity for the new arm), so a contrast does
not depend on the divisor. Table P (experiment 450's factors, per series and lead band, fitted on
earlier scored months only) is applied in MW with experiment 451's clip as coded there: the upper
bound is the larger of the capacity in use and the series' largest actual in the earlier scored
months; the lower bound is 0. Rows are true days 1 to 10 (lead 24 to 240 hours); wind uses the
rows kept by experiment 210's mask. Seeds are averaged quantile by quantile and sorted. Output:
`work/cells.parquet` (one row per technology, arm, series, month, and variant) and
`work/report.txt`.

    source ml-experiments/00429-*/env.sh   # then add the 00450, 00461, 00430, 00310 folders
    uv run --no-sync python ml-experiments/00461-*/e00461_score.py
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00450_score as score450  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import e00453_data as data453  # ty: ignore[unresolved-import]
import e00461_common as common
import e00461_solar_fit as solar_fit
import e00461_wind_fit as wind_fit
import numpy as np
import polars as pl

LEVELS: Final = step1.LEVELS
TAILS: Final = step1.TAILS
LEAD_MIN_H, LEAD_MAX_H = 24.0, 240.0
DRAWS: Final[int] = 1000
WORK_DIR: Final[Path] = common.DATA_DIR / "work"
SOLAR_OLD: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR / "00430-unified-forecaster-solar-and-pretraining-ablations"
    / "predictions"
)  # fmt: skip
SOLAR_ARMS: Final[tuple[str, ...]] = ("a2_member", "g0_onepass", "a5_plain")
MARGIN_WIND_PTS: Final[float] = 0.10
MARGIN_SOLAR_PCT: Final[float] = 1.0
LIMIT: Final[float] = 0.015


def _finish(*, frame: pl.DataFrame, tech: str, arm: str, rule: str) -> pl.DataFrame:
    """Sort the quantiles, convert to MW with `cap_mw`, and filter to the scored rows."""
    q = np.sort(frame.select(list(LEVELS)).to_numpy(), axis=1)
    cap = frame["cap_mw"].to_numpy()
    frame = frame.with_columns(*[pl.Series(c, q[:, i] * cap) for i, c in enumerate(LEVELS)])
    return (
        frame.with_columns(actual=pl.col("actual") * pl.col("cap_mw"))
        .filter(pl.col("lead_hours").is_between(LEAD_MIN_H, LEAD_MAX_H))
        .with_columns(lead_band=step1.lead_band_expr(), tech=pl.lit(tech), arm=pl.lit(arm),
                      rule=pl.lit(rule))
        .filter(pl.col("lead_band").is_not_null() & pl.col("actual").is_finite())
        .select("tech", "arm", "rule", "site", "month", "init", "lead_hours", "lead_band", "cap_mw",
                "actual", *LEVELS)
    )  # fmt: skip


def wind_frame(*, rule: str) -> pl.DataFrame:
    """Return `member_plain`'s kept scored rows in MW under the old or the p99.9 capacity."""
    old = c230.site_table()
    old_cap = {r["site"]: float(r["effective_capacity_mw"]) for r in old.iter_rows(named=True)}
    if rule == "old":
        tables = score450.load_seed_tables(name="base", seeds=wind_fit.SEEDS)
        cap = old_cap
    else:
        tables = [
            np.load(wind_fit.PARTS_DIR / f"member_plain_p999_h{s}.npy") for s in wind_fit.SEEDS
        ]
        table = json.loads((common.DATA_DIR / "wind_capacity.json").read_text())
        cap = {farm: float(row["new_mw"]) for farm, row in table.items()}
    q = np.mean(np.stack(tables), axis=0)
    frame = (
        data347.scored_frame()
        .select(
            "site",
            "month",
            "masked",
            "lead_hours",
            "actual_norm",
            init=pl.col("power_fcst_init_time"),
        )
        .with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])
        .filter(~pl.col("masked") & pl.col("actual_norm").is_finite())
    )
    # `actual_norm` is power over the old capacity; the new arm's actual in new units is that
    # divided by new over old, so the common factor `cap_mw` below returns MW.
    to_new_units = {f: cap[f] / old_cap[f] for f in cap}
    frame = frame.with_columns(
        cap_mw=pl.col("site").replace_strict(cap, return_dtype=pl.Float64),
        actual=pl.col("actual_norm")
        / pl.col("site").replace_strict(to_new_units, return_dtype=pl.Float64),
        site=pl.col("site").replace_strict(common.WIND_LABEL),
    )
    return _finish(frame=frame, tech="wind", arm="member_plain", rule=rule)


def solar_frame(*, arm: str, rule: str) -> pl.DataFrame:
    """Return a solar arm's scored rows in MW under the old or the p99.9 capacity."""
    keys = ["time", "init_time", "time_series_id", "lead_hours", "actual"]
    if rule == "old":
        labels = data453.labels().filter(pl.col("tech") == "solar")
        cap = {r["site"].removeprefix("Solar "): r["cap"] for r in labels.iter_rows(named=True)}
        files = sorted((SOLAR_OLD / arm).glob("s*_*.parquet"))
        parts = [
            pl.read_parquet(f).with_columns(
                cap_mw=pl.col("time_series_id").replace_strict(cap, return_dtype=pl.Float64),
                refit=pl.lit(f.stem.split("_")[1]),
            )
            for f in files
        ]
    else:
        table = json.loads((common.DATA_DIR / solar_fit.CAPACITY_FILE).read_text())
        files = sorted((solar_fit.PRED_DIR / "p999" / arm).glob("s*_*.parquet"))
        parts = []
        for f in files:
            refit = f.stem.split("_")[1]
            cap = {s: float(v["new_mw"]) for s, v in table[refit].items()}
            parts.append(
                pl.read_parquet(f).with_columns(
                    cap_mw=pl.col("time_series_id").replace_strict(cap, return_dtype=pl.Float64),
                    refit=pl.lit(refit),
                )
            )
    frame = pl.concat(parts)
    mean = (
        frame.group_by([*keys, "refit", "cap_mw"])
        .agg(*[pl.col(c).mean() for c in LEVELS])
        .sort(keys)
    )
    mean = mean.with_columns(
        init=pl.col("init_time"),
        site="Solar " + pl.col("time_series_id"),
        month=pl.col("time").dt.strftime("%Y-%m"),
    )
    return _finish(frame=mean, tech="solar", arm=arm, rule=rule)


def apply_table_p(*, rows: pl.DataFrame, factors: dict, upper: dict[str, float]) -> pl.DataFrame:
    """Scale each tail quantile's distance from the median by its factor and clip in MW."""
    out = rows
    for name in TAILS:
        lookup = pl.DataFrame(
            [(s, b, f) for (s, b, n), f in factors.items() if n == name],
            schema={"site": pl.String, "lead_band": pl.String, "factor": pl.Float64},
            orient="row",
        )
        out = (
            out.join(lookup, on=["site", "lead_band"], how="left")
            .with_columns(
                (pl.col("q50") + pl.col("factor").fill_null(1.0) * (pl.col(name) - pl.col("q50")))
                .clip(0.0, None)
                .alias(name)
            )
            .drop("factor")
        )
    upper_expr = pl.col("site").replace_strict(upper, return_dtype=pl.Float64)
    return out.with_columns(*[pl.col(c).clip(upper_bound=upper_expr) for c in LEVELS])


def cells(*, frame: pl.DataFrame) -> list[dict]:
    """Return one dict per (series, scored month, variant) for one arm and rule."""
    out = []
    months = sorted(frame["month"].unique())
    tech, arm, rule = (frame[c][0] for c in ("tech", "arm", "rule"))
    for i, month in enumerate(months[1:], start=1):
        earlier = frame.filter(pl.col("month").is_in(months[:i]))
        held = frame.filter(pl.col("month") == month)
        factors = step1.fit_factors(fit=earlier)
        cap_now = {r["site"]: float(r["c"]) for r in held.group_by("site").agg(
            pl.col("cap_mw").max().alias("c")).iter_rows(named=True)}  # fmt: skip
        earlier_max = {r["site"]: float(r["m"]) for r in earlier.group_by("site").agg(
            pl.col("actual").max().alias("m")).iter_rows(named=True)}  # fmt: skip
        upper = {s: max(cap_now[s], earlier_max.get(s, 0.0)) for s in cap_now}
        for variant, rows in (
            ("raw", held),
            ("P", apply_table_p(rows=held, factors=factors, upper=upper)),
        ):
            for site, part in rows.group_by("site", maintain_order=True):
                q = part.select(list(LEVELS)).to_numpy()
                a = part["actual"].to_numpy()
                score = c230.quantile_score(quantiles=q, actual=np.clip(a, 0.0, None))
                row = {
                    "tech": tech, "arm": arm, "rule": rule, "site": site[0], "month": month,
                    "variant": variant, "n": len(a), "score_mw_sum": float(score.sum()),
                    "cap_mw": float(part["cap_mw"].to_numpy().mean()),
                    "mean_q50_minus_mean_actual": float(
                        (q[:, LEVELS.index("q50")] - a).sum()),
                    "n_gt_1": int((a > part["cap_mw"].to_numpy()).sum()),
                }  # fmt: skip
                for name, tau in TAILS.items():
                    qi = q[:, LEVELS.index(name)]
                    row[f"hit_{name}"] = int((a <= qi).sum())
                    d = a - qi
                    row[f"pin_{name}"] = float(np.where(d >= 0, tau * d, (tau - 1) * d).sum())
                out.append(row)
    return out


def bootstrap(*, table: pl.DataFrame, sites: list[str], rng: np.random.Generator) -> np.ndarray:
    """Return DRAWS month-block draws of (new minus old) pooled mean score in MW over `sites`."""
    wide = (
        table.filter(pl.col("site").is_in(sites) & (pl.col("variant") == "P"))
        .group_by("rule", "month")
        .agg(pl.col("score_mw_sum").sum(), pl.col("n").sum())
        .sort("month")
    )
    months = sorted(wide["month"].unique())
    cell = {
        (r["rule"], r["month"]): (r["score_mw_sum"], r["n"]) for r in wide.iter_rows(named=True)
    }
    draws = np.empty(DRAWS)
    for d in range(DRAWS):
        pick = rng.integers(0, len(months), size=len(months))
        mean = {}
        for rule in ("old", "p999"):
            s = sum(cell[rule, months[i]][0] for i in pick)
            n = sum(cell[rule, months[i]][1] for i in pick)
            mean[rule] = s / n
        draws[d] = mean["p999"] - mean["old"]
    return draws


def seed_spread() -> dict[str, float | list[float]]:
    """Return single-seed pooled raw scores (MW) and mean pair gaps (wind old/new, solar old)."""
    out: dict[str, float | list[float]] = {}
    full = data347.scored_frame()
    old_cap = {
        r["site"]: float(r["effective_capacity_mw"])
        for r in c230.site_table().iter_rows(named=True)
    }
    new_cap = {
        f: float(v["new_mw"])
        for f, v in json.loads((common.DATA_DIR / "wind_capacity.json").read_text()).items()
    }
    sel = (
        (
            ~full["masked"]
            & full["actual_norm"].is_finite()
            & full["lead_hours"].is_between(LEAD_MIN_H, LEAD_MAX_H)
        )
        .to_numpy()
    )  # fmt: skip
    actual_mw = (
        np.clip(full["actual_norm"].to_numpy()[sel], 0.0, None)
        * (full["site"].replace_strict(old_cap, return_dtype=pl.Float64).to_numpy()[sel])
    )
    for tag, tables in (
        ("old", score450.load_seed_tables(name="base", seeds=wind_fit.SEEDS)),
        (
            "new",
            [np.load(wind_fit.PARTS_DIR / f"member_plain_p999_h{s}.npy") for s in wind_fit.SEEDS],
        ),
    ):
        cap = full["site"].replace_strict(
            old_cap if tag == "old" else new_cap, return_dtype=pl.Float64
        )
        cap_sel = cap.to_numpy()[sel]
        scores = [
            float(
                np.mean(
                    c230.quantile_score(
                        quantiles=np.sort(t[sel], axis=1) * cap_sel[:, None], actual=actual_mw
                    )
                )
            )
            for t in tables
        ]
        gaps = [abs(a - b) for a, b in combinations(scores, 2)]
        out[f"wind_{tag}_mean_pair_gap_mw"] = float(np.mean(gaps))
        out[f"wind_{tag}_seed_scores_mw"] = scores
    # solar a2_member: old arm single-seed pooled raw score
    labels = data453.labels().filter(pl.col("tech") == "solar")
    cap_s = {r["site"].removeprefix("Solar "): r["cap"] for r in labels.iter_rows(named=True)}
    scores = []
    for seed in (0, 1, 2):
        frame = pl.concat(
            [
                pl.read_parquet(f)
                for f in sorted((SOLAR_OLD / "a2_member").glob(f"s{seed}_*.parquet"))
            ]
        ).filter(
            pl.col("lead_hours").is_between(LEAD_MIN_H, LEAD_MAX_H) & pl.col("actual").is_finite()
        )
        q = np.sort(frame.select(list(LEVELS)).to_numpy(), axis=1)
        cap = frame["time_series_id"].replace_strict(cap_s, return_dtype=pl.Float64).to_numpy()
        a = np.clip(frame["actual"].to_numpy(), 0.0, None)
        scores.append(float(np.mean(c230.quantile_score(quantiles=q, actual=a) * cap)))
    gaps = [abs(a - b) for a, b in combinations(scores, 2)]
    out["solar_old_a2_member_mean_pair_gap_mw"] = float(np.mean(gaps))
    out["solar_old_a2_member_seed_scores_mw"] = scores
    return out


def tail_line(*, piece: pl.DataFrame) -> str:
    """Return the share above q99, each tail's coverage and pinball loss, and the median bias."""
    n = float(piece["n"].sum())
    total = {c: float(piece[c].sum()) for c in piece.columns if c.startswith(("hit_", "pin_"))}
    tails = " ".join(
        f"{t}: cov {total[f'hit_{t}'] / n * 100:.2f}% pin {total[f'pin_{t}'] / n:.4f}"
        for t in TAILS
    )
    bias = float(piece["mean_q50_minus_mean_actual"].sum()) / n
    return (
        f"above q99 {(1 - total['hit_q99'] / n) * 100:.3f}% {tails} mean(q50-actual) {bias:+.4f} MW"
    )


def main() -> None:
    """Build every arm's frame, the cells, and the report."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    frames = [wind_frame(rule="old"), wind_frame(rule="p999")]
    for arm in SOLAR_ARMS:
        frames += [solar_frame(arm=arm, rule="old"), solar_frame(arm=arm, rule="p999")]
    rows = []
    for frame in frames:
        rows += cells(frame=frame)
    table = pl.DataFrame(rows)
    table.write_parquet(WORK_DIR / "cells.parquet")
    for frame in frames:
        frame.write_parquet(WORK_DIR / f"rows_{frame['arm'][0]}_{frame['rule'][0]}.parquet")
    rng = np.random.default_rng(0)
    lines: list[str] = []
    lines.append(f"seed spread: {seed_spread()}")
    for arm in ("member_plain", *SOLAR_ARMS):
        sub = table.filter(pl.col("arm") == arm)
        tech = sub["tech"][0]
        old_n = (
            sub.filter(pl.col("rule") == "old").group_by("site").agg(pl.col("n").sum()).sort("site")
        )
        new_n = (
            sub.filter(pl.col("rule") == "p999")
            .group_by("site")
            .agg(pl.col("n").sum())
            .sort("site")
        )
        assert old_n.equals(new_n), f"unequal rows for {arm}: {old_n} {new_n}"
        sites = sorted(sub["site"].unique())
        lines.append(f"== {arm} ({tech}); rows per series (equal across rules): "
                     f"{dict(zip(old_n['site'], old_n['n'], strict=True))}")  # fmt: skip
        for label, group in (("pooled", sites), *[(s, [s]) for s in sites]):
            draws = bootstrap(table=sub, sites=group, rng=rng)
            piece = sub.filter(pl.col("site").is_in(group) & (pl.col("variant") == "P"))
            agg = piece.group_by("rule").agg(pl.col("score_mw_sum").sum(), pl.col("n").sum())
            m = {r["rule"]: r["score_mw_sum"] / r["n"] for r in agg.iter_rows(named=True)}
            diff = m["p999"] - m["old"]
            lines.append(
                f"{label}: old {m['old']:.4f} MW, new {m['p999']:.4f} MW, diff {diff:+.4f} MW "
                f"({diff / m['old'] * 100:+.2f}%), 95% [{np.percentile(draws, 2.5):+.4f}, "
                f"{np.percentile(draws, 97.5):+.4f}], 99.55% [{np.percentile(draws, 0.225):+.4f}, "
                f"{np.percentile(draws, 99.775):+.4f}]"
            )
        for rule in ("old", "p999"):
            for variant in ("raw", "P"):
                for label, group in (("pooled", sites), *[(s, [s]) for s in sites]):
                    p = sub.filter(
                        pl.col("site").is_in(group) & (pl.col("rule") == rule)
                        & (pl.col("variant") == variant)
                    )  # fmt: skip
                    lines.append(f"  [{rule} {variant}] {label}: {tail_line(piece=p)}")
    (WORK_DIR / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
