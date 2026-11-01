"""Score experiment 466: recency-weighted arms against the equal-weight p99.9 references, in MW.

The scoring is experiment 461's (`e00461_score`): quantiles and actuals in MW, table P fitted per
series and lead band on earlier scored months only, experiment 451's clip, true days 1 to 10 (lead
24 to 240 hours), experiment 210's idle-band mask for wind, seeds averaged quantile by quantile.
The references are experiment 461's stored p99.9 forecasts for solar and this experiment's
all-weights-1 wind fits (identical to experiment 461's). Every contrast is arm minus reference
in MW after table P, with month-block bootstrap intervals, per series and pooled.

    source ml-experiments/00429-*/env.sh   # then add the 00310, 00430, 00450, 00461, 00466 folders
    uv run --no-sync python ml-experiments/00466-*/e00466_score.py [--mde-only]
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Final

import e00230_common as c230  # ty: ignore[unresolved-import]
import e00347_s4_data as data347  # ty: ignore[unresolved-import]
import e00450_step1 as step1  # ty: ignore[unresolved-import]
import e00461_common as common461  # ty: ignore[unresolved-import]
import e00461_score as score461  # ty: ignore[unresolved-import]
import e00466_common as common
import numpy as np
import polars as pl

LEVELS: Final = step1.LEVELS
TAILS: Final = step1.TAILS
DRAWS: Final[int] = 10_000
WORK_DIR: Final[Path] = common.DATA_DIR / "work"
PARTS_DIR: Final[Path] = common.DATA_DIR / "wind_parts"
SOLAR_DIR: Final[Path] = common.DATA_DIR / "solar_predictions" / "p999"
SOLAR_461: Final[Path] = common461.DATA_DIR / "solar_predictions" / "p999"
WIND_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SOLAR_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MARGIN_WIND_PTS: Final[float] = 0.10
MARGIN_SOLAR_PCT: Final[float] = 1.0
NO_WEIGHT: Final[str] = "none"
SOLAR_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "g0_onepass": ("hl05", "hl1", "hl2", "hl3", "ind49"),
    "a2_member": ("hl2", "hl3"),
}


def wind_frame(*, label: str) -> pl.DataFrame:
    """Return `member_plain`'s kept scored rows in MW for one weighting arm (`none` = reference)."""
    tables = [np.load(PARTS_DIR / f"member_plain_{label}_h{s}.npy") for s in WIND_SEEDS]
    table = json.loads((common.DATA_DIR / "wind_capacity.json").read_text())
    cap = {farm: float(row["new_mw"]) for farm, row in table.items()}
    old_cap = {farm: float(row["old_mw"]) for farm, row in table.items()}
    q = np.mean(np.stack(tables), axis=0)
    frame = (
        data347.scored_frame()
        .select("site", "month", "masked", "lead_hours", "actual_norm",
                init=pl.col("power_fcst_init_time"))
        .with_columns(*[pl.Series(c, q[:, i]) for i, c in enumerate(LEVELS)])
        .filter(~pl.col("masked") & pl.col("actual_norm").is_finite())
    )  # fmt: skip
    ratio = {f: cap[f] / old_cap[f] for f in cap}
    frame = frame.with_columns(
        cap_mw=pl.col("site").replace_strict(cap, return_dtype=pl.Float64),
        actual=pl.col("actual_norm")
        / pl.col("site").replace_strict(ratio, return_dtype=pl.Float64),
        site=pl.col("site").replace_strict(common461.WIND_LABEL),
    )
    return score461._finish(frame=frame, tech="wind", arm=f"member_plain_{label}", rule="p999")


def solar_frame(*, base: str, label: str) -> pl.DataFrame:
    """Return a solar arm's scored rows in MW (`none` reads experiment 461's stored forecasts)."""
    keys = ["time", "init_time", "time_series_id", "lead_hours", "actual"]
    if label == NO_WEIGHT:
        files = sorted((SOLAR_461 / base).glob("s*_*.parquet"))
        table = json.loads((common461.DATA_DIR / "solar_capacity.json").read_text())
    else:
        files = sorted((SOLAR_DIR / f"{base}_{label}").glob("s*_*.parquet"))
        table = json.loads((common.DATA_DIR / "solar_capacity.json").read_text())
    assert len(files) == len(SOLAR_SEEDS) * 4, (base, label, len(files))
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
    mean = (
        pl.concat(parts)
        .group_by([*keys, "refit", "cap_mw"])
        .agg(*[pl.col(c).mean() for c in LEVELS])
        .sort(keys)
    )
    mean = mean.with_columns(
        init=pl.col("init_time"),
        site="Solar " + pl.col("time_series_id"),
        month=pl.col("time").dt.strftime("%Y-%m"),
    )
    return score461._finish(frame=mean, tech="solar", arm=f"{base}_{label}", rule="p999")


def seed_gaps() -> dict[str, float]:
    """Return each reference's mean pairwise gap between single-seed pooled raw scores (MW).

    This is the minimum detectable effect: the score difference two fits of one design differ by.
    """
    out = {}
    full = data347.scored_frame()
    table = json.loads((common.DATA_DIR / "wind_capacity.json").read_text())
    cap = (
        full["site"]
        .replace_strict({f: float(r["new_mw"]) for f, r in table.items()}, return_dtype=pl.Float64)
        .to_numpy()
    )
    ratio = (
        full["site"]
        .replace_strict({f: float(r["ratio"]) for f, r in table.items()}, return_dtype=pl.Float64)
        .to_numpy()
    )
    sel = (
        ~full["masked"] & full["actual_norm"].is_finite()
        & full["lead_hours"].is_between(24.0, 240.0)
    ).to_numpy()  # fmt: skip
    actual = np.clip(full["actual_norm"].to_numpy() / ratio, 0.0, None)[sel]
    scores = []
    for s in WIND_SEEDS:
        q = np.load(PARTS_DIR / f"member_plain_{NO_WEIGHT}_h{s}.npy")[sel]
        scores.append(
            float(
                np.mean(c230.quantile_score(quantiles=np.sort(q, axis=1), actual=actual) * cap[sel])
            )
        )
    out["wind"] = float(np.mean([abs(a - b) for a, b in combinations(scores, 2)]))
    stored = json.loads((common461.DATA_DIR / "solar_capacity.json").read_text())
    for base in SOLAR_ARMS:
        scores = []
        for seed in SOLAR_SEEDS:
            frames = []
            for f in sorted((SOLAR_461 / base).glob(f"s{seed}_*.parquet")):
                caps = {s: float(v["new_mw"]) for s, v in stored[f.stem.split("_")[1]].items()}
                frames.append(
                    pl.read_parquet(f).with_columns(
                        cap_mw=pl.col("time_series_id").replace_strict(
                            caps, return_dtype=pl.Float64
                        )
                    )
                )
            frame = pl.concat(frames).filter(
                pl.col("lead_hours").is_between(24.0, 240.0) & pl.col("actual").is_finite()
            )
            q = np.sort(frame.select(list(LEVELS)).to_numpy(), axis=1)
            a = np.clip(frame["actual"].to_numpy(), 0.0, None)
            scores.append(
                float(
                    np.mean(c230.quantile_score(quantiles=q, actual=a) * frame["cap_mw"].to_numpy())
                )
            )
        out[base] = float(np.mean([abs(a - b) for a, b in combinations(scores, 2)]))
    return out


def month_table(*, table: pl.DataFrame, sites: list[str]) -> pl.DataFrame:
    """Return the sum of score and rows per (arm, month) over `sites` for the table-P variant."""
    return (
        table.filter(pl.col("site").is_in(sites) & (pl.col("variant") == "P"))
        .group_by("arm", "month")
        .agg(pl.col("score_mw_sum").sum(), pl.col("n").sum())
    )


def contrast(
    *, table: pl.DataFrame, arm: str, reference: str, sites: list[str], rng: np.random.Generator
) -> dict:
    """Return the pooled mean score of each arm over `sites` and their difference."""
    wide = month_table(table=table, sites=sites).filter(pl.col("arm").is_in([arm, reference]))
    months = sorted(wide["month"].unique())
    cell = {(r["arm"], r["month"]): (r["score_mw_sum"], r["n"]) for r in wide.iter_rows(named=True)}
    mean = {a: sum(cell[a, m][0] for m in months) / sum(cell[a, m][1] for m in months)
            for a in (arm, reference)}  # fmt: skip
    per_month = {}
    for m in months:
        arm_mean = cell[arm, m][0] / cell[arm, m][1]
        ref_mean = cell[reference, m][0] / cell[reference, m][1]
        per_month[m] = arm_mean - ref_mean
    draws = np.empty(DRAWS)
    for d in range(DRAWS):
        pick = rng.integers(0, len(months), size=len(months))
        resampled = {
            a: sum(cell[a, months[i]][0] for i in pick) / sum(cell[a, months[i]][1] for i in pick)
            for a in (arm, reference)
        }
        draws[d] = resampled[arm] - resampled[reference]
    return {
        "arm_score": mean[arm],
        "ref_score": mean[reference],
        "diff": mean[arm] - mean[reference],
        "draws": draws,
        "per_month": per_month,
    }


def main() -> None:
    """Build every frame, the cells, and the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mde-only", action="store_true")
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    gaps = seed_gaps()
    lines = [f"minimum detectable effect (mean pair gap of reference seeds, pooled MW): {gaps}"]
    print(lines[0], flush=True)
    if args.mde_only:
        return
    frames = {"member_plain": [wind_frame(label=lab) for lab in common.WIND_HALF_LIVES_YEARS]}
    for base, labels in SOLAR_ARMS.items():
        present = [
            lab
            for lab in (NO_WEIGHT, *labels)
            if lab == NO_WEIGHT or (SOLAR_DIR / f"{base}_{lab}").exists()
        ]
        frames[base] = [solar_frame(base=base, label=lab) for lab in present]
    rng = np.random.default_rng(0)
    table = pl.DataFrame([r for fs in frames.values() for f in fs for r in score461.cells(frame=f)])
    table.write_parquet(WORK_DIR / "cells.parquet")
    for fs in frames.values():
        for f in fs:
            f.write_parquet(WORK_DIR / f"rows_{f['arm'][0]}.parquet")
    n_tests = sum(
        (len(fs) - 1) * (1 + len(table.filter(pl.col("arm") == fs[0]["arm"][0])["site"].unique()))
        for fs in frames.values()
    )
    lines.append(f"Bonferroni count (arms x (pooled + series)): {n_tests}")
    for base, fs in frames.items():
        reference = fs[0]["arm"][0]
        sub = table.filter(pl.col("arm").is_in([f["arm"][0] for f in fs]))
        sites = sorted(sub["site"].unique())
        n_by_arm = {}
        for a in sub["arm"].unique():
            rows = sub.filter(pl.col("arm") == a).group_by("site").agg(pl.col("n").sum())
            n_by_arm[a] = dict(rows.iter_rows())
        assert all(v == n_by_arm[reference] for v in n_by_arm.values()), f"unequal rows for {base}"
        lines.append(f"== {base}; rows per series (equal across arms): {n_by_arm[reference]}")
        for f in fs[1:]:
            arm = f["arm"][0]
            for label, group in (("pooled", sites), *[(s, [s]) for s in sites]):
                c = contrast(table=sub, arm=arm, reference=reference, sites=group, rng=rng)
                d = c["draws"]
                better = sum(v < 0 for v in c["per_month"].values())
                lines.append(
                    f"{arm} {label}: ref {c['ref_score']:.4f} MW, arm {c['arm_score']:.4f} MW, "
                    f"diff {c['diff']:+.4f} MW ({c['diff'] / c['ref_score'] * 100:+.2f}%), "
                    f"95% [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], "
                    f"Bonferroni [{np.percentile(d, 2.5 / n_tests * 2):+.4f}, "
                    f"{np.percentile(d, 100 - 2.5 / n_tests * 2):+.4f}], "
                    f"better in {better} of {len(c['per_month'])} months"
                )
                if label == "pooled":
                    lines.append(
                        "  months: " + " ".join(f"{m} {v:+.4f}" for m, v in c["per_month"].items())
                    )
        for a in sub["arm"].unique():
            for label, group in (("pooled", sites), *[(s, [s]) for s in sites]):
                p = sub.filter(
                    pl.col("site").is_in(group) & (pl.col("arm") == a) & (pl.col("variant") == "P")
                )
                lines.append(f"  [{a} P] {label}: {score461.tail_line(piece=p)}")
        # 12 to 36 hour band and per-year, from the scored rows
        for f in fs:
            rows = f.filter(pl.col("lead_band") == "12-36h")
            lines.append(f"  band 12-36h rows in {f['arm'][0]}: {rows.height}")
    (WORK_DIR / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
