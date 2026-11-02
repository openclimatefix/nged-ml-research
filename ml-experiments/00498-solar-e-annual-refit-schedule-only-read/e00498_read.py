# ruff: noqa: E501
"""Experiment 498: read experiment 493's stored seeds 0 to 5 for Solar E, annual against quarterly refits.

CPU only, no fits. Contrast C2 is `a2_member` annual against `a2_member` quarterly (schedule only).
Contrast C3 is `g0_onepass` annual against `a2_member` quarterly (schedule and network design).

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00493-solar-annual-against-quarterly-refits \
        uv run --no-sync python ml-experiments/00498-*/e00498_read.py
"""

from typing import Final

import e00430_score as s430  # ty: ignore[unresolved-import]
import e00493_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

FARM: Final[str] = "E"
BONFERRONI_COUNT: Final[int] = 2
"""Solar E, all bands (C2) and Solar E, the 12 to 36 hour band (C2), as the queue-order check set."""
LEVEL_PCT: Final[float] = 100.0 * (1.0 - 0.05 / BONFERRONI_COUNT)
LIMIT_PERCENT: Final[float] = 1.0
DRAWS: Final[int] = 10_000
ARMS: Final[dict[str, str]] = {
    "annual_a2": "annual_a2_member",
    "q_a2": "q_a2_member",
    "annual_g0": "annual_g0_onepass",
}


def folder(*, name: str):
    """Return the stored prediction folder of an arm (`annual_*` is 493's, `q_*` is 430's)."""
    if name.startswith("annual_"):
        return common.PRED_DIR / name
    return common.E430_PRED_DIR / name.removeprefix("q_")


def load_seed(*, name: str, seed: int) -> pl.DataFrame:
    """Return one arm's rows for Solar E and one seed, finite quantiles only."""
    files = sorted(folder(name=name).glob(f"s{seed}_*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files]).filter(pl.col("time_series_id") == FARM)
    return frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in s430.QCOLS]))


def score_rows(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses with month, quarter, and half-hour of day added."""
    rows = s430.losses(frame=frame.with_columns(arm=pl.lit("x"), setting=pl.lit("own_history")))
    return rows.with_columns(
        init_month=pl.col("init_time").dt.strftime("%Y-%m"),
        hour=pl.col("time").dt.hour(),
    )


def paired(*, a: pl.DataFrame, b: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return both frames cut to the rows both hold, in one order."""
    keys = a.select(s430.KEYS).join(b.select(s430.KEYS), on=s430.KEYS, how="inner")
    return (
        keys.join(a, on=s430.KEYS, how="left").sort(s430.KEYS),
        keys.join(b, on=s430.KEYS, how="left").sort(s430.KEYS),
    )


def block_interval(*, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, int]:
    """Return percent change of a against b, its Bonferroni interval over blocks, and the block count."""
    rng = np.random.default_rng(0)
    n = len(a)
    point = 100.0 * (a.mean() / b.mean() - 1.0)
    index = rng.integers(0, n, size=(DRAWS, n))
    draws = 100.0 * (a[index].mean(axis=1) / b[index].mean(axis=1) - 1.0)
    lo = (100.0 - LEVEL_PCT) / 2.0
    return float(point), float(np.percentile(draws, lo)), float(np.percentile(draws, 100 - lo)), n


def by_block(*, rows: pl.DataFrame, column: str) -> tuple[np.ndarray, list]:
    """Return the mean quantile score in each block of the column."""
    per = rows.group_by(column).agg(pl.col("score").mean()).sort(column)
    return per["score"].to_numpy(), per[column].to_list()


def main() -> None:
    """Print the minimum detectable effect, the per-seed contrasts, and the breakdowns."""
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(14)
    seeds = common.SEEDS
    raw = {(n, s): load_seed(name=ARMS[n], seed=s) for n in ARMS for s in seeds}

    # Seed-averaged frames (six seeds), then all pairs on shared rows.
    avg = {
        n: s430.average_seeds(
            frame=pl.concat([raw[(n, s)] for s in seeds]), arm=n
        ).with_columns(time_series_id=pl.lit(FARM))
        for n in ARMS
    }
    cut = s430.common_rows(frames=avg)
    rows = {n: score_rows(frame=f) for n, f in cut.items()}
    print(f"Solar E rows per arm {rows['annual_a2'].height}; interval level {LEVEL_PCT:.2f}% (Bonferroni count {BONFERRONI_COUNT})")
    lead = rows["annual_a2"]["lead_hours"]
    print(f"lead hours min {lead.min()} max {lead.max()}")

    # Minimum detectable effect from per-seed paired contrasts (annual seed s against quarterly seed s).
    pairs = {"C2": ("annual_a2", "q_a2"), "C3": ("annual_g0", "q_a2")}
    per_seed: dict[str, list[float]] = {"C2": [], "C3": []}
    for cid, (na, nb) in pairs.items():
        for s in seeds:
            a, b = paired(a=raw[(na, s)], b=raw[(nb, s)])
            ra, rb = score_rows(frame=a), score_rows(frame=b)
            per_seed[cid].append(100.0 * (ra["score"].mean() / rb["score"].mean() - 1.0))
    print("\nPER-SEED CONTRASTS (percent, Solar E, all bands, paired by seed)")
    out = []
    for cid, vals in per_seed.items():
        sd = float(np.std(vals, ddof=1))
        mde = 1.96 * sd / np.sqrt(len(vals))
        print(cid, [f"{v:+.2f}" for v in vals], f"mean {np.mean(vals):+.3f} sd {sd:.3f} min {min(vals):+.2f} max {max(vals):+.2f}  MDE (1.96 sd/sqrt 6) {mde:.3f}%")
        out.append({"id": cid, "mean": float(np.mean(vals)), "sd": sd, "mde": mde, "min": min(vals), "max": max(vals)})
    print(f"limit: {LIMIT_PERCENT}% (at or above the C2 MDE: {LIMIT_PERCENT >= out[0]['mde']})")

    print("\nSIX-SEED AVERAGE CONTRASTS, month-block and quarter-block")
    for cid, (na, nb) in pairs.items():
        for cut_name, sel in (("all", None), ("d1", "d1")):
            ra = rows[na] if sel is None else rows[na].filter(pl.col("band") == sel)
            rb = rows[nb] if sel is None else rows[nb].filter(pl.col("band") == sel)
            va, months = by_block(rows=ra, column="month")
            vb, _ = by_block(rows=rb, column="month")
            p, lo, hi, n = block_interval(a=va, b=vb)
            print(f"{cid} {cut_name} month-block ({n} months): {p:+.3f}% ({lo:+.3f} to {hi:+.3f})")
            qa = ra.with_columns(quarter=pl.col("month").map_elements(lambda m: (months.index(m) // 3), return_dtype=pl.Int64))
            qb = rb.with_columns(quarter=pl.col("month").map_elements(lambda m: (months.index(m) // 3), return_dtype=pl.Int64))
            qva, _ = by_block(rows=qa, column="quarter")
            qvb, _ = by_block(rows=qb, column="quarter")
            p, lo, hi, n = block_interval(a=qva, b=qvb)
            print(f"{cid} {cut_name} quarter-block ({n} quarters, 35 distinct resamples at most): {p:+.3f}% ({lo:+.3f} to {hi:+.3f})")

    print("\nBY QUARTER OF THE SCORED YEAR (init-month groups), per seed contrast for C2")
    months = sorted(rows["annual_a2"]["month"].unique().to_list())
    qrows = []
    for k in range(4):
        sel = months[3 * k : 3 * k + 3]
        for cid, (na, nb) in pairs.items():
            va = rows[na].filter(pl.col("month").is_in(sel))["score"].mean()
            vb = rows[nb].filter(pl.col("month").is_in(sel))["score"].mean()
            seed_vals = []
            for s in seeds:
                a, b = paired(a=raw[(na, s)], b=raw[(nb, s)])
                ra, rb = score_rows(frame=a), score_rows(frame=b)
                seed_vals.append(
                    100.0
                    * (
                        ra.filter(pl.col("month").is_in(sel))["score"].mean()
                        / rb.filter(pl.col("month").is_in(sel))["score"].mean()
                        - 1.0
                    )
                )
            qrows.append(
                {"quarter": k + 1, "months": f"{sel[0]}..{sel[-1]}", "id": cid, "pct": 100.0 * (va / vb - 1.0), "seed_sd": float(np.std(seed_vals, ddof=1)), "seed_min": min(seed_vals), "seed_max": max(seed_vals)}
            )
    print(pl.DataFrame(qrows))

    print("\nBY MONTH (C2 and C3, six-seed average)")
    mrows = []
    for m in months:
        rec: dict = {"month": m}
        for cid, (na, nb) in pairs.items():
            va = rows[na].filter(pl.col("month") == m)["score"].mean()
            vb = rows[nb].filter(pl.col("month") == m)["score"].mean()
            rec[f"{cid}_pct"] = 100.0 * (va / vb - 1.0)
            rec[f"{cid}_loss_pts"] = va - vb
        mrows.append(rec)
    print(pl.DataFrame(mrows))

    print("\nBY HOUR OF DAY (UTC), C2 six-seed average; share of the total score difference")
    ra, rb = rows["annual_a2"], rows["q_a2"]
    total = ra["score"].sum() - rb["score"].sum()
    hrows = []
    for h in sorted(ra["hour"].unique().to_list()):
        sa, sb = ra.filter(pl.col("hour") == h), rb.filter(pl.col("hour") == h)
        diff = sa["score"].sum() - sb["score"].sum()
        hrows.append({"hour_utc": h, "rows": sa.height, "pct": 100.0 * (sa["score"].mean() / sb["score"].mean() - 1.0), "share_of_total_diff": diff / total})
    print(pl.DataFrame(hrows))

    print("\nCAPACITY AND CLIP (actual output as a fraction of capacity in the scored year)")
    act = cut["annual_a2"].with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
    for m in months:
        a_ = act.filter(pl.col("month") == m)["actual"]
        print(m, f"p99.9 {a_.quantile(0.999):.3f} max {a_.max():.3f} share >=0.98 of p99.9: {float((a_ >= 0.98 * a_.quantile(0.999)).mean()):.4f}")
    top = float(act["actual"].quantile(0.999))
    near = act.filter(pl.col("actual") >= 0.98 * top)
    sa = score_rows(frame=cut["annual_a2"]).join(near.select(s430.KEYS), on=s430.KEYS, how="semi")
    sq = score_rows(frame=cut["q_a2"]).join(near.select(s430.KEYS), on=s430.KEYS, how="semi")
    all_diff = rows["annual_a2"]["score"].sum() - rows["q_a2"]["score"].sum()
    clip_diff = sa["score"].sum() - sq["score"].sum()
    print(f"rows with actual >= 0.98 of the year's p99.9 ({near.height} rows): C2 clipped-rows share of total score difference {clip_diff / all_diff:+.3f}")
    print(f"upper-tail pinball q95 and q99 contrasts (C2, all rows):")
    for t in ("pin95", "pin99", "pin01", "pin05"):
        print(t, f"annual {rows['annual_a2'][t].mean():.4f} quarterly {rows['q_a2'][t].mean():.4f}")
    for t in ("cover95", "cover99"):
        print(t, f"annual {rows['annual_a2'][t].mean():.4f} quarterly {rows['q_a2'][t].mean():.4f}")
    print("median minus mean (points):", f"annual {rows['annual_a2']['median_minus_mean'].mean():+.3f} quarterly {rows['q_a2']['median_minus_mean'].mean():+.3f}")
    print("mean-forecast absolute error (points):", f"annual {rows['annual_a2']['ae_mean'].mean():.3f} quarterly {rows['q_a2']['ae_mean'].mean():.3f}")
    print("absolute scores:", {n: round(float(rows[n]['score'].mean()), 3) for n in rows})
    pl.DataFrame(out).write_csv(common.DATA_DIR / "scores" / "e00498_seed_contrasts.csv")


if __name__ == "__main__":
    main()
