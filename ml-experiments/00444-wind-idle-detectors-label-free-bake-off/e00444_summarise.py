"""Experiment 444: turn the bake-off's long tables into the gate, intervals, and census tables.

    PYTHONPATH=ml-experiments/00431-wind-idle-data-look \
        OMP_NUM_THREADS=1 \
        uv run --no-sync python \
        ml-experiments/00444-wind-idle-detectors-label-free-bake-off/e00444_summarise.py

Reads the tables that `e00444_bakeoff.py` wrote and writes published-safe summary tables (recall,
false-drop rates, shares by wind bin, counts; no farm-day and no date) beside them. Every interval
resamples calendar-year blocks (2,000 draws) with one set of weights per farm, so paired
differences use the same draws.
"""

import os
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00431_analysis import stratified_auc  # ty: ignore[unresolved-import]

OUT_DIR: Final[Path] = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments"
    / os.environ.get("E444_OUT", "00444-wind-idle-detectors-label-free-bake-off")
)
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
N_BOOT: Final[int] = 2000
MIN_BLOCKS_PER_FOLD: Final[int] = 10
BLOCK_LENGTHS: Final[tuple[int, ...]] = (6, 24, 48, 96, 192)
GATE_RECALL: Final[float] = 0.80
GATE_FD: Final[float] = 0.05
GATE_OVER_R0: Final[float] = 0.10
NEG_SETS: Final[tuple[str, ...]] = ("SG", "SGm", "G0s")
MDE_FACTOR: Final[float] = 2.8
SEED: Final[int] = 4440


def year_weights(*, years: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return an (N_BOOT, n_years) matrix of year-block resampling counts."""
    draws = rng.integers(0, len(years), size=(N_BOOT, len(years)))
    return np.stack([np.bincount(d, minlength=len(years)) for d in draws]).astype(float)


def recall_draws(
    *, blocks: pl.DataFrame, years: np.ndarray, weights: np.ndarray, column: str
) -> tuple[float, np.ndarray, pl.DataFrame]:
    """Return the cell-mean recall (point and draws) and the per-cell table for one candidate."""
    sums = {}
    for (length, bin_name), grp in blocks.group_by("length", "bin"):
        s = np.zeros(len(years))
        c = np.zeros(len(years))
        for y, v in zip(grp["year"].to_numpy(), grp[column].to_numpy(), strict=True):
            if np.isfinite(v):
                s[np.searchsorted(years, y)] += v
                c[np.searchsorted(years, y)] += 1
        sums[(length, bin_name)] = (s, c)
    cell_rows = [
        {"length": k[0], "bin": k[1], "n": int(c.sum()), "recall": float(s.sum() / max(c.sum(), 1))}
        for k, (s, c) in sorted(sums.items())
    ]
    point_cells = [r["recall"] for r in cell_rows]
    point = float(np.mean(point_cells))
    cell_draws = []
    for s, c in sums.values():
        num, den = weights @ s, weights @ c
        cell_draws.append(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan))
    return point, np.mean(np.stack(cell_draws), axis=0), pl.DataFrame(cell_rows)


def fd_draws(
    *, neg: pl.DataFrame, years: np.ndarray, weights: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Return per-set false-drop rates (point and draws) and the equal-weight gate rate."""
    point: dict[str, float] = {}
    draws: dict[str, np.ndarray] = {}
    for set_name in (*NEG_SETS, "SG_not_D3", "clear"):
        grp = neg.filter(pl.col("set") == set_name)
        n = np.zeros(len(years))
        f = np.zeros(len(years))
        for y, nn, ff in zip(
            grp["year"].to_numpy(), grp["n"].to_numpy(), grp["n_flag"].to_numpy(), strict=True
        ):
            n[np.searchsorted(years, y)] += nn
            f[np.searchsorted(years, y)] += ff
        if n.sum() == 0:
            continue
        point[set_name] = float(f.sum() / n.sum())
        den = weights @ n
        draws[set_name] = np.where(den > 0, (weights @ f) / np.maximum(den, 1e-9), np.nan)
    present = [s for s in NEG_SETS if s in point]
    point["gate"] = float(np.mean([point[s] for s in present]))
    draws["gate"] = np.mean(np.stack([draws[s] for s in present]), axis=0)
    return point, draws


def interval(draws: np.ndarray) -> tuple[float, float]:
    """Return the 2.5% and 97.5% quantiles of finite draws."""
    finite = draws[np.isfinite(draws)]
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def main() -> None:  # noqa: PLR0915
    """Write the gate, cell, curve, pair, census, and agreement tables."""
    inj = pl.read_parquet(OUT_DIR / "inj.parquet")
    neg = pl.read_parquet(OUT_DIR / "neg.parquet")
    bins = pl.read_parquet(OUT_DIR / "bins.parquet")
    high_year = pl.read_parquet(OUT_DIR / "high_year.parquet")
    special = pl.read_parquet(OUT_DIR / "special.parquet")
    census = pl.read_parquet(OUT_DIR / "census.parquet")
    rng = np.random.default_rng(SEED)
    gate_rows, cell_tables, pair_rows, thin_list, matched_rows = [], [], [], [], []
    for site in SITES:
        site_inj = inj.filter(pl.col("site") == site)
        base = (
            site_inj.filter(pl.col("cand") == "R0")
            .select("length", "bin", "fold", "year", "start")
            .unique()
        )
        counts = base.group_by("length", "bin", "fold").agg(pl.len().alias("n"))
        grid = pl.DataFrame(
            [
                {"length": ln, "bin": bn, "fold": fo}
                for ln in BLOCK_LENGTHS
                for bn in ("moderate", "high")
                for fo in (0, 1)
            ]
        )
        counts = grid.join(counts, on=["length", "bin", "fold"], how="left").with_columns(
            pl.col("n").fill_null(0)
        )
        thin = (
            counts.group_by("length", "bin")
            .agg(pl.col("n").min().alias("min_fold"))
            .filter(pl.col("min_fold") < MIN_BLOCKS_PER_FOLD)
        )
        thin_list.extend({"site": site, **r} for r in thin.iter_rows(named=True))
        thin_cells = {(r["length"], r["bin"]) for r in thin.iter_rows(named=True)}
        keep = pl.Series(
            [
                (a, b) not in thin_cells
                for a, b in zip(site_inj["length"], site_inj["bin"], strict=True)
            ]
        )
        site_inj = site_inj.filter(keep)
        years = np.array(
            sorted(
                set(site_inj["year"].unique().to_list())
                | set(neg.filter(pl.col("site") == site)["year"].unique().to_list())
            )
        )
        weights = year_weights(years=years, rng=rng)
        point_by = {}
        for cand in sorted(site_inj["cand"].unique().to_list()):
            blocks = site_inj.filter(pl.col("cand") == cand)
            pt, dr, cells = recall_draws(
                blocks=blocks, years=years, weights=weights, column="recall"
            )
            pt_nj, _, _ = recall_draws(
                blocks=blocks, years=years, weights=weights, column="recall_nojoin"
            )
            fpt, fdr = fd_draws(
                neg=neg.filter((pl.col("site") == site) & (pl.col("cand") == cand)),
                years=years,
                weights=weights,
            )
            sp = special.filter((pl.col("site") == site) & (pl.col("cand") == cand))

            def share(name: str, frame: pl.DataFrame = sp) -> tuple[float, int]:
                part = frame.filter(pl.col("set") == name)
                n = int(part["n"].sum())
                return (float(part["n_flag"].sum() / n) if n else float("nan")), n

            rec_lo, rec_hi = interval(dr)
            fd_lo, fd_hi = interval(fdr["gate"])
            point_by[cand] = (pt, dr, fpt, fdr)
            cut_s, cut_n = share("cutout_idle")
            neg_h, neg_hn = share("negative_high_wind")
            neg_a, neg_an = share("negative_any_wind")
            gate_rows.append(
                {
                    "site": site,
                    "cand": cand,
                    "recall": pt,
                    "recall_lo": rec_lo,
                    "recall_hi": rec_hi,
                    "recall_nojoin": pt_nj,
                    "fd_gate": fpt["gate"],
                    "fd_lo": fd_lo,
                    "fd_hi": fd_hi,
                    "fd_SG": fpt.get("SG", float("nan")),
                    "fd_SGm": fpt.get("SGm", float("nan")),
                    "fd_G0s": fpt.get("G0s", float("nan")),
                    "fd_SG_not_D3": fpt.get("SG_not_D3", float("nan")),
                    "flag_rate_clear": fpt.get("clear", float("nan")),
                    "recall_cutout": cut_s,
                    "n_cutout": cut_n,
                    "recall_negative_high_wind": neg_h,
                    "n_negative_high_wind": neg_hn,
                    "recall_negative_any_wind": neg_a,
                    "n_negative_any_wind": neg_an,
                    "dropped_thin_cells": len(thin_cells),
                }
            )
            cell_tables.append(cells.with_columns(site=pl.lit(site), cand=pl.lit(cand)))
        r0_recall, r0_fd = point_by["R0"][0], point_by["R0"][2]["gate"]
        curve = sorted(
            (point_by[c][2]["gate"], point_by[c][0])
            for c in ("D1_fd01", "D1_fd02", "D1", "D1_fd10")
            if c in point_by
        )
        matched = float(np.interp(r0_fd, [x[0] for x in curve], [x[1] for x in curve]))
        matched_rows.append(
            {
                "site": site,
                "r0_fd_gate": r0_fd,
                "r0_recall": r0_recall,
                "d1_recall_at_r0_fd": matched,
                "d1_curve_fd": [x[0] for x in curve],
                "d1_curve_recall": [x[1] for x in curve],
            }
        )
        for row in gate_rows:
            if row["site"] == site:
                row["gate_recall"] = row["recall"] >= GATE_RECALL
                row["gate_fd"] = row["fd_gate"] <= GATE_FD
                row["gate_over_r0"] = row["recall"] - r0_recall >= GATE_OVER_R0
                row["gate_over_r0_matched_fd"] = (
                    (matched - r0_recall >= GATE_OVER_R0) if row["cand"] == "D1" else None
                )
                gated = row["cand"] in ("D1", "D4")
                over = (
                    row["gate_over_r0_matched_fd"] if row["cand"] == "D1" else row["gate_over_r0"]
                )
                row["gate_pass"] = (
                    bool(row["gate_recall"] and row["gate_fd"] and over) if gated else None
                )
        for a, b in (("D4", "D1"), ("D1", "R0"), ("D1a", "D1"), ("D1&D3", "D1"), ("D4", "R0")):
            if a in point_by and b in point_by:
                d_rec = point_by[a][1] - point_by[b][1]
                d_fd = point_by[a][3]["gate"] - point_by[b][3]["gate"]
                lo, hi = interval(d_rec)
                flo, fhi = interval(d_fd)
                pair_rows.append(
                    {
                        "site": site,
                        "a": a,
                        "b": b,
                        "recall_diff": point_by[a][0] - point_by[b][0],
                        "recall_lo": lo,
                        "recall_hi": hi,
                        "recall_mde": MDE_FACTOR * float(np.nanstd(d_rec)),
                        "fd_diff": point_by[a][2]["gate"] - point_by[b][2]["gate"],
                        "fd_lo": flo,
                        "fd_hi": fhi,
                        "fd_mde": MDE_FACTOR * float(np.nanstd(d_fd)),
                    }
                )
    pl.DataFrame(gate_rows).write_parquet(OUT_DIR / "gate.parquet")
    pl.concat(cell_tables).write_parquet(OUT_DIR / "cells.parquet")
    pl.DataFrame(pair_rows).write_parquet(OUT_DIR / "pairs.parquet")
    pl.DataFrame(matched_rows).write_parquet(OUT_DIR / "matched_fd.parquet")
    pl.DataFrame(
        thin_list,
        schema={"site": pl.String, "length": pl.Int64, "bin": pl.String, "min_fold": pl.Int64},
    ).write_parquet(OUT_DIR / "thin_cells.parquet")
    curve = (
        bins.group_by("site", "cand", "wind_bin")
        .agg(pl.col("n").sum(), pl.col("n_flag").sum())
        .with_columns(share=pl.col("n_flag") / pl.col("n"))
    )
    curve.write_parquet(OUT_DIR / "cut_in_curve.parquet")
    high_year.group_by("site", "cand", "year").agg(
        pl.col("n").sum(), pl.col("n_flag").sum()
    ).with_columns(share=pl.col("n_flag") / pl.col("n")).write_parquet(
        OUT_DIR / "high_wind_by_year.parquet"
    )
    census.group_by("site", "kind", "cell").agg(pl.col("n").sum()).write_parquet(
        OUT_DIR / "agreement_d1_d3.parquet"
    )
    _auc_table()
    _flag_census()


def _auc_table() -> None:
    """Level-matched AUC of D1's score for sure-idle against sure-generating rows, per farm."""
    rows = pl.read_parquet(OUT_DIR / "auc_rows_private.parquet")
    rng = np.random.default_rng(SEED + 1)
    out = []
    for site in SITES:
        sub = rows.filter(pl.col("site") == site)
        years = sorted(sub["year"].unique().to_list())
        by_year = {y: sub.filter(pl.col("year") == y) for y in years}

        def matched(ys: list[int], by_year: dict = by_year) -> float:
            parts = [by_year[y] for y in ys]
            sel = pl.concat(parts)
            if sel.height > 60000:
                sel = sel.sample(60000, seed=SEED)
            return stratified_auc(
                score=sel["score"].to_numpy(),
                label=sel["idle"].to_numpy(),
                level=sel["level"].to_numpy(),
            )

        point = matched(years)
        draws = np.array([matched(list(rng.choice(years, size=len(years)))) for _ in range(200)])
        finite = draws[np.isfinite(draws)]
        out.append(
            {
                "site": site,
                "auc_level_matched": point,
                "lo": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
                "hi": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
                "n_draws_used": len(finite),
            }
        )
    pl.DataFrame(out).write_parquet(OUT_DIR / "d1_auc.parquet")


def _runs(*, flag: np.ndarray, contiguous: np.ndarray) -> list[int]:
    """Return the lengths in half-hours of unbroken runs of `flag`."""
    out, length = [], 0
    for k in range(len(flag)):
        if flag[k] and (length == 0 or contiguous[k]):
            length += 1
        else:
            if length:
                out.append(length)
            length = 1 if flag[k] else 0
    if length:
        out.append(length)
    return out


def _flag_census() -> None:
    """Census of out-of-fold flags at high wind, B against C agreement, and hand-label widths."""
    oof = pl.read_parquet(OUT_DIR / "oof_flags_private.parquet").sort("site", "valid_time")
    rows, day_rows = [], []
    for site in SITES:
        sub = oof.filter((pl.col("site") == site) & ~pl.col("edge"))
        t = sub["valid_time"].dt.replace_time_zone(None).to_numpy()
        contiguous = np.concatenate([[False], np.diff(t) == np.timedelta64(30, "m")])
        high = sub["high"].to_numpy()
        for cand in ("D1", "D4", "D3", "R0"):
            flag = sub[cand].to_numpy() & high
            runs = _runs(flag=flag, contiguous=contiguous)
            rows.append(
                {
                    "site": site,
                    "cand": cand,
                    "high_wind_rows": int(high.sum()),
                    "flagged": int(flag.sum()),
                    "share": float(flag.sum() / max(high.sum(), 1)),
                    "runs_24h_or_more": int(sum(r >= 48 for r in runs)),
                    "runs_48h_or_more": int(sum(r >= 96 for r in runs)),
                    "longest_hours": float(max(runs, default=0) / 2),
                }
            )
        days = (
            sub.with_columns(day=pl.col("valid_time").dt.date())
            .group_by("day")
            .agg(
                (pl.col("D4") & pl.col("high")).sum().alias("n_flag"),
                pl.col("high").sum().alias("n_high"),
                pl.len().alias("n"),
            )
        )
        eligible = days.filter(pl.col("n_high") >= 24)
        prevalence = float(np.mean(eligible["n_flag"].to_numpy() >= 6))
        day_rows.append(
            {
                "site": site,
                "eligible_days": eligible.height,
                "share_days_with_flag": prevalence,
                "idle_days_in_100_random": 100 * prevalence,
                "recall_halfwidth_random_100": float(
                    1.96 * np.sqrt(0.8 * 0.2 / max(100 * prevalence, 1))
                ),
                "recall_halfwidth_enriched_100": float(1.96 * np.sqrt(0.8 * 0.2 / 100)),
            }
        )
    pl.DataFrame(rows).write_parquet(OUT_DIR / "flag_census.parquet")
    pl.DataFrame(day_rows).write_parquet(OUT_DIR / "hand_label_width.parquet")
    b = oof.filter((pl.col("site") == "W2") & ~pl.col("edge")).select(
        "valid_time", d1_b="D1", high_b="high"
    )
    c = oof.filter((pl.col("site") == "W3") & ~pl.col("edge")).select(
        "valid_time", d1_c="D1", high_c="high"
    )
    both = b.join(c, on="valid_time").filter(pl.col("high_b") & pl.col("high_c"))
    pl.DataFrame(
        {
            "cell": ["both", "only_farm_b", "only_farm_c", "neither"],
            "n": [
                int((both["d1_b"] & both["d1_c"]).sum()),
                int((both["d1_b"] & ~both["d1_c"]).sum()),
                int((~both["d1_b"] & both["d1_c"]).sum()),
                int((~both["d1_b"] & ~both["d1_c"]).sum()),
            ],
        }
    ).write_parquet(OUT_DIR / "agreement_b_c.parquet")


if __name__ == "__main__":
    main()
