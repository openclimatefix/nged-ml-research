# ruff: noqa: D103, E501, PLR0915, PLR0912, D107, PTH123, SIM115
"""Experiment 702: score the arms, the five planned contrasts, and the match checks of the control.

Usage (after `e00702_export.py` has written the arms' backtest files):

    python e00702_score.py score [--months 2025-11 2026-01 2026-04 2026-07]
    python e00702_score.py match-stored      # rescore experiment 677's file; compare with its stored monthly scores
    python e00702_score.py match-refit       # the pilot fold's refit of the global arm against the stored seeds
    python e00702_score.py bank              # copied share of the held-out injected shifts
    python e00702_score.py pilot --files NAME [NAME ...]   # first run: scores of backtest files on 2026-05, at the three Primaries
    python e00702_score.py match-bank --months 2026-05 --seed 0   # this experiment's bank on the stored global models against experiment 677's bank file

`score` writes CSV tables to `scores/four_months/` in this experiment's data folder: `arms.csv`,
`contrasts.csv`, `per_primary.csv`, `leave_one_out.csv`, `lead_bands.csv`, `groups.csv`,
`tails.csv`, and `row_counts.csv`. One function (`lib.losses`, experiment 668's scoring library, as
experiment 677 used) scores every arm on the same rows.
"""

import argparse
import csv
from pathlib import Path
from typing import Final

import e00668_score_lib as lib  # ty: ignore[unresolved-import]
import e00702_core as core
import numpy as np
import polars as pl

KEYS: Final = ("time_series_id", "power_fcst_init_time", "valid_time")
QCOLS: Final = tuple(lib.QCOLS)
TAILS: Final[dict[str, int]] = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
"""Index of each tail quantile in the 11 levels."""
SCORES: Final[Path] = core.ROOT / "scores"
SITE_MAP: Final[Path] = (
    next(core.ROOT.parent.glob("00521-*")) / "site_map.csv"
)  # one row per Primary: label, time_series_id, capacity_mw, group
BANK_MATCH_TOLERANCE: Final[float] = 1e-3
"""Largest accepted difference between two runs' median forecasts on the same stored network (GPU arithmetic is not bit-exact)."""
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 split-at-1 minus global": ("split1", "global"),
    "C2 split-at-2 minus global": ("split2", "global"),
    "C3 split-at-3 minus global": ("split3", "global"),
    "C4 blend minus global pair": ("blend", "pair"),
    "C5 split-at-2-plus-encoders minus split-at-2": ("hybrid3", "split2"),
}
"""The five planned contrasts: (arm tested, reference). The seed-level pairing is `seed_forecast`."""


def sites() -> pl.DataFrame:
    return pl.read_csv(SITE_MAP).select(
        "primary", "label", "time_series_id", "capacity_mw", "group"
    )


def read_backtest(
    *, name: str, seeds: tuple[int, ...], months: tuple[str, ...]
) -> dict[int, pl.DataFrame]:
    """Return one sorted frame per seed of the arm's contract rows whose origin lies in `months`."""
    raw = pl.read_parquet(core.ROOT / "backtests" / f"{name}.parquet")
    raw = raw.filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(list(months)))
    out = {}
    for s in seeds:
        part = raw.filter(pl.col("seed") == s).sort(KEYS)
        assert part.height, (name, s)
        out[s] = part
    return out


def read_stored_677(*, seeds: tuple[int, ...], months: tuple[str, ...]) -> dict[int, pl.DataFrame]:
    """Return experiment 677's stored global forecasts (step 5 with injected events, monthly refits)."""
    raw = pl.read_parquet(core.ROOT_677 / "backtests" / "step5w_with_injected_sp1.parquet")
    raw = raw.filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(list(months)))
    return {s: raw.filter(pl.col("seed") == s).sort(KEYS) for s in seeds}


class Rows:
    """The scored rows: keys, observed power, the manual heuristic and climatology, all in points of capacity."""

    def __init__(self, *, keys: pl.DataFrame) -> None:
        capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
        self.keys = keys
        self.ids = sorted(keys["time_series_id"].unique().to_list())
        cap = capacity.filter(pl.col("time_series_id").is_in(self.ids)).sort("time_series_id")[
            "capacity"
        ]
        cap = cap.to_numpy().astype(np.float64)
        grid = lib.cleaned_grid(ids=self.ids)
        self.row = (
            keys["time_series_id"].replace_strict({t: i for i, t in enumerate(self.ids)}).to_numpy()
        )
        vidx = lib.grid_index(times=keys["valid_time"])
        oidx = lib.grid_index(times=keys["power_fcst_init_time"])
        if vidx.max() >= grid.shape[1]:
            grid = np.pad(
                grid, ((0, 0), (0, int(vidx.max()) + 1 - grid.shape[1])), constant_values=np.nan
            )
        self.lead_h = (vidx - oidx) / 2.0
        self.month = keys["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
        self.origin_day = oidx // 48
        scale = 100.0 / cap[self.row]
        self.scale = scale
        self.y = grid[self.row, vidx] * scale
        starts = keys["power_fcst_init_time"].dt.truncate("1mo")
        cutoff = lib.grid_index(times=pl.Series(starts))
        self.q_heur = (
            lib.manual_heuristic(grid=grid, row=self.row, vidx=vidx, lead_h=self.lead_h)
            * scale[:, None]
        )
        self.q_clim = (
            lib.climatology(grid=grid, row=self.row, vidx=vidx, cutoff_idx=cutoff) * scale[:, None]
        )
        self.ok = (
            np.isfinite(self.y)
            & np.isfinite(self.q_heur).all(axis=1)
            & np.isfinite(self.q_clim).all(axis=1)
        )
        label_of = dict(
            zip(sites()["time_series_id"].to_list(), sites()["label"].to_list(), strict=True)
        )
        self.label = np.array([label_of[t] for t in self.ids])[self.row]

    def quantiles(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return a frame's 11 quantiles in points of capacity, in this object's row order."""
        assert frame.select(KEYS).equals(self.keys), "rows differ from the scored keys"
        return frame.select(QCOLS).to_numpy().astype(np.float64) * self.scale[:, None]


def align(
    *, frames: dict[tuple[str, int], pl.DataFrame]
) -> tuple[pl.DataFrame, dict[tuple[str, int], pl.DataFrame]]:
    """Restrict every (arm, seed) frame to the keys they all share, in one order."""
    keys = None
    for f in frames.values():
        k = f.select(KEYS)
        keys = k if keys is None else keys.join(k, on=list(KEYS), how="inner")
    assert keys is not None
    keys = keys.sort(KEYS)
    out = {
        n: keys.join(f.select(*KEYS, *QCOLS), on=list(KEYS), how="left", maintain_order="left")
        for n, f in frames.items()
    }
    return keys, out


def row_losses(*, y: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row CRPS, median absolute error, and each tail's pinball loss and coverage."""
    base = lib.losses(y=y, q=q)
    pin = lib.pinball(y=y, q=q)
    out = {"crps": base["crps"], "mae": base["mae"]}
    for name, i in TAILS.items():
        out[f"pin_{name}"] = pin[:, i]
        out[f"cov_{name}"] = (y <= q[:, i]).astype(np.float64) * 100.0
    return out


def mean_sorted(*, parts: list[np.ndarray]) -> np.ndarray:
    return np.sort(np.mean(parts, axis=0), axis=-1)


def seed_forecast(*, q: dict[tuple[str, int], np.ndarray], name: str, seed: int) -> np.ndarray:
    """Return the quantiles that the contrasts use for arm `name` at seed `seed`.

    `blend` is the equal-weight mean of global seed `s` and split-at-1 seed `s`, sorted. `pair` is
    the mean of global seeds `s` and `s + 3`, sorted. Every other name is the arm's own seed.
    """
    if name == "blend":
        return core.blend_quantiles(first=q[("global", seed)], second=q[("split1", seed)])
    if name == "pair":
        return mean_sorted(parts=[q[("global", seed)], q[("global", seed + len(core.SEEDS))]])
    return q[(name, seed)]


def run_score(*, months: tuple[str, ...]) -> None:
    arms = ("split1", "split2", "split3", "hybrid3")
    frames: dict[tuple[str, int], pl.DataFrame] = {}
    for s, fr in read_stored_677(seeds=core.SEEDS, months=months).items():
        frames[("global", s)] = fr
    for s, fr in read_backtest(
        name="global_pair", seeds=core.GLOBAL_PAIR_SEEDS, months=months
    ).items():
        frames[("global", s)] = fr
    for arm in arms:
        for s, fr in read_backtest(name=arm, seeds=core.SEEDS, months=months).items():
            frames[(arm, s)] = fr
    keys, aligned = align(frames=frames)
    rows = Rows(keys=keys)
    q = {n: rows.quantiles(frame=f) for n, f in aligned.items()}
    ok = rows.ok.copy()
    for arr in q.values():
        ok &= np.isfinite(arr).all(axis=1)
    headline = ok & (rows.lead_h < core.HEADLINE_LEAD_HOURS)
    out = SCORES / "four_months"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"month": rows.month, "scored": ok, "headline": headline}).group_by("month").agg(
        rows=pl.len(), scored=pl.col("scored").sum(), headline=pl.col("headline").sum()
    ).sort("month").write_csv(out / "row_counts.csv")

    # per-seed row losses of every name the contrasts use
    names = ("global", *arms, "blend", "pair")
    loss: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for name in names:
        for s in core.SEEDS:
            loss[(name, s)] = row_losses(
                y=rows.y, q=np.where(ok[:, None], seed_forecast(q=q, name=name, seed=s), 0.0)
            )
    # arm-level scores use the seed-mean forecast, as the plan states
    mean_q = {
        name: mean_sorted(parts=[seed_forecast(q=q, name=name, seed=s) for s in core.SEEDS])
        for name in names
    }
    mean_q["global six seeds"] = mean_sorted(parts=[q[("global", s)] for s in range(6)])
    mean_q["manual heuristic"], mean_q["climatology"] = rows.q_heur, rows.q_clim
    block = core.block_ids(origin_day=rows.origin_day, month=rows.month)
    table, tails = [], []
    for name, arr in mean_q.items():
        ls = row_losses(y=rows.y, q=np.where(ok[:, None], arr, 0.0))
        table.append({"arm": core.ARM_LABEL.get(name, name), "rows": int(headline.sum()),
                      "crps": float(ls["crps"][headline].mean()), "nmae_median": float(ls["mae"][headline].mean())})  # fmt: skip
        tails.append({"arm": core.ARM_LABEL.get(name, name), **{
            f"{kind}_{t}": float(ls[f"{kind}_{t}"][headline].mean()) for t in TAILS for kind in ("cov", "pin")
        }})  # fmt: skip
    pl.DataFrame(table).write_csv(out / "arms.csv")
    pl.DataFrame(tails).write_csv(out / "tails.csv")

    def contrast(*, a: str, b: str, mask: np.ndarray, metric: str = "crps") -> dict[str, float]:
        la = np.stack([loss[(a, s)][metric][mask] for s in core.SEEDS])
        lb = np.stack([loss[(b, s)][metric][mask] for s in core.SEEDS])
        return core.block_seed_bootstrap(loss_a=la, loss_b=lb, block=block[mask])

    d15, d18 = rows.label == "D15", rows.label == "D18"
    result = []
    for title, (a, b) in CONTRASTS.items():
        main = contrast(a=a, b=b, mask=headline)
        by_month = []
        for m in months:
            sel = headline & (rows.month == m)
            by_month.append(
                float(
                    np.mean(
                        [
                            (loss[(a, s)]["crps"][sel] - loss[(b, s)]["crps"][sel]).mean()
                            for s in core.SEEDS
                        ]
                    )
                )
            )
        no15, no18 = (
            contrast(a=a, b=b, mask=headline & ~d15),
            contrast(a=a, b=b, mask=headline & ~d18),
        )
        no_jan = contrast(a=a, b=b, mask=headline & (rows.month != "2026-01"))
        calendar = core.block_seed_bootstrap(
            loss_a=np.stack([loss[(a, s)]["crps"][headline] for s in core.SEEDS]),
            loss_b=np.stack([loss[(b, s)]["crps"][headline] for s in core.SEEDS]),
            block=np.unique(rows.month[headline], return_inverse=True)[1],
        )  # the second interval: whole calendar months and one seed
        result.append({
            "contrast": title, "mean": main["mean"], "lo": main["lo"], "hi": main["hi"],
            **{f"mean_{m}": v for m, v in zip(months, by_month, strict=True)},
            "hi_without_D15": no15["hi"], "hi_without_D18": no18["hi"], "lo_without_D15": no15["lo"], "lo_without_D18": no18["lo"],
            "mean_without_2026-01": no_jan["mean"], "hi_without_2026-01": no_jan["hi"],
            "month_interval_lo": calendar["lo"], "month_interval_hi": calendar["hi"],
            "verdict": core.verdict(
                hi=main["hi"], lo=main["lo"], month_means=by_month, hi_without_d15=no15["hi"], hi_without_d18=no18["hi"],
                lo_without_d15=no15["lo"], lo_without_d18=no18["lo"],
            ),
        })  # fmt: skip
    pl.DataFrame(result).write_csv(out / "contrasts.csv")

    per_primary, loo = [], []
    labels = [r["label"] for r in sites().iter_rows(named=True)]
    for label in labels:
        sel = headline & (rows.label == label)
        for title, (a, b) in CONTRASTS.items():
            c = contrast(a=a, b=b, mask=sel)
            per_primary.append(
                {
                    "primary": label,
                    "contrast": title,
                    "mean": c["mean"],
                    "lo": c["lo"],
                    "hi": c["hi"],
                }
            )
            c_out = contrast(a=a, b=b, mask=headline & (rows.label != label))
            loo.append(
                {
                    "left_out": label,
                    "contrast": title,
                    "mean": c_out["mean"],
                    "lo": c_out["lo"],
                    "hi": c_out["hi"],
                }
            )
    pl.DataFrame(per_primary).write_csv(out / "per_primary.csv")
    pl.DataFrame(loo).write_csv(out / "leave_one_out.csv")
    per = pl.DataFrame(per_primary)
    print(per.group_by("contrast").agg(improved=(pl.col("mean") < 0).sum()))

    bands, groups = [], []
    band_of = {
        b: (rows.lead_h >= lo) & (rows.lead_h < hi) & ok for b, (lo, hi) in lib.LEAD_BANDS.items()
    }
    for b, mask in band_of.items():
        for title, (a, b2) in CONTRASTS.items():
            bands.append({"band": b, "contrast": title, "mean": float(np.mean([(loss[(a, s)]["crps"][mask] - loss[(b2, s)]["crps"][mask]).mean() for s in core.SEEDS]))})  # fmt: skip
        for name, arr in mean_q.items():
            bands.append({"band": b, "contrast": f"CRPS of {core.ARM_LABEL.get(name, name)}", "mean": float(row_losses(y=rows.y, q=np.where(ok[:, None], arr, 0.0))["crps"][mask].mean())})  # fmt: skip
    pl.DataFrame(bands).write_csv(out / "lead_bands.csv")
    history = {
        r["label"]: r for r in csv.DictReader(open(core.ROOT / "history_first_clean_reading.csv"))
    }
    site = {r["label"]: r["group"] for r in sites().iter_rows(named=True)}
    history_group = {
        label: "D15 and D18"
        if label in {"D15", "D18"}
        else "D16 and D9"
        if label in {"D16", "D9"}
        else "six years or more"
        for label in history
    }
    for kind, group_of in (("metering group", site), ("history group", history_group)):
        for g in sorted(set(group_of.values())):
            sel = headline & np.isin(rows.label, [lab for lab, v in group_of.items() if v == g])
            for title, (a, b) in CONTRASTS.items():
                c = contrast(a=a, b=b, mask=sel)
                groups.append(
                    {
                        "grouping": kind,
                        "group": g,
                        "contrast": title,
                        "mean": c["mean"],
                        "lo": c["lo"],
                        "hi": c["hi"],
                    }
                )
    pl.DataFrame(groups).write_csv(out / "groups.csv")
    print("wrote", out, flush=True)


def run_match_stored(*, months: tuple[str, ...]) -> None:
    """Rescore experiment 677's stored file on experiment 677's own scored rows; its stored monthly CRPS must be reproduced to within 0.01 points.

    The observed power comes from the cleaned-power Delta table, which the production pipeline
    overwrites every 6 hours. The current table drops, as `initial_analogue_noise`, the readings of
    D15 before 2025-05-07 and of D18 before 2025-03-30, which experiment 677's scoring still had.
    The climatology of those two Primaries therefore has no history for some calendar months, and
    some rows that experiment 677 scored no longer have a climatology. The check therefore takes experiment 677's row set from its `rows.parquet` and
    reports the rows whose status changed, which must all be D15 or D18 rows that lost their
    climatology.
    """
    frames = read_stored_677(seeds=core.SEEDS, months=months)
    keys, aligned = align(frames={("m", s): f for s, f in frames.items()})
    rows = Rows(keys=keys)
    arr = mean_sorted(parts=[rows.quantiles(frame=f) for f in aligned.values()])
    finite = np.isfinite(rows.y) & np.isfinite(arr).all(axis=1)
    crps = lib.losses(y=np.where(finite, rows.y, 0.0), q=np.where(finite[:, None], arr, 0.0))[
        "crps"
    ]
    ours = pl.DataFrame(
        {
            "time_series_id": keys["time_series_id"].cast(pl.Int64),
            "origin_idx": lib.grid_index(times=keys["power_fcst_init_time"]).astype(np.int64),
            "valid_idx": lib.grid_index(times=keys["valid_time"]).astype(np.int64),
            "month": rows.month, "label": rows.label, "crps": crps, "finite": finite,
            "ok_now": rows.ok & finite,
            "climatology_now": np.isfinite(rows.q_clim).all(axis=1),
        }
    )  # fmt: skip
    folder = core.ROOT_677 / "scores" / "step5w_with_injected_sp1_s012"
    theirs = pl.read_parquet(
        folder / "rows.parquet", columns=["time_series_id", "origin_idx", "valid_idx", "ok"]
    ).with_columns(pl.col("time_series_id", "origin_idx", "valid_idx").cast(pl.Int64))
    joined = ours.join(
        theirs, on=["time_series_id", "origin_idx", "valid_idx"], how="inner", validate="1:1"
    )
    assert joined.height == ours.height, (
        "a stored forecast row is missing from experiment 677's rows"
    )
    changed = joined.filter(pl.col("ok") != pl.col("ok_now"))
    print("rows whose scored status changed since experiment 677 scored them:")
    print(changed.group_by("month", "label", "ok", "climatology_now").len().sort("month", "label"))
    explained = changed.filter(
        pl.col("ok") & ~pl.col("ok_now") & pl.col("finite") & ~pl.col("climatology_now")
        & pl.col("label").is_in(["D15", "D18"])
    )  # fmt: skip
    assert explained.height == changed.height, (
        "a row changed status for a reason other than D15's or D18's lost climatology"
    )
    assert joined.filter(pl.col("ok"))["finite"].all(), (
        "a row that experiment 677 scored has no truth now"
    )
    rescored = (
        joined.filter(pl.col("ok")).group_by("month").agg(crps=pl.col("crps").mean()).sort("month")
    )
    stored = (
        pl.read_csv(folder / "scores.csv")
        .filter(
            (pl.col("slice") == "origin month")
            & (pl.col("system") == "Model")
            & (pl.col("metric") == "crps")
        )
        .select(month=pl.col("group"), stored=pl.col("mean"))
    )
    result = rescored.join(stored, on="month").with_columns(
        difference=pl.col("crps") - pl.col("stored")
    )
    now = (
        joined.filter(pl.col("ok_now"))
        .group_by("month")
        .agg(crps_on_current_rows=pl.col("crps").mean())
    )
    result = result.join(now, on="month").join(
        changed.group_by("month").agg(rows_changed=pl.len()), on="month", how="left"
    )
    print(result)
    assert result.height == len(months), "a month is missing from the stored scores"
    worst = float(result["difference"].abs().max())  # ty: ignore[invalid-argument-type]
    SCORES.mkdir(parents=True, exist_ok=True)
    result.write_csv(SCORES / "match_stored.csv")
    assert worst <= 0.01, (
        f"the rescore differs from experiment 677's stored monthly scores by {worst:.4f} points"
    )
    print("MATCH OK: largest difference", worst)


def pilot_primaries() -> list[str]:
    """Return the first run's three Primaries by the plan's rule: the largest signed-MW Primary, the largest MVA-only Primary, and D15."""
    site = sites()
    signed = site.filter(pl.col("group") == "signed-MW six").sort("capacity_mw", descending=True)[
        "label"
    ][0]
    mva = site.filter(pl.col("group") == "MVA-only eight").sort("capacity_mw", descending=True)[
        "label"
    ][0]
    return [signed, mva, "D15"]


def run_pilot(*, files: list[str]) -> None:
    """Score backtest files (seed 0) on the first run's fold and three Primaries, beside climatology and the manual heuristic.

    Checks that every file beats climatology (the plan's guard against a broken fit), and prints each
    file's CRPS and normalised MAE of the median, which the batch diagnostic compares. Every number
    is interim.
    """
    month = core.PILOT_MONTH
    site = sites()
    chosen = pilot_primaries()
    ids = site.filter(pl.col("label").is_in(chosen))["time_series_id"].to_list()
    frames = {}
    for name in files:
        raw = pl.read_parquet(core.ROOT / "backtests" / f"{name}.parquet")
        frames[(name, 0)] = (
            raw.filter(pl.col("seed") == 0)
            .filter(pl.col("power_fcst_init_time").dt.strftime("%Y-%m") == month)
            .filter(pl.col("time_series_id").is_in(ids))
            .sort(KEYS)
        )
    keys, aligned = align(frames=frames)
    rows = Rows(keys=keys)
    mask = rows.ok & (rows.lead_h < core.HEADLINE_LEAD_HOURS)
    table = []
    for name in files:
        frame = aligned[(name, 0)]
        ls = row_losses(y=rows.y, q=np.where(rows.ok[:, None], rows.quantiles(frame=frame), 0.0))
        table.append({"file": name, "crps": float(ls["crps"][mask].mean()), "nmae_median": float(ls["mae"][mask].mean())})  # fmt: skip
    for name, arr in (("climatology", rows.q_clim), ("manual heuristic", rows.q_heur)):
        ls = row_losses(y=rows.y, q=np.where(rows.ok[:, None], arr, 0.0))
        table.append({"file": name, "crps": float(ls["crps"][mask].mean()), "nmae_median": float(ls["mae"][mask].mean())})  # fmt: skip
    result = pl.DataFrame(table)
    out = SCORES / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    result.write_csv(out / "pilot.csv")
    print("interim; fold", month, "Primaries", chosen, "rows", int(mask.sum()))
    print(result)
    clim = float(result.filter(pl.col("file") == "climatology")["crps"][0])
    beaten = {n: bool(float(result.filter(pl.col("file") == n)["crps"][0]) < clim) for n in files}
    print("beats climatology:", beaten)
    assert all(beaten.values()), "a fit does not beat climatology on the pilot fold"


def run_match_bank(*, month: str, seed: int) -> None:
    """Compare this experiment's bank run on the stored global models with experiment 677's own bank file.

    The file comes from `e00702_fit.py global bank --stored-models --suffix _stored`. The medians
    and the injected shifts of both files must agree row for row (the medians within
    `BANK_MATCH_TOLERANCE` of capacity, the shifts exactly).
    """
    mine = pl.read_parquet(
        core.ROOT
        / "arms"
        / "global_stored"
        / month
        / "pooled"
        / "work"
        / "bank"
        / f"{month}_d0_s{seed}.parquet"
    )
    theirs = pl.read_parquet(
        core.ROOT_677 / f"P0_D50_{month}_5w_sp1" / "work" / "bank" / f"{month}_d0_s{seed}.parquet"
    )
    k3 = ["time_series_id", "power_fcst_init_time", "valid_time"]
    mine, theirs = (f.with_columns(pl.col("time_series_id").cast(pl.Int64)) for f in (mine, theirs))
    joined = mine.join(theirs, on=k3, how="inner", suffix="_677")
    assert joined.height == mine.height == theirs.height, (
        joined.height,
        mine.height,
        theirs.height,
    )
    diffs = {
        col: float((joined[col] - joined[f"{col}_677"]).abs().max())  # ty: ignore[invalid-argument-type]
        for col in ("median_base", "median_bank", "lag_shift")
    }
    print(diffs)
    assert diffs["lag_shift"] == 0.0, "the injected shifts differ from experiment 677's"
    assert max(diffs["median_base"], diffs["median_bank"]) <= BANK_MATCH_TOLERANCE, diffs
    print("BANK MATCH OK")


def run_match_refit() -> None:
    """Compare the pilot fold's refit of the global arm (seed 0) with the stored seeds 0 to 2 on the same fold and Primaries."""
    month = core.PILOT_MONTH
    site = sites()
    chosen = pilot_primaries()
    ids = site.filter(pl.col("label").is_in(chosen))["time_series_id"].to_list()
    refit = (
        pl.read_parquet(core.ROOT / "backtests" / "global_pilot.parquet")
        .filter(pl.col("seed") == 0)
        .filter(pl.col("time_series_id").is_in(ids))
        .sort(KEYS)
    )
    stored = {
        s: f.filter(pl.col("time_series_id").is_in(ids))
        for s, f in read_stored_677(seeds=core.SEEDS, months=(month,)).items()
    }
    frames = {("stored", s): f for s, f in stored.items()} | {("refit", 0): refit}
    keys, aligned = align(frames=frames)
    rows = Rows(keys=keys)
    mask = rows.ok & (rows.lead_h < core.HEADLINE_LEAD_HOURS)
    score = {
        n: float(
            lib.losses(y=rows.y, q=np.where(rows.ok[:, None], rows.quantiles(frame=f), 0.0))[
                "crps"
            ][mask].mean()
        )
        for n, f in aligned.items()
    }
    seeds = [score[("stored", s)] for s in core.SEEDS]
    spread = max(seeds) - min(seeds)
    gap = abs(score[("refit", 0)] - score[("stored", 0)])
    print(
        {
            "chosen": chosen,
            "scores": score,
            "largest pairwise difference of stored seeds": spread,
            "refit versus stored seed 0": gap,
        }
    )
    assert gap <= spread, (
        "the refit differs from the stored seed 0 by more than the stored seeds differ from each other"
    )
    print("MATCH OK")


def run_bank(*, months: tuple[str, ...]) -> None:
    """Return each arm's copied share of the held-out injected shifts (least-squares slope of the median's change on the shift)."""
    labels = [r["label"] for r in sites().iter_rows(named=True)]
    out = []
    for arm in (*core.ARMS, "global"):
        for month in months:
            for seed in core.SEEDS:
                if arm == "global":
                    files = [
                        core.ROOT_677
                        / f"P0_D50_{month}_5w_sp1"
                        / "work"
                        / "bank"
                        / f"{month}_d0_s{seed}.parquet"
                    ]
                else:
                    files = [
                        core.ROOT
                        / "arms"
                        / arm
                        / month
                        / lab
                        / "work"
                        / "bank"
                        / f"{month}_d0_s{seed}.parquet"
                        for lab in labels
                    ]
                frame = pl.concat(
                    [
                        pl.read_parquet(
                            f,
                            columns=[
                                "power_fcst_init_time",
                                "valid_time",
                                "median_base",
                                "median_bank",
                                "lag_shift",
                            ],
                        )
                        for f in files
                    ]
                )
                frame = frame.filter(pl.col("lag_shift") != 0).with_columns(
                    lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
                    change=pl.col("median_bank") - pl.col("median_base"),
                )
                for lo, hi in ((0, 336), (0, 24), (24, 72), (72, 168), (168, 336)):
                    part = frame.filter((pl.col("lead_h") >= lo) & (pl.col("lead_h") < hi))
                    out.append({"arm": arm, "month": month, "seed": seed, "band_h": f"{lo}-{hi}", "rows": part.height,
                                "num": float((part["change"] * part["lag_shift"]).sum()), "den": float((part["lag_shift"] ** 2).sum())})  # fmt: skip
    table = (
        pl.DataFrame(out)
        .group_by("arm", "seed", "band_h")
        .agg(num=pl.col("num").sum(), den=pl.col("den").sum(), rows=pl.col("rows").sum())
    )
    table = (
        table.with_columns(follow=pl.col("num") / pl.col("den"))
        .group_by("arm", "band_h")
        .agg(
            follow_mean=pl.col("follow").mean(),
            follow_sd_over_seeds=pl.col("follow").std(),
            rows_per_seed=pl.col("rows").mean(),
        )
        .sort("band_h", "arm")
    )
    SCORES.mkdir(parents=True, exist_ok=True)
    table.write_csv(SCORES / "bank_follow.csv")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("score", "match-stored", "match-refit", "bank", "pilot", "match-bank")
    )
    parser.add_argument("--months", nargs="+", default=list(core.TEST_MONTHS))
    parser.add_argument("--files", nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    months = tuple(args.months)
    if args.mode == "score":
        run_score(months=months)
    elif args.mode == "match-stored":
        run_match_stored(months=months)
    elif args.mode == "match-refit":
        run_match_refit()
    elif args.mode == "pilot":
        assert args.files, "name the backtest files with --files"
        run_pilot(files=args.files)
    elif args.mode == "match-bank":
        run_match_bank(month=months[0], seed=args.seed)
    else:
        run_bank(months=months)


if __name__ == "__main__":
    main()
