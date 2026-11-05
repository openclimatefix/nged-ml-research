# ruff: noqa: E501, PLR0912, PLR0915, D103, D107
r"""Score the arms of experiment 567 on every slice of the plan.

Forecasts are read from `work/seed_preds/<arm>[-<variant>]/<month>_s<seed>.parquet` (two fit
seeds, pooled by averaging the quantiles, as experiment 470 does). CRPS is in points of
capacity, from the 11 stored quantiles, against the metered actual (the clean series), or
against experiment 435's NRA-corrected actual (depth factor 1.0) inside labelled events.

Slices: all rows by lead band at the eight, the six, and the two report-only Primaries; rows
with no labelled event; rows whose target lies inside a labelled event (padded 1 day before and
after, plus the 2 days after), split by whether the recipient is named and is another Primary;
rows whose own lag lies inside a labelled event; held-out synthetic events (variants `hx` exact
conservation and `hi` 60% to 100%); every arm's lags-removed, aged, and neighbour-lags-removed
variants.

The Bonferroni count is `TESTS` (the counted contrasts of the plan). The minimum detectable
effect is 1.96 * sqrt(2) times the pooled within-month standard deviation of the contrast across
the two fit seeds, over sqrt(2), as in experiment 566's scorer.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_report as report404  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import e00560_data as d  # ty: ignore[unresolved-import]
import e00567_world as w  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = w.WORK
W: Final[Path] = WORK / "work"
W566: Final[Path] = (
    DD
    / "00566-combination-block-lag-encoder-with-three-step-schedule-and-embargo-real-event-leak/work"
)
NRA_SHARES: Final[Path] = (
    DD
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails/work/nra_correction_share.npz"
)
MONTHS: Final[tuple[str, ...]] = c.MONTHS
SEEDS: Final[tuple[int, ...]] = (0, 1)
TESTS: Final[int] = 10
EVENT_PAD: Final[int] = 48
POST_STEPS: Final[int] = 96
MAX_LEAD_HOURS: Final[float] = 240.0
BAND_24_120: Final[str] = "24-120h"
US_STEP: Final[int] = c.prep422.STEP_US
GRID0: Final[int] = c.prep422.GRID_START_US
LEVELS: Final[np.ndarray] = np.asarray(common.LEVELS, dtype=np.float64)
MASSES: Final[np.ndarray] = np.asarray(common.ATOM_MASSES, dtype=np.float64)
COLS: Final[list[str]] = list(common.LEVEL_COLUMNS)
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
VARIANTS: Final[tuple[str, ...]] = (
    "",
    "-removed",
    "-aged1",
    "-aged2",
    "-aged3",
    "-nbremoved",
    "-hx",
    "-hi",
)


def folder_of(*, arm: str) -> Path:
    """Return the folder holding an arm's forecasts.

    That is this experiment's work folder, or experiment 566's for C0 and C1.
    """
    return W566 if arm.split("-", maxsplit=1)[0] in {"C0", "C1"} else W


def pooled(*, arm: str, month: str, seeds: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Return the quantile average over the seeds of an arm (or arm-variant) for a month."""
    folder = folder_of(arm=arm) / "seed_preds" / arm
    frames = [
        pl.read_parquet(folder / f"{month}_s{sd}.parquet").sort(KEYS) for sd in (seeds or SEEDS)
    ]
    base = frames[0].select(KEYS)
    for fr in frames[1:]:
        assert fr.select(KEYS).equals(base), f"{arm} {month}: seeds cover different rows"
    mean = np.mean([fr.select(COLS).to_numpy().astype(np.float64) for fr in frames], axis=0)
    return base.hstack(pl.DataFrame(mean, schema=COLS, orient="row"))


def crps_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted pinball loss per row, in points of capacity (experiment 404's `pinball_points`)."""
    diff = actual[:, None] - q
    loss = np.maximum(LEVELS[None, :] * diff, (LEVELS[None, :] - 1.0) * diff)
    return 200.0 * (loss * MASSES[None, :]).sum(axis=1)


class Events:
    """Labelled events by Primary: padded spans, whether the recipient is named, and per-row masks."""

    def __init__(self) -> None:
        import e00400_blend as blend400  # ty: ignore[unresolved-import]
        import e00402_counterfactual as cf  # ty: ignore[unresolved-import]

        names = c.sites()
        table = pl.read_csv(blend400.WORK / "label_table_corrected.csv")
        onset = table["onset_utc"].str.to_datetime(time_zone="UTC").dt.epoch("us").to_numpy()
        first = table["first_primary"].to_list()
        named = table["n_named"].to_numpy()
        ev = pl.read_parquet(cf.EVENTS)
        self.spans: dict[int, list[tuple[int, int, bool]]] = {p: [] for p in range(len(names))}
        for r in ev.iter_rows(named=True):
            t_us = GRID0 + r["onset_idx"] * US_STEP
            near = [
                i for i, (f, o) in enumerate(zip(first, onset, strict=True))
                if f == r["label"] and abs(o - t_us) <= 6 * 3_600_000_000
            ]  # fmt: skip
            visible = bool(named[near[0]] >= 2) if near else True
            self.spans[names.index(r["label"])].append((r["onset_idx"], r["end_idx"], visible))
        self.names = names

    def mask(
        self, *, primary: np.ndarray, idx: np.ndarray, pad_before: int, pad_after: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (inside any span with the padding, inside a span whose recipient is named)."""
        inside = np.zeros(len(primary), dtype=bool)
        visible = np.zeros(len(primary), dtype=bool)
        for p, spans in self.spans.items():
            sel = primary == p
            if not sel.any():
                continue
            t = idx[sel]
            hit = np.zeros(len(t), dtype=bool)
            vis = np.zeros(len(t), dtype=bool)
            for onset, end, v in spans:
                h = (t >= onset - pad_before) & (t <= end + pad_after)
                hit |= h
                vis |= h & v
            inside[sel], visible[sel] = hit, vis
        return inside, visible


def month_rows(
    *, month: str, ev: Events, clean: np.ndarray, shares: np.ndarray, nbr: np.ndarray
) -> pl.DataFrame:
    """Return a month's scored rows with every mask the slices need (rows shared by all arms)."""
    keys = pl.read_parquet(W566 / "seed_preds/C1" / f"{month}_s0.parquet").sort(KEYS).select(KEYS)
    sites = prep.primaries().select("primary", "time_series_id")
    keys = keys.join(sites, on="time_series_id", how="left", maintain_order="left")
    primary = keys["primary"].to_numpy().astype(np.int64)
    origin = keys["power_fcst_init_time"].dt.epoch("us").to_numpy()
    valid = keys["valid_time"].dt.epoch("us").to_numpy()
    target_idx = (valid - GRID0) // US_STEP
    origin_idx = (origin - GRID0) // US_STEP
    day, week = d.lag_indices(target_idx=target_idx, origin_idx=origin_idx)
    lead = (valid - origin) / 3.6e9
    t_in, t_vis = ev.mask(
        primary=primary, idx=target_idx, pad_before=EVENT_PAD, pad_after=EVENT_PAD + POST_STEPS
    )
    lag_in = np.zeros(len(primary), dtype=bool)
    lag_vis = np.zeros(len(primary), dtype=bool)
    any_nb_event = np.zeros(len(primary), dtype=bool)
    for idx in (day, week):
        a, v = ev.mask(primary=primary, idx=idx, pad_before=EVENT_PAD, pad_after=EVENT_PAD)
        lag_in |= a
        lag_vis |= v
        for j in range(nbr.shape[1]):
            nid = nbr[primary, j]
            ok = nid >= 0
            a, _ = ev.mask(
                primary=np.where(ok, nid, 0), idx=idx, pad_before=EVENT_PAD, pad_after=EVENT_PAD
            )
            any_nb_event |= a & ok
    actual = clean[primary, np.clip(target_idx, 0, clean.shape[1] - 1)].astype(np.float64)
    nra = actual - shares[primary, np.clip(target_idx, 0, shares.shape[1] - 1)]
    groups = report404.group_of(sites=c.sites())[primary]
    month_s = pl.Series([month] * len(primary))
    days = ((valid - c.us(t=c.month_start(month=month))) // (86_400 * 1_000_000)).astype(np.int64)
    return pl.DataFrame(
        {"month": month_s, "day": days, "primary": primary, "target_idx": target_idx, "origin_idx": origin_idx,
         "day_idx": day, "week_idx": week, "lead_hours": lead, "group": groups, "actual": actual, "actual_nra": nra,
         "t_in": t_in, "t_vis": t_vis, "lag_in": lag_in, "lag_vis": lag_vis, "no_event": ~(t_in | lag_in | any_nb_event)}
    )  # fmt: skip


def scored(*, arm: str, month: str, rows: pl.DataFrame, nra: bool = False) -> np.ndarray:
    """Return per-row CRPS of the pooled arm on a month's rows (NaN where the actual is missing)."""
    fr = pooled(arm=arm, month=month)
    assert fr.height == rows.height, f"{arm} {month}: {fr.height} rows, expected {rows.height}"
    actual = rows["actual_nra" if nra else "actual"].to_numpy()
    return crps_points(q=fr.select(COLS).to_numpy().astype(np.float64), actual=actual)


def per_seed(
    *, arm: str, month: str, seed: int, rows: pl.DataFrame, nra: bool = False
) -> np.ndarray:
    """Return per-row CRPS of one seed's forecast."""
    folder = folder_of(arm=arm) / "seed_preds" / arm
    fr = pl.read_parquet(folder / f"{month}_s{seed}.parquet").sort(KEYS)
    actual = rows["actual_nra" if nra else "actual"].to_numpy()
    return crps_points(q=fr.select(COLS).to_numpy().astype(np.float64), actual=actual)


def mde(
    *,
    a: str,
    b: str,
    month_rows_: dict[str, pl.DataFrame],
    mask_name: str | None,
    band: tuple[float, float] | None,
    group: str | None,
    nra: bool,
) -> float:
    """Return the minimum detectable effect of the contrast `a - b` from its spread across the two fit seeds."""
    per_month: list[list[float]] = []
    for m in MONTHS:
        r = month_rows_[m]
        sel = np.ones(r.height, dtype=bool)
        if mask_name:
            sel &= r[mask_name].to_numpy()
        if group:
            sel &= (r["group"] == group).to_numpy()
        if band:
            lead = r["lead_hours"].to_numpy()
            sel &= (lead >= band[0]) & (lead < band[1])
        if not sel.any():
            continue
        per_month.append(
            [
                float(
                    np.nanmean(
                        (
                            per_seed(arm=a, month=m, seed=sd, rows=r, nra=nra)
                            - per_seed(arm=b, month=m, seed=sd, rows=r, nra=nra)
                        )[sel]
                    )
                )
                for sd in SEEDS
            ]
        )
    if not per_month:
        return float("nan")
    mm = np.array(per_month)
    f, k = mm.shape
    if k < 2:
        return float("nan")
    resid = mm - mm.mean(axis=1, keepdims=True)
    return 1.96 * math.sqrt(2) * math.sqrt(float((resid**2).sum()) / (f * (k - 1))) / math.sqrt(k)


def cluster_stats(
    *, diff: np.ndarray, cluster: np.ndarray, rng: np.random.Generator, tests: int = TESTS
) -> dict:
    """Return the mean of `diff` and event-cluster bootstrap intervals.

    The intervals are 95% and Bonferroni, and the number of clusters is returned too.
    """
    ok = np.isfinite(diff)
    diff, cluster = diff[ok], cluster[ok]
    codes, inverse = np.unique(cluster, return_inverse=True)
    total = np.bincount(inverse, weights=diff, minlength=len(codes))
    count = np.bincount(inverse, minlength=len(codes)).astype(float)
    out: dict = {
        "n": len(diff),
        "clusters": len(codes),
        "difference": float(diff.mean()) if len(diff) else float("nan"),
    }
    if len(codes) < 2:
        return out | {k: float("nan") for k in ("low", "high", "low_bonferroni", "high_bonferroni")}
    draws = rng.integers(0, len(codes), (s.RESAMPLES, len(codes)))
    boot = total[draws].sum(axis=1) / count[draws].sum(axis=1)
    out |= s.percentiles(boot=boot, tests=tests)
    biggest = int(np.argmax(np.abs(total)))
    keep = inverse != biggest
    out["difference_drop_largest"] = float(diff[keep].mean())
    return out


def heldout_mask(
    *, month: str, seed: int, variant: str, rows: pl.DataFrame, nbr: np.ndarray
) -> np.ndarray:
    """Return rows whose own or neighbour lag lies inside a synthetic held-out-pair event of that fold's world."""
    kind = {"hx": "exact", "hi": "imperfect"}[variant]
    spec = w.WorldSpec(kind=kind, rate="r21", month=month, seed=seed, gain=f"heldfold{month}")
    code = np.load(spec.file())["code"]
    primary = rows["primary"].to_numpy()
    hit = np.zeros(len(primary), dtype=bool)
    for idx in (rows["day_idx"].to_numpy(), rows["week_idx"].to_numpy()):
        i = np.clip(idx, 0, code.shape[1] - 1)
        hit |= (code[primary, i] != 0) & (idx >= 0)
        for j in range(nbr.shape[1]):
            nid = nbr[primary, j]
            hit |= (nid >= 0) & (code[np.where(nid >= 0, nid, 0), i] != 0) & (idx >= 0)
    return hit


def heldout_tables(
    *, arms: list[str], contrasts: list[str], month_rows_: dict[str, pl.DataFrame], nbr: np.ndarray
) -> None:
    """Score the held-out synthetic-event variants (`hx`, `hi`) per seed; write `heldout.csv` (arms) and `heldout_contrasts.csv`."""
    rng = np.random.default_rng(1)
    names = list(dict.fromkeys([*arms, *[x for p in contrasts for x in p.split(":")[:2]]]))
    absolute, out_c = [], []
    for var in ("hx", "hi"):
        for group in ("mva_eight", "signed_six"):
            vals: dict[str, list[np.ndarray]] = {n: [] for n in names}
            diffs: dict[str, list[np.ndarray]] = {p: [] for p in contrasts}
            cl: list[np.ndarray] = []
            for m in MONTHS:
                r = month_rows_[m]
                lead = r["lead_hours"].to_numpy()
                for sd in SEEDS:
                    sel = (
                        heldout_mask(month=m, seed=sd, variant=var, rows=r, nbr=nbr)
                        & (r["group"] == group).to_numpy()
                        & (lead >= 24)
                        & (lead < 120)
                    )
                    crps = {
                        n: per_seed(arm=f"{n}-{var}", month=m, seed=sd, rows=r)[sel] for n in names
                    }
                    for n in names:
                        vals[n].append(crps[n])
                    for p in contrasts:
                        a, b = p.split(":")[:2]
                        diffs[p].append(crps[a] - crps[b])
                    cl.append(
                        r["primary"].to_numpy()[sel] * 1000
                        + r["day"].to_numpy()[sel] // 14
                        + 100 * MONTHS.index(m)
                        + 10_000 * sd
                    )
            for n in names:
                v = np.concatenate(vals[n])
                absolute.append(
                    {
                        "variant": var,
                        "group": group,
                        "band": BAND_24_120,
                        "arm": n,
                        "n": len(v),
                        "crps": float(np.nanmean(v)),
                    }
                )
            for p in contrasts:
                a, b = p.split(":")[:2]
                out_c.append({"variant": var, "group": group, "band": BAND_24_120, "contrast": f"{a} - {b}", "tests": TESTS,
                              **cluster_stats(diff=np.concatenate(diffs[p]), cluster=np.concatenate(cl), rng=rng)})  # fmt: skip
    pl.DataFrame(absolute).write_csv(WORK / "scores/heldout.csv")
    pl.DataFrame(out_c).write_csv(WORK / "scores/heldout_contrasts.csv")


def main() -> None:
    global MONTHS, SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument(
        "--contrasts", nargs="*", default=[], help="a:b pairs; add :counted to count"
    )
    parser.add_argument(
        "--heldout", action="store_true", help="also score the -hx and -hi variants"
    )
    parser.add_argument("--months", nargs="+", default=list(MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    MONTHS, SEEDS = tuple(args.months), tuple(args.seeds)  # ty: ignore[invalid-assignment]
    arms = args.arms
    ev = Events()
    clean = c.clean_stack()
    shares = np.load(NRA_SHARES)["f1.0"]
    ctx_nbr = w.neighbour_index(graph=_graph(), labels=c.sites())
    month_rows_ = {
        m: month_rows(month=m, ev=ev, clean=clean, shares=shares, nbr=ctx_nbr) for m in MONTHS
    }
    absolute: list[dict] = []
    contrasts: list[dict] = []
    rng = np.random.default_rng(0)
    slices: dict[str, dict] = {
        "all": {"mask": None, "nra": False},
        "no_event": {"mask": "no_event", "nra": False},
        "target_in_event": {"mask": "t_in", "nra": True},
        "target_in_event_visible": {"mask": "t_vis", "nra": True},
        "lag_in_event": {"mask": "lag_in", "nra": False},
    }
    bands = {**s.BANDS, BAND_24_120: (24.0, 120.0), "all": (0.0, MAX_LEAD_HOURS + 1e-6)}
    cache: dict[tuple, np.ndarray] = {}

    def crps_of(arm: str, nra: bool) -> dict[str, np.ndarray]:
        key = (arm, nra)
        if key not in cache:
            cache[key] = {m: scored(arm=arm, month=m, rows=month_rows_[m], nra=nra) for m in MONTHS}  # ty: ignore[invalid-assignment]
        return cache[key]  # ty: ignore[invalid-return-type]

    names = list(dict.fromkeys([*arms, *[x for p in args.contrasts for x in p.split(":")[:2]]]))
    for sl, spec in slices.items():
        for group in ("mva_eight", "signed_six", "report_only"):
            for band, (lo, hi) in bands.items():
                for arm in names:
                    vals, _n = [], 0
                    for m in MONTHS:
                        r = month_rows_[m]
                        sel = (
                            (r["group"] == group).to_numpy()
                            & (r["lead_hours"].to_numpy() >= lo)
                            & (r["lead_hours"].to_numpy() < hi)
                        )
                        if spec["mask"]:
                            sel &= r[spec["mask"]].to_numpy()
                        v = crps_of(arm, spec["nra"])[m][sel]
                        vals.append(v[np.isfinite(v)])
                    allv = np.concatenate(vals)
                    if len(allv):
                        absolute.append(
                            {
                                "slice": sl,
                                "group": group,
                                "band": band,
                                "arm": arm,
                                "n": len(allv),
                                "crps": float(allv.mean()),
                            }
                        )
    for spec_text in args.contrasts:
        a, b, *rest = spec_text.split(":")
        counted = "counted" in rest
        for sl, spec in slices.items():
            for group in ("mva_eight", "signed_six"):
                for band in (BAND_24_120, "all"):
                    lo, hi = bands[band]
                    diffs, cl, month_col, day_col = [], [], [], []
                    for m in MONTHS:
                        r = month_rows_[m]
                        sel = (
                            (r["group"] == group).to_numpy()
                            & (r["lead_hours"].to_numpy() >= lo)
                            & (r["lead_hours"].to_numpy() < hi)
                        )
                        if spec["mask"]:
                            sel &= r[spec["mask"]].to_numpy()
                        dd = (crps_of(a, spec["nra"])[m] - crps_of(b, spec["nra"])[m])[sel]
                        diffs.append(dd)
                        month_col += [m] * int(sel.sum())
                        day_col.append(r["day"].to_numpy()[sel])
                        cl.append(
                            r["primary"].to_numpy()[sel] * 1000
                            + r["day"].to_numpy()[sel] // 14
                            + 100 * MONTHS.index(m)
                        )
                    dd, day_col, cl = (
                        np.concatenate(diffs),
                        np.concatenate(day_col),
                        np.concatenate(cl),
                    )
                    if not len(dd):
                        continue
                    if sl in {"all", "no_event"}:
                        frame = pl.DataFrame({"month": month_col, "day": day_col, "d": dd}).filter(
                            pl.col("d").is_finite()
                        )
                        stats = s.block_stats(diff=frame, rng=rng, tests=TESTS)
                    else:
                        stats = cluster_stats(diff=dd, cluster=cl, rng=rng)
                    contrasts.append({"slice": sl, "group": group, "band": band, "contrast": f"{a} - {b}",
                                      "counted": counted and band == BAND_24_120 and group == "mva_eight" and sl in {"target_in_event_visible", "no_event"},
                                      "tests": TESTS,
                                      "mde": mde(a=a, b=b, month_rows_=month_rows_, mask_name=spec["mask"], band=(lo, hi), group=group, nra=spec["nra"]),
                                      **stats})  # fmt: skip
    WORK.joinpath("scores").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(absolute).write_csv(WORK / "scores/absolute.csv")
    pl.DataFrame(contrasts).write_csv(WORK / "scores/contrasts.csv")
    if args.heldout:
        heldout_tables(arms=arms, contrasts=args.contrasts, month_rows_=month_rows_, nbr=ctx_nbr)
    print(json.dumps({"absolute": len(absolute), "contrasts": len(contrasts)}))


def _graph() -> dict[str, list[str]]:
    import e00402_worlds as worlds402  # ty: ignore[unresolved-import]

    return w.neighbour_graph(ctx=worlds402.load_context())


if __name__ == "__main__":
    main()
