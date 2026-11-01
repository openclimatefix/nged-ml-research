"""Score G1 against B-mix, B1, and the in-sample control on the real-data world (no fit).

Every arm is read from a parquet of 11 quantiles on experiment 404's rows (leads of 0 to 240
hours). Row sets follow experiments 455 and 458: no-label rows (no labelled event in the day
windows,
the lags, or the target), harm rows (a day window inside a padded event, the target outside), and
real-event rows (the target inside a padded event or in the 96 half-hours after one). Real-event
rows are also scored against experiment 435's NRA-corrected actuals at depth factors 0.75, 1.0
(headline), and 1.25. The score is experiment 404's mass-weighted pinball loss in points of
capacity. A contrast is the first arm minus the second on the same rows, so negative is better.

```bash
PYTHONPATH=<as e00463_crossfit.py shows> DATA_PATH_INTERNAL=<private data folder> \
    uv run --no-sync python ml-experiments/00463-*/e00463_score.py cells
uv run --no-sync python ml-experiments/00463-*/e00463_score.py pool
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_report as report  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_404: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00404-mdn-lagless-and-calibration-module-trust-weights"
    / "work"
    / "predictions"
)
DATA_445: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00445-flag-oracle-o-rule-ceiling-for-lag-event-detectors"
    / "work"
    / "predictions"
)
NRA_SHARES: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00435-stage-0-recalibration-table-for-untrusted-inputs-and-tails"
    / "work"
    / "nra_correction_share.npz"
)
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00463-lag-trust-cross-fitted-trust-fit-seed-zero"
ARM_DIRS: Final[dict[str, Path]] = {
    "B-mix": DATA_404 / "B-mix",
    "B1": DATA_404 / "B1",
    "A0-inj": DATA_404 / "A0-inj",
    "lagless": DATA_445 / "O-lagless",
    "G1-insample": OUT_DIR / "predictions" / "G1-insample",
    "G1-trust": OUT_DIR / "predictions" / "G1-trust",
}
MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
GROUPS: Final[tuple[str, ...]] = ("signed_six", "mva_eight")
KEY: Final[list[str]] = ["primary", "origin", "target"]
LEVELS: Final[np.ndarray] = np.array(models.LEVELS)
LEVEL_NAMES: Final[tuple[str, ...]] = tuple(f"q{round(x * 100):02d}" for x in LEVELS)
ATOM_MASSES: Final[np.ndarray] = np.asarray(common.ATOM_MASSES, dtype=np.float64)
TAILS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
BAND_EDGES: Final[np.ndarray] = np.array([12.0, 36.0, 72.0, 120.0])
BAND_NAMES: Final[tuple[str, ...]] = ("0-12h", "12-36h", "36-72h", "days4-5", "days6-10")
DEPTHS: Final[tuple[str, ...]] = ("0.75", "1.0", "1.25")
SETS: Final[tuple[tuple[str, str], ...]] = (
    ("all", "metered"),
    ("nolabel", "metered"),
    ("harm", "metered"),
    ("event", "metered"),
    ("event", "0.75"),
    ("event", "1.0"),
    ("event", "1.25"),
)
"""Row set and actual of every scored slice; (event, 1.0) is the headline."""
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("G1-trust", "B-mix"),
    ("G1-trust", "B1"),
    ("G1-trust", "G1-insample"),
    ("G1-insample", "B-mix"),
    ("B1", "B-mix"),
    ("A0-inj", "B-mix"),
    ("lagless", "B-mix"),
)
"""The first three are the pre-registered contrasts; the rest are descriptive."""
CONTRAST_COUNT: Final[int] = 20
"""Pre-registered: G1-trust against B-mix (no-label, harm, event with NRA 1.0, event metered) at
both groups = 8, plus G1-trust against B1 and against G1-insample on the first three at both
groups = 12."""
RESAMPLES: Final[int] = 20000
MDE_MULTIPLIER: Final[float] = 2.8


def pinball_points(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return experiment 404's score in points of capacity: 200 x mass-weighted pinball loss."""
    diff = actual[:, None] - q
    loss = np.maximum(LEVELS * diff, (LEVELS - 1.0) * diff)
    return 200.0 * (loss * ATOM_MASSES).sum(axis=1)


def tail_stats(*, q: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return coverage and pinball loss at the four tail quantiles."""
    out: dict[str, float] = {}
    for name, tau in TAILS.items():
        qi = q[:, LEVEL_NAMES.index(name)]
        err = actual - qi
        out[f"cov_{name}"] = float((actual <= qi).mean())
        out[f"pin_{name}"] = float(np.where(err >= 0, tau * err, (tau - 1) * err).mean())
    return out


def event_clusters(*, primary: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an event id per row: the Primary and the run of consecutive event half-hours."""
    ids = np.zeros(len(primary), dtype=np.int64)
    for p in np.unique(primary):
        sel = np.flatnonzero(primary == p)
        uniq = np.unique(target[sel])
        run = np.concatenate([[0], np.cumsum(np.diff(uniq) > 1)])
        ids[sel] = int(p) * 10_000 + run[np.searchsorted(uniq, target[sel])]
    return ids


def load_arm(*, arm: str, month: str) -> pl.DataFrame:
    """Return an arm's forecasts for a month with leads of 0 to 240 hours and an observed actual."""
    frame = pl.read_parquet(ARM_DIRS[arm] / f"{month}_s0_w-1.parquet")
    return frame.filter((pl.col("lead_hours") <= 240) & pl.col("actual").is_finite())


def row_set(*, frame: pl.DataFrame, name: str) -> np.ndarray:
    """Return the mask of a row set: all, nolabel, harm, or event."""
    event = (frame["tgt_real"] | frame["tgt_post"]).to_numpy()
    if name == "event":
        return event
    if name == "harm":
        return frame["win_real"].to_numpy() & ~event
    if name == "nolabel":
        return ~(frame["win_real"] | frame["lag_real"]).to_numpy() & ~event
    return np.ones(frame.height, dtype=bool)


def cells() -> None:
    """Write one score row per arm, month, group, slice, and one contrast row per pair and slice."""
    npz = np.load(NRA_SHARES)
    shares = {d: npz[f"f{d}"] for d in DEPTHS}
    groups = report.group_of(sites=report.fit.prep.primaries()["site"].to_list())
    scores, contrasts = [], []
    for month in MONTHS:
        frames = {arm: load_arm(arm=arm, month=month) for arm in ARM_DIRS}
        ref = frames["B-mix"]
        for arm, frame in frames.items():
            if not frame.select(KEY).equals(ref.select(KEY)):
                msg = f"{arm} rows differ from B-mix in {month}"
                raise ValueError(msg)
        primary = ref["primary"].to_numpy().astype(np.int64)
        target = ref["target"].to_numpy().astype(np.int64)
        group = groups[primary]
        actual = ref["actual"].to_numpy().astype(np.float64)
        q = {a: f.select(LEVEL_NAMES).to_numpy().astype(np.float64) for a, f in frames.items()}
        crps: dict[tuple[str, str], np.ndarray] = {}
        for arm in frames:
            for depth in ("metered", *DEPTHS):
                a = actual if depth == "metered" else actual - shares[depth][primary, target]
                crps[arm, depth] = pinball_points(q=q[arm], actual=a)
        for g in GROUPS:
            for setname, depth in SETS:
                mask = (group == g) & row_set(frame=ref, name=setname)
                if not mask.any():
                    continue
                cluster = event_clusters(primary=primary[mask], target=target[mask])
                n_events = len(np.unique(cluster))
                for arm in frames:
                    row = {
                        "arm": arm, "month": month, "group": g, "set": setname, "actual": depth,
                        "n": int(mask.sum()), "events": n_events,
                        "score": float(crps[arm, depth][mask].mean()),
                    }  # fmt: skip
                    if depth == "metered":
                        row.update(tail_stats(q=q[arm][mask], actual=actual[mask]))
                    scores.append(row)
                for a, b in PAIRS:
                    diff = (crps[a, depth] - crps[b, depth])[mask]
                    codes, inverse = np.unique(cluster, return_inverse=True)
                    totals = np.abs(np.bincount(inverse, weights=diff, minlength=len(codes)))
                    keep = inverse != int(np.argmax(totals))
                    contrasts.append(
                        {
                            "a": a, "b": b, "month": month, "group": g, "set": setname,
                            "actual": depth, "n": int(mask.sum()), "events": n_events,
                            "diff": float(diff.mean()),
                            "diff_drop_largest": float(diff[keep].mean())
                            if setname == "event" and n_events > 1
                            else np.nan,
                        }
                    )  # fmt: skip
        print(f"{month} scored", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(scores).write_parquet(OUT_DIR / "scores.parquet")
    pl.DataFrame(contrasts).write_parquet(OUT_DIR / "contrasts.parquet")


def month_bootstrap(*, values: np.ndarray, level: float, seed: int = 0) -> tuple[float, float]:
    """Return the percentile interval of the mean of month values, resampling months."""
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def pool() -> None:
    """Pool the contrasts over months and write the table with intervals and the MDE."""
    table = pl.read_parquet(OUT_DIR / "contrasts.parquet")
    level = 1.0 - 0.05 / CONTRAST_COUNT
    mde: dict[tuple[str, str, str], float] = {}
    for key, part in table.filter((pl.col("a") == "A0-inj") & (pl.col("b") == "B-mix")).group_by(
        "group", "set", "actual"
    ):
        d = part["diff"].to_numpy()
        mde[key] = MDE_MULTIPLIER * float(d.std(ddof=1)) / np.sqrt(len(d))  # ty: ignore[invalid-argument-type]
    rows = []
    for key, part in table.group_by("a", "b", "group", "set", "actual", maintain_order=True):
        d = part["diff"].to_numpy()
        dd = part["diff_drop_largest"].drop_nans().to_numpy()
        lo95, hi95 = month_bootstrap(values=d, level=0.95)
        lob, hib = month_bootstrap(values=d, level=level)
        loo = [float(np.delete(d, i).mean()) for i in range(len(d))]
        rows.append(
            {
                "a": key[0], "b": key[1], "group": key[2], "set": key[3], "actual": key[4],
                "months": len(d), "mean": float(d.mean()), "low95": lo95, "high95": hi95,
                "low_bonf": lob, "high_bonf": hib, "lomo_min": min(loo), "lomo_max": max(loo),
                "months_negative": int((d < 0).sum()),
                "drop_largest_mean": float(dd.mean()) if len(dd) else np.nan,
                "sd_months": float(d.std(ddof=1)),
                "mde": mde[key[2], key[3], key[4]],  # ty: ignore[invalid-argument-type]
            }
        )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(OUT_DIR / "pooled.csv", float_precision=3)
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(20)
    print(f"Bonferroni contrast count: {CONTRAST_COUNT}, level {level:.5f}")
    print(out.filter(pl.col("actual").is_in(["metered", "1.0"])))


def weights() -> None:
    """Write the mean learned lagged weight by arm, group, lead band, and month."""
    rows = []
    for month in MONTHS:
        for arm in ("G1-trust", "G1-insample", "B1"):
            frame = load_arm(arm=arm, month=month)
            w = (
                frame["trust_w"].to_numpy()
                if arm != "B1"
                else frame.select(w=pl.col("trust_0") * pl.col("trust_1") * pl.col("trust_2"))[
                    "w"
                ].to_numpy()
            )
            groups = report.group_of(sites=report.fit.prep.primaries()["site"].to_list())
            group = groups[frame["primary"].to_numpy().astype(np.int64)]
            band = np.searchsorted(BAND_EDGES, frame["lead_hours"].to_numpy(), side="right")
            for g in GROUPS:
                for b, name in enumerate(BAND_NAMES):
                    sel = (group == g) & (band == b)
                    if sel.any():
                        rows.append(
                            {
                                "arm": arm, "month": month, "group": g, "band": name,
                                "weight": float(w[sel].mean()), "n": int(sel.sum()),
                            }
                        )  # fmt: skip
    out = pl.DataFrame(rows)
    out.write_csv(OUT_DIR / "weights.csv", float_precision=3)
    swing = out.group_by("arm", "group", "band").agg(
        swing=pl.col("weight").max() - pl.col("weight").min(), mean=pl.col("weight").mean()
    )
    print(swing.sort("arm", "group", "band"))


def checks() -> None:
    """Write the median-error contrasts and the leave-one-Primary-out event contrast.

    The harness stores quantiles only, so the mean forecast is not available; the error of the
    median (q50) forecast stands in for it (audit trigger 3). The leave-one-Primary-out rows
    (trigger 4) are rows-pooled over the five months, at the signed-MW six.
    """
    npz = np.load(NRA_SHARES)
    share = npz["f1.0"]
    groups = report.group_of(sites=report.fit.prep.primaries()["site"].to_list())
    median_rows, loo_rows = [], []
    pooled: dict[str, list] = {a: [] for a in ("G1-trust", "B-mix")}
    for month in MONTHS:
        frames = {a: load_arm(arm=a, month=month) for a in pooled}
        ref = frames["B-mix"]
        primary = ref["primary"].to_numpy().astype(np.int64)
        target = ref["target"].to_numpy().astype(np.int64)
        actual = ref["actual"].to_numpy().astype(np.float64)
        for setname in ("nolabel", "harm", "event"):
            mask = row_set(frame=ref, name=setname)
            for g in GROUPS:
                sel = mask & (groups[primary] == g)
                if not sel.any():
                    continue
                err = {
                    a: np.abs(f["q50"].to_numpy().astype(np.float64) - actual)[sel]
                    for a, f in frames.items()
                }
                median_rows.append(
                    {
                        "month": month, "group": g, "set": setname, "n": int(sel.sum()),
                        "mae_median_diff": float((err["G1-trust"] - err["B-mix"]).mean()),
                        "mae_median_bmix": float(err["B-mix"].mean()),
                    }
                )  # fmt: skip
        event = row_set(frame=ref, name="event") & (groups[primary] == "signed_six")
        a1 = actual - share[primary, target]
        for a, f in frames.items():
            q = f.select(LEVEL_NAMES).to_numpy().astype(np.float64)
            pooled[a].append(
                pl.DataFrame(
                    {
                        "month": month,
                        "primary": primary[event],
                        "crps": pinball_points(q=q[event], actual=a1[event]),
                    }
                )
            )
    out = pl.DataFrame(median_rows)
    out.write_csv(OUT_DIR / "median_error.csv", float_precision=4)
    diff = pooled["G1-trust"][0]
    both = pl.concat(
        [
            pl.concat(pooled["G1-trust"]).rename({"crps": "g1"}),
            pl.concat(pooled["B-mix"]).select(pl.col("crps").alias("bmix")),
        ],
        how="horizontal",
    ).with_columns(d=pl.col("g1") - pl.col("bmix"))
    del diff
    loo_rows.append({"left_out": "none", "rows": both.height, "diff": float(both["d"].mean())})  # ty: ignore[invalid-argument-type]
    for p in sorted(both["primary"].unique().to_list()):
        rest = both.filter(pl.col("primary") != p)
        loo_rows.append(
            {"left_out": int(p), "rows": rest.height, "diff": float(rest["d"].mean())}  # ty: ignore[invalid-argument-type]
        )
    loo = pl.DataFrame(loo_rows)
    loo.write_csv(OUT_DIR / "leave_one_primary_out.csv", float_precision=4)
    pl.Config.set_tbl_rows(100)
    print(loo)
    print(
        out.group_by("group", "set").agg(
            n=pl.col("n").sum(), mae_diff=pl.col("mae_median_diff").mean(),
            mae_bmix=pl.col("mae_median_bmix").mean(),
        )
    )  # fmt: skip


def main() -> None:
    """Run one step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cells", "pool", "weights", "checks"))
    args = parser.parse_args()
    {"cells": cells, "pool": pool, "weights": weights, "checks": checks}[args.command]()


if __name__ == "__main__":
    main()
