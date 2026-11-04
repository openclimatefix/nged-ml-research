# ruff: noqa: E501, PTH123, SIM115
"""Experiment 565, stage 1: design 6, a slow residual correction of stored forecasts (CPU, no fit).

For a base arm's forecast row with origin `o` and Primary `p`, the correction is the exponentially
weighted mean of the residuals (actual minus `q50`) of that Primary's earlier stored forecast rows
whose target is at most `o` minus 24 h and whose own origin is at most `o`. The weight of a row is
`0.5 ** (age_days / half_life)`, with age the days from the row's target to `o`. The weighted sum is
divided by `max(sum of weights, MIN_WEIGHT)`, so the correction shrinks to zero where history is
short. The same shift is added to all 11 quantiles.

Only the fold's own month supplies residuals (the stored forecasts cover nothing earlier).

Commands (each reads what the earlier one wrote under `data/ml-experiments/00565-*/work/`):

- `correct`: write the corrected quantiles of every base arm and half-life, and print the row-by-row
  24 h assertion and one example row.
- `score`: write per-row CRPS of every arm and the contrasts against the base arms.
- `leak`: the closed-form injected-shift leak of the correction (experiment 511's stress world).
"""

import argparse
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00454_score as score454  # ty: ignore[unresolved-import]
import e00470_build as build470  # ty: ignore[unresolved-import]
import e00470_heads as heads470  # ty: ignore[unresolved-import]
import e00470_score as score470  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00521_stats as s  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DD: Final[Path] = ML_EXPERIMENTS_DATA_DIR
W: Final[Path] = DD / "00565-drift-embedding-variants-six-designs" / "work"
W545: Final[Path] = (
    DD / "00545-modular-lag-free-demand-network-weather-and-datetime-encoders" / "work"
)
W559: Final[Path] = (
    DD / "00559-m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty" / "work"
)
SOURCES: Final[dict[str, Path]] = {
    "M0": W545 / "predictions" / "M0",
    "A1": W559 / "predictions" / "A1",
    "A10": W559 / "predictions" / "A10",
    "A11": W559 / "predictions" / "A11",
}
HALF_LIVES: Final[tuple[int, ...]] = (28, 90)
STEPS_PER_DAY: Final[int] = 48
MIN_AGE_STEPS: Final[int] = STEPS_PER_DAY
"""Residuals count only if their target is at least 24 h (48 half-hours) before the origin."""
MIN_WEIGHT: Final[float] = 48.0
LEVELS: Final[tuple[str, ...]] = build470.LEVELS
KEYS: Final[list[str]] = build470.KEYS
MONTHS: Final[tuple[str, ...]] = build470.MONTHS
assert MONTHS == ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06"), MONTHS
N_PRIMARIES: Final[int] = 16
BASE_COUNTED: Final[str] = "M0"
BANDS: Final[tuple[str, ...]] = ("24-48h", "48-72h", "72-120h")
TESTS: Final[int] = 12
"""Counted contrasts of stage 1: 2 half-lives x 2 row sets x 3 lead bands, all against M0."""


def arm_name(*, base: str, half_life: int) -> str:
    """Return the name of a corrected arm."""
    return f"D6-{base}-h{half_life}"


def month_frame(*, base: str, month: str, scored: pl.DataFrame) -> pl.DataFrame:
    """Return the base arm's quantiles joined to the scored rows' actuals, sorted by origin."""
    pred = pl.read_parquet(SOURCES[base] / f"{month}_c0.parquet")
    frame = scored.join(pred, on=KEYS, how="inner", validate="1:1")
    assert frame.height == scored.height, (base, month, frame.height, scored.height)
    return frame.sort("origin", "primary", "target")


def corrections(
    *, frame: pl.DataFrame, half_life: int, block: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the shift of every row and, if `block` is given, the weight share of block rows.

    The share is the sum of the weights of the rows whose target lies in `block` (grid indices,
    lower bound inclusive, upper bound exclusive) over `max(sum of weights, MIN_WEIGHT)`, per origin
    and Primary, as an array of shape (origins, primaries) in the order of the sorted origins.
    """
    org = frame["origin"].to_numpy().astype(np.int64)
    tgt = frame["target"].to_numpy().astype(np.int64)
    prim = frame["primary"].to_numpy().astype(np.int64)
    res = (frame["actual"] - frame["q50"]).to_numpy().astype(np.float64)
    shift = np.zeros(frame.height)
    origins = np.unique(org)
    share = np.zeros((len(origins), N_PRIMARIES)) if block is not None else None
    for i, o in enumerate(origins):
        use = (tgt <= o - MIN_AGE_STEPS) & (org <= o)
        assert not use.any() or int(tgt[use].max()) <= o - MIN_AGE_STEPS
        w = 0.5 ** ((o - tgt[use]) / (STEPS_PER_DAY * half_life))
        denom = np.maximum(np.bincount(prim[use], w, minlength=N_PRIMARIES), MIN_WEIGHT)
        mean = np.bincount(prim[use], w * res[use], minlength=N_PRIMARIES) / denom
        here = org == o
        shift[here] = mean[prim[here]]
        if share is not None and block is not None:
            inside = (tgt[use] >= block[0]) & (tgt[use] < block[1])
            share[i] = np.bincount(prim[use][inside], w[inside], minlength=N_PRIMARIES) / denom
    return shift, share


def correct() -> None:
    """Write corrected quantiles for every base arm and half-life; assert the 24 h rule."""
    heads470.WORK_DIR = W
    for month in MONTHS:
        scored = score454.month_base(month=month).select(*KEYS, "actual")
        for base in SOURCES:
            frame = month_frame(base=base, month=month, scored=scored)
            for hl in HALF_LIVES:
                shift, _ = corrections(frame=frame, half_life=hl)
                out = frame.with_columns(
                    *[(pl.col(q) + pl.Series(shift)).alias(q) for q in LEVELS]
                ).select(*KEYS, *LEVELS)
                build470.write_arm(
                    arm=arm_name(base=base, half_life=hl), month=month, cell=0, frame=out
                )
                if base == "M0" and month == MONTHS[0] and hl == 28:
                    pick = int(np.argmax(np.abs(shift) > 0))
                    row = frame.row(pick, named=True)
                    print(
                        "example row (M0, h28):",
                        {k: row[k] for k in (*KEYS, "actual", "q50")},
                        "shift",
                        shift[pick],
                    )
                    print(
                        "  every residual used has target <= origin - 48 half-hours (asserted per origin);"
                        " rows with shift != 0:",
                        int((shift != 0).sum()),
                        "of",
                        frame.height,
                    )
        print(month, "done", flush=True)


def load_scored() -> pl.DataFrame:
    """Return experiment 559's per-row table (M0, A1, A10, A11, A-pool, L-1d CRPS) joined to the new arms."""
    wanted = [f"{a}__crps" for a in ("A1", "A10", "A11", "A-pool", "L-1d", "manual_heuristic")]
    frames = []
    for month in MONTHS:
        rows = pl.read_parquet(
            W559 / "scores" / f"rows_{month}.parquet",
            columns=[*KEYS, "group", "cls", "lead_hours", "tgt_real", "cold", "main16", *wanted],
        )
        m0 = pl.read_parquet(W545 / "scores" / f"rows_{month}.parquet", columns=[*KEYS, "M0__crps"])
        rows = rows.join(m0, on=KEYS, how="left", maintain_order="left", validate="1:1")
        assert rows["M0__crps"].null_count() == 0
        base = score454.month_base(month=month).select(*KEYS, "actual")
        rows = rows.join(base, on=KEYS, how="left", maintain_order="left", validate="1:1")
        cols: dict[str, np.ndarray] = {}
        for b in SOURCES:
            pb = pl.read_parquet(SOURCES[b] / f"{month}_c0.parquet")
            fb = rows.select(*KEYS, "actual").join(pb, on=KEYS, how="left", maintain_order="left")
            fb = score470.widen_arm(arm=b, frame=fb, factor=score470.HEADLINE)
            diff = np.abs(score470.row_metrics(frame=fb)["crps"] - rows[f"{b}__crps"].to_numpy())
            assert np.nanmax(diff) < 1e-4, (b, month, np.nanmax(diff))
            for hl in HALF_LIVES:
                name = arm_name(base=b, half_life=hl)
                pred = pl.read_parquet(build470.prediction_path(arm=name, month=month, cell=0))
                fr = rows.select(*KEYS, "actual").join(
                    pred, on=KEYS, how="left", maintain_order="left"
                )
                ok = fr["q50"].is_not_null().to_numpy()
                fr = fr.with_columns(*[pl.col(q).fill_null(0.0) for q in LEVELS])
                fr = score470.widen_arm(arm=name, frame=fr, factor=score470.HEADLINE)
                crps = score470.row_metrics(frame=fr)["crps"]
                cols[f"{name}__crps"] = np.where(ok, crps, np.nan)
        rows = rows.hstack(
            pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()})
        )
        frames.append(
            rows.with_columns(
                month=pl.lit(month),
                day=(pl.col("origin") // 48).cast(pl.Int32),
                band=s.band_expr(),
            ).drop("actual")
        )
    return pl.concat(frames)


def score() -> None:
    """Write absolute CRPS and contrasts (corrected minus its base; corrected minus A-pool gap closed)."""
    t = load_scored()
    arms = list(SOURCES) + [arm_name(base=b, half_life=h) for b in SOURCES for h in HALF_LIVES]
    arms += ["A-pool", "L-1d", "manual_heuristic"]
    need = [f"{a}__crps" for a in arms if a != "manual_heuristic"]
    fin = pl.all_horizontal([pl.col(c).is_finite() for c in need])
    t = t.with_columns(main16=pl.col("main16") & fin)
    W.mkdir(parents=True, exist_ok=True)
    (W / "scores").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    absolute, contrasts = [], []
    for name in ("eight", "six", "report"):
        part = s.row_set(table=t, name=name)
        for band in (*BANDS, "24-120h"):
            sub = (
                part.filter(pl.col("band").is_in(BANDS))
                if band == "24-120h"
                else s.in_band(part=part, band=band)
            )
            if sub.height == 0:
                continue
            for arm in arms:
                col = f"{arm}__crps"
                if col in sub.columns:
                    absolute.append(
                        {
                            "set": name,
                            "band": band,
                            "arm": arm,
                            "n": sub.height,
                            "crps": float(sub[col].mean()),
                        }
                    )
            for b in SOURCES:
                for hl in HALF_LIVES:
                    new = f"{arm_name(base=b, half_life=hl)}__crps"
                    d = sub.select("month", "day", d=pl.col(new) - pl.col(f"{b}__crps")).filter(
                        pl.col("d").is_finite()
                    )
                    st = s.block_stats(diff=d, rng=rng, tests=TESTS)
                    contrasts.append(
                        {
                            "set": name,
                            "band": band,
                            "base": b,
                            "half_life": hl,
                            "contrast": f"{arm_name(base=b, half_life=hl)} - {b}",
                            "counted": b == BASE_COUNTED and band != "24-120h" and name != "report",
                            "tests": TESTS,
                            **st,
                        }
                    )
    pl.DataFrame(absolute).write_csv(W / "scores/design6_absolute.csv")
    pl.DataFrame(contrasts).write_csv(W / "scores/design6_contrasts.csv")
    gap = []
    absolute_frame = pl.DataFrame(absolute)
    for hl in HALF_LIVES:
        for b in SOURCES:
            for name in ("eight", "six"):
                a = absolute_frame.filter((pl.col("set") == name) & (pl.col("band") == "24-120h"))
                crps = dict(zip(a["arm"].to_list(), a["crps"].to_list(), strict=True))
                base_gap = crps[b] - crps["A-pool"]
                closed = (
                    (crps[b] - crps[arm_name(base=b, half_life=hl)]) / base_gap
                    if base_gap > 0
                    else math.nan
                )
                gap.append(
                    {
                        "base": b,
                        "half_life": hl,
                        "set": name,
                        "base_gap_to_A_pool": base_gap,
                        "share_closed": closed,
                    }
                )
    pl.DataFrame(gap).write_csv(W / "scores/design6_gap_closed.csv")
    json.dump(
        {"rows": t.height, "main16 with all arms finite": int(t["main16"].sum())},
        open(W / "scores/design6_checks.json", "w"),
        indent=1,
    )
    with pl.Config(tbl_rows=80):
        print(pl.DataFrame(gap).with_columns(pl.col(pl.Float64).round(3)))


def leak() -> None:
    """Write the closed-form injected-shift leak of the correction, in points of capacity."""
    clean = c510.clean_stack()
    rows = []
    for month in MONTHS:
        start = c510.month_start(month=month)
        end = start + timedelta(days=14)
        size = c510.shift_sizes(clean=clean, cutoff=end)
        to_idx = lambda t: (c510.us(t=t) - heads470.GRID_START_US) // heads470.STEP_US  # noqa: E731
        block = (
            int(to_idx(end - timedelta(days=7))),
            int(to_idx(end) - c510.CUTOFF_AGE_US // heads470.STEP_US),
        )
        scored = score454.month_base(month=month).select(*KEYS, "actual")
        for base in SOURCES:
            frame = month_frame(base=base, month=month, scored=scored)
            origins = np.unique(frame["origin"].to_numpy())
            for hl in HALF_LIVES:
                _, share = corrections(frame=frame, half_life=hl, block=block)
                assert share is not None
                stress_origins = [
                    o for o in origins if to_idx(end) <= o < to_idx(end + timedelta(days=7))
                ]
                assert len(stress_origins) == 7, (month, len(stress_origins))
                for age, o in enumerate(stress_origins):
                    i = int(np.searchsorted(origins, o))
                    rows.append(
                        {
                            "base": base,
                            "half_life": hl,
                            "month": month,
                            "block_age_days": age,
                            "leak": float(np.mean(100.0 * size * share[i])),
                        }
                    )
    t = pl.DataFrame(rows)
    out = (
        t.group_by("base", "half_life", "month")
        .agg(pl.col("leak").mean())
        .group_by("base", "half_life")
        .agg(leak=pl.col("leak").mean(), lo=pl.col("leak").min(), hi=pl.col("leak").max())
        .sort("base", "half_life")
    )
    t.write_csv(W / "scores/design6_leak_long.csv")
    out.write_csv(W / "scores/design6_leak.csv")
    print(out.with_columns(pl.col(pl.Float64).round(3)))


def main() -> None:
    """Run one command."""
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("correct", "score", "leak"))
    args = p.parse_args()
    {"correct": correct, "score": score, "leak": leak}[args.command]()


if __name__ == "__main__":
    main()
