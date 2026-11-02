"""Experiment 506, Part 4: a gated physical rule for reverse flow, with a stored capacity.

The rule calls a sunny-midday half-hour reverse flow when the unit PV shape `u` is at least 0.3 and
the expected solar output `C * u` exceeds the expected gross demand `D` (the dark-hour demand level
of the same month and day type; no temperature adjustment). `C` is experiment 474's AC capacity
(p10, p50, p90) of the latest season-year up to 2025, converted to points of capacity. **The test is
circular**, because that table was fitted on the signed power of the Primary it labels, and 474's
table covers only the six signed-MW Primaries. At level 2.0 the injected farm's size is added to C
(a register that lists the farm). The eight have no stored capacity, so for them the rule is applied
at the six's capacities and the output is the share of sunny half-hours it calls.

    PYTHONPATH=ml-experiments/00087-*:ml-experiments/00506-* \
        uv run --no-sync python ml-experiments/00506-*/e00506_part4.py
"""

from typing import Final

import e00506_data as data  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

U_GATE: Final[float] = 0.3
LEVELS: Final[tuple[float, ...]] = (0.0, 2.0)
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 506
TABLE: Final = (
    data.ML_EXPERIMENTS_DATA_DIR
    / "00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals"
    / "work"
    / "nged_table.parquet"
)


def capacity_points(*, site: str) -> dict[str, float]:
    """Return 474's p10, p50, and p90 AC capacity of the 2025 season-year, in points of capacity."""
    cap_mw = float(data.series_table().filter(pl.col("site") == site)["effective_capacity_mw"][0])
    row = (
        pl.read_parquet(TABLE)
        .filter((pl.col("site") == site) & (pl.col("season_year") == 2025))
        .row(0, named=True)
    )
    return {k: row[f"ac_{k}"] / cap_mw * 100 for k in ("p10", "p50", "p90")}


def half_hours(*, site: str, level: float) -> pl.DataFrame:
    """Return the sunny-midday half-hours with signed truth, shape, demand level, and block."""
    stamps = data.grid()
    p = data.power_points(site=site)
    u, k = data.cams(site=site)
    hour = data.local_hour_start(stamps=stamps)
    local = stamps.dt.offset_by("-30m").dt.convert_time_zone("Europe/London")
    month = local.dt.month().to_numpy()
    year = local.dt.year().to_numpy()
    weekend = (local.dt.weekday().to_numpy() >= 6).astype(int)
    window = (hour >= 10.0) & (hour < 15.0) & (np.nan_to_num(k) >= data.SUNNY_K) & np.isfinite(p)
    dark = (np.nan_to_num(u, nan=1.0) < 0.01) & np.isfinite(p)
    key = month * 2 + weekend
    lev = {
        int(c): float(np.median(p[dark & (key == c)]))
        for c in np.unique(key)
        if (dark & (key == c)).sum() > 50
    }
    demand = np.array([lev.get(int(c), np.nan) for c in key])
    signed, extra = p.copy(), 0.0
    if level > 0 and site in data.FARM_OF_SITE:
        farm = data.farm_norm(farm=data.FARM_OF_SITE[site])
        before = (stamps < data.LEVEL_WINDOW_END).to_numpy() & window & np.isin(month, range(4, 10))
        extra = level * float(np.median(np.abs(p[before])))
        signed = p - extra * farm
    keep = window & (np.nan_to_num(u) >= U_GATE) & np.isfinite(signed) & np.isfinite(demand)
    return pl.DataFrame(
        {
            "site": site,
            "level": level,
            "u": u[keep],
            "demand": demand[keep],
            "signed": signed[keep],
            "reading": np.abs(p[keep]),
            "block": (year[keep] * 100 + month[keep]).astype(int),
            "extra": extra,
        }
    )


def rates(
    *, called: np.ndarray, truth: np.ndarray, block: np.ndarray, rng: np.random.Generator
) -> dict[str, float | int]:
    """Return recall, false-reverse rate, and balanced accuracy with month-block 95% intervals."""
    ids, inv = np.unique(block, return_inverse=True)
    nb = len(ids)
    stack = np.zeros((nb, 4))
    for col, m in enumerate((called & truth, ~called & truth, called & ~truth, ~called & ~truth)):
        np.add.at(stack[:, col], inv, m)
    w = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(N_RESAMPLES)])
    out = {}
    for name, tot in (("point", stack.sum(0)), ("draws", w @ stack)):
        tp, fn, fp, tn = (tot[..., i] for i in range(4))
        with np.errstate(invalid="ignore", divide="ignore"):
            recall, fr = tp / (tp + fn), fp / (fp + tn)
        out[name] = (recall, fr, 0.5 * (recall + 1 - fr))
    res = {"n_truth_reverse": int(stack[:, 0].sum() + stack[:, 1].sum()), "n": int(stack.sum())}
    for i, name in enumerate(("recall", "false_reverse", "balanced_accuracy")):
        d = out["draws"][i]
        res[name] = float(out["point"][i])
        res[f"{name}_lo"] = float(np.nanpercentile(d, 2.5))
        res[f"{name}_hi"] = float(np.nanpercentile(d, 97.5))
    return res


def main() -> None:
    """Score the rule at the six, then apply it to the eight at the six's capacities."""
    rng = np.random.default_rng(SEED)
    rows = []
    for level in LEVELS:
        for site in data.SIGNED_SIX:
            frame = half_hours(site=site, level=level)
            cap = capacity_points(site=site)
            extra = float(frame["extra"][0])
            for tag in ("p10", "p50", "p90"):
                solar = (cap[tag] + extra) * frame["u"].to_numpy()
                called = solar > frame["demand"].to_numpy()
                truth = frame["signed"].to_numpy() < 0
                if truth.sum() == 0:
                    rows.append(
                        {
                            "level": level,
                            "site": site,
                            "capacity": tag,
                            "n": len(frame),
                            "n_truth_reverse": 0,
                            "false_reverse": float(called.mean()),
                            "recall": float("nan"),
                        }
                    )
                    continue
                rows.append(
                    {"level": level, "site": site, "capacity": tag}
                    | rates(called=called, truth=truth, block=frame["block"].to_numpy(), rng=rng)
                )
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    table = pl.DataFrame(rows, infer_schema_length=None)
    table.write_parquet(data.WORK / "part4_six.parquet")
    print("== rule at the six, per Primary (circular: 474's capacity is fitted on these rows)")
    print(
        table.select(pl.all().exclude("n_truth_reverse")).with_columns(pl.col(pl.Float64).round(3))
    )
    # Pooled over the six, p50 capacity.
    for level in LEVELS:
        parts = [half_hours(site=s, level=level) for s in data.SIGNED_SIX]
        called, truth, block = [], [], []
        for s, f in zip(data.SIGNED_SIX, parts, strict=True):
            cap = capacity_points(site=s)
            called.append(
                (cap["p50"] + float(f["extra"][0])) * f["u"].to_numpy() > f["demand"].to_numpy()
            )
            truth.append(f["signed"].to_numpy() < 0)
            block.append(f["block"].to_numpy())
        print(
            "pooled six, p50 capacity, level",
            level,
            rates(
                called=np.concatenate(called),
                truth=np.concatenate(truth),
                block=np.concatenate(block),
                rng=rng,
            ),
        )
    # The eight at the six's capacities (p50 range) and at twice the largest.
    caps = {s: capacity_points(site=s)["p50"] for s in data.SIGNED_SIX}
    lo, hi = min(caps.values()), max(caps.values())
    print(f"== six's 474 p50 capacities in points of capacity: {lo:.1f} to {hi:.1f} ({caps})")
    out = []
    for site in data.MVA_ONLY:
        f = half_hours(site=site, level=0.0)
        u, d = f["u"].to_numpy(), f["demand"].to_numpy()
        out.append(
            {"site": site, "n_sunny_gated": len(f)}
            | {
                f"share_called_C{c:.0f}": float(((c * u) > d).mean())
                for c in (lo, np.median(list(caps.values())), hi)
            }
            | {
                "median_demand_level_points": float(np.median(d)),
                "capacity_needed_median_points": float(np.median(d / u)),
            }
        )
    eight = pl.DataFrame(out)
    eight.write_parquet(data.WORK / "part4_eight.parquet")
    print(eight.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
