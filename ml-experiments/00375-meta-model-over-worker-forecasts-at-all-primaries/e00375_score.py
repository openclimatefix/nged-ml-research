# ruff: noqa: E501
"""Score the meta-model against the 50/50 blend and each worker at every Primary (experiment 375).

Per-row quantile scores are in points of capacity. Intervals resample whole scored months (the same
months at every Primary), 20,000 resamples, seed 0. The pooled figure is the mean of the Primaries'
means over the Primaries with a stored GAM stack. Prints every number the write-up quotes and writes
`work/scores.json`.

    PYTHONPATH=<the folders listed in e00375_meta.py> uv run --no-sync python ml-experiments/00375-*/e00375_score.py
"""

import json
import os
from typing import Final

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

import e00252_common as common  # ty: ignore[unresolved-import]
import e00252_score as s252  # ty: ignore[unresolved-import]
import e00375_meta as meta
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

WORK: Final = meta.WORK
STACK_SITES: Final = tuple(s for s in meta.SITES if s not in ("D15", "D18"))
NO_STACK_SITES: Final = ("D15", "D18")
Q: Final = s252.QUANTILES
KEY: Final = s252.KEY
ARM_PREFIXES: Final[dict[str, str]] = {
    "meta": "meta__", "meta_nolag": "meta_nolag__", "meta_nocal": "meta_nocal__",
    "blend": "b50__", "single": "single__", "bg2": "bg2_", "gam": "gam__",
    "W": "only_W__", "M": "only_M__", "AN": "only_AN__", "PT": "only_PT__", "ST": "only_ST__",
    "climatology": "climatology_", "manual": "manual_",
}  # fmt: skip
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1_meta_minus_blend": ("meta", "blend"),
    "C2_meta_minus_best_single": ("meta", "single"),
    "C3_meta_minus_bg2": ("meta", "bg2"),
    "C4_meta_minus_nolag": ("meta", "meta_nolag"),
    "C5_meta_minus_nocal": ("meta", "meta_nocal"),
}
SLICES: Final = ("all", "event_day", "other_day", "cold_winter_night", "sunny_apr_jun")


def site_frame(*, cache: FeatureCache, site: str) -> pl.DataFrame:
    """Return the scored frame of one Primary with references, slice flags, and per-row scores."""
    frame = pl.read_parquet(WORK / f"meta_wide_{site}.parquet")
    for name in ("climatology", "manual"):
        ref = s252.reference_frame(name=name, site=site).rename(
            {f"{name}_{q}": f"{name}_{q}" for q in Q}
        )
        frame = frame.join(ref, on=KEY, how="left", validate="1:1")
    features = cache.sites[site]
    index = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    unit = features.pv_unit_shape[index]
    local_month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % 12 + 1
    night = features.elevation_deg[index] < -5
    daylight = unit > s252.SUN_UP
    edge = float(np.quantile(unit[daylight], 2 / 3))
    frame = frame.with_columns(
        all=pl.lit(True),
        other_day=~pl.col("event_day"),
        cold_winter_night=pl.Series(
            np.isin(local_month, [11, 12, 1, 2, 3]) & night & (features.temperature_c[index] < 2.0)
        ),
        sunny_apr_jun=pl.Series((unit >= edge) & np.isin(local_month, [4, 5, 6])),
    )
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        *[
            pl.Series(
                f"s_{arm}",
                s252.score(
                    actual=actual,
                    quantiles=frame.select([f"{p}{q}" for q in Q]).to_numpy().astype(np.float64),
                ),
            )
            for arm, p in ARM_PREFIXES.items()
            if f"{p}q01" in frame.columns
        ]
    )


def bootstrap(*, sums: np.ndarray, counts: np.ndarray) -> dict:
    """Resample months (seed 0); a Primary with no rows in the drawn months is left out of that draw."""
    rng = np.random.default_rng(0)
    n_months = sums.shape[1]
    draws = rng.integers(0, n_months, size=(s252.N_RESAMPLES, n_months))
    drawn_counts = counts[:, draws].sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        site_means = np.where(drawn_counts > 0, sums[:, draws].sum(axis=2) / drawn_counts, np.nan)
    pooled = np.nanmean(site_means, axis=0)
    point = sums.sum(axis=1) / np.maximum(counts.sum(axis=1), 1)
    low, high = np.nanpercentile(site_means, s252.INTERVAL, axis=1)
    return {
        "site": [(float(p), float(a), float(b)) for p, a, b in zip(point, low, high, strict=True)],
        "pooled": (
            float(point.mean()),
            float(np.nanpercentile(pooled, s252.INTERVAL[0])),
            float(np.nanpercentile(pooled, s252.INTERVAL[1])),
        ),
    }


def contrast(*, frames: dict[str, pl.DataFrame], column: str, flag: str) -> dict:
    """Return the paired month-resampled contrast of per-row differences over a slice (seed 0)."""
    months = sorted({m for f in frames.values() for m in f["month"].unique().to_list()})
    sums = np.zeros((len(frames), len(months)))
    counts = np.zeros_like(sums)
    for i, f in enumerate(frames.values()):
        part = f.filter(pl.col(flag) & pl.col(column).is_finite())
        for month, total, count in (
            part.group_by("month").agg(s=pl.col(column).sum(), n=pl.len()).iter_rows()
        ):
            sums[i, months.index(month)] = total
            counts[i, months.index(month)] = count
    return bootstrap(sums=sums, counts=counts) | {"sites": list(frames)}


def main() -> None:  # noqa: PLR0912
    """Score every Primary, then print and write the tables."""
    cache = common.load_cache()
    frames = {site: site_frame(cache=cache, site=site) for site in meta.SITES}
    for site, f in frames.items():
        if f["climatology_q50"].null_count() or f["manual_q50"].null_count():
            frames[site] = f.filter(
                pl.col("climatology_q50").is_not_null() & pl.col("manual_q50").is_not_null()
            )
    out: dict = {"rows": {}, "absolute": {}, "contrast": {}, "checks": {}}
    for site, f in frames.items():
        out["rows"][site] = f.height
        out["absolute"][site] = {
            arm: float(f[f"s_{arm}"].to_numpy().mean())
            for arm in ARM_PREFIXES
            if f"s_{arm}" in f.columns
        }
        actual = f["actual"].to_numpy().astype(np.float64)
        out["checks"][site] = {
            "median_minus_blend_median": float((f["meta__q50"] - f["b50__q50"]).to_numpy().mean()),
            "mae_meta": float(np.abs(actual - f["meta__q50"].to_numpy()).mean()),
            "mae_blend": float(np.abs(actual - f["b50__q50"].to_numpy()).mean()),
            "mae_single": float(np.abs(actual - f["single__q50"].to_numpy()).mean()),
            "mae_bg2": float(np.abs(actual - f["bg2_q50"].to_numpy()).mean()),
            "tails": {
                arm: {
                    q: {
                        "below": float((actual <= f[f"{ARM_PREFIXES[arm]}{q}"].to_numpy()).mean()),
                        "pinball": float(
                            np.mean(
                                np.maximum(
                                    lv * (actual - f[f"{ARM_PREFIXES[arm]}{q}"].to_numpy()),
                                    (lv - 1) * (actual - f[f"{ARM_PREFIXES[arm]}{q}"].to_numpy()),
                                )
                            )
                        ),
                    }
                    for q, lv in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99))
                }
                for arm in ("meta", "blend", "bg2")
            },
        }
    for name, (a, b) in CONTRASTS.items():
        for site, f in frames.items():
            frames[site] = f.with_columns((pl.col(f"s_{a}") - pl.col(f"s_{b}")).alias(name))
    for name in CONTRASTS:
        out["contrast"][name] = {}
        for flag in SLICES:
            stack = {s: frames[s] for s in STACK_SITES}
            nostack = {s: frames[s] for s in NO_STACK_SITES}
            res = contrast(frames=stack, column=name, flag=flag)
            out["contrast"][name][flag] = {
                "stack_sites": res,
                "no_stack_sites": contrast(frames=nostack, column=name, flag=flag),
            }
    wins = {}
    for name in ("C1_meta_minus_blend", "C2_meta_minus_best_single", "C3_meta_minus_bg2"):
        res = out["contrast"][name]["all"]["stack_sites"]
        wins[name] = {
            "below_zero": sum(p < 0 for p, _, _ in res["site"]),
            "wholly_below": sum(h < 0 for _, _, h in res["site"]),
            "wholly_above": sum(lo > 0 for _, lo, _ in res["site"]),
            "sites": len(res["site"]),
        }
    out["wins"] = wins
    leave = {}
    for left in STACK_SITES:
        rest = {s: frames[s] for s in STACK_SITES if s != left}
        leave[left] = contrast(frames=rest, column="C1_meta_minus_blend", flag="all")["pooled"]
    out["leave_one_out_c1"] = leave
    (WORK / "scores.json").write_text(json.dumps(out, indent=1))
    for name, by_flag in out["contrast"].items():
        for flag, res in by_flag.items():
            for kind, r in res.items():
                pooled = tuple(round(x, 3) for x in r["pooled"])
                per = {
                    s: tuple(round(x, 3) for x in v)
                    for s, v in zip(r["sites"], r["site"], strict=True)
                }
                print(f"{name}|{flag}|{kind}|pooled {pooled}|{per}")
    print("wins", wins)
    for site in meta.SITES:
        print(site, out["rows"][site], {k: round(v, 3) for k, v in out["absolute"][site].items()})
    print("leave one out C1", {k: tuple(round(x, 3) for x in v) for k, v in leave.items()})


if __name__ == "__main__":
    main()
