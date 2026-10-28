# ruff: noqa: E501
"""Score the band heads against the stored forecasts and experiment 354's D8 control.

Frames at each site: `stored` (the stack's donated band), `arm2` and `arm3` (experiment 354's
own-residual bands), and the heads `x`, `m`, `xshuf`, `mshuf`. A head's rows before its first fit
(origins before 2025-08-01) keep the stored band. At D8 the control is `arm3` on the stack's fit `b`
monthly (gate 8.382).

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history:ml-experiments/00368-d8-conditional-band-head-on-gam-stack-residuals \
        uv run --no-sync python ml-experiments/00368-*/e00368_score.py
"""

import json
import os
from datetime import UTC, datetime

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from typing import Final

import e00252_common as common252  # ty: ignore[unresolved-import]
import e00252_score as score_module  # ty: ignore[unresolved-import]
import e00354_bands as bands  # ty: ignore[unresolved-import]
import e00354_lib as lib  # ty: ignore[unresolved-import]
import e00368_features as feat
import numpy as np
import polars as pl

HEADS: Final[tuple[str, ...]] = ("x", "m", "xshuf", "mshuf")
OTHERS: Final[tuple[str, ...]] = tuple(s for s in feat.SITES if s != "D8")
GATE_D8: Final[float] = 8.382
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "lead_0_24": (0.0, 24.0),
    "lead_24_72": (24.0, 72.0),
    "lead_72_120": (72.0, 120.0),
    "lead_120_168": (120.0, 168.0),
}
OUT_FILE: Final = feat.WORK / "scores.json"


def head_frame(*, stored: pl.DataFrame, head: pl.DataFrame) -> pl.DataFrame:
    """Return `stored` with the head's quantiles where the head has them."""
    renamed = head.select(*lib.KEY, *[pl.col(q).alias(f"h_{q}") for q in lib.QUANTILES])
    joined = stored.join(renamed, on=lib.KEY, how="left", validate="1:1")
    return joined.with_columns([pl.coalesce(f"h_{q}", q).alias(q) for q in lib.QUANTILES]).drop(
        [f"h_{q}" for q in lib.QUANTILES]
    )


def site_wide(*, site: str, cache, heads: dict[str, pl.DataFrame]) -> pl.DataFrame:  # noqa: ANN001
    """Return the site's shared rows with each arm's quantiles, scores, flags, and lead bands."""
    stored = feat.stored_frame(site=site)
    frames = {
        "stored": stored,
        "arm2": lib.own_residual_band(frame=stored),
        "arm3": lib.cell_residual_band(frame=stored),
    } | {
        name: head_frame(stored=stored, head=head.filter(pl.col("site") == site))
        for name, head in heads.items()
    }
    wide = bands.wide_frame(site=site, cache=cache, frames=frames)
    wide = wide.with_columns(
        *[
            ((pl.col("lead_hours") > lo) & (pl.col("lead_hours") <= hi)).alias(n)
            for n, (lo, hi) in LEAD_BANDS.items()
        ],
        (pl.col("power_fcst_init_time") >= pl.lit(datetime(2025, 8, 1, tzinfo=UTC))).alias(
            "fitted"
        ),
    )
    arms = list(frames)
    return wide.with_columns(
        *[(pl.col(f"s_{a}") - pl.col("s_stored")).alias(f"c_{a}_stored") for a in arms],
        *[(pl.col(f"s_{a}") - pl.col("s_arm3")).alias(f"c_{a}_arm3") for a in arms],
        *[(pl.col(f"s_{a}") - pl.col("s_bg2")).alias(f"g_{a}") for a in arms],
    )


def summarise(*, wide: pl.DataFrame, arms: list[str], capacity_mw: float) -> dict:
    """Return absolute scores by slice, tail shares, coverage by lead band, and mean-error checks."""
    names = [*arms, *score_module.REFERENCES]
    flags = ["all", "fitted", "sunny", "cold_night", "winter_night", *LEAD_BANDS]
    out: dict = {"rows": wide.height}
    out["absolute"] = {
        f: {n: float(wide.filter(pl.col(f))[f"s_{n}"].mean()) for n in names}  # ty: ignore[invalid-argument-type]
        for f in flags
    }
    out["tails"] = {
        f: {
            a: lib.coverage_and_tails(frame=wide.filter(pl.col(f)), prefix=f"{a}__")
            for a in [*arms]
        }
        | {"bg2": lib.coverage_and_tails(frame=wide.filter(pl.col(f)), prefix="bg2_")}
        for f in ("all", "fitted", *LEAD_BANDS)
    }
    out["checks"] = {
        "median_minus_mean": {
            a: float(np.mean((wide[f"{a}__q50"] - wide[f"{a}__mean"]).to_numpy())) for a in arms
        },
        "mae_median_mw": {
            a: float(np.abs((wide["actual"] - wide[f"{a}__q50"]).to_numpy()).mean())
            * capacity_mw
            / 100.0
            for a in arms
        },
        "mae_mean_mw": {
            a: float(np.abs((wide["actual"] - wide[f"{a}__mean"]).to_numpy()).mean())
            * capacity_mw
            / 100.0
            for a in arms
        },
    }
    return out


def main() -> None:
    """Score every site, then the planned contrasts, and write `work/scores.json`."""
    cache = common252.load_cache()
    heads = {
        h: pl.read_parquet(feat.WORK / "heads" / f"{h}.parquet")
        for h in HEADS
        if (feat.WORK / "heads" / f"{h}.parquet").exists()
    }
    wides = {s: site_wide(site=s, cache=cache, heads=heads) for s in feat.SITES}
    arms = ["stored", "arm2", "arm3", *heads]
    results: dict = {"sites": {}}
    for site, wide in wides.items():
        capacity = float(cache.sites[site].table.effective_capacity_mw)
        results["sites"][site] = summarise(wide=wide, arms=arms, capacity_mw=capacity)
    gate = results["sites"]["D8"]["absolute"]["all"]["arm3"]
    print("gate D8 arm3", round(gate, 3), "expected", GATE_D8)
    best = (
        min(("x", "m"), key=lambda h: results["sites"]["D8"]["absolute"]["all"][h])
        if {"x", "m"} <= set(heads)
        else "x"
    )
    results["best_head"] = best
    flags = ("all", "fitted", "lead_0_24", "lead_120_168")
    contrasts: dict = {}
    for head in list(heads):
        contrasts[head] = {
            "D8_vs_arm3": {
                f: lib.contrast_both(wide=wides["D8"], column=f"c_{head}_arm3", flag=f, site="D8")
                for f in flags
            },
            "D8_vs_stored": {
                f: lib.contrast_both(wide=wides["D8"], column=f"c_{head}_stored", flag=f, site="D8")
                for f in flags
            },
            "D2_vs_stored": {
                f: lib.contrast_both(wide=wides["D2"], column=f"c_{head}_stored", flag=f, site="D2")
                for f in flags
            },
            "D4_vs_stored": {
                f: lib.contrast_both(wide=wides["D4"], column=f"c_{head}_stored", flag=f, site="D4")
                for f in flags
            },
        }
        for label, sites in [("others", OTHERS)] + [
            (f"others_without_{o}", tuple(s for s in OTHERS if s != o)) for o in OTHERS
        ]:
            contrasts[head][f"{label}_vs_stored"] = {}
            for f in ("all", "fitted"):
                row = {}
                for lab, interval in (("ci95", (2.5, 97.5)), ("ci99", (0.5, 99.5))):
                    score_module.INTERVAL = interval
                    row[lab] = list(
                        score_module.contrast(
                            frames={s: wides[s] for s in sites}, column=f"c_{head}_stored", flag=f
                        )["pooled"]
                    )
                score_module.INTERVAL = (2.5, 97.5)
                contrasts[head][f"{label}_vs_stored"][f] = row
    contrasts["arm2"] = {
        site: {
            f: lib.contrast_both(wide=wides[site], column="c_arm2_stored", flag=f, site=site)
            for f in ("all",)
        }
        for site in feat.SITES
    }
    results["contrasts"] = contrasts
    results["others_mean_absolute"] = {
        a: float(np.mean([results["sites"][s]["absolute"]["all"][a] for s in OTHERS])) for a in arms
    }
    OUT_FILE.write_text(json.dumps(results, indent=2, default=float))
    for site in feat.SITES:
        print(site, {k: round(v, 3) for k, v in results["sites"][site]["absolute"]["all"].items()})


if __name__ == "__main__":
    main()
