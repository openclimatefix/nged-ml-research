"""Score the stack at all 16 Primaries, with the MVA treatment as explicit arms.

Groups:

- `signed` (D5, D9, D10, D11, D14, D16): `stack`, this experiment's copy of experiment 252's chain
  with stage 2 (arm d).
- `mva` (D2, D3, D4, D8, D12, D19, D21, D22): `chosen2` and `n2`, experiment 374's rule-chosen and
  naive fits with stage 2, and `a1`, experiment 252's stored arm.
- `short` (D18): `stack` with the spread borrowed from the six signed-MW Primaries.

D15 has no ensemble store and so only the references are scored there (`reference_only`).
Rows, metric, bootstrap, and slices are experiment 374's; the extra slice `e7_step` holds the
half-hours that end at 00:30, 01:00, 07:30, and 08:00 UK local time (the Economy 7 off-peak
edges). Writes `work/scores.json` and prints every number the write-up quotes.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:\
ml-experiments/00374-mva-fold-stage-two-at-eight-primaries:\
ml-experiments/00376-all-primaries-gam-stack-with-mva-treatment-arms \
        uv run --no-sync python ml-experiments/00376-*/e00376_all_score.py
"""

import json
import os

os.environ.setdefault("E00376_SOURCE", "control")
os.environ.setdefault("E00376_ARM", "a0")

from pathlib import Path
from typing import Final

import e00374_score as e374  # ty: ignore[unresolved-import]
import e00376_common as common
import e00376_score as score_module
import e00376s2_common as s2common
import numpy as np
import polars as pl

SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA: Final[tuple[str, ...]] = common.SCORED_SITES
SHORT: Final[tuple[str, ...]] = ("D18",)
REFERENCE_ONLY: Final[tuple[str, ...]] = ("D15",)
QUANTILES: Final[tuple[str, ...]] = score_module.QUANTILES
KEY: Final[list[str]] = score_module.KEY
REFERENCES: Final[tuple[str, ...]] = tuple(score_module.REFERENCES)
SLICES: Final[tuple[str, ...]] = (*score_module.SLICES, "e7_step")
WORK: Final[Path] = common.DATA_DIR / "work"
STACK_ARM_D: Final[str] = "d"


def stack_frames(*, site: str) -> dict[str, pl.DataFrame]:
    """Return the stack (stage-2 arm d) scored-year frame at a signed-MW or short Primary."""
    frame = pl.read_parquet(s2common.out_path(arm=STACK_ARM_D, period="scored", site=site))
    return {"stack": frame.select(list(s2common.BASE_COLUMNS))}


def e7_flag(*, frame: pl.DataFrame) -> pl.Series:
    """Return whether each valid time is one of the four half-hours at the Economy 7 edges."""
    local = frame["valid_time"].dt.convert_time_zone("Europe/London")
    key = local.dt.hour().cast(pl.Int32) * 100 + local.dt.minute().cast(pl.Int32)
    return key.is_in([30, 100, 730, 800]).alias("e7_step")


def wide(*, site: str, frames: dict[str, pl.DataFrame], cache, base_arm: str) -> pl.DataFrame:  # noqa: ANN001
    """Return the rows every arm and reference share at one Primary, with scores and flags."""
    base = (
        frames[base_arm]
        .filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= score_module.HORIZON_HOURS)
        )
        .select([*KEY, "lead_hours", "actual", "run"])
    )
    for arm, frame in frames.items():
        base = base.join(
            frame.select(
                *KEY,
                pl.col("mean").alias(f"{arm}__mean"),
                *[pl.col(q).alias(f"{arm}__{q}") for q in QUANTILES],
            ),
            on=KEY,
            how="inner",
            validate="1:1",
        )
    for name in REFERENCES:
        base = base.join(
            score_module.reference_frame(name=name, site=site), on=KEY, how="inner", validate="1:1"
        )
    base = e374.with_flags(frame=base, cache=cache, site=site)
    base = base.with_columns(e7_step=e7_flag(frame=base))
    actual = base["actual"].to_numpy().astype(np.float64)
    names = (*frames, *REFERENCES)
    scores = [
        pl.Series(
            f"s_{name}",
            score_module.score(
                actual=actual,
                quantiles=base.select(
                    [f"{name}__{q}" for q in QUANTILES]
                    if name in frames
                    else [f"{name}_{q}" for q in QUANTILES]
                ).to_numpy(),
            ),
        )
        for name in names
    ]
    return base.with_columns(*scores)


def tails(*, w: pl.DataFrame, names: tuple[str, ...], frames: dict, flag: str) -> dict:
    """Return each arm's share of readings at or below q01, q05, q95, and q99 over a slice."""
    part = w.filter(pl.col(flag))
    actual = part["actual"].to_numpy()
    out: dict = {}
    for name in names:
        prefix = f"{name}__" if name in frames else f"{name}_"
        out[name] = {
            q: float((actual <= part[f"{prefix}{q}"].to_numpy()).mean())
            for q in ("q01", "q05", "q95", "q99")
        }
    return out


def group_scores(*, wides: dict[str, pl.DataFrame], arms: tuple[str, ...], pairs: dict) -> dict:
    """Return absolute scores, contrasts, tails, and leave-one-out for one group of Primaries."""
    sites = list(wides)
    out: dict = {"rows": {s: w.height for s, w in wides.items()}}
    out["absolute"] = {
        s: {
            flag: {
                n: float(w.filter(pl.col(flag))[f"s_{n}"].to_numpy().mean())
                for n in (*arms, *REFERENCES)
            }
            for flag in SLICES
        }
        for s, w in wides.items()
    }
    with_diffs = {
        s: w.with_columns(
            *[(pl.col(f"s_{a}") - pl.col(f"s_{b}")).alias(label) for label, (a, b) in pairs.items()]
        )
        for s, w in wides.items()
    }
    out["contrast"] = {
        label: {
            flag: score_module.contrast(frames=with_diffs, column=label, flag=flag)
            for flag in SLICES
        }
        for label in pairs
    }
    out["leave_one_out"] = {
        label: {
            left: score_module.contrast(
                frames={s: w for s, w in with_diffs.items() if s != left}, column=label, flag="all"
            )["pooled"]
            for left in sites
        }
        for label in pairs
        if len(sites) > 1
    }
    out["tails"] = {
        s: {
            flag: tails(w=w, names=(*arms, *REFERENCES), frames=dict.fromkeys(arms, 1), flag=flag)
            for flag in ("all", "sunny", "cold_night", "winter_night")
        }
        for s, w in wides.items()
    }
    out["checks"] = {
        s: {
            "median_minus_mean": {
                a: float((w[f"{a}__q50"] - w[f"{a}__mean"]).to_numpy().mean()) for a in arms
            },
            "mae_mean_forecast": {
                a: float((w["actual"] - w[f"{a}__mean"]).abs().to_numpy().mean()) for a in arms
            },
            "mae_bg2_q50": float((w["actual"] - w["bg2_q50"]).abs().to_numpy().mean()),
        }
        for s, w in wides.items()
    }
    return out


def reference_only_scores(*, site: str, cache) -> dict:  # noqa: ANN001
    """Return the references' absolute scores at a Primary that has no stack forecast."""
    frame = None
    for name in REFERENCES:
        part = score_module.reference_frame(name=name, site=site)
        frame = part if frame is None else frame.join(part, on=KEY, how="inner")
    return {"rows": frame.height if frame is not None else 0}


def main() -> None:
    """Score every group and write `work/scores.json`."""
    cache = common.load_cache()
    out: dict = {}
    signed = {
        s: wide(site=s, frames=stack_frames(site=s), cache=cache, base_arm="stack") for s in SIGNED
    }
    out["signed"] = group_scores(
        wides=signed,
        arms=("stack",),
        pairs={
            "stack_minus_bg2": ("stack", "bg2"),
            "stack_minus_manual": ("stack", "manual"),
            "stack_minus_clim": ("stack", "climatology"),
        },
    )
    mva = {}
    for s in MVA:
        frames = e374.site_arms(site=s)
        keep = {k: frames[k] for k in ("chosen2", "n2", "a1")}
        mva[s] = wide(site=s, frames=keep, cache=cache, base_arm="a1")
    out["mva"] = group_scores(
        wides=mva,
        arms=("chosen2", "n2", "a1"),
        pairs={
            "chosen_minus_bg2": ("chosen2", "bg2"),
            "naive_minus_bg2": ("n2", "bg2"),
            "chosen_minus_naive": ("chosen2", "n2"),
            "chosen_minus_a1": ("chosen2", "a1"),
            "a1_minus_bg2": ("a1", "bg2"),
        },
    )
    mva_no_d8 = {s: w for s, w in mva.items() if s != "D8"}
    with_diffs = {
        s: w.with_columns(
            c=pl.col("s_chosen2") - pl.col("s_bg2"), n=pl.col("s_n2") - pl.col("s_bg2")
        )
        for s, w in mva_no_d8.items()
    }
    out["mva_without_d8"] = {
        label: score_module.contrast(frames=with_diffs, column=column, flag="all")["pooled"]
        for label, column in (("chosen_minus_bg2", "c"), ("naive_minus_bg2", "n"))
    }
    short = {}
    for s in SHORT:
        short[s] = wide(site=s, frames=stack_frames(site=s), cache=cache, base_arm="stack")
    out["short"] = group_scores(
        wides=short,
        arms=("stack",),
        pairs={"stack_minus_bg2": ("stack", "bg2")},
    )
    out["reference_only"] = {s: reference_only_scores(site=s, cache=cache) for s in REFERENCE_ONLY}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    for group in ("signed", "mva", "short"):
        for label, by_flag in out[group]["contrast"].items():
            for flag, result in by_flag.items():
                p, lo, hi = result["pooled"]
                print(f"{group} {label} {flag}: {p:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print("mva without D8:", out["mva_without_d8"])


if __name__ == "__main__":
    main()
