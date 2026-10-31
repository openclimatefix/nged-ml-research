# ruff: noqa: E501
"""Score experiment 424's fits against the registered bars S1 to S3 and the descriptive tables.

The primary contrast is the signed CRPS of the hidden pair's forecast on rows with reverse flow
(signed truth below zero, injected or real), in points of effective capacity, days 1 to 10, A on ENS
inputs against P0 on ENS inputs. CRPS is twice the unweighted mean pinball loss over the 11 stored
quantile levels (it approximates the CRPS; paired contrasts are unaffected). Intervals come from
experiment 420's paired month-block bootstrap (`e00420_score.block_bootstrap`, 20,000 resamples,
blocks are one hidden Primary in one test month).

```bash
. /tmp/env424.sh   # or the PYTHONPATH and DATA_PATH_INTERNAL lines of e00424_fit.py
uv run --no-sync python ml-experiments/00424-*/e00424_score.py mde-stage1  # before any stage-1 arm contrast
uv run --no-sync python ml-experiments/00424-*/e00424_score.py stage1      # then: mde-full, full
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00417_inputs as i417  # ty: ignore[unresolved-import]
import e00420_score as s420  # ty: ignore[unresolved-import]
import e00424_fit as fit  # ty: ignore[unresolved-import]
import polars as pl

OUT_DIR: Final[Path] = fit.OUT_DIR
FITS: Final[Path] = OUT_DIR / "fits"
FITS_420: Final[Path] = s420.FITS
KEY: Final[list[str]] = s420.KEY
REVERSE: Final[pl.Expr] = s420.REVERSE
R1_POOLED_FLOOR: Final[float] = 4.0
FULL_POOLED_FLOOR: Final[float] = 10.0
"""S1 floors in points: about half the laboratory gain on r1 (8.29) and on all rotations (19.2)."""
LEAD_HOURS: Final[pl.Expr] = (pl.col("valid") - pl.col("origin")) / 3_600_000_000.0
"""Lead in hours from the join keys (the contrast frame does not carry the `lead` column)."""
LEAD_BANDS: Final[dict[str, tuple[float, float]]] = {
    "days 1 to 2": (0.0, 48.0),
    "days 3 to 6 (lead up to 144 hours)": (48.0, 144.0),
    "days 6 to 10 (lead above 144 hours)": (144.0, 240.01),
}
"""Lead bands in hours; the headline is all leads up to 240 hours (days 1 to 10)."""


def read_arm(
    *,
    arm: str,
    setting: str,
    seeds: tuple[int, ...],
    eight: bool = False,
    rotations: tuple[str, ...] | None = None,
    root: Path = FITS,
) -> pl.DataFrame:
    """Return an arm's rows with a per-row CRPS from `root` (experiment 424's fits, or 420's)."""
    old = s420.FITS
    s420.FITS = root
    try:
        return s420.read_arm(
            arm=arm, setting=setting, seeds=seeds, eight=eight, rotations=rotations
        )
    finally:
        s420.FITS = old


def minimum_detectable_effect(*, rotations: tuple[str, ...]) -> dict:
    """Return the minimum detectable effect from A's two seeds (ENS inputs) at level 2.0."""
    a0 = read_arm(arm="A", setting="L2", seeds=(0,), rotations=rotations)
    a1 = read_arm(arm="A", setting="L2", seeds=(1,), rotations=rotations)
    null = s420.contrast(reference=a0, candidate=a1, rows=REVERSE)
    return {
        "null_difference": null,
        "mde_points": s420.MDE_Z * null["sd"],
        "underpowered_above": s420.MDE_LIMIT_POINTS,
        "rotations": list(rotations),
    }


def harm(*, p0: pl.DataFrame, cand: pl.DataFrame, sites: list[str] | None) -> tuple[dict, dict]:
    """Return the pooled harm (candidate CRPS minus P0 CRPS, all rows) and the harm by Primary."""
    if sites is not None:
        p0, cand = (f.filter(pl.col("site").is_in(sites)) for f in (p0, cand))
    pooled = s420.contrast(reference=p0, candidate=cand)
    single = s420.per_site(reference=p0, candidate=cand, rows=None)
    flipped = {**pooled, "mean": -pooled["mean"], "lo": -pooled["hi"], "hi": -pooled["lo"]}
    return flipped, {s: -v["mean"] for s, v in single.items()}


def bars(*, seeds_a: tuple[int, ...], rotations: tuple[str, ...], mde: float) -> dict:
    """Read S1 to S3 for A on ENS inputs from the stored forecasts, plus the reported-only rows."""
    p0 = read_arm(arm="P0", setting="L2", seeds=(0,), rotations=rotations)
    a = read_arm(arm="A", setting="L2", seeds=seeds_a, rotations=rotations)
    out: dict = {"mde_points": mde, "rotations": list(rotations), "seeds_for_A": list(seeds_a)}
    s1 = s420.contrast(reference=p0, candidate=a, rows=REVERSE)
    by_primary = s420.per_site(reference=p0, candidate=a, rows=REVERSE)
    out["s1_vs_p0_pooled"] = s1
    out["s1_vs_p0_by_primary"] = by_primary
    out["s1_vs_p0_by_lead_band"] = {
        name: s420.contrast(
            reference=p0,
            candidate=a,
            rows=REVERSE & (lo <= LEAD_HOURS) & (hi > LEAD_HOURS),
        )
        for name, (lo, hi) in LEAD_BANDS.items()
    }
    out["crps_reverse_rows"] = {
        "A": s420.mean_crps(frame=a, rows=REVERSE),
        "P0": s420.mean_crps(frame=p0, rows=REVERSE),
    }
    floor = R1_POOLED_FLOOR if tuple(rotations) == ("r1",) else FULL_POOLED_FLOOR
    out["s1_pooled_floor_points"] = floor
    out["s1_interval_pass"] = bool(
        s1["mean"] >= mde
        and s1["lo"] > 0.0
        and all(v["mean"] >= mde and v["lo"] > 0.0 for v in by_primary.values())
    )
    out["s1_floor_met"] = bool(s1["mean"] >= floor)
    out["s1_pass"] = out["s1_interval_pass"] and out["s1_floor_met"]
    p0_placebo = read_arm(arm="P0", setting="L0", seeds=(0,), rotations=rotations)
    placebo = read_arm(arm="A", setting="L0", seeds=(0,), rotations=rotations)
    if len(p0_placebo) and len(placebo):
        pooled, single = harm(p0=p0_placebo, cand=placebo, sites=list(s420.PLACEBO_PRIMARIES))
        out["s2_harm_pooled"] = pooled
        out["s2_harm_by_primary"] = single
        out["s2_pass"] = bool(
            pooled["mean"] <= s420.NO_HARM_POOLED
            and all(v <= s420.NO_HARM_SINGLE for v in single.values())
        )
    eight_p0 = read_arm(arm="P0", setting="L2", seeds=(0,), eight=True, rotations=rotations)
    eight_a = read_arm(arm="A", setting="L2", seeds=seeds_a, eight=True, rotations=rotations)
    if len(eight_p0) and len(eight_a):
        pooled, _ = harm(p0=eight_p0, cand=eight_a, sites=None)
        out["s3_reading_harm_pooled"] = pooled
        out["s3_pass"] = bool(pooled["mean"] <= s420.NO_HARM_POOLED)
    out["reported_only"] = reported_only(a=a, p0=p0, rotations=rotations, seeds_a=seeds_a)
    return out


def reported_only(
    *, a: pl.DataFrame, p0: pl.DataFrame, rotations: tuple[str, ...], seeds_a: tuple[int, ...]
) -> dict:
    """Return experiment 420's laboratory rows and the applied-only diagnostic, with no bar."""
    out: dict = {}
    lab_a = read_arm(arm="A", setting="L2", seeds=seeds_a, rotations=rotations, root=FITS_420)
    lab_ref = read_arm(arm="ref", setting="L2", seeds=(0,), rotations=rotations, root=FITS_420)
    lab_p0 = read_arm(arm="P0", setting="L2", seeds=(0,), rotations=rotations, root=FITS_420)
    for name, frame in (("lab_A", lab_a), ("lab_ref", lab_ref), ("lab_P0", lab_p0)):
        if len(frame):
            out[f"crps_reverse_rows_{name}"] = s420.mean_crps(frame=frame, rows=REVERSE)
    if len(lab_a):
        diff = s420.contrast(reference=lab_a, candidate=a, rows=REVERSE)
        out["ens_A_minus_lab_A_reverse"] = {
            "note": "positive when ENS inputs score worse than the laboratory inputs",
            "mean": -diff["mean"],
            "lo": -diff["hi"],
            "hi": -diff["lo"],
        }
    applied = read_arm(arm=fit.APPLIED, setting="L2", seeds=(0,), rotations=rotations)
    if len(applied):
        a0 = read_arm(arm="A", setting="L2", seeds=(0,), rotations=rotations)
        out["applied_vs_p0"] = s420.contrast(reference=p0, candidate=applied, rows=REVERSE)
        out["applied_vs_trained_A"] = s420.contrast(reference=a0, candidate=applied, rows=REVERSE)
        out["crps_reverse_rows_applied"] = s420.mean_crps(frame=applied, rows=REVERSE)
    return out


def descriptive(*, rotations: tuple[str, ...]) -> dict:
    """Return the sign, tail, and gate-decision tables for every arm and setting."""
    out: dict = {}
    for arm in (*fit.ARMS, fit.APPLIED):
        for setting in fit.SETTINGS:
            frame = read_arm(arm=arm, setting=setting, seeds=(0,), rotations=rotations)
            if not len(frame):
                continue
            sites = sorted(frame["site"].unique().to_list())
            entry = {
                "sign_pooled": s420.sign_metrics(frame=frame),
                "sign_by_primary": {
                    s: s420.sign_metrics(frame=frame.filter(pl.col("site") == s)) for s in sites
                },
                "tails_reverse_rows": s420.tail_metrics(frame=frame),
                "crps_reverse_rows": s420.mean_crps(frame=frame, rows=REVERSE),
                "crps_all_rows": s420.mean_crps(frame=frame),
                "crps_all_rows_by_primary": {
                    s: s420.mean_crps(frame=frame, rows=pl.col("site") == s) for s in sites
                },
            }
            infos = {}
            tag = fit.SETTINGS[setting].tag
            for path in sorted((FITS / arm).glob(f"{tag}_r*_s0.json")):
                infos[path.stem] = json.loads(path.read_text())
            entry["gates"] = {k: v.get("gates") for k, v in infos.items()}
            entry["dropped_half_hours"] = {
                k: sum(v.get("dropped", {}).values()) for k, v in infos.items()
            }
            out[f"{arm} {setting}"] = entry
    return out


def main() -> None:
    """Score a stage and write `scores_<stage>.json` (or `mde_<stage>.json`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mde-stage1", "mde-full", "stage1", "full"))
    args = parser.parse_args()
    if args.stage.startswith("mde"):
        rotations = ("r1",) if args.stage == "mde-stage1" else tuple(i417.ROTATIONS)
        result = minimum_detectable_effect(rotations=rotations)
        path = OUT_DIR / f"{args.stage.replace('-', '_')}.json"
        if path.exists():
            msg = f"{path.name} exists: the figure is fixed once written"
            raise SystemExit(msg)
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    mde_file = OUT_DIR / ("mde_stage1.json" if args.stage == "stage1" else "mde_full.json")
    mde = json.loads(mde_file.read_text())["mde_points"]
    rotations = ("r1",) if args.stage == "stage1" else tuple(i417.ROTATIONS)
    seeds_a = (0,) if args.stage == "stage1" else (0, 1)
    result = {
        "bars": bars(seeds_a=seeds_a, rotations=rotations, mde=mde),
        "descriptive": descriptive(rotations=rotations),
    }
    (OUT_DIR / f"scores_{args.stage}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["bars"], indent=1))


if __name__ == "__main__":
    main()
