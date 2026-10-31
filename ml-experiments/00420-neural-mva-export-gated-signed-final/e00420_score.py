# ruff: noqa: E501
"""Score experiment 420's fits against the registered bars B1 to B5.

The primary contrast is the signed CRPS of the hidden pair's forecast on rows with reverse flow
(signed truth below zero, injected or real), in points of effective capacity, days 1 to 10. CRPS is
taken as twice the unweighted mean pinball loss over the 11 stored quantile levels (spaced
unequally, so it approximates the CRPS; paired contrasts are unaffected). Intervals come from a
paired month-block bootstrap: each block is one hidden Primary in one test month, resampled with
replacement, 20,000 times. Rows are not independent half-hours (each valid half-hour appears in
about 15 runs), so every count states whether it counts rows or unique half-hours.

```bash
. env.sh && uv run --no-sync python ml-experiments/00420-*/e00420_score.py mde-stage1   # before any stage-1 arm contrast
. env.sh && uv run --no-sync python ml-experiments/00420-*/e00420_score.py stage1       # stage 1 bars
. env.sh && uv run --no-sync python ml-experiments/00420-*/e00420_score.py mde-full     # then: ... full
```
"""

import argparse
import json
from pathlib import Path
from typing import Final

import e00412_models as m412  # ty: ignore[unresolved-import]
import e00417_inputs as inputs  # ty: ignore[unresolved-import]
import e00420_fit as fit
import numpy as np
import polars as pl
from scipy.stats import rankdata

OUT_DIR: Final[Path] = fit.OUT_DIR
FITS: Final[Path] = OUT_DIR / "fits"
LEVELS: Final[np.ndarray] = np.array(m412.LEVELS)
LEVEL_COLUMNS: Final[tuple[str, ...]] = fit.LEVEL_COLUMNS
KEY: Final[list[str]] = ["rotation", "month", "site", "origin", "valid"]
N_RESAMPLES: Final[int] = 20_000
MDE_Z: Final[float] = 2.8
"""80% power at 5% two-sided: 1.96 + 0.84."""
MDE_LIMIT_POINTS: Final[float] = 0.5
NO_HARM_POOLED: Final[float] = 0.05
NO_HARM_SINGLE: Final[float] = 0.20
PLACEBO_PRIMARIES: Final[tuple[str, ...]] = ("D5", "D9", "D11", "D14", "D16")
SUNNY_K: Final[float] = 0.7
DEAD_BAND_POINTS: Final[float] = 2.0
TAILS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
SEED: Final[int] = 420


def row_crps(*, frame: pl.DataFrame, truth: str) -> np.ndarray:
    """Return each row's CRPS, twice the mean pinball loss over the 11 quantile levels."""
    y = frame[truth].to_numpy()[:, None]
    q = frame.select(LEVEL_COLUMNS).to_numpy()
    diff = y - q
    return 2.0 * np.maximum(LEVELS[None, :] * diff, (LEVELS[None, :] - 1.0) * diff).mean(axis=1)


def read_arm(
    *,
    arm: str,
    setting: str,
    seeds: tuple[int, ...],
    eight: bool = False,
    rotations: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Return an arm's rows with a per-row CRPS, averaged over the seeds that exist."""
    tag = fit.SETTINGS[setting].tag
    suffix = ".eight.parquet" if eight else ".parquet"
    pieces = []
    for path in sorted((FITS / arm).glob(f"{tag}_r*_s*{suffix}")):
        if eight == (not path.name.endswith(".eight.parquet")):
            continue
        stem = path.name.removesuffix(suffix)
        _tag, rotation, month, seed_text = stem.split("_")[-4:]
        if int(seed_text[1:]) not in seeds or (rotations and rotation not in rotations):
            continue
        frame = pl.read_parquet(path).with_columns(
            pl.lit(rotation).alias("rotation"),
            pl.lit(month).alias("month"),
            pl.lit(int(seed_text[1:])).alias("seed"),
        )
        truth = "reading_points" if eight else "truth_points"
        frame = frame.filter(pl.col(truth).is_finite())
        pieces.append(frame.with_columns(pl.Series("crps", row_crps(frame=frame, truth=truth))))
    if not pieces:
        return pl.DataFrame()
    cat = pl.concat(pieces, how="diagonal")
    found = cat.group_by("rotation", "month").agg(pl.col("seed").n_unique().alias("n"))
    if (found["n"] != len(seeds)).any():
        msg = (
            f"{arm} {setting}: some rotation-months lack seeds {seeds}; wait for the run to finish"
        )
        raise ValueError(msg)
    agg = [pl.col("crps").mean()] + [
        pl.col(c).first() for c in cat.columns if c not in (*KEY, "crps", "seed")
    ]
    return cat.group_by(KEY, maintain_order=True).agg(agg)


def block_bootstrap(
    *, blocks: np.ndarray, total: np.ndarray, count: np.ndarray
) -> dict[str, float]:
    """Return the pooled mean of `total / count` with a 95% month-block bootstrap interval.

    `blocks` names each row group (a hidden Primary in a test month), `total` is the group's sum of
    the per-row difference, and `count` the number of rows in the group.
    """
    rng = np.random.default_rng(SEED)
    n = len(total)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n))
    den = count[idx].sum(axis=1)
    stat = np.where(den > 0, total[idx].sum(axis=1) / np.maximum(den, 1), np.nan)
    return {
        "mean": float(total.sum() / max(count.sum(), 1)),
        "lo": float(np.nanpercentile(stat, 2.5)),
        "hi": float(np.nanpercentile(stat, 97.5)),
        "sd": float(np.nanstd(stat)),
        "n_blocks": int(n),
        "n_rows": int(count.sum()),
    }


def contrast(
    *, reference: pl.DataFrame, candidate: pl.DataFrame, rows: pl.Expr | None = None
) -> dict[str, float]:
    """Return `mean(reference CRPS - candidate CRPS)` over the paired rows, positive when the candidate is better."""
    if not len(reference) or not len(candidate):
        return {"mean": float("nan")}
    both = reference.select(*KEY, pl.col("crps").alias("ref"), "truth_points").join(
        candidate.select(*KEY, pl.col("crps").alias("cand")), on=KEY, how="inner"
    )
    if rows is not None:
        both = both.filter(rows)
    if not len(both):
        return {"mean": float("nan")}
    grouped = (
        both.with_columns((pl.col("ref") - pl.col("cand")).alias("d"))
        .group_by("rotation", "month", "site")
        .agg(pl.col("d").sum().alias("total"), pl.len().alias("count"))
    )
    return block_bootstrap(
        blocks=grouped["site"].to_numpy(),
        total=grouped["total"].to_numpy(),
        count=grouped["count"].to_numpy().astype(np.float64),
    )


def per_site(
    *, reference: pl.DataFrame, candidate: pl.DataFrame, rows: pl.Expr | None
) -> dict[str, dict[str, float]]:
    """Return `contrast` for each hidden Primary separately."""
    out = {}
    for site in sorted(reference["site"].unique().to_list()):
        out[site] = contrast(
            reference=reference.filter(pl.col("site") == site),
            candidate=candidate.filter(pl.col("site") == site),
            rows=rows,
        )
    return out


REVERSE: Final[pl.Expr] = pl.col("truth_points") < 0.0


def mean_crps(*, frame: pl.DataFrame, rows: pl.Expr | None = None) -> float:
    """Return the mean CRPS of an arm's rows (optionally filtered)."""
    f = frame if rows is None else frame.filter(rows)
    return float(f["crps"].to_numpy().mean()) if len(f) else float("nan")


def auc(*, positive: np.ndarray, negative: np.ndarray) -> float:
    """Return the area under the receiver operating characteristic curve (ties count half)."""
    if not len(positive) or not len(negative):
        return float("nan")
    ranks = rankdata(np.concatenate([positive, negative]))
    return float(
        (ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def sign_metrics(*, frame: pl.DataFrame) -> dict:
    """Return descriptive sign metrics on the headline slice, with both false-reverse denominators.

    The headline slice is the real scaled CAMS unit shape at least 0.3 and the clear-sky index at least
    0.7. `false_reverse_slice` divides by every non-reverse slice row (what experiments 412 and 417
    printed). `false_reverse_sunny_midday` divides by the non-reverse slice rows in UTC summer middays
    (the registered denominator). Counts are rows, and unique half-hours are given beside them.
    """
    s = frame.filter((pl.col("u_gate") >= m412.GATE_U) & (pl.col("k") >= SUNNY_K))
    if not len(s):
        return {}
    truth = s["truth_points"].to_numpy() < 0.0
    p = s["p_negative"].to_numpy()
    pred = p > 0.5
    midday = inputs.is_midday(valid_us=s["valid"].to_numpy())
    near_zero = np.abs(s["truth_points"].to_numpy()) < DEAD_BAND_POINTS

    def rate(*, mask: np.ndarray) -> float:
        return float(pred[mask].mean()) if mask.any() else float("nan")

    recall = rate(mask=truth)
    false_slice = rate(mask=~truth)
    threshold_5 = float(np.percentile(p[~truth], 95.0)) if (~truth).any() else float("nan")
    return {
        "rows_slice": len(s),
        "half_hours_slice": int(s.select("site", "valid").unique().height),
        "rows_reverse": int(truth.sum()),
        "half_hours_reverse": int(s.filter(REVERSE).select("site", "valid").unique().height),
        "recall": recall,
        "false_reverse_slice": false_slice,
        "false_reverse_sunny_midday": rate(mask=~truth & midday),
        "n_sunny_midday_non_reverse_rows": int((~truth & midday).sum()),
        "false_reverse_dead_band_2pt": float(pred[~truth & ~near_zero].mean())
        if (~truth & ~near_zero).any()
        else float("nan"),
        "balanced_accuracy": (recall + 1.0 - false_slice) / 2.0,
        "auc": auc(positive=p[truth], negative=p[~truth]),
        "recall_at_5pc_false_reverse": float((p[truth] > threshold_5).mean())
        if truth.any()
        else float("nan"),
    }


def tail_metrics(*, frame: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Return the coverage and pinball loss at the tail levels on reverse-flow rows."""
    r = frame.filter(REVERSE)
    if not len(r):
        return {}
    truth = r["truth_points"].to_numpy()
    out = {}
    for tau in TAILS:
        q = r[f"q{round(100 * tau):02d}"].to_numpy()
        diff = truth - q
        out[f"{tau:g}"] = {
            "coverage": float(np.mean(truth <= q)),
            "pinball": float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff))),
        }
    return out


def fit_infos(*, arm: str, setting: str) -> dict[str, dict]:
    """Return each fit's stored gate decisions, capacities, learned variance scale, and epochs."""
    tag = fit.SETTINGS[setting].tag
    out = {}
    for path in sorted((FITS / arm).glob(f"{tag}_r*_s*.json")):
        out[path.stem] = json.loads(path.read_text())
    return out


def minimum_detectable_effect(*, rotations: tuple[str, ...] | None = None) -> dict:
    """Return the minimum detectable effect from A's two seeds at level 2.0 on reverse-flow rows."""
    a0 = read_arm(arm="A", setting="L2", seeds=(0,))
    a1 = read_arm(arm="A", setting="L2", seeds=(1,))
    if rotations:
        a0, a1 = (f.filter(pl.col("rotation").is_in(rotations)) for f in (a0, a1))
    null = contrast(reference=a0, candidate=a1, rows=REVERSE)
    return {
        "null_difference": null,
        "mde_points": MDE_Z * null["sd"],
        "underpowered_above": MDE_LIMIT_POINTS,
        "rotations": list(rotations) if rotations else "all with both seeds",
    }


def bars(*, seeds_ab: tuple[int, ...], rotations: tuple[str, ...], mde: float) -> dict:
    """Read B1 to B5 for AV and for A from the stored forecasts."""
    p0 = read_arm(arm="P0", setting="L2", seeds=(0,), rotations=rotations)
    out: dict = {
        "mde_points": mde,
        "rotations": list(rotations),
        "seeds_for_A_and_AV": list(seeds_ab),
    }
    cand = {
        name: read_arm(arm=name, setting="L2", seeds=seeds_ab, rotations=rotations)
        for name in ("A", "AV")
    }
    shuffle = read_arm(arm="A-shuffle", setting="L2", seeds=(0,), rotations=rotations)
    for name, frame in cand.items():
        if not len(frame):
            continue
        entry: dict = {}
        entry["b1_vs_p0_pooled"] = contrast(reference=p0, candidate=frame, rows=REVERSE)
        entry["b1_vs_p0_by_primary"] = per_site(reference=p0, candidate=frame, rows=REVERSE)
        entry["crps_reverse_rows"] = {
            "candidate": mean_crps(frame=frame, rows=REVERSE),
            "p0": mean_crps(frame=p0, rows=REVERSE),
        }
        pooled = entry["b1_vs_p0_pooled"]
        entry["b1_pass"] = bool(
            pooled["mean"] >= mde
            and pooled["lo"] > 0.0
            and all(
                v["mean"] >= mde and v["lo"] > 0.0 for v in entry["b1_vs_p0_by_primary"].values()
            )
        )
        if name == "AV":
            entry["b2_vs_a"] = contrast(reference=cand["A"], candidate=frame, rows=REVERSE)
            entry["b2_pass"] = bool(
                entry["b2_vs_a"]["mean"] >= mde and entry["b2_vs_a"]["lo"] > 0.0
            )
            entry["b2_underpowered"] = bool(mde > MDE_LIMIT_POINTS)
        if name == "A" and len(shuffle):
            a0 = read_arm(arm="A", setting="L2", seeds=(0,), rotations=rotations)
            entry["b3_vs_shuffle"] = contrast(reference=shuffle, candidate=a0, rows=REVERSE)
            entry["b3_pass"] = bool(
                entry["b3_vs_shuffle"]["mean"] >= mde and entry["b3_vs_shuffle"]["lo"] > 0.0
            )
        p0_placebo = read_arm(arm="P0", setting="L0", seeds=(0,), rotations=rotations)
        placebo = read_arm(arm=name, setting="L0", seeds=seeds_ab, rotations=rotations)
        if len(placebo) and len(p0_placebo):
            five = pl.col("site").is_in(PLACEBO_PRIMARIES)
            p0_five, cand_five = p0_placebo.filter(five), placebo.filter(five)
            pooled_harm = contrast(reference=p0_five, candidate=cand_five)
            single = per_site(reference=p0_five, candidate=cand_five, rows=None)
            # `contrast` is reference minus candidate, so harm (candidate worse) is its negative
            # and the interval ends swap.
            entry["b4_harm_pooled"] = {
                **pooled_harm,
                "mean": -pooled_harm["mean"],
                "lo": -pooled_harm["hi"],
                "hi": -pooled_harm["lo"],
            }
            entry["b4_harm_by_primary"] = {s: -v["mean"] for s, v in single.items()}
            entry["b4_pass"] = bool(
                -pooled_harm["mean"] <= NO_HARM_POOLED
                and all(-v["mean"] <= NO_HARM_SINGLE for v in single.values())
            )
        eight_p0 = read_arm(arm="P0", setting="L2", seeds=(0,), eight=True, rotations=rotations)
        eight = read_arm(arm=name, setting="L2", seeds=seeds_ab, eight=True, rotations=rotations)
        if len(eight) and len(eight_p0):
            both = eight_p0.select(*KEY, pl.col("crps").alias("ref")).join(
                eight.select(*KEY, pl.col("crps").alias("cand")), on=KEY
            )
            grouped = (
                both.with_columns((pl.col("cand") - pl.col("ref")).alias("d"))
                .group_by("rotation", "month", "site")
                .agg(pl.col("d").sum().alias("total"), pl.len().alias("count"))
            )
            harm = block_bootstrap(
                blocks=grouped["site"].to_numpy(),
                total=grouped["total"].to_numpy(),
                count=grouped["count"].to_numpy().astype(np.float64),
            )
            entry["b5_reading_harm_pooled"] = harm
            entry["b5_pass"] = bool(harm["mean"] <= NO_HARM_POOLED)
        out[name] = entry
    return out


def descriptive(*, rotations: tuple[str, ...]) -> dict:
    """Return the descriptive sign, tail, and fit-information tables for every arm and setting."""
    out: dict = {}
    for arm in fit.ARMS:
        for setting in fit.SETTINGS:
            seeds = (0,)
            frame = read_arm(arm=arm, setting=setting, seeds=seeds, rotations=rotations)
            if not len(frame):
                continue
            entry = {
                "sign_pooled": sign_metrics(frame=frame),
                "sign_by_primary": {
                    s: sign_metrics(frame=frame.filter(pl.col("site") == s))
                    for s in sorted(frame["site"].unique().to_list())
                },
                "tails_reverse_rows": tail_metrics(frame=frame),
                "crps_reverse_rows": mean_crps(frame=frame, rows=REVERSE),
                "crps_all_rows": mean_crps(frame=frame),
                "crps_all_rows_by_primary": {
                    s: mean_crps(frame=frame, rows=pl.col("site") == s)
                    for s in sorted(frame["site"].unique().to_list())
                },
            }
            infos = fit_infos(arm=arm, setting=setting)
            entry["variance_scale"] = {k: v["variance_scale"] for k, v in infos.items()}
            entry["gates"] = {k: v.get("gates") for k, v in infos.items()}
            entry["capacity_points"] = {k: v["capacity_points"] for k, v in infos.items()}
            out[f"{arm} {setting}"] = entry
    return out


def main() -> None:
    """Score a stage and write `scores.json` (or `mde.json`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mde-stage1", "mde-full", "stage1", "full"))
    args = parser.parse_args()
    if args.stage.startswith("mde"):
        rotations = ("r1",) if args.stage == "mde-stage1" else tuple(inputs.ROTATIONS)
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
    if args.stage == "stage1":
        result = {
            "bars": bars(seeds_ab=(0,), rotations=("r1",), mde=mde),
            "descriptive": descriptive(rotations=("r1",)),
        }
    else:
        result = {
            "bars": bars(seeds_ab=(0, 1), rotations=tuple(inputs.ROTATIONS), mde=mde),
            "descriptive": descriptive(rotations=tuple(inputs.ROTATIONS)),
        }
    (OUT_DIR / f"scores_{args.stage}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["bars"], indent=1))


if __name__ == "__main__":
    main()
