# ruff: noqa: E501
"""Score experiment 430: seed-averaged arms, farm-equal means, month-block intervals at 99%.

Rows are experiment 310's scored-year runs that every compared arm holds. An arm's seeds are
averaged quantile by quantile, the loss is computed per row on sorted quantiles
(`e00310_score.row_losses`, points of capacity), and the contrast between two arms is the
percentage change of the farm-equal mean score, with the 12 scored months resampled in blocks.
Also printed: each arm's tail pinball losses and coverage, and the minimum detectable effect from
the spread of the seeds.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring \
        uv run --no-sync python ml-experiments/00430-*/e00430_score.py gate
        uv run --no-sync python ml-experiments/00430-*/e00430_score.py contrast ARM REF [ARM REF ...]
"""

import sys
from pathlib import Path
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_score as s310  # ty: ignore[unresolved-import]
import e00430_common as common
import numpy as np
import polars as pl

DRAWS: Final[int] = 10_000
LO_PCT, HI_PCT = 0.5, 99.5
QCOLS: Final[list[str]] = s310.QCOLS
KEYS: Final[list[str]] = ["time_series_id", "time", "init_time", "band"]
OUT_DIR: Final[Path] = common.DATA_DIR / "scores"
BAND_SETS: Final[dict[str, tuple[str, ...]]] = {
    "all": c310.BAND_ORDER,
    "d3_10": c310.BAND_ORDER[2:],
    "d6_10": c310.BAND_ORDER[4:],
    **{b: (b,) for b in c310.BAND_ORDER},
}
"""Row sets of the contrasts: every band, days 3 to 10 (A2), days 6 to 10 (A5), and each band."""
GATE_TOLERANCE_PERCENT: Final[float] = 0.5
"""The reproduction gate: experiment 345's solar-only pooled network, within 0.5% (the design note)."""


SEEDS: tuple[int, ...] = common.SEEDS_ABLATION
"""Seeds averaged for every arm of a contrast; the same set on both sides (set with `--seeds`)."""


def load_mine(*, arm: str) -> pl.DataFrame:
    """Return one arm's predictions with the seeds averaged quantile by quantile."""
    files = sorted(f for s in SEEDS for f in (common.PRED_DIR / arm).glob(f"s{s}_*.parquet"))
    if not files:
        msg = f"no predictions for {arm}"
        raise FileNotFoundError(msg)
    return average_seeds(frame=pl.concat([pl.read_parquet(f) for f in files]), arm=arm)


def load_reference(*, name: str) -> pl.DataFrame:
    """Return a stored reference arm: `e345` (the solar-only pooled network) or `blend310`."""
    if name == "e345":
        files = sorted(common.E345_PRED_DIR.glob("*/own_history/solar_pooled_s*.parquet"))
    else:
        files = sorted(c310.PRED_DIR.glob("*/own_history/blend_s*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files])
    return average_seeds(frame=frame, arm=name)


def average_seeds(*, frame: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return the frame with seeds averaged per row (a row with any non-finite quantile is dropped)."""
    frame = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
    frame = (
        frame.join(frame.group_by(KEYS).agg(pl.col("seed").n_unique().alias("_n")), on=KEYS)
        .filter(pl.col("_n") == pl.col("_n").max())
        .drop("_n")
    )
    return (
        frame.group_by(KEYS, maintain_order=True)
        .agg(
            pl.col("lead_hours").first(),
            pl.col("actual").first(),
            pl.col("seed").n_unique().alias("n_seeds"),
            *[pl.col(c).cast(pl.Float64).mean() for c in QCOLS],
        )
        .with_columns(arm=pl.lit(arm), setting=pl.lit("own_history"))
    )


def common_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Return the frames cut to the rows every frame holds, in one shared order."""
    shared = None
    for frame in frames.values():
        keys = frame.select(KEYS)
        shared = keys if shared is None else shared.join(keys, on=KEYS, how="inner")
    assert shared is not None
    return {n: shared.join(f, on=KEYS, how="left").sort(KEYS) for n, f in frames.items()}


def losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses with the month column added."""
    return s310.row_losses(frame=frame).with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def farm_equal_by_month(*, rows: pl.DataFrame, metric: str) -> tuple[np.ndarray, list[str]]:
    """Return the farm-equal mean of the metric in each month (farms weigh the same)."""
    per = rows.group_by("month", "time_series_id").agg(pl.col(metric).mean())
    months = sorted(per["month"].unique().to_list())
    values = np.array(
        [per.filter(pl.col("month") == m)[metric].mean() for m in months], dtype=np.float64
    )
    return values, months


def block_bootstrap(
    *, per_month_a: np.ndarray, per_month_b: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    """Return the percentage change of a against b and its 99% interval over month blocks."""
    rng = np.random.default_rng(seed)
    n = len(per_month_a)
    point = 100.0 * (per_month_a.mean() / per_month_b.mean() - 1.0)
    index = rng.integers(0, n, size=(DRAWS, n))
    draws = 100.0 * (per_month_a[index].mean(axis=1) / per_month_b[index].mean(axis=1) - 1.0)
    return float(point), float(np.percentile(draws, LO_PCT)), float(np.percentile(draws, HI_PCT))


def contrast_table(*, arms: dict[str, pl.DataFrame], pairs: list[tuple[str, str]]) -> pl.DataFrame:
    """Return, per pair and day band (and all bands), the percentage change and its 99% interval."""
    aligned = common_rows(frames=arms)
    per_arm = {n: losses(frame=f) for n, f in aligned.items()}
    out = []
    for arm, ref in pairs:
        for band, members in BAND_SETS.items():
            a = per_arm[arm].filter(pl.col("band").is_in(members))
            b = per_arm[ref].filter(pl.col("band").is_in(members))
            va, _ = farm_equal_by_month(rows=a, metric="score")
            vb, _ = farm_equal_by_month(rows=b, metric="score")
            point, lo, hi = block_bootstrap(per_month_a=va, per_month_b=vb)
            out.append(
                {
                    "arm": arm,
                    "ref": ref,
                    "band": band,
                    "score_arm": float(va.mean()),
                    "score_ref": float(vb.mean()),
                    "pin01_diff": float(a["pin01"].mean()) - float(b["pin01"].mean()),  # ty: ignore[invalid-argument-type]
                    "pin99_diff": float(a["pin99"].mean()) - float(b["pin99"].mean()),  # ty: ignore[invalid-argument-type]
                    "pct_change": point,
                    "lo99": lo,
                    "hi99": hi,
                    "rows": a.height,
                }
            )
    return pl.DataFrame(out)


def absolute_table(*, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return each arm's farm-equal score, tail pinball losses, and tail coverage over all bands."""
    aligned = common_rows(frames=arms)
    out = []
    for name, frame in aligned.items():
        rows = losses(frame=frame)
        record: dict[str, float | str] = {"arm": name, "rows": rows.height}
        for metric in (
            "score",
            "pin01",
            "pin99",
            "pin05",
            "pin95",
            "cover01",
            "cover99",
            "median_minus_mean",
        ):
            values, _ = farm_equal_by_month(rows=rows, metric=metric)
            record[metric] = float(values.mean())
        out.append(record)
    return pl.DataFrame(out)


def seed_spread(*, arm: str) -> dict[str, float]:
    """Return the minimum detectable effect, in percent, from the seed-to-seed spread of an arm.

    Each seed's full-year farm-equal score is computed on the rows every seed holds. The effect a
    paired comparison of two arms with n seeds each can resolve is 1.96 x sqrt(2) x sd / sqrt(n),
    as a percentage of the mean score. It covers seed noise only, so it is a lower bound.
    """
    files = sorted((common.PRED_DIR / arm).glob("s*.parquet"))
    frame = pl.concat([pl.read_parquet(f) for f in files])
    seeds = sorted(frame["seed"].unique().to_list())
    scores = []
    keys = None
    per_seed = {s: frame.filter(pl.col("seed") == s) for s in seeds}
    for s in seeds:
        k = per_seed[s].select(KEYS)
        keys = k if keys is None else keys.join(k, on=KEYS, how="inner")
    assert keys is not None
    for s in seeds:
        rows = losses(
            frame=keys.join(per_seed[s], on=KEYS, how="left").with_columns(
                arm=pl.lit(arm), setting=pl.lit("own_history")
            )
        )
        values, _ = farm_equal_by_month(rows=rows, metric="score")
        scores.append(float(values.mean()))
    sd = float(np.std(scores, ddof=1))
    mean = float(np.mean(scores))
    return {
        "seeds": len(seeds),
        "mean_score": mean,
        "sd": sd,
        "mde_pct": 100.0 * 1.96 * np.sqrt(2.0) * sd / np.sqrt(len(seeds)) / mean,
    }


def main() -> None:
    """Run the gate or a list of contrasts and print the tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(12)
    global SEEDS  # noqa: PLW0603
    if "--seeds" in sys.argv:
        i = sys.argv.index("--seeds")
        SEEDS = tuple(int(x) for x in sys.argv[i + 1].split(","))
        del sys.argv[i : i + 2]
    mode = sys.argv[1]
    if mode == "gate":
        arms = {"g0_onepass": load_mine(arm="g0_onepass"), "e345": load_reference(name="e345")}
        table = contrast_table(arms=arms, pairs=[("g0_onepass", "e345")])
        print(table)
        print(absolute_table(arms=arms))
        worst = float(table["pct_change"].abs().max())  # ty: ignore[invalid-argument-type]
        overall = table.filter(pl.col("band") == "all")["pct_change"][0]
        print(
            f"gate: all-band change {overall:+.3f}%, worst band {worst:.3f}% (tolerance {GATE_TOLERANCE_PERCENT}%)"
        )
        print(
            "GATE",
            "PASSED"
            if abs(overall) <= GATE_TOLERANCE_PERCENT and worst <= 2 * GATE_TOLERANCE_PERCENT
            else "MISSED",
        )
        table.write_csv(OUT_DIR / "gate.csv")
    else:
        names = sys.argv[2:]
        pairs = [(names[i], names[i + 1]) for i in range(0, len(names), 2)]
        arms = {}
        for n in {x for p in pairs for x in p}:
            arms[n] = load_reference(name=n) if n in ("e345", "blend310") else load_mine(arm=n)
        table = contrast_table(arms=arms, pairs=pairs)
        print(table)
        print(absolute_table(arms=arms))
        for n in arms:
            if n not in ("e345", "blend310"):
                print(n, seed_spread(arm=n))
        table.write_csv(OUT_DIR / ("contrast_" + "_".join(names)[:80] + ".csv"))


if __name__ == "__main__":
    main()
