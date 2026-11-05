# ruff: noqa: E501
"""Experiment 575: collate the scores into the tables and figures of the write-up.

Prints every number the page quotes, and writes the figures to `docs/ml-experiments/00575-*/`.
The headline pool is the latest fold (2026-06) with all five world draws; the other folds
(nested histories, so not independent) are shown as per-fold signs for draws 0 to 2.
"""

import json
from pathlib import Path

import e00573_steps as s573  # ty: ignore[unresolved-import]
import e00575_run as run
import e00575_world as w
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = run.WORK / "scores"
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00575-stepwise-offset-large-step-and-mirrored-pair-recovery"
)
R573 = s573.WORK
ARMS = ["S0", "S1@3000", "S1@1000", "S2@3000/300", "S2@3000/30", "S2@1000/300", "S2@1000/30"]
SIZE_BINS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.50001)]
DUR_BINS = [(1, 3), (3, 7), (7, 14), (14, 30), (30, 181)]
LATEST = "2026-06"


def load() -> pl.DataFrame:
    """Return one row per fold, world, seed, arm, and truth event."""
    rows = []
    for f in sorted(SCORES.glob("*_K[lp]?.json")):
        month, rest = f.stem.split("_")
        world, seed = rest[:2], int(rest[2:])
        for arm, res in json.loads(f.read_text()).items():
            rows.extend(
                {"month": month, "world": world, "seed": seed, "arm": arm, **e}
                for e in res["events"]
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("size").abs().alias("abs_size"), pl.col("donor_size").abs().alias("abs_donor")
    )


def untouched_false(
    *, world: str, months: list[str], seeds: list[int]
) -> dict[str, dict[str, float]]:
    """Return per arm the false steps of 5% or more per Primary-year on the ten untouched Primaries."""
    out: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        n = y = 0.0
        for m in months:
            for sd in seeds:
                f = SCORES / f"{m}_{world}{sd}.json"
                if f.exists():
                    u = json.loads(f.read_text())[arm]["untouched_ten"]
                    n, y = n + u["steps_5pct"], y + u["years"]
        out[arm] = {"steps": n, "years": y, "per_year": n / y if y else float("nan")}
    return out


def k0_false(*, month: str) -> dict[str, float]:
    """Return per arm the false steps of 5% or more per Primary-year on all 16 Primaries in 573's K0 paths."""
    fold = s573.load_fold(month=month, device="cpu")
    limit = fold.span_end_day - s573.END_MARGIN
    out = {}
    for arm in ARMS:
        path = np.load(R573 / "paths_run_r2" / f"{month}_K0_{arm.replace('/', '_')}.npy")
        n = years = 0.0
        for p in range(16):
            first = int(np.argmax(fold.rows_per_day[p] > 0))
            n += sum(
                abs(sz) >= run.FALSE_MIN
                for d, sz in s573.extract(path=path[p], std=fold.std[p])
                if first <= d <= limit
            )
            years += w.w573.offset_years(start_day=first, end_day=limit)
        out[arm] = n / years
    return out


def recall(*, df: pl.DataFrame, col: str = "hit_onset") -> tuple[float, float, int]:
    """Return recall, the binomial half-width, and n."""
    n = len(df)
    if n == 0:
        return float("nan"), float("nan"), 0
    p = float(np.mean(df[col].to_numpy()))
    return p, 1.96 * float(np.sqrt(p * (1 - p) / n)), n


def fmt(*, r: tuple[float, float, int]) -> str:
    """Format a recall with its half-width, n, and the unresolved mark."""
    p, h, n = r
    mark = " (unresolved)" if n < 20 else ""
    return f"{p:.2f} ±{h:.2f} n={n}{mark}"


def main() -> None:
    """Print the tables and draw the figures."""
    df = load().filter(pl.col("scored"))
    print("labelled-event quartiles:", json.dumps(w.label_quartiles()))
    kl = df.filter((pl.col("world") == "Kl") & (pl.col("role") == "single"))
    late = kl.filter(pl.col("month") == LATEST)
    print("\n== scored Kl events per size x duration bin (latest fold, 5 draws; one arm) ==")
    one = late.filter(pl.col("arm") == "S0")
    for lo, hi in SIZE_BINS:
        print(
            f"size {lo:.2f}-{hi:.2f}:",
            [
                len(
                    one.filter(
                        (pl.col("abs_size") >= lo)
                        & (pl.col("abs_size") < hi)
                        & (pl.col("duration") >= a)
                        & (pl.col("duration") < b)
                    )
                )
                for a, b in DUR_BINS
            ],
        )
    big_long = lambda d: d.filter((pl.col("abs_size") >= 0.20) & (pl.col("duration") >= 7))  # noqa: E731
    print(
        "\n== prediction 1: recall of steps >=20% lasting >=7 days; false steps >=5% per Primary-year =="
    )
    k0 = k0_false(month=LATEST)
    fl = untouched_false(world="Kl", months=[LATEST], seeds=list(range(5)))
    fp = untouched_false(world="Kp", months=[LATEST], seeds=list(range(5)))
    for arm in ARMS:
        r = recall(df=big_long(late.filter(pl.col("arm") == arm)))
        per_fold = [
            round(
                recall(
                    df=big_long(
                        kl.filter(
                            (pl.col("arm") == arm) & (pl.col("month") == m) & (pl.col("seed") < 3)
                        )
                    )
                )[0],
                2,
            )
            for m in s573.MONTHS
            if m in kl["month"].unique().to_list()
        ]
        ok = r[0] >= 0.8 and max(k0[arm], fl[arm]["per_year"], fp[arm]["per_year"]) <= 1.0
        print(
            f"{arm}: recall {fmt(r=r)} | per-fold (draws 0-2) {per_fold} | false/yr K0 {k0[arm]:.2f} Kl-untouched {fl[arm]['per_year']:.2f} Kp-untouched {fp[arm]['per_year']:.2f} | passes {ok}"
        )
    print(
        "\n== recall by size bin (all durations) and by duration bin (sizes >=20%), latest fold =="
    )
    for arm in ARMS:
        sz = [
            fmt(
                r=recall(
                    df=late.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("abs_size") >= lo)
                        & (pl.col("abs_size") < hi)
                    )
                )
            )
            for lo, hi in SIZE_BINS
        ]
        du = [
            fmt(
                r=recall(
                    df=late.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("abs_size") >= 0.2)
                        & (pl.col("duration") >= a)
                        & (pl.col("duration") < b)
                    )
                )
            )
            for a, b in DUR_BINS
        ]
        print(arm, "\n   size:", sz, "\n   dur(>=20%):", du)
    print("\n== prediction 2: recall of 1-day steps (duration 1 to 2 days) by size bin ==")
    for arm in ARMS:
        print(
            arm,
            [
                fmt(
                    r=recall(
                        df=late.filter(
                            (pl.col("arm") == arm)
                            & (pl.col("duration") < 2)
                            & (pl.col("abs_size") >= lo)
                            & (pl.col("abs_size") < hi)
                        )
                    )
                )
                for lo, hi in SIZE_BINS
            ],
        )
    print("\n== return-step recall (sizes >=20%, >=7 days) ==")
    for arm in ARMS:
        print(arm, fmt(r=recall(df=big_long(late.filter(pl.col("arm") == arm)), col="hit_return")))
    print(
        "\n== prediction 4: both halves of a pair (donor size >=20%, >=7 days; both halves scored) =="
    )
    kp = df.filter(pl.col("world") == "Kp")
    for m in (LATEST,):
        sub = kp.filter(pl.col("month") == m)
        for arm in ARMS:
            a = sub.filter(pl.col("arm") == arm)
            both = (
                a.group_by(["seed", "pair_id"])
                .agg(
                    pl.len().alias("k"),
                    pl.col("hit_onset").all().alias("all"),
                    pl.col("abs_donor").first(),
                    pl.col("duration").first(),
                )
                .filter(
                    (pl.col("k") == 2) & (pl.col("abs_donor") >= 0.2) & (pl.col("duration") >= 7)
                )
            )
            print(arm, "pairs", fmt(r=recall(df=both, col="all")))
    print("\n== prediction 5: S0 recall of steps lasting under 30 days (all sizes) ==")
    print("S0", fmt(r=recall(df=late.filter((pl.col("arm") == "S0") & (pl.col("duration") < 30)))))
    sd = {
        m: json.loads((SCORES / f"{m}_residual_sd.json").read_text())
        for m in s573.MONTHS
        if (SCORES / f"{m}_residual_sd.json").exists()
    }
    sd_late = np.array([sd[LATEST][str(p)] for p in w.INJECTED])
    print(
        "\nresidual sd of the daily mean (fraction of capacity) per injected Primary, latest fold:",
        np.round(sd_late, 4).tolist(),
    )
    heat(df=late, sd=float(np.median(sd_late)))


def heat(*, df: pl.DataFrame, sd: float) -> None:
    """Draw recall against duration for each size bin with the detectability curve, and heat maps."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, arm in zip(axes, ["S1@1000", "S2@3000/300", "S0"], strict=True):
        for (lo, hi), c in zip(SIZE_BINS, ["#fdb462", "#ec5a24", "#1d4e89"], strict=True):
            ys, xs = [], []
            for a, b in DUR_BINS:
                sub = df.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("abs_size") >= lo)
                    & (pl.col("abs_size") < hi)
                    & (pl.col("duration") >= a)
                    & (pl.col("duration") < b)
                )
                if len(sub) >= 5:
                    ys.append(float(np.mean(sub["hit_onset"].to_numpy())))
                    xs.append(np.sqrt(a * b))
            ax.plot(xs, ys, "o-", color=c, label=f"{lo:.0%} to {min(hi, 0.5):.0%}")
        ax.set_xscale("log")
        ax.set_title(arm)
        ax.set_xlabel("event length (days)")
    axes[0].set_ylabel("share of steps recovered")
    axes[0].legend(title="size", fontsize=7)
    fig.tight_layout()
    fig.savefig(DOCS / "recall_by_duration.svg")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    n = np.logspace(0, np.log10(180), 50)
    ax.plot(n, 3 * sd / np.sqrt(n), color="#1d4e89")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("event length (days)")
    ax.set_ylabel("smallest step above 3 standard errors\n(share of capacity)")
    for y in (0.1, 0.5):
        ax.axhline(y, color="grey", lw=0.5)
    fig.tight_layout()
    fig.savefig(DOCS / "detectability_curve.svg")


if __name__ == "__main__":
    main()
