# ruff: noqa: D103, E501
"""Experiment 681, step 4: paired reaction-time and score contrasts against L0 (CPU).

Reads `score/reaction.parquet` and `score/summary.json` and prints, for each arm against L0: the
number of paired injected events, the median reaction time after the start and after the end, the
mean paired difference with a 95% interval from a bootstrap over events, the share of events that
react at all, and the score differences. Reaction times are in hours and resolve to 24 hours.

Usage: python e00681_contrasts.py ARM [ARM ...]    (the first arm named is the reference)
"""

import json
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

DATA: Final = (
    Path(__import__("os").environ["DATA_PATH_INTERNAL"])
    / "ml-experiments"
    / "00681-extra-lagged-power-inputs-for-the-prevailing-forecast"
    / "score"
)
KEYS: Final = ["set", "primary", "month", "pass"]
N_BOOT: Final = 4000


def boot(*, x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    arms = sys.argv[1:]
    ref = arms[0]
    r = pl.read_parquet(DATA / "reaction.parquet").with_columns(pl.col("pass").fill_null(-1))
    summ = json.loads((DATA / "summary.json").read_text())
    base = r.filter(pl.col("arm") == ref)
    for arm in arms:
        a = r.filter(pl.col("arm") == arm)
        for kind in ("injected", "real"):
            for col in ("start_h", "end_h"):
                j = a.filter(pl.col("kind") == kind).join(
                    base.filter(pl.col("kind") == kind), on=KEYS, suffix="_ref"
                )
                pair = j.filter(pl.col(col).is_not_nan() & pl.col(f"{col}_ref").is_not_nan())
                n_all = j.height
                react = float(j[col].is_not_nan().mean()) if n_all else float("nan")  # ty: ignore[invalid-argument-type]
                react_ref = float(j[f"{col}_ref"].is_not_nan().mean()) if n_all else float("nan")  # ty: ignore[invalid-argument-type]
                if pair.height >= 2:
                    d = (pair[col] - pair[f"{col}_ref"]).to_numpy()
                    mean, lo, hi = boot(x=d)
                    print(
                        f"{arm:3s} {kind:8s} {col:7s} events {n_all:5d} paired {pair.height:5d} median {pair[col].median():7.1f} (ref {pair[f'{col}_ref'].median():7.1f}) diff {mean:+7.1f} ({lo:+.1f} to {hi:+.1f}) reacting {react:.3f} (ref {react_ref:.3f})"
                    )
                else:
                    print(
                        f"{arm:3s} {kind:8s} {col:7s} events {n_all:5d} paired {pair.height:5d} too few to read"
                    )
        j = a.filter(pl.col("kind") == "injected").join(
            base.filter(pl.col("kind") == "injected"), on=KEYS, suffix="_ref"
        )
        j = j.filter(pl.col("start_h").is_not_nan() & pl.col("start_h_ref").is_not_nan())
        loo = [
            float(
                (  # ty: ignore[invalid-argument-type]
                    j.filter(pl.col("primary") != p)["start_h"]
                    - j.filter(pl.col("primary") != p)["start_h_ref"]
                ).mean()
            )
            for p in sorted(set(j["primary"].to_list()))
        ]
        print(
            f"    injected start diff leaving one Primary out: min {min(loo):+.1f}, max {max(loo):+.1f} hours"
        )
        s, s0 = summ[arm], summ[ref]
        print(
            f"{arm:3s} scores: all CRPS {s['all_crps']:.4f} ({s['all_crps'] - s0['all_crps']:+.4f}), event-free CRPS {s['event_free_crps']:.4f} ({s['event_free_crps'] - s0['event_free_crps']:+.4f}), "
            f"false alarms {s['false_alarm_share_event_free']:.4f} ({s['false_alarm_share_event_free'] - s0['false_alarm_share_event_free']:+.4f}), NRA all CRPS {s['nra_all_crps']:.4f}"
        )
        keys = sorted(s["event_free_blocks"])
        for m in ("crps", "mae"):
            diff = np.array(
                [s["event_free_blocks"][k][m] - s0["event_free_blocks"][k][m] for k in keys]
            )
            n = np.array([s["event_free_blocks"][k]["n"] for k in keys])
            rng = np.random.default_rng(1)
            boots = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(keys), len(keys))
                boots.append(diff[i].sum() / n[i].sum())
            print(
                f"    event-free {m} vs {ref}: {diff.sum() / n.sum():+.4f} (95% {np.percentile(boots, 2.5):+.4f} to {np.percentile(boots, 97.5):+.4f}; blocks are Primary x month)"
            )
        print(
            "    bank CRPS by state:",
            {
                k: round(v - s0["bank_crps_by_state"][k], 3)
                for k, v in s["bank_crps_by_state"].items()
            },
        )


if __name__ == "__main__":
    main()
