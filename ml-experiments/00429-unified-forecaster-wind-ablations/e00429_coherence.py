# ruff: noqa: E501
"""Coherence of 51-member paths (experiment 00429, A1): energy and variogram scores.

    source ml-experiments/00429-*/env.sh
    uv run --no-sync python ml-experiments/00429-*/e00429_coherence.py ARM [ARM ...]

For each arm and seed, three path constructions give 51 member paths per valid time:

- `ecc`: member k takes the arm's quantile (11 levels, linear in between, clamped to q01 to q99)
  at the level `(rank + 0.5) / 51`, where `rank` is the rank of the member's stage-3 plant power
  among the 51 at that valid time (ties broken by a fixed random jitter);
- `shifted`: member k keeps its stage-3 plant power and adds the arm's median minus the plant's
  median, clamped to [0, 1];
- `independent`: every member and valid time takes the quantile at its own random level.

A block is one farm, one run, and one calendar day of valid times, kept half-hours only, at least
36 of them. The energy score is the mean over members of the root-mean-square distance of the path
to the observed path, minus half the mean root-mean-square distance between two members' paths;
the variogram score (order 0.5) is the mean over half-hour pairs of the squared difference
between the observed and the ensemble-mean `|difference|^0.5`. Both are in points of capacity
(percent of capacity), lower is better, and are averaged over seeds.
"""

import os
import sys
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00347_s4_data as data347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00429_data as data  # noqa: E402
import e00429_run as run  # noqa: E402
import numpy as np  # noqa: E402

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEMBERS: Final[int] = 51
MIN_BLOCK: Final[int] = 36
SCALE: Final[float] = 100.0
CONSTRUCTIONS: Final[tuple[str, ...]] = ("ecc", "shifted", "independent")
RESAMPLES: Final[int] = 2_000


def quantile_at(*, quantiles: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear quantile function of each row at `level` (rows, members)."""
    idx = np.clip(np.searchsorted(LEVELS, level) - 1, 0, len(LEVELS) - 2)
    low, high = LEVELS[idx], LEVELS[idx + 1]
    frac = np.clip((level - low) / (high - low), 0.0, 1.0)
    rows = np.arange(len(quantiles))[:, None]
    return quantiles[rows, idx] * (1 - frac) + quantiles[rows, idx + 1] * frac


def paths(
    *, kind: str, quantiles: np.ndarray, power: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return the member values of every row: (rows, 51)."""
    if kind == "independent":
        return quantile_at(quantiles=quantiles, level=rng.uniform(0.01, 0.99, power.shape))
    if kind == "shifted":
        shift = quantiles[:, 5] - np.median(power, axis=1)
        return np.clip(power + shift[:, None], 0.0, 1.0)
    jitter = rng.uniform(0, 1e-9, power.shape)
    rank = np.argsort(np.argsort(power + jitter, axis=1), axis=1)
    return quantile_at(quantiles=quantiles, level=(rank + 0.5) / MEMBERS)


def block_scores(*, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (energy score, variogram score) of members `x` (51, n) against observed `y` (n,)."""
    n = x.shape[1]
    to_obs = np.sqrt(((x - y) ** 2).mean(axis=1)).mean()
    spread = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).mean(axis=2)).mean()
    energy = to_obs - 0.5 * spread
    obs = np.abs(y[:, None] - y[None, :]) ** 0.5
    ens = (np.abs(x[:, :, None] - x[:, None, :]) ** 0.5).mean(axis=0)
    upper = np.triu_indices(n, k=1)
    variogram = ((obs - ens)[upper] ** 2).mean()
    return float(energy * SCALE), float(variogram * SCALE)


def main() -> None:
    """Score the named arms and print the table."""
    arms = sys.argv[1:] or ["one_pass_ms", "member_plain"]
    scored = data347.scored_frame()
    _, power = data.load(set_name="scored")
    actual = scored["actual_norm"].to_numpy().astype(np.float64)
    kept = ~scored["masked"].to_numpy()
    key = (
        scored["site"].cast(str)
        + "|"
        + scored["power_fcst_init_time"].dt.strftime("%Y%m%d%H")
        + "|"
        + scored["valid_time"].dt.strftime("%Y%m%d")
    ).to_numpy()
    month = scored["month"].to_numpy()
    lead_day = scored["lead_hours"].to_numpy() // 24
    blocks = []
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        rows = order[a:b]
        rows = rows[kept[rows]]
        if len(rows) >= MIN_BLOCK:
            blocks.append(rows)
    print(f"{len(blocks)} blocks of at least {MIN_BLOCK} kept half-hours", flush=True)
    result: dict[tuple[str, str], np.ndarray] = {}
    for arm in arms:
        parts = np.stack([np.load(run.PARTS_DIR / f"{arm}_h{s}.npy") for s in run.SEEDS])
        for kind in CONSTRUCTIONS:
            values = np.zeros((len(blocks), 2))
            for s in range(len(parts)):
                q = np.asarray(parts[s], dtype=np.float64)
                rng = np.random.default_rng(1000 + s)
                member_values = paths(
                    kind=kind, quantiles=q, power=power.astype(np.float64), rng=rng
                )
                for i, rows in enumerate(blocks):
                    values[i] += block_scores(x=member_values[rows].T, y=actual[rows])
            result[arm, kind] = values / len(parts)
    block_month = np.array([month[r[0]] for r in blocks])
    block_day = np.array([int(lead_day[r[0]]) for r in blocks])
    months = np.unique(block_month)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(months), size=(RESAMPLES, len(months)))

    def interval(*, diff: np.ndarray) -> str:
        sums = np.array([diff[block_month == m].sum(axis=0) for m in months])
        counts = np.array([(block_month == m).sum() for m in months])
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
        low, high = np.quantile(means, [0.005, 0.995], axis=0)
        return " ; ".join(
            f"{name} {diff[:, k].mean():+.3f} (99% {low[k]:+.3f} to {high[k]:+.3f})"
            for k, name in enumerate(("energy", "variogram"))
        )

    lines = ["-- coherence scores, mean over blocks (percent of capacity; lower is better) --"]
    for (arm, kind), values in result.items():
        lines.append(
            f"{arm.ljust(16)}{kind.ljust(12)}energy {values[:, 0].mean():.3f}  variogram {values[:, 1].mean():.3f}"
        )
    lines.append("-- by lead day band (energy / variogram) --")
    for lo, hi in ((0, 1), (2, 3), (4, 6), (7, 10), (11, 14)):
        sel = (block_day >= lo) & (block_day <= hi)
        if not sel.any():
            continue
        for (arm, kind), values in result.items():
            lines.append(
                f"days {lo}-{hi} {arm.ljust(16)}{kind.ljust(12)}{values[sel, 0].mean():.3f} / {values[sel, 1].mean():.3f}  ({int(sel.sum())} blocks)"
            )
    lines.append(
        "-- paired differences, first minus second (negative favours the first), month-block 99% --"
    )
    if "one_pass_ms" in arms and "member_plain" in arms:
        pairs = {
            "one_pass_ms ecc - member_plain shifted": (
                ("one_pass_ms", "ecc"),
                ("member_plain", "shifted"),
            ),
            "one_pass_ms ecc - member_plain ecc": (("one_pass_ms", "ecc"), ("member_plain", "ecc")),
            "member_plain shifted - member_plain ecc": (
                ("member_plain", "shifted"),
                ("member_plain", "ecc"),
            ),
            "one_pass_ms ecc - one_pass_ms independent": (
                ("one_pass_ms", "ecc"),
                ("one_pass_ms", "independent"),
            ),
        }
        for label, (first, second) in pairs.items():
            lines.append(f"{label}: {interval(diff=result[first] - result[second])}")
    text = "\n".join(lines)
    print(text)
    (data.DATA_DIR / "coherence_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
