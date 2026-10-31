# ruff: noqa: E501
"""Coherence of 51-member paths across the three farms (experiment 00429, A1).

    source ml-experiments/00429-*/env.sh
    uv run --no-sync python ml-experiments/00429-*/e00429_coherence.py ARM [ARM ...]

For each arm and seed, four path constructions give 51 member values per farm, run, and lead:

- `ecc`: ensemble copula coupling. Member k takes the arm's quantile (11 levels, linear in between,
  clamped to q01 to q99) at the level `(rank + 0.5) / 51`, where `rank` is the rank of the member's
  own 100 m wind speed among the 51 at that farm and valid time (ties broken by a fixed jitter);
- `schaake`: the Schaake shuffle. For each scored run, the 51 hindcast runs nearest in day of year
  supply observed paths; member k takes the arm's quantile at the rank of historical path k among
  the 51 at that farm and lead, so one historical date orders all three farms and all leads;
- `shifted`: member k keeps its own stage-3 plant power and adds the arm's median minus the
  plant's median, clamped to [0, 1] (the member route's own paths);
- `independent`: every member, farm, and lead takes its own random level (no coherence).

Two kinds of block are scored, both across W1 to W3 jointly and over kept half-hours only. A *day
block* is one run and one calendar day of valid times (at least 36 shared half-hours, three farms).
A *run block* is one run's every twelfth lead (6-hourly) across all leads. The energy score is the
mean over members of the root-mean-square distance of the path to the observed path minus half the
mean root-mean-square distance between two members' paths. The variogram score (order 0.5) is the
mean over pairs of path points of the squared difference between the observed and the ensemble-mean
`|difference|^0.5`. Both are in points of capacity, lower is better, averaged over seeds.
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
MIN_DAY_BLOCK: Final[int] = 36
RUN_STRIDE: Final[int] = 12
SCALE: Final[float] = 100.0
CONSTRUCTIONS: Final[tuple[str, ...]] = ("ecc", "schaake", "shifted", "independent")
RESAMPLES: Final[int] = 2_000
MEDIAN_INDEX: Final[int] = 5
WIND_100M: Final[int] = 1
"""Column of the member weather array that holds the 100 m wind speed."""


def quantile_at(*, quantiles: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear quantile function of each row at `level` (rows, members)."""
    idx = np.clip(np.searchsorted(LEVELS, level) - 1, 0, len(LEVELS) - 2)
    low, high = LEVELS[idx], LEVELS[idx + 1]
    frac = np.clip((level - low) / (high - low), 0.0, 1.0)
    rows = np.arange(len(quantiles))[:, None]
    return quantiles[rows, idx] * (1 - frac) + quantiles[rows, idx + 1] * frac


def ranks(values: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Return the rank (0 to 50) of each column within each row, ties broken at random."""
    jitter = rng.uniform(0, 1e-9, values.shape)
    return np.argsort(np.argsort(values + jitter, axis=1), axis=1)


def schaake_ranks(*, scored, hindcast, grid: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Return (scored rows, 51) ranks that follow 51 observed hindcast paths, one set per run."""
    steps = {float(h): i for i, h in enumerate(grid)}
    farms = {f: i for i, f in enumerate(data.FARMS)}
    runs_h = hindcast["power_fcst_init_time"].unique().sort()
    run_index = {t: i for i, t in enumerate(runs_h.to_list())}
    table = np.full((len(runs_h), len(farms), len(grid)), np.nan)
    ri = np.array([run_index[t] for t in hindcast["power_fcst_init_time"].to_list()])
    fi = np.array([farms[f] for f in hindcast["site"].to_list()])
    li = np.array([steps[float(h)] for h in hindcast["lead_hours"].to_numpy()])
    usable = ~hindcast["masked"].to_numpy()
    table[ri[usable], fi[usable], li[usable]] = hindcast["actual_norm"].to_numpy()[usable]
    fill = np.nanmedian(table, axis=0)
    table = np.where(np.isnan(table), fill[None], table)
    doy_h = np.array([t.timetuple().tm_yday for t in runs_h.to_list()])
    out = np.empty((scored.height, MEMBERS), dtype=np.int16)
    runs_s = scored["power_fcst_init_time"].to_numpy()
    site_s = np.array([farms[f] for f in scored["site"].to_list()])
    lead_s = np.array([steps[float(h)] for h in scored["lead_hours"].to_numpy()])
    scored_doy = np.array([t.timetuple().tm_yday for t in scored["power_fcst_init_time"].to_list()])
    for t in np.unique(runs_s):
        rows = np.flatnonzero(runs_s == t)
        dist = np.abs(doy_h - scored_doy[rows[0]])
        dist = np.minimum(dist, 365 - dist)
        chosen = np.argsort(dist, kind="stable")[:MEMBERS]
        history = table[chosen][:, site_s[rows], lead_s[rows]].T
        out[rows] = np.argsort(np.argsort(history, axis=1, kind="stable"), axis=1)
    return out


def member_values(
    *,
    kind: str,
    quantiles: np.ndarray,
    power: np.ndarray,
    wind: np.ndarray,
    shuffle: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the member values of every row: (rows, 51)."""
    if kind == "independent":
        return quantile_at(quantiles=quantiles, level=rng.uniform(0.01, 0.99, power.shape))
    if kind == "shifted":
        shift = quantiles[:, 5] - np.median(power, axis=1)
        return np.clip(power + shift[:, None], 0.0, 1.0)
    rank = ranks(wind, rng=rng) if kind == "ecc" else shuffle
    return quantile_at(quantiles=quantiles, level=(rank + 0.5) / MEMBERS)


def block_scores(*, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (energy score, variogram score) of members `x` (51, n) against observed `y` (n,)."""
    n = x.shape[1]
    to_obs = np.sqrt(((x - y) ** 2).mean(axis=1)).mean()
    gram = x @ x.T
    sq = np.diag(gram)
    dist2 = np.maximum(sq[:, None] + sq[None, :] - 2 * gram, 0.0) / n
    energy = to_obs - 0.5 * np.sqrt(dist2).mean()
    obs = np.abs(y[:, None] - y[None, :]) ** 0.5
    ens = (np.abs(x[:, :, None] - x[:, None, :]) ** 0.5).mean(axis=0)
    upper = np.triu_indices(n, k=1)
    variogram = ((obs - ens)[upper] ** 2).mean()
    return float(energy * SCALE), float(variogram * SCALE)


def build_blocks(*, scored, grid: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:  # noqa: ANN001
    """Return the row-index arrays (3 farms, steps) of each block and its description."""
    steps = {float(h): i for i, h in enumerate(grid)}
    farms = {f: i for i, f in enumerate(data.FARMS)}
    runs = {t: i for i, t in enumerate(sorted(set(scored["power_fcst_init_time"].to_list())))}
    row_map = np.full((len(runs), len(farms), len(grid)), -1, dtype=np.int64)
    kept = ~scored["masked"].to_numpy()
    ri = np.array([runs[t] for t in scored["power_fcst_init_time"].to_list()])
    fi = np.array([farms[f] for f in scored["site"].to_list()])
    li = np.array([steps[float(h)] for h in scored["lead_hours"].to_numpy()])
    row_map[ri[kept], fi[kept], li[kept]] = np.flatnonzero(kept)
    month = scored["month"].to_numpy()
    blocks, info = [], []
    for run_i in range(len(runs)):
        present = (row_map[run_i] >= 0).all(axis=0)
        if not (row_map[run_i] >= 0).any():
            continue
        first_row = row_map[run_i][row_map[run_i] >= 0][0]
        day_of_lead = (grid // 24).astype(int)
        for day in np.unique(day_of_lead):
            sel = np.flatnonzero(present & (day_of_lead == day))
            if len(sel) >= MIN_DAY_BLOCK:
                blocks.append(row_map[run_i][:, sel])
                info.append({"kind": "day", "lead_day": int(day), "month": month[first_row]})
        sel = np.flatnonzero(present)[::RUN_STRIDE]
        if len(sel) >= MIN_DAY_BLOCK:
            blocks.append(row_map[run_i][:, sel])
            info.append({"kind": "run", "lead_day": -1, "month": month[first_row]})
    return blocks, info


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score the named arms and print the tables."""
    arms = sys.argv[1:] or ["one_pass_ms", "one_pass_m", "member_plain"]
    scored = data347.scored_frame()
    hindcast = data347.hindcast_frame()
    weather, power = data.load(set_name="scored")
    wind = weather[:, :, WIND_100M].astype(np.float64)
    power = power.astype(np.float64)
    actual = scored["actual_norm"].to_numpy().astype(np.float64)
    grid = np.sort(scored["lead_hours"].unique().to_numpy())
    blocks, info = build_blocks(scored=scored, grid=grid)
    kinds = np.array([i["kind"] for i in info])
    lead_day = np.array([i["lead_day"] for i in info])
    month = np.array([i["month"] for i in info])
    print(
        f"{int((kinds == 'day').sum())} day blocks, {int((kinds == 'run').sum())} run blocks",
        flush=True,
    )
    shuffle = schaake_ranks(scored=scored, hindcast=hindcast, grid=grid)
    result: dict[tuple[str, str], np.ndarray] = {}
    for arm in arms:
        parts = np.stack([np.load(run.PARTS_DIR / f"{arm}_h{s}.npy") for s in run.SEEDS])
        kinds_for_arm = (
            (*CONSTRUCTIONS, "memberwise") if arm.startswith("memberwise") else CONSTRUCTIONS
        )
        for kind in kinds_for_arm:
            values = np.zeros((len(blocks), 2))
            for s in range(len(parts)):
                q = np.asarray(parts[s], dtype=np.float64)
                rng = np.random.default_rng(1000 + s)
                if kind == "memberwise":
                    path = np.load(run.PARTS_DIR / f"{arm}_members_h{s}.npy", mmap_mode="r")
                    x = np.asarray(path[:, :, MEDIAN_INDEX], dtype=np.float64)
                else:
                    x = member_values(
                        kind=kind, quantiles=q, power=power, wind=wind, shuffle=shuffle, rng=rng
                    )
                for i, block in enumerate(blocks):
                    rows = block.reshape(-1)
                    values[i] += block_scores(x=x[rows].T, y=actual[rows])
            result[arm, kind] = values / len(parts)
            print(f"{arm} {kind} done", flush=True)
    months = np.unique(month)
    draw = np.random.default_rng(0).integers(0, len(months), size=(RESAMPLES, len(months)))

    def interval(*, diff: np.ndarray, base: np.ndarray, select: np.ndarray) -> str:
        sums = np.array([diff[select & (month == m)].sum(axis=0) for m in months])
        counts = np.array([(select & (month == m)).sum() for m in months])
        means = sums[draw].sum(axis=1) / np.maximum(counts[draw].sum(axis=1), 1)[:, None]
        low, high = np.quantile(means, [0.005, 0.995], axis=0)
        point = diff[select].mean(axis=0)
        relative = point / base[select].mean(axis=0)
        return " ; ".join(
            f"{name} {point[k]:+.3f} ({relative[k]:+.1%}; 99% {low[k]:+.3f} to {high[k]:+.3f})"
            for k, name in enumerate(("energy", "variogram"))
        )

    lines = [
        "-- coherence scores across W1 to W3, mean over blocks (points of capacity; lower is better) --"
    ]
    for block_kind in ("day", "run"):
        sel = kinds == block_kind
        lines.append(f"[{block_kind} blocks, {int(sel.sum())}]")
        for (arm, kind), values in result.items():
            lines.append(
                f"{arm.ljust(16)}{kind.ljust(12)}energy {values[sel, 0].mean():.3f}  variogram {values[sel, 1].mean():.3f}"
            )
    lines.append("-- day blocks by lead day (energy / variogram) --")
    for lo, hi in ((0, 1), (2, 3), (4, 6), (7, 10), (11, 14)):
        sel = (kinds == "day") & (lead_day >= lo) & (lead_day <= hi)
        for (arm, kind), values in result.items():
            lines.append(
                f"days {lo}-{hi} {arm.ljust(16)}{kind.ljust(12)}{values[sel, 0].mean():.3f} / {values[sel, 1].mean():.3f}  ({int(sel.sum())} blocks)"
            )
    lines.append(
        "-- paired differences, first minus second (negative favours the first); relative change and 99% "
        "month-block interval. Pairs across arms mix in marginal differences, and `shifted` keeps the plant's own spread; only the member_plain ecc-versus-shifted pair isolates the path construction. The 5% coherence margin is the Opus note's proposal, not the maintainer's --"
    )
    pairs = {
        "member_plain shifted - one_pass_ms ecc": (
            ("member_plain", "shifted"),
            ("one_pass_ms", "ecc"),
        ),
        "member_plain shifted - one_pass_ms schaake": (
            ("member_plain", "shifted"),
            ("one_pass_ms", "schaake"),
        ),
        "one_pass_ms schaake - one_pass_ms ecc": (
            ("one_pass_ms", "schaake"),
            ("one_pass_ms", "ecc"),
        ),
        "memberwise memberwise - one_pass_ms ecc": (
            ("memberwise", "memberwise"),
            ("one_pass_ms", "ecc"),
        ),
        "memberwise memberwise - one_pass_ms schaake": (
            ("memberwise", "memberwise"),
            ("one_pass_ms", "schaake"),
        ),
        "memberwise memberwise - memberwise ecc": (
            ("memberwise", "memberwise"),
            ("memberwise", "ecc"),
        ),
        "one_pass_m ecc - one_pass_ms ecc": (("one_pass_m", "ecc"), ("one_pass_ms", "ecc")),
        "one_pass_ms ecc - one_pass_ms independent": (
            ("one_pass_ms", "ecc"),
            ("one_pass_ms", "independent"),
        ),
        "member_plain ecc - member_plain shifted": (
            ("member_plain", "ecc"),
            ("member_plain", "shifted"),
        ),
    }
    for label, (first, second) in pairs.items():
        if first not in result or second not in result:
            continue
        for block_kind in ("day", "run"):
            sel = kinds == block_kind
            lines.append(
                f"{label} | {block_kind}: {interval(diff=result[first] - result[second], base=result[second], select=sel)}"
            )
    text = "\n".join(lines)
    print(text)
    (data.DATA_DIR / "coherence_report.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
