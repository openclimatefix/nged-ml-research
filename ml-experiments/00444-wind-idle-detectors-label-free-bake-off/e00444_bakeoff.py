"""Experiment 444: label-free bake-off of idle-half-hour detectors at three metered wind farms.

    PYTHONPATH=\
        ml-experiments/00431-wind-idle-data-look:ml-experiments/00210-wind-idle-noise-mva-artefact \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue nice -n 19 \
        uv run --no-sync python \
        ml-experiments/00444-wind-idle-detectors-label-free-bake-off/e00444_bakeoff.py

Reads experiment 210's `joined.parquet` and `report.json`, imports experiment 431's feature code and
experiment 210's wind-threshold fit, and writes per-farm summary tables (published-safe: recall,
false-drop counts, shares by wind bin) and a private out-of-fold flag series under
`data/ml-experiments/00444-wind-idle-detectors-label-free-bake-off/`. Farms are W1, W2, W3 in the
stored tables and Farm A, Farm B, Farm C in every written page. The plan in `result.md` fixes
every definition here before any score is read.
"""

import json
import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from e00210_analysis import detector_rates  # ty: ignore[unresolved-import]
from e00431_analysis import add_features, auc, stratified_auc  # ty: ignore[unresolved-import]
from sklearn.linear_model import LogisticRegression

DATA_ROOT: Final[Path] = Path(os.environ["DATA_PATH_INTERNAL"]) / "ml-experiments"
SRC_DIR: Final[Path] = DATA_ROOT / "00210-wind-idle-noise-mva-artefact"
OUT_DIR: Final[Path] = DATA_ROOT / os.environ.get(
    "E444_OUT", "00444-wind-idle-detectors-label-free-bake-off"
)
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
SEED: Final[int] = 444
N_REPLICATES: Final[int] = int(os.environ.get("E444_REPS", "10"))
IDLE_WIND_MAX: Final[float] = 2.0
GEN_WIND_MIN: Final[float] = 6.0
GEN_SHARE: Final[tuple[float, float]] = (0.03, 0.08)
HIGH_FACTOR: Final[float] = 1.5
EDGE_ROWS: Final[int] = 48
PIECE: Final[int] = 12
BLOCK_LENGTHS: Final[tuple[int, ...]] = (6, 24, 48, 96, 192)
BLOCKS_PER_CELL: Final[int] = 4
MOD_MAX_WIND: Final[float] = 8.0
G0_RUNS_PER_PARITY: Final[int] = 20
G0_PAD: Final[int] = 48
G0_LEN: Final[tuple[int, int]] = (12, 96)
G0_TARGET: Final[tuple[float, float]] = (0.02, 0.06)
G0_MIN_SHARE: Final[float] = 0.15
SEPARATION: Final[int] = 96
TARGET_FD: Final[float] = 0.05
FD_LEVELS: Final[tuple[float, ...]] = (0.01, 0.02, 0.10)
RHOS: Final[tuple[float, ...]] = (0.95, 0.90, 0.98)
D3_RATIOS: Final[tuple[float, ...]] = (0.10, 0.05, 0.20)
D3_MIN_EXPECTED: Final[float] = 0.15
MAX_TRAIN_PER_SET: Final[int] = 30000
FEATURE_PREFIXES: Final[tuple[str, ...]] = ("sd_diff_", "ac1_", "ac2_", "repeat_share_")
BIAS_GRID: Final[np.ndarray] = np.arange(-3.0, 4.01, 0.25)
HALF_HOUR: Final[np.timedelta64] = np.timedelta64(30, "m")


def window_ok(*, ok: np.ndarray, length: int) -> np.ndarray:
    """Return, for each start index, whether `ok` is true on all of `[s, s + length)`."""
    csum = np.concatenate([[0], np.cumsum(~ok)])
    n = len(ok)
    out = np.zeros(n, dtype=bool)
    if length <= n:
        starts = np.arange(n - length + 1)
        out[starts] = (csum[starts + length] - csum[starts]) == 0
    return out


def window_sum(*, x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return the sum of `x` on `[lo, hi)` for arrays of clipped bounds."""
    csum = np.concatenate([[0.0], np.cumsum(x.astype(float))])
    return csum[hi] - csum[lo]


def feature_matrix(
    *, valid_time: pl.Series, power: np.ndarray, capacity: float
) -> tuple[np.ndarray, list[str]]:
    """Return experiment 431's level-free statistics for a power series."""
    frame = pl.DataFrame(
        {
            "site": ["x"] * len(power),
            "valid_time": valid_time,
            "power": power,
            "capacity_mw": [capacity] * len(power),
        }
    )
    out = add_features(frame=frame, idle_p95={"x": 1.0})
    names = [c for c in out.columns if c.startswith(FEATURE_PREFIXES)]
    return out.select(names).to_numpy().astype(float), names


def segments(*, contiguous: np.ndarray) -> list[tuple[int, int]]:
    """Return `[start, end)` index pairs of unbroken half-hourly stretches."""
    breaks = np.flatnonzero(~contiguous)
    starts = np.unique(np.concatenate([[0], breaks]))
    ends = np.concatenate([starts[1:], [len(contiguous)]])
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def viterbi_idle(*, logodds: np.ndarray, segs: list[tuple[int, int]], rho: float) -> np.ndarray:
    """Return the most likely idle path of a two-state chain with stay probability `rho`."""
    ln_stay, ln_move = np.log(rho), np.log(1.0 - rho)
    em_gen = -np.logaddexp(0.0, logodds)
    em_idle = -np.logaddexp(0.0, -logodds)
    out = np.zeros(len(logodds), dtype=bool)
    for a, b in segs:
        n = b - a
        if n == 0:
            continue
        g = em_gen[a:b].tolist()
        i = em_idle[a:b].tolist()
        back_g = [False] * n
        back_i = [False] * n
        dg, di = g[0], i[0]
        for k in range(1, n):
            from_g_g, from_i_g = dg + ln_stay, di + ln_move
            from_g_i, from_i_i = dg + ln_move, di + ln_stay
            back_g[k] = from_i_g > from_g_g
            back_i[k] = from_i_i > from_g_i
            dg = max(from_g_g, from_i_g) + g[k]
            di = max(from_g_i, from_i_i) + i[k]
        state_idle = di > dg
        for k in range(n - 1, -1, -1):
            out[a + k] = state_idle
            state_idle = back_i[k] if state_idle else back_g[k]
    return out


def runs_at_least(*, flag: np.ndarray, contiguous: np.ndarray, minimum: int) -> np.ndarray:
    """Return rows that sit in an unbroken run of `flag` of at least `minimum` half-hours."""
    out = np.zeros(len(flag), dtype=bool)
    start = None
    for k in range(len(flag) + 1):
        inside = k < len(flag) and flag[k] and (start is None or contiguous[k])
        if inside and start is None:
            start = k
        elif not inside:
            if start is not None and k - start >= minimum:
                out[start:k] = True
            start = k if (k < len(flag) and flag[k]) else None
    return out


class Farm:
    """One farm's arrays, true series, features, and the labels that do not depend on the fold."""

    def __init__(self, *, frame: pl.DataFrame, cut_in: float) -> None:
        """Read the farm's rows and compute its labels-free arrays and features."""
        self.site = frame["site"][0]
        self.valid_time = frame["valid_time"]
        self.t = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
        self.power = frame["power"].to_numpy().astype(float)
        self.wind = frame["wind_speed"].to_numpy().astype(float)
        self.cap = float(frame["capacity_mw"][0])
        self.cut_in = cut_in
        self.n = len(self.power)
        self.year = frame["valid_time"].dt.year().to_numpy()
        self.contig = np.concatenate([[False], np.diff(self.t) == HALF_HOUR])
        self.segs = segments(contiguous=self.contig)
        change = np.flatnonzero(np.diff(self.year) != 0) + 1
        self.edge = np.zeros(self.n, dtype=bool)
        for c in change:
            self.edge[max(c - EDGE_ROWS, 0) : c + EDGE_ROWS] = True
        self.share = self.power / self.cap
        self.high = self.wind >= HIGH_FACTOR * cut_in
        self.feats, self.feat_names = feature_matrix(
            valid_time=self.valid_time, power=self.power, capacity=self.cap
        )


def make_context(*, farm: Farm, parity: int) -> dict[str, Any]:
    """Fit everything that depends on the tune years only (band, wind rule, power curve)."""
    tune = farm.year % 2 == parity
    low = farm.wind < IDLE_WIND_MAX
    lo, hi = np.quantile(farm.power[tune & low], [0.05, 0.95])
    train_frame = pl.DataFrame(
        {
            "valid_time": farm.valid_time.filter(pl.Series(tune)),
            "wind_speed": farm.wind[tune],
            "power": farm.power[tune],
        }
    )
    fit = detector_rates(train=train_frame, test=train_frame, capacity=farm.cap)
    bins = np.floor(farm.wind[tune] / 0.5).astype(int)
    centres, medians = [], []
    for b in np.unique(bins):
        sel = bins == b
        if sel.sum() >= 30:
            centres.append((b + 0.5) * 0.5)
            medians.append(float(np.median(farm.power[tune][sel])))
    return {
        "tune": tune,
        "band": (float(lo), float(hi)),
        "wind_threshold": float(fit["threshold"]),
        "idle_hi_210": float(fit["idle_hi"]),
        "curve": (np.array(centres), np.array(medians)),
    }


def smooth(*, x: np.ndarray, width: int = 6) -> np.ndarray:
    """Centred moving mean of `width` rows, shorter at the ends."""
    kernel = np.ones(width)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones(len(x)), kernel, mode="same")


def place_replicate(  # noqa: PLR0915
    *, farm: Farm, ctx: dict[str, Any], parity: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build one modified series: injected idle blocks in test years, rescaled low-output runs.

    Returns the modified power, the injected-block id per row (-1 none), the synthetic low-output
    run id per row (-1 none), and one dict per injected block.
    """
    n, tune = farm.n, ctx["tune"]
    lo, hi = ctx["band"]
    in_band = (farm.power >= lo) & (farm.power <= hi)
    si = (farm.wind < IDLE_WIND_MAX) & in_band & tune
    pieces_ok = window_ok(ok=si & np.concatenate([[False], farm.contig[1:]]), length=PIECE)
    piece_starts: dict[int, np.ndarray] = {
        int(y): np.flatnonzero(pieces_ok & (farm.year == y)) for y in np.unique(farm.year[tune])
    }
    power = farm.power.copy()
    inj = np.full(n, -1, dtype=int)
    g0 = np.full(n, -1, dtype=int)
    occupied = np.zeros(n, dtype=bool)
    in_band_count = np.concatenate([[0], np.cumsum(in_band)])
    blocks: list[dict[str, Any]] = []
    wind_cum = np.concatenate([[0.0], np.cumsum(farm.wind)])
    high_ok = farm.high & farm.contig
    block_id = 0
    for length in BLOCK_LENGTHS:
        ok = window_ok(ok=high_ok | (np.arange(n) == 0), length=length)
        starts = np.flatnonzero(ok)
        starts = starts[starts + length <= n]
        mean_wind = (wind_cum[starts + length] - wind_cum[starts]) / length
        keep = (
            (farm.year[starts] == farm.year[starts + length - 1])
            & (farm.year[starts] % 2 != parity)
            & ~np.array([farm.edge[s : s + length].any() for s in starts])
        )
        lo_b = np.clip(starts - 12, 0, n)
        hi_b = np.clip(starts + length + 12, 0, n)
        keep &= (in_band_count[hi_b] - in_band_count[lo_b]) == 0
        starts, mean_wind = starts[keep], mean_wind[keep]
        for bin_name, sel in (
            ("moderate", mean_wind < MOD_MAX_WIND),
            ("high", mean_wind >= MOD_MAX_WIND),
        ):
            cand = rng.permutation(starts[sel])
            placed = 0
            for s in cand:
                if placed >= BLOCKS_PER_CELL:
                    break
                a, b = max(s - SEPARATION, 0), min(s + length + SEPARATION, n)
                if occupied[a:b].any():
                    continue
                y = int(farm.year[s])
                pool = [
                    piece_starts.get(y - 1, np.array([], int)),
                    piece_starts.get(y + 1, np.array([], int)),
                ]
                pool_all = np.concatenate(pool)
                if len(pool_all) == 0:
                    continue
                n_pieces = -(-length // PIECE)
                picks = rng.choice(pool_all, size=n_pieces)
                fill = np.concatenate([farm.power[p : p + PIECE] for p in picks])[:length]
                power[s : s + length] = fill
                inj[s : s + length] = block_id
                occupied[a:b] = True
                blocks.append(
                    {
                        "block_id": block_id,
                        "length": length,
                        "bin": bin_name,
                        "year": y,
                        "start": int(s),
                        "join_rows": [
                            int(s + k * PIECE + d)
                            for k in range(n_pieces)
                            for d in (-1, 0)
                            if k > 0 and s + k * PIECE + d < s + length
                        ],
                    }
                )
                block_id += 1
                placed += 1
    run_id = 0
    good = farm.contig & (farm.wind >= GEN_WIND_MIN) & (farm.share >= G0_MIN_SHARE)
    for want_tune in (True, False):
        placed = 0
        tries = 0
        while placed < G0_RUNS_PER_PARITY and tries < 3000:
            tries += 1
            length = int(rng.integers(G0_LEN[0], G0_LEN[1] + 1))
            s = int(rng.integers(G0_PAD, n - length - G0_PAD))
            if (farm.year[s] % 2 == parity) != want_tune:
                continue
            a, b = s - G0_PAD, s + length + G0_PAD
            if occupied[max(a - SEPARATION, 0) : min(b + SEPARATION, n)].any():
                continue
            if farm.edge[a:b].any() or not farm.contig[a + 1 : b].all():
                continue
            if not good[s : s + length].all() or farm.year[a] != farm.year[b - 1]:
                continue
            factor = rng.uniform(*G0_TARGET) * farm.cap / farm.power[s : s + length].mean()
            power[a:b] = farm.power[a:b] * factor
            g0[s : s + length] = run_id
            occupied[max(a - SEPARATION, 0) : min(b + SEPARATION, n)] = True
            run_id += 1
            placed += 1
    return power, inj, g0, blocks


def weighted_threshold(*, scores: list[np.ndarray], fd: float) -> float:
    """Return the score above which a share `fd` of the equal-weight pooled sets lies."""
    values = np.concatenate(scores)
    weights = np.concatenate([np.full(len(s), 1.0 / (len(scores) * len(s))) for s in scores])
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    idx = int(np.searchsorted(cum, 1.0 - fd))
    return float(values[order][min(idx, len(values) - 1)])


def fit_d1(
    *, x_pos: np.ndarray, neg_sets: list[np.ndarray], rng: np.random.Generator
) -> dict[str, Any]:
    """Fit the standardised, class-balanced logistic regression of idle against the negatives."""

    def cap(x: np.ndarray) -> np.ndarray:
        return (
            x[rng.choice(len(x), MAX_TRAIN_PER_SET, replace=False)]
            if len(x) > MAX_TRAIN_PER_SET
            else x
        )

    pos = cap(x_pos)
    negs = [cap(x) for x in neg_sets]
    x = np.vstack([pos, *negs])
    y = np.concatenate([np.ones(len(pos)), np.zeros(sum(len(n) for n in negs))])
    w = np.concatenate(
        [np.full(len(pos), 1.0 / len(pos))]
        + [np.full(len(n), 1.0 / (len(negs) * len(n))) for n in negs]
    )
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit((x - mean) / std, y, sample_weight=w * len(y))
    return {
        "med": med,
        "mean": mean,
        "std": std,
        "coef": model.coef_[0],
        "icpt": model.intercept_[0],
    }


def d1_score(*, model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Return D1's log-odds of idle."""
    x = np.where(np.isfinite(x), x, model["med"])
    return ((x - mean_std(model, "mean")) / model["std"]) @ model["coef"] + model["icpt"]


def mean_std(model: dict[str, Any], key: str) -> np.ndarray:
    """Return a stored array."""
    return model[key]


def all_flags(
    *,
    farm: Farm,
    ctx: dict[str, Any],
    fitted: dict[str, Any],
    power: np.ndarray,
    feats: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return every candidate's boolean flags for a series (the real one or a modified one)."""
    lo, hi = ctx["band"]
    in_band = (power >= lo) & (power <= hi)
    flags: dict[str, np.ndarray] = {}
    hi210 = ctx["idle_hi_210"]
    cut = farm.cut_in
    outage210 = runs_at_least(
        flag=(farm.wind > HIGH_FACTOR * cut) & (power <= hi210), contiguous=farm.contig, minimum=24
    )
    flags["R0"] = (farm.wind < ctx["wind_threshold"]) | outage210
    flags["R1"] = in_band
    flags["I2"] = runs_at_least(
        flag=in_band & (farm.wind > HIGH_FACTOR * cut), contiguous=farm.contig, minimum=48
    )
    score = d1_score(model=fitted["d1"], x=feats)
    flags["D1"] = score > fitted["t_d1"]
    for p, name in ((0.3, "D1_p30"), (0.7, "D1_p70")):
        flags[name] = score > np.log(p / (1 - p))
    for fd, name in zip(FD_LEVELS, ("D1_fd01", "D1_fd02", "D1_fd10"), strict=True):
        flags[name] = score > fitted["t_fd"][fd]
    j = farm.feat_names.index(fitted["fa_name"])
    flags["D1a"] = fitted["fa_sign"] * feats[:, j] > fitted["t_fa"]
    flags["D1a"] &= np.isfinite(feats[:, j])
    centres, medians = ctx["curve"]
    expected = np.interp(farm.wind, centres, medians)
    exp_s, pow_s = smooth(x=expected), smooth(x=power)
    for ratio in D3_RATIOS:
        name = "D3" if ratio == D3_RATIOS[0] else f"D3_{int(ratio * 100):02d}"
        flags[name] = (exp_s >= D3_MIN_EXPECTED * farm.cap) & (
            pow_s <= ratio * np.maximum(exp_s, 1e-9)
        )
    flags["D1&D3"] = flags["D1"] & flags["D3"]
    for rho in RHOS:
        name = "D4" if rho == RHOS[0] else f"D4_{int(rho * 100):02d}"
        flags[name] = viterbi_idle(
            logodds=score - fitted["t_d1"] + fitted["bias"][rho], segs=farm.segs, rho=rho
        )
    return flags


def tune_biases(
    *,
    farm: Farm,
    ctx: dict[str, Any],
    fitted: dict[str, Any],
    neg_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> dict[float, float]:
    """Choose each stay probability's bias so the tune-year false-drop rate is closest to 5%."""
    out: dict[float, float] = {}
    for rho in RHOS:
        best = (9.9, 0.0)
        for b in BIAS_GRID:
            rates = []
            for feats, rows_a, rows_b in neg_frames:
                score = d1_score(model=fitted["d1"], x=feats)
                state = viterbi_idle(logodds=score - fitted["t_d1"] + b, segs=farm.segs, rho=rho)
                rates.append(
                    state[rows_a].mean() * 0.5 + state[rows_b].mean() * 0.5
                    if rows_b is not None
                    else state[rows_a].mean()
                )
            pooled = (
                0.5 * rates[0] + 0.5 * float(np.mean(rates[1:])) if len(rates) > 1 else rates[0]
            )
            if os.environ.get("E444_DEBUG"):
                print(f"rho={rho} b={b} rates={[round(float(r), 4) for r in rates]}")
            gap = abs(pooled - TARGET_FD)
            if gap < best[0]:
                best = (gap, float(b))
        out[rho] = best[1]
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run both folds for the three farms and write the tables."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((SRC_DIR / "report.json").read_text())
    joined = pl.read_parquet(SRC_DIR / "joined.parquet").sort("site", "valid_time")
    results: dict[str, list[dict[str, Any]]] = {
        k: [] for k in ("inj", "neg", "bins", "high_year", "special", "census", "placed")
    }
    oof_flags: list[pl.DataFrame] = []
    auc_rows: list[pl.DataFrame] = []
    thresholds: list[dict[str, Any]] = []
    for site in SITES:
        farm = Farm(frame=joined.filter(pl.col("site") == site), cut_in=report[site]["cut_in_used"])
        rng = np.random.default_rng(SEED + int(site[1]))
        farm_flags = {}
        for parity in (1, 0):
            ctx = make_context(farm=farm, parity=parity)
            tune = ctx["tune"]
            lo, hi = ctx["band"]
            in_band = (farm.power >= lo) & (farm.power <= hi)
            ok_rows = ~farm.edge
            si = (farm.wind < IDLE_WIND_MAX) & in_band
            sg = (
                (farm.wind >= GEN_WIND_MIN)
                & (farm.share >= GEN_SHARE[0])
                & (farm.share <= GEN_SHARE[1])
            )
            sgm = (
                farm.high
                & (farm.wind < GEN_WIND_MIN)
                & (farm.share >= GEN_SHARE[0])
                & (farm.share <= GEN_SHARE[1])
            )
            clear = farm.high & (farm.share >= 0.10)
            reps = [
                place_replicate(farm=farm, ctx=ctx, parity=parity, rng=rng)
                for _ in range(N_REPLICATES)
            ]
            rep_feats = [
                feature_matrix(valid_time=farm.valid_time, power=r[0], capacity=farm.cap)[0]
                for r in reps
            ]
            # --- fit D1 on tune years
            g0_train = np.vstack(
                [f[(r[2] >= 0) & tune] for f, r in zip(rep_feats, reps, strict=True)]
            )
            x_pos = farm.feats[si & tune & ok_rows]
            x_sg = farm.feats[sg & tune & ok_rows]
            d1 = fit_d1(x_pos=x_pos, neg_sets=[x_sg, g0_train], rng=rng)
            s_sg = d1_score(model=d1, x=x_sg)
            s_g0 = d1_score(model=d1, x=g0_train)
            t_d1 = weighted_threshold(scores=[s_sg, s_g0], fd=TARGET_FD)
            t_fd = {fd: weighted_threshold(scores=[s_sg, s_g0], fd=fd) for fd in FD_LEVELS}
            # --- D1a: best single statistic from experiment 431
            fa_name = "sd_diff_3h_centre" if site == "W2" else "repeat_share_24h_centre"
            j = farm.feat_names.index(fa_name)
            pos_a = farm.feats[si & tune & ok_rows, j]
            neg_a = farm.feats[sg & tune & ok_rows, j]
            sign = (
                1.0
                if auc(
                    score=np.concatenate([pos_a[np.isfinite(pos_a)], neg_a[np.isfinite(neg_a)]]),
                    label=np.concatenate(
                        [np.ones(np.isfinite(pos_a).sum()), np.zeros(np.isfinite(neg_a).sum())]
                    ).astype(int),
                )
                >= 0.5
                else -1.0
            )
            g0_a = np.concatenate(
                [f[(r[2] >= 0) & tune, j] for f, r in zip(rep_feats, reps, strict=True)]
            )
            fitted: dict[str, Any] = {
                "d1": d1,
                "t_d1": t_d1,
                "t_fd": t_fd,
                "fa_name": fa_name,
                "fa_sign": sign,
            }
            sa = sign * np.concatenate([neg_a[np.isfinite(neg_a)]])
            sg0a = sign * g0_a[np.isfinite(g0_a)]
            fitted["t_fa"] = weighted_threshold(scores=[sa, sg0a], fd=TARGET_FD)
            # --- D4 biases on the real series and the first three replicates
            neg_frames = [(farm.feats, sg & tune & ok_rows, None)] + [
                (rep_feats[i], (reps[i][2] >= 0) & tune, None) for i in range(min(5, N_REPLICATES))
            ]
            fitted["bias"] = tune_biases(farm=farm, ctx=ctx, fitted=fitted, neg_frames=neg_frames)
            thresholds.append(
                {
                    "site": site,
                    "parity": parity,
                    "band_lo": ctx["band"][0],
                    "band_hi": ctx["band"][1],
                    "wind_threshold": ctx["wind_threshold"],
                    "t_d1": t_d1,
                    "bias_d4_95": fitted["bias"][0.95],
                    "bias_d4_90": fitted["bias"][0.90],
                    "bias_d4_98": fitted["bias"][0.98],
                }
            )
            # --- score on the test years
            test = ~tune & ok_rows
            flags0 = all_flags(
                farm=farm, ctx=ctx, fitted=fitted, power=farm.power, feats=farm.feats
            )
            for name, fl in flags0.items():
                farm_flags.setdefault(name, np.zeros(farm.n, dtype=bool))[~tune] = fl[~tune]
            d3_flag = flags0["D3"]
            years_test = np.unique(farm.year[~tune])
            for name, fl in flags0.items():
                for set_name, mask in (("SG", sg), ("SGm", sgm), ("clear", clear)):
                    for y in years_test:
                        sel = mask & test & (farm.year == y)
                        results["neg"].append(
                            {
                                "site": site,
                                "cand": name,
                                "set": set_name,
                                "year": int(y),
                                "n": int(sel.sum()),
                                "n_flag": int(fl[sel].sum()),
                            }
                        )
                        if set_name == "SG":
                            sel2 = sel & ~d3_flag
                            results["neg"].append(
                                {
                                    "site": site,
                                    "cand": name,
                                    "set": "SG_not_D3",
                                    "year": int(y),
                                    "n": int(sel2.sum()),
                                    "n_flag": int(fl[sel2].sum()),
                                }
                            )
                # real-series wind curve and high-wind flag rate per year
                bins = np.floor(farm.wind).astype(int)
                for b in np.unique(bins[test]):
                    sel = test & (bins == b)
                    results["bins"].append(
                        {
                            "site": site,
                            "cand": name,
                            "wind_bin": int(b),
                            "n": int(sel.sum()),
                            "n_flag": int(fl[sel].sum()),
                        }
                    )
                for y in years_test:
                    sel = test & farm.high & (farm.year == y)
                    results["high_year"].append(
                        {
                            "site": site,
                            "cand": name,
                            "year": int(y),
                            "n": int(sel.sum()),
                            "n_flag": int(fl[sel].sum()),
                        }
                    )
                cutout = test & (farm.wind >= 22) & in_band
                neg_read = test & farm.high & (farm.power < -0.005)
                neg_all = test & (farm.power < -0.005)
                for sname, sel in (
                    ("cutout_idle", cutout),
                    ("negative_high_wind", neg_read),
                    ("negative_any_wind", neg_all),
                ):
                    results["special"].append(
                        {
                            "site": site,
                            "cand": name,
                            "set": sname,
                            "n": int(sel.sum()),
                            "n_flag": int(fl[sel].sum()),
                        }
                    )
            # level-matched AUC rows for D1 on SG against SI in the test years
            sc = d1_score(model=d1, x=farm.feats)
            sel = test & (si | sg)
            auc_rows.append(
                pl.DataFrame(
                    {
                        "site": site,
                        "year": farm.year[sel],
                        "score": sc[sel],
                        "idle": si[sel].astype(int),
                        "level": farm.share[sel],
                    }
                )
            )
            # agreement census at high wind
            hw = test & farm.high
            for a_name, a_val in (("flag", True), ("noflag", False)):
                for b_name, b_val in (("D3flag", True), ("D3noflag", False)):
                    results["census"].append(
                        {
                            "site": site,
                            "kind": "D1_vs_D3",
                            "cell": f"D1_{a_name}_{b_name}",
                            "n": int((hw & (flags0["D1"] == a_val) & (d3_flag == b_val)).sum()),
                        }
                    )
            # --- injected and synthetic series
            for r_i, ((power, inj, g0, blocks), feats) in enumerate(
                zip(reps, rep_feats, strict=True)
            ):
                g0_ids = np.unique(g0[g0 >= 0])
                n_test_runs = sum(
                    int(farm.year[np.flatnonzero(g0 == g)[0]] % 2 != parity) for g in g0_ids
                )
                results["placed"].append(
                    {
                        "site": site,
                        "fold": parity,
                        "rep": r_i,
                        "g0s_test_runs": n_test_runs,
                        "g0s_tune_runs": len(g0_ids) - n_test_runs,
                        "inj_blocks": len(blocks),
                    }
                )
                fl_all = all_flags(farm=farm, ctx=ctx, fitted=fitted, power=power, feats=feats)
                for name, fl in fl_all.items():
                    for b in blocks:
                        rows = np.flatnonzero(inj == b["block_id"])
                        if farm.year[rows[0]] % 2 == parity:
                            continue
                        joinset = set(b["join_rows"])
                        keep = np.array([x not in joinset for x in rows])
                        results["inj"].append(
                            {
                                "site": site,
                                "cand": name,
                                "fold": parity,
                                "rep": r_i,
                                "length": b["length"],
                                "start": b["start"],
                                "bin": b["bin"],
                                "year": b["year"],
                                "recall": float(fl[rows].mean()),
                                "recall_nojoin": float(fl[rows[keep]].mean())
                                if keep.any()
                                else float("nan"),
                            }
                        )
                    for g in np.unique(g0[g0 >= 0]):
                        rows = np.flatnonzero(g0 == g)
                        if farm.year[rows[0]] % 2 == parity:
                            continue
                        results["neg"].append(
                            {
                                "site": site,
                                "cand": name,
                                "set": "G0s",
                                "year": int(farm.year[rows[0]]),
                                "n": len(rows),
                                "n_flag": int(fl[rows].sum()),
                            }
                        )
        flag_frame = pl.DataFrame(
            {
                "site": site,
                "valid_time": farm.valid_time,
                "year": farm.year,
                "edge": farm.edge,
                "high": farm.high,
                "wind": farm.wind,
                "power_share": farm.share,
            }
        )
        for name in ("D1", "D4", "D3", "R0"):
            flag_frame = flag_frame.with_columns(pl.Series(name, farm_flags[name]))
        oof_flags.append(flag_frame)
    for key, rows in results.items():
        pl.DataFrame(rows).write_parquet(OUT_DIR / f"{key}.parquet")
    pl.concat(auc_rows).write_parquet(OUT_DIR / "auc_rows_private.parquet")
    pl.concat(oof_flags).write_parquet(OUT_DIR / "oof_flags_private.parquet")
    pl.DataFrame(thresholds).write_parquet(OUT_DIR / "thresholds.parquet")
    _ = stratified_auc


if __name__ == "__main__":
    main()
