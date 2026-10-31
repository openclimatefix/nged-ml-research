"""Experiment 426 stage 0: do neighbouring Primaries carry usable residual information? (no fits).

Reads experiment 397's stored `plain_pooled` forecasts, experiment 415's stored N1 forecasts at the
four nodes, experiment 386's stored monthly XGBoost control, experiment 87's cleaned observed panel,
and NGED's private interconnection and hierarchy files (read only, only D-labels leave this script).

Run: `PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations uv run
--no-sync python ml-experiments/00426-*/e00426_stage0.py` with `DATA_PATH_INTERNAL` set.
"""

import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR
OUT_DIR: Final[Path] = (
    DATA_DIR / "00426-mdn-neighbouring-primaries-graph-and-bottom-up-design" / "stage0"
)
E397: Final[Path] = (
    DATA_DIR / "00397-plain-and-attention-mixture-density-network-for-primary-demand" / "work"
)
E415: Final[Path] = (
    DATA_DIR / "00415-pooled-mixture-density-network-at-bulk-and-grid-supply-points" / "work"
)
E386: Final[Path] = (
    DATA_DIR / "00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost" / "work" / "arm_preds"
)
HIERARCHY_DIR: Final[Path] = DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
METADATA_PATH: Final[Path] = DATA_DIR.parent / "NGED" / "metadata.parquet"
LABEL_TABLE: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "00395-xgboost-level-state-winter-spread-anchor-stored-forecasts"
    / "sw_block_detector_label_table_corrected.csv"
)

MONTHS: Final[tuple[str, ...]] = ("2025-10", "2025-12", "2026-01", "2026-03", "2026-06")
NODE_MONTHS: Final[tuple[str, ...]] = ("2025-10", "2026-01", "2026-03", "2026-06")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
NODES: Final[tuple[str, ...]] = ("D13", "D17", "D1", "D6")
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
CUT_SITES: Final[tuple[str, ...]] = ("D15", "D18")
CUT_TIME: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
N_GRAPHS: Final[int] = 1000
N_GRAPHS_RIDGE: Final[int] = 30
MAX_ATTEMPTS: Final[int] = 200_000
RIDGE_GRID: Final[tuple[float, ...]] = (0.1, 1.0, 10.0, 100.0, 1000.0)
BANDS: Final[dict[str, tuple[float, float]]] = {"d1": (12.0, 36.0), "d5_10": (120.0, 240.0)}
RNG_SEED: Final[int] = 426
T0: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
N_STEPS: Final[int] = 48 * 365


# --- loading ---------------------------------------------------------------------------------


def series_info() -> pl.DataFrame:
    """Return the 20 series with site, type, units, coordinates, capacity, and index."""
    return common.series_table().sort("site").with_row_index("idx")


def observed_mw_free(*, info: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned capacity-normalised power with the D15 and D18 start cut applied."""
    obs = (
        pl.read_parquet(common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
        .join(info.select("time_series_id", "site"), on="time_series_id", validate="m:1")
        .drop_nulls("power")
    )
    return obs.filter(~(pl.col("site").is_in(CUT_SITES) & (pl.col("valid_time") < CUT_TIME)))


def load_pooled(*, info: pl.DataFrame, obs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return residual rows of stored `plain_pooled` forecasts and a per-month drop count.

    The q50 is averaged over the 3 seeds. The residual is the actual minus that q50, in capacity
    units. Rows with no cleaned actual are dropped and counted.
    """
    frames, counts = [], []
    for month in MONTHS:
        seeds = [
            pl.read_parquet(
                E397 / "seed_preds" / "summary" / "plain_pooled" / f"{month}_s{s}.parquet"
            )
            .select("time_series_id", "power_fcst_init_time", "valid_time", "q50")
            .rename({"power_fcst_init_time": "origin"})
            for s in SEEDS
        ]
        mean = (
            pl.concat(seeds)
            .group_by("time_series_id", "origin", "valid_time")
            .agg(pl.col("q50").mean())
        )
        joined = mean.join(
            obs.select("time_series_id", "valid_time", "power"),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
        ).join(info.select("time_series_id", "site", "idx"), on="time_series_id", validate="m:1")
        counts.append(
            {
                "month": month,
                "rows": joined.height,
                "no_actual": int(joined["power"].null_count()),
                "no_actual_d15_d18": int(
                    joined.filter(
                        pl.col("power").is_null() & pl.col("site").is_in(CUT_SITES)
                    ).height
                ),
            }
        )
        frames.append(
            joined.drop_nulls("power")
            .with_columns(
                y=pl.col("power") - pl.col("q50"),
                month=pl.lit(month),
                lead_h=(pl.col("valid_time") - pl.col("origin")).dt.total_minutes() / 60.0,
            )
            .select("idx", "site", "month", "origin", "valid_time", "lead_h", "y")
        )
    return pl.concat(frames), pl.DataFrame(counts)


def step_index(*, times: pl.Series) -> np.ndarray:
    """Return half-hour grid indices of `times` from `T0`."""
    micro = times.dt.epoch("us").to_numpy()
    return ((micro - int(T0.timestamp()) * 1_000_000) // (30 * 60 * 1_000_000)).astype(np.int64)


def lag_residual_grid(*, res: pl.DataFrame, n_series: int) -> np.ndarray:
    """Return residuals of the shortest-lead (lead under 24 h) stored forecast on the time grid."""
    short = res.filter(pl.col("lead_h") < 24.0)
    grid = np.full((n_series, N_STEPS), np.nan)
    grid[short["idx"].to_numpy(), step_index(times=short["valid_time"])] = short["y"].to_numpy()
    return grid


def target_rows(*, res: pl.DataFrame, band: tuple[float, float]) -> pl.DataFrame:
    """Return targets in a lead band with the two lag time indices.

    Lag 1 is the newest half-hour before the origin. Lag 2 is the target's time of day on the
    latest day at or before lag 1.
    """
    low, high = band
    rows = res.filter((pl.col("lead_h") >= low) & (pl.col("lead_h") < high))
    s1 = rows["origin"] - STEP
    gap_h = (rows["valid_time"] - s1).dt.total_minutes() / 60.0
    k = np.ceil(gap_h.to_numpy() / 24.0)
    s1_idx = step_index(times=s1)
    t_idx = step_index(times=rows["valid_time"])
    return rows.with_columns(
        s1=pl.Series(s1_idx), s2=pl.Series(t_idx - (k * 48).astype(np.int64)), t=pl.Series(t_idx)
    )


# --- graphs ----------------------------------------------------------------------------------


def haversine_km(*, lat: np.ndarray, lon: np.ndarray, a: int, b: int) -> float:
    """Return the great-circle distance between two series in kilometres."""
    p1, p2 = math.radians(lat[a]), math.radians(lat[b])
    dl = math.radians(lon[b] - lon[a])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(h))


def true_edges(*, info: pl.DataFrame) -> list[tuple[int, int]]:
    """Return the undirected interconnections among the 16 modelled Primaries (private file)."""
    primaries = info.filter(pl.col("substation_type") == "Primary")
    numbers = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "substation_number"])
    lab = primaries.join(numbers, on="time_series_id", validate="1:1")
    by_number = dict(zip(lab["substation_number"], lab["idx"], strict=True))
    pairs = pl.read_csv(HIERARCHY_DIR / "Primary Substation Interconnections.csv")
    edges = {
        tuple(sorted((by_number[a], by_number[b])))
        for a, b in pairs.iter_rows()
        if a in by_number and b in by_number
    }
    return sorted(edges)


def random_graphs(
    *,
    edges: list[tuple[int, int]],
    lengths: np.ndarray,
    length_matched: bool,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[list[tuple[int, int]]], int]:
    """Return degree-preserving random graphs by double edge swaps, and the attempts used.

    With `length_matched`, a graph is kept only if its mean edge length is within 10% and its
    median edge length within 15% of the interconnected graph's.
    """
    true_len = [lengths[a, b] for a, b in edges]
    mean_t, med_t = float(np.mean(true_len)), float(np.median(true_len))
    current = [list(e) for e in edges]
    present = {tuple(e) for e in edges}
    out: list[list[tuple[int, int]]] = []
    attempts = 0
    while len(out) < count and attempts < MAX_ATTEMPTS:
        attempts += 1
        for _ in range(10):
            i, j = rng.choice(len(current), size=2, replace=False)
            a, b = current[i]
            c, d = current[j]
            if rng.random() < 0.5:
                c, d = d, c
            new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
            if a == d or c == b or new1 in present or new2 in present or new1 == new2:
                continue
            present -= {tuple(current[i]), tuple(current[j])}
            present |= {new1, new2}
            current[i], current[j] = list(new1), list(new2)
        if length_matched:
            lens = [lengths[a, b] for a, b in current]
            if (
                abs(np.mean(lens) - mean_t) > 0.10 * mean_t
                or abs(np.median(lens) - med_t) > 0.15 * med_t
            ):
                continue
        out.append([(a, b) for a, b in current])
    return out, attempts


def adjacency(*, graphs: Sequence[Sequence[tuple[int, int]]], n: int) -> np.ndarray:
    """Return symmetric boolean adjacency matrices, shape (graphs, n, n)."""
    adj = np.zeros((len(graphs), n, n), dtype=bool)
    for g, edges in enumerate(graphs):
        for a, b in edges:
            adj[g, a, b] = adj[g, b, a] = True
    return adj


# --- stage 0a --------------------------------------------------------------------------------


def corr_matrix(*, rows: pl.DataFrame, grid: np.ndarray, lag_col: str, n: int) -> np.ndarray:
    """Return C[i, j]: correlation of target i's residual with series j's residual at the lag."""
    out = np.full((n, n), np.nan)
    for i in range(n):
        sub = rows.filter(pl.col("idx") == i)
        if sub.height < 50:
            continue
        y = sub["y"].to_numpy()
        lag = sub[lag_col].to_numpy()
        for j in range(n):
            if j == i:
                continue
            x = grid[j, lag]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 50 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                out[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def graph_means(*, corr: np.ndarray, adj: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return the mean correlation over ordered edges whose target is in `targets`."""
    weight = adj & targets[None, :, None] & np.isfinite(corr)[None]
    total = np.where(weight, np.nan_to_num(corr)[None], 0.0).sum(axis=(1, 2))
    return total / np.maximum(weight.sum(axis=(1, 2)), 1)


def lag_average(
    *, lag_stats: dict[str, np.ndarray], adj: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Return the mean over the two lag types of each graph's mean lagged correlation."""
    return np.mean([graph_means(corr=c, adj=adj, targets=mask) for c in lag_stats.values()], axis=0)


def stage_0a(
    *,
    res: pl.DataFrame,
    grid: np.ndarray,
    adj_true: np.ndarray,
    adj_rand: np.ndarray,
    adj_len: np.ndarray,
    info: pl.DataFrame,
) -> pl.DataFrame:
    """Return the per-month, per-band, per-group lagged cross-correlation table."""
    n = info.height
    groups = {
        "six": np.isin(info["site"].to_numpy(), SIX),
        "eight": np.array(
            [
                (u == "MVA") and (t == "Primary")
                for u, t in zip(info["units"], info["substation_type"], strict=True)
            ]
        ),
    }
    records = []
    for band, window in BANDS.items():
        rows_all = target_rows(res=res, band=window)
        for month in MONTHS:
            rows = rows_all.filter(pl.col("month") == month)
            lag_stats = {
                lag: corr_matrix(rows=rows, grid=grid, lag_col=lag, n=n) for lag in ("s1", "s2")
            }
            for group, mask in groups.items():
                true_v = float(lag_average(lag_stats=lag_stats, adj=adj_true, mask=mask)[0])
                rand_v = lag_average(lag_stats=lag_stats, adj=adj_rand, mask=mask)
                len_v = lag_average(lag_stats=lag_stats, adj=adj_len, mask=mask)
                records.append(
                    {
                        "band": band,
                        "month": month,
                        "group": group,
                        "true": true_v,
                        "random_mean": float(rand_v.mean()),
                        "random_sd": float(rand_v.std()),
                        "length_mean": float(len_v.mean()),
                        "length_sd": float(len_v.std()),
                        "excess_random": true_v - float(rand_v.mean()),
                        "excess_length": true_v - float(len_v.mean()),
                    }
                )
    table = pl.DataFrame(records)
    return table.with_columns(
        pass_month=(
            (pl.col("excess_random") >= 0.10)
            & (pl.col("excess_random") >= 2 * pl.col("random_sd"))
            & (pl.col("excess_length") >= 0.05)
            & (pl.col("excess_length") >= 2 * pl.col("length_sd"))
        )
    )


# --- stage 0b --------------------------------------------------------------------------------


def events_by_site() -> dict[str, list[tuple[int, int]]]:
    """Return labelled event windows (half-hour grid indices) for each Primary named in `pair`."""
    labels = pl.read_csv(LABEL_TABLE)
    out: dict[str, list[tuple[int, int]]] = {}
    for row in labels.iter_rows(named=True):
        on = datetime.strptime(row["onset_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        end = datetime.strptime(row["end_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        window = (
            int((on - T0).total_seconds() // 1800),
            int((end - T0).total_seconds() // 1800),
        )
        for token in row["pair"].replace("with", " ").replace(",", " ").split():
            if token.startswith("D") and token[1:].isdigit():
                out.setdefault(token, []).append(window)
    return out


def recipient_share() -> tuple[int, int]:
    """Return (rows naming exactly one Primary, rows in the label table)."""
    labels = pl.read_csv(LABEL_TABLE)
    one = sum(
        1
        for pair in labels["pair"]
        if sum(
            1
            for t in pair.replace("with", " ").replace(",", " ").split()
            if t[:1] == "D" and t[1:].isdigit()
        )
        == 1
    )
    return one, labels.height


def neighbour_features(
    *, grid: np.ndarray, idx: np.ndarray, s: np.ndarray, adj_row: np.ndarray, own: int
) -> np.ndarray:
    """Return mean, min, max of available neighbours' lag residuals at each row, plus a flag."""
    nbrs = np.flatnonzero(adj_row[own])
    if nbrs.size == 0:
        return np.zeros((idx.size, 4))
    vals = grid[nbrs][:, s]
    with np.errstate(all="ignore"):
        have = np.isfinite(vals)
        any_ok = have.any(axis=0)
        mean = np.where(any_ok, np.nanmean(np.where(have, vals, np.nan), axis=0), 0.0)
        mn = np.where(any_ok, np.nanmin(np.where(have, vals, np.nan), axis=0), 0.0)
        mx = np.where(any_ok, np.nanmax(np.where(have, vals, np.nan), axis=0), 0.0)
    return np.column_stack([mean, mn, mx, any_ok.astype(float)])


def design(
    *, rows: pl.DataFrame, grid: np.ndarray, adj_row: np.ndarray | None, own: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, keep) for one Primary's rows; adj_row None gives the own-lags-only design."""
    s1, s2 = rows["s1"].to_numpy(), rows["s2"].to_numpy()
    y = rows["y"].to_numpy()
    cols = [grid[own, s1], grid[own, s2]]
    keep = np.isfinite(cols[0]) & np.isfinite(cols[1]) & np.isfinite(y)
    feats = [np.column_stack(cols)]
    if adj_row is not None:
        n1 = neighbour_features(
            grid=grid, idx=rows["idx"].to_numpy(), s=s1, adj_row=adj_row, own=own
        )
        n2 = neighbour_features(
            grid=grid, idx=rows["idx"].to_numpy(), s=s2, adj_row=adj_row, own=own
        )
        feats += [n1[:, :3], n2[:, :3], np.maximum(n1[:, 3], n2[:, 3])[:, None]]
    x = np.column_stack(feats)
    return x, y, keep


def ridge_fit(*, xtx: np.ndarray, xty: np.ndarray, lam: float) -> np.ndarray:
    """Return ridge coefficients with an unpenalised intercept in the last column."""
    pen = np.eye(xtx.shape[0]) * lam
    pen[-1, -1] = 0.0
    return np.linalg.solve(xtx + pen, xty)


def sse(*, beta: np.ndarray, xtx: np.ndarray, xty: np.ndarray, yty: float) -> float:
    """Return the sum of squared errors from sufficient statistics."""
    return float(yty - 2 * beta @ xty + beta @ xtx @ beta)


def month_blocks(
    *,
    rows: pl.DataFrame,
    grid: np.ndarray,
    adj_row: np.ndarray | None,
    own: int,
    mask: np.ndarray | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, float, int]]:
    """Return sufficient statistics (X'X, X'y, y'y, n) per month for one Primary and one arm."""
    x, y, keep = design(rows=rows, grid=grid, adj_row=adj_row, own=own)
    if mask is not None:
        keep = keep & mask
    months = rows["month"].to_numpy()
    out = {}
    for month in MONTHS:
        sel = keep & (months == month)
        xa = np.column_stack([x[sel], np.ones(sel.sum())])
        out[month] = (xa.T @ xa, xa.T @ y[sel], float(y[sel] @ y[sel]), int(sel.sum()))
    return out


def test_sse(
    *, blocks: dict[str, tuple[np.ndarray, np.ndarray, float, int]]
) -> dict[str, tuple[float, int]]:
    """Return held-out SSE and row count per test month (penalty by leave-one-month-out)."""
    out = {}
    for test in MONTHS:
        train = [m for m in MONTHS if m != test]

        def pooled(months: Sequence[str]) -> tuple[np.ndarray, np.ndarray, float]:
            return (
                np.sum([blocks[m][0] for m in months], axis=0),
                np.sum([blocks[m][1] for m in months], axis=0),
                float(sum(blocks[m][2] for m in months)),
            )

        best, best_err = RIDGE_GRID[0], math.inf
        for lam in RIDGE_GRID:
            err = 0.0
            for hold in train:
                inner = [m for m in train if m != hold]
                xtx, xty, _ = pooled(inner)
                beta = ridge_fit(xtx=xtx, xty=xty, lam=lam)
                err += sse(beta=beta, xtx=blocks[hold][0], xty=blocks[hold][1], yty=blocks[hold][2])
            if err < best_err:
                best, best_err = lam, err
        xtx, xty, _ = pooled(train)
        beta = ridge_fit(xtx=xtx, xty=xty, lam=best)
        out[test] = (
            sse(beta=beta, xtx=blocks[test][0], xty=blocks[test][1], yty=blocks[test][2]),
            blocks[test][3],
        )
    return out


def stage_0b(
    *,
    res: pl.DataFrame,
    grid: np.ndarray,
    adj_true: np.ndarray,
    adj_rand: np.ndarray,
    adj_len: np.ndarray,
    info: pl.DataFrame,
) -> pl.DataFrame:
    """Return held-out SSE per group, month, arm, and event-row handling."""
    events = events_by_site()
    sites = info["site"].to_list()
    groups = {
        "six": [info["site"].to_list().index(s) for s in SIX],
        "eight": [
            i
            for i, (u, t) in enumerate(zip(info["units"], info["substation_type"], strict=True))
            if u == "MVA" and t == "Primary"
        ],
        "d15_d18": [sites.index(s) for s in CUT_SITES],
    }
    arms: dict[str, list[np.ndarray | None]] = {
        "own": [None],
        "true": [adj_true[0]],
        "random": list(adj_rand[:N_GRAPHS_RIDGE]),
        "length": list(adj_len[:N_GRAPHS_RIDGE]),
    }
    records = []
    rows_all = target_rows(res=res, band=BANDS["d1"])
    for group, members in groups.items():
        for own in members:
            rows = rows_all.filter(pl.col("idx") == own).sort("origin", "valid_time")
            if rows.height == 0:
                continue
            in_event = np.zeros(rows.height, dtype=bool)
            for lo, hi in events.get(sites[own], []):
                for col in ("t", "s1", "s2"):
                    v = rows[col].to_numpy()
                    in_event |= (v >= lo) & (v <= hi)
            for handling, mask in (("all_rows", None), ("no_event_rows", ~in_event)):
                for arm, graphs in arms.items():
                    for draw, adj in enumerate(graphs):
                        adj_row = None if adj is None else adj
                        errs = test_sse(
                            blocks=month_blocks(
                                rows=rows, grid=grid, adj_row=adj_row, own=own, mask=mask
                            )
                        )
                        for month, (e, n_rows) in errs.items():
                            records.append(
                                {
                                    "group": group,
                                    "series": sites[own],
                                    "handling": handling,
                                    "arm": arm,
                                    "draw": draw,
                                    "month": month,
                                    "sse": e,
                                    "n": n_rows,
                                }
                            )
    return pl.DataFrame(records)


def arm_month(*, by: pl.DataFrame, arm: str) -> np.ndarray:
    """Return an arm's per-month SSE, averaged over its random-graph draws."""
    sub = by.filter(pl.col("arm") == arm).group_by("month").agg(pl.col("sse").mean()).sort("month")
    return sub["sse"].to_numpy()


def reduction(*, true: np.ndarray, control: np.ndarray, idx: np.ndarray) -> float:
    """Return the percentage SSE reduction of the true graph against a control over months `idx`."""
    return float(100 * (1 - true[idx].sum() / control[idx].sum()))


def summarise_0b(*, table: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return the SSE reduction of the true graph against each control, with intervals."""
    out = []
    for (group, handling), part in table.group_by(["group", "handling"]):
        by = (
            part.group_by("arm", "draw", "month")
            .agg(pl.col("sse").sum())
            .sort("arm", "draw", "month")
        )

        own, true = arm_month(by=by, arm="own"), arm_month(by=by, arm="true")
        rand, length = arm_month(by=by, arm="random"), arm_month(by=by, arm="length")
        for label, control in (("vs_own", own), ("vs_random", rand), ("vs_length", length)):
            point = reduction(true=true, control=control, idx=np.arange(len(true)))
            boots = [
                reduction(true=true, control=control, idx=rng.integers(0, len(true), len(true)))
                for _ in range(2000)
            ]
            out.append(
                {
                    "group": group,
                    "handling": handling,
                    "contrast": label,
                    "reduction_pct": point,
                    "lo": float(np.percentile(boots, 2.5)),
                    "hi": float(np.percentile(boots, 97.5)),
                }
            )
    return pl.DataFrame(out)


# --- stage 0c --------------------------------------------------------------------------------


def children_of_nodes(*, info: pl.DataFrame) -> dict[str, list[str]]:
    """Return the modelled Primaries beneath each node (private hierarchy file; labels only)."""
    numbers = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "substation_number"])
    lab = info.join(numbers, on="time_series_id", validate="1:1")
    hier = pl.read_csv(HIERARCHY_DIR / "Substations.csv").join(
        lab.select("site", "substation_type", "substation_number"), on="substation_number"
    )
    out = {}
    for row in lab.filter(pl.col("site").is_in(NODES)).iter_rows(named=True):
        col = (
            "bsp_substation_number" if row["substation_type"] == "BSP" else "gsp_substation_number"
        )
        kids = hier.filter(
            (pl.col(col) == row["substation_number"]) & (pl.col("substation_type") == "Primary")
        )
        out[row["site"]] = sorted(kids["site"].to_list(), key=lambda s: int(s[1:]))
    return out


def _read_preds(*, paths: Sequence[Path], extra: str | None = None) -> pl.DataFrame:
    frames = [
        pl.read_parquet(p).select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "q50",
            *([extra] if extra else []),
        )
        for p in paths
    ]
    return (
        pl.concat(frames)
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(pl.col("q50").mean(), *([pl.col(extra).first()] if extra else []))
    )


def stage_0c(*, info: pl.DataFrame, obs: pl.DataFrame, res_pooled: pl.DataFrame) -> pl.DataFrame:
    """Return node MAE (points of capacity) for N1, the XGBoost control, and bottom-up variants."""
    kids = children_of_nodes(info=info)
    cap = dict(zip(info["site"], info["effective_capacity_mw"], strict=True))
    ids = dict(zip(info["site"], info["time_series_id"], strict=True))
    wide = obs.pivot(on="site", index="valid_time", values="power", aggregate_function="first")
    mw = wide.with_columns(
        [(pl.col(s) * cap[s]).alias(s) for s in wide.columns if s != "valid_time"]
    )
    child_fc = (
        pl.concat(
            [
                pl.read_parquet(
                    E397 / "seed_preds" / "summary" / "plain_pooled" / f"{m}_s{s}.parquet"
                )
                .select("time_series_id", "power_fcst_init_time", "valid_time", "q50")
                .with_columns(month=pl.lit(m))
                for m in NODE_MONTHS
                for s in SEEDS
            ]
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time", "month")
        .agg(pl.col("q50").mean())
        .join(info.select("time_series_id", "site"), on="time_series_id")
    )
    records = []
    for node in NODES:
        n1 = (
            pl.concat(
                [
                    pl.read_parquet(E415 / "seed_preds" / "N1" / f"{m}_s{s}.parquet")
                    .filter(pl.col("time_series_id") == ids[node])
                    .select("power_fcst_init_time", "valid_time", "q50", "actual")
                    .with_columns(month=pl.lit(m))
                    for m in NODE_MONTHS
                    for s in SEEDS
                ]
            )
            .group_by("power_fcst_init_time", "valid_time", "month")
            .agg(pl.col("q50").mean().alias("n1"), pl.col("actual").first())
        )
        control = (
            _read_preds(paths=sorted((E386 / "monthly" / "control").glob(f"{node}_s0_*.parquet")))
            .filter(pl.col("time_series_id") == ids[node])
            .rename({"q50": "xgb"})
            .drop("time_series_id")
        )
        csum = (
            child_fc.filter(pl.col("site").is_in(kids[node]))
            .with_columns(
                mw=pl.col("q50") * pl.col("site").replace_strict(cap, return_dtype=pl.Float64)
            )
            .group_by("power_fcst_init_time", "valid_time", "month")
            .agg(pl.col("mw").sum().alias("child_fc"), pl.len().alias("n_kids"))
            .filter(pl.col("n_kids") == len(kids[node]))
        )
        shared = (
            n1.join(control, on=["power_fcst_init_time", "valid_time"])
            .join(csum, on=["power_fcst_init_time", "valid_time", "month"])
            .with_columns(
                lead_h=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60.0
            )
            .filter(pl.col("lead_h") < 240.0)
            .drop_nulls("actual")
        )
        node_mw = mw.select("valid_time", pl.col(node).alias("node_mw"), *kids[node]).drop_nulls()
        node_mw = node_mw.with_columns(child_act=pl.sum_horizontal(kids[node]))
        for month in NODE_MONTHS:
            start = datetime.strptime(month + "-01", "%Y-%m-%d").replace(tzinfo=UTC)
            train = node_mw.filter(pl.col("valid_time") < start)
            b = np.polyfit(train["child_act"].to_numpy(), train["node_mw"].to_numpy(), 1)
            lag_src = mw.select("valid_time", pl.col(node).alias("v")).drop_nulls()
            lookup = dict(zip(lag_src["valid_time"].to_list(), lag_src["v"].to_list(), strict=True))
            part = shared.filter(pl.col("month") == month)
            if part.height == 0:
                continue
            origin = part["power_fcst_init_time"].to_list()
            valid = part["valid_time"].to_list()

            lagm = np.array(
                [lag_values(lookup=lookup, o=o, t=t) for o, t in zip(origin, valid, strict=True)]
            )
            child = part["child_fc"].to_numpy()
            lead = part["lead_h"].to_numpy()
            variant_a = b[0] * child + b[1]
            # variant B: per-lead-band ridge on [child, lag_origin, lag_day, lag_week], training
            # rows from origins before the scored month with the children's actual sum in place of
            # their forecast.
            variant_b = np.full(part.height, np.nan)
            for lo, hi in ((0, 48), (48, 120), (120, 240)):
                tr = _train_band(node_mw=node_mw, lookup=lookup, start=start, lo=lo, hi=hi)
                if tr is None:
                    continue
                x_tr, y_tr = tr
                mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
                xs = (x_tr - mu) / sd
                beta = np.linalg.solve(xs.T @ xs + 1.0 * np.eye(4), xs.T @ (y_tr - y_tr.mean()))
                sel = (lead >= lo) & (lead < hi)
                feat = np.column_stack([child, lagm])[sel]
                variant_b[sel] = ((feat - mu) / sd) @ beta + y_tr.mean()
            actual = part["actual"].to_numpy() * cap[node]
            keep = np.isfinite(variant_b) & np.isfinite(lagm).all(axis=1)
            for name, pred in (
                ("N1", part["n1"].to_numpy() * cap[node]),
                ("xgb_control", part["xgb"].to_numpy() * cap[node]),
                ("bottom_up_A", variant_a),
                ("bottom_up_B_with_node_lags", variant_b),
            ):
                records.append(
                    {
                        "node": node,
                        "month": month,
                        "arm": name,
                        "mae_points": float(
                            100 * np.abs(pred[keep] - actual[keep]).mean() / cap[node]
                        ),
                        "n": int(keep.sum()),
                    }
                )
    return pl.DataFrame(records)


def lag_values(*, lookup: dict, o: datetime, t: datetime) -> tuple[float, float, float]:
    """Return the node's actual at the newest half-hour before the origin, day lag, and week lag."""
    s1 = o - STEP
    k_day = math.ceil((t - s1) / timedelta(hours=24))
    k_wk = math.ceil((t - s1) / timedelta(hours=168))
    return (
        lookup.get(s1, math.nan),
        lookup.get(t - timedelta(hours=24 * k_day), math.nan),
        lookup.get(t - timedelta(hours=168 * k_wk), math.nan),
    )


def _train_band(
    *, node_mw: pl.DataFrame, lookup: dict, start: datetime, lo: float, hi: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return training features and targets for one lead band from days before `start`."""
    rows_x, rows_y = [], []
    origin = datetime(2025, 3, 2, 9, tzinfo=UTC)
    child = dict(zip(node_mw["valid_time"].to_list(), node_mw["child_act"].to_list(), strict=True))
    while origin < start:
        for step in range(int(lo * 2), int(hi * 2), 6):
            t = origin + STEP * step
            if t >= start:
                break
            s1 = origin - STEP
            k_day = math.ceil((t - s1) / timedelta(hours=24))
            k_wk = math.ceil((t - s1) / timedelta(hours=168))
            vals = (
                child.get(t, math.nan),
                lookup.get(s1, math.nan),
                lookup.get(t - timedelta(hours=24 * k_day), math.nan),
                lookup.get(t - timedelta(hours=168 * k_wk), math.nan),
                lookup.get(t, math.nan),
            )
            if all(math.isfinite(v) for v in vals):
                rows_x.append(vals[:4])
                rows_y.append(vals[4])
        origin += timedelta(days=1)
    if len(rows_y) < 200:
        return None
    return np.array(rows_x), np.array(rows_y)


def children_share(*, info: pl.DataFrame, obs: pl.DataFrame) -> pl.DataFrame:
    """Return, per node, the children's share of mean demand and their correlation with it."""
    kids = children_of_nodes(info=info)
    cap = dict(zip(info["site"], info["effective_capacity_mw"], strict=True))
    wide = obs.pivot(on="site", index="valid_time", values="power", aggregate_function="first")
    mw = wide.with_columns(
        [(pl.col(s) * cap[s]).alias(s) for s in wide.columns if s != "valid_time"]
    )
    mw = mw.filter(pl.col("valid_time") >= CUT_TIME)
    out = []
    for node, members in kids.items():
        d = mw.select(node, *members).drop_nulls()
        s, y = d.select(members).to_numpy().sum(1), d[node].to_numpy()
        out.append(
            {
                "node": node,
                "n_children": len(members),
                "rows": len(y),
                "child_share_of_mean": float(s.mean() / y.mean()),
                "corr": float(np.corrcoef(s, y)[0, 1]),
            }
        )
    return pl.DataFrame(out)


# --- main ------------------------------------------------------------------------------------


def main() -> None:
    """Run stage 0 and write the tables and a JSON summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    info = series_info()
    obs = observed_mw_free(info=info)
    res, drops = load_pooled(info=info, obs=obs)
    drops.write_csv(OUT_DIR / "dropped_rows.csv")
    print("rows without a cleaned actual, by fold:\n", drops)
    grid = lag_residual_grid(res=res, n_series=info.height)
    edges = true_edges(info=info)
    lat, lon = info["latitude"].to_numpy(), info["longitude"].to_numpy()
    n = info.height
    lengths = np.array(
        [[haversine_km(lat=lat, lon=lon, a=a, b=b) for b in range(n)] for a in range(n)]
    )
    rand, att_r = random_graphs(
        edges=edges, lengths=lengths, length_matched=False, count=N_GRAPHS, rng=rng
    )
    matched, att_l = random_graphs(
        edges=edges, lengths=lengths, length_matched=True, count=N_GRAPHS, rng=rng
    )
    print(
        f"edges {len(edges)}; random graphs {len(rand)} ({att_r} attempts); "
        f"length-matched {len(matched)} ({att_l} attempts)"
    )
    adj_true = adjacency(graphs=[edges], n=n)
    adj_rand = adjacency(graphs=rand, n=n)
    adj_len = adjacency(graphs=matched, n=n)
    table_a = stage_0a(
        res=res, grid=grid, adj_true=adj_true, adj_rand=adj_rand, adj_len=adj_len, info=info
    )
    table_a.write_csv(OUT_DIR / "stage0a.csv")
    six_d1 = table_a.filter((pl.col("group") == "six") & (pl.col("band") == "d1"))
    gate_a = int(six_d1["pass_month"].sum())
    print(table_a)
    table_b = stage_0b(
        res=res, grid=grid, adj_true=adj_true, adj_rand=adj_rand, adj_len=adj_len, info=info
    )
    table_b.write_csv(OUT_DIR / "stage0b_raw.csv")
    summary_b = summarise_0b(table=table_b, rng=rng)
    summary_b.write_csv(OUT_DIR / "stage0b.csv")
    print(summary_b.sort("group", "handling", "contrast"))
    share = children_share(info=info, obs=obs)
    share.write_csv(OUT_DIR / "stage0c_share.csv")
    print(share)
    table_c = stage_0c(info=info, obs=obs, res_pooled=res)
    table_c.write_csv(OUT_DIR / "stage0c.csv")
    print(table_c.group_by("node", "arm").agg(pl.col("mae_points").mean()).sort("node", "arm"))
    one, total = recipient_share()
    six_b = summary_b.filter(pl.col("group") == "six")
    gate_b = bool(
        all(
            six_b.filter((pl.col("contrast") == c) & (pl.col("handling") == h))["reduction_pct"][0]
            >= thr
            for h in ("all_rows", "no_event_rows")
            for c, thr in (("vs_own", 2.0), ("vs_random", 1.0), ("vs_length", 1.0))
        )
    )
    summary = {
        "gate_0a_months_passed_of_5": gate_a,
        "gate_0a_pass": gate_a >= 3,
        "gate_0b_pass": gate_b,
        "advance_to_stage_1": gate_a >= 3 and gate_b,
        "events_naming_one_primary": one,
        "events_total": total,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
