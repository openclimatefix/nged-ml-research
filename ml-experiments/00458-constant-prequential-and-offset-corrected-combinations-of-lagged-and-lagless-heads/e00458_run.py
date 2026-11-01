"""Experiment 458: fit the combination rules on earlier months and score every arm (no network fit).

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00402-lag-gate-learned-weight-injected-load-shifts-part-1:\
ml-experiments/00458-constant-prequential-and-offset-corrected-combinations-of-lagged-and-lagless-heads\
DATA_PATH_INTERNAL=<private data folder> OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00458-*/e00458_run.py [--check]
```

`--check` runs the fidelity check (a 50/50 mixture of the stored heads must reproduce the stored
B-mix) on one cell and stops.
"""

import argparse
import time

import e00458_core as core  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

DEPTHS: tuple[str, ...] = ("0.75", "1.0", "1.25")
PAIRS: tuple[str, ...] = ("A0", "S1")
TEMPERATURES: np.ndarray = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
OFFSET_GRID: np.ndarray = np.arange(0.0, 1.51, 0.05)
MIN_FIT_ROWS: int = 200
SAVED_ARMS: tuple[str, ...] = ("B-mix", "half-A0", "b-A0", "c-A0", "d")


def check_fidelity() -> None:
    """Compare a recomputed 50/50 mixture with the stored B-mix on one cell."""
    cell = core.load_cell(month="2026-01", seed=0)
    q = core.mixture_quantiles(
        lagged=cell.q["A0"], lagless=cell.q["Z"], weights={"half": np.full(len(cell.actual), 0.5)}
    )["half"]
    err = np.abs(q - cell.q["B-mix"])
    score_new = core.pinball_points(q=q, actual=cell.actual)
    score_old = core.pinball_points(q=cell.q["B-mix"], actual=cell.actual)
    print("max abs quantile difference:", float(err.max()), "mean:", float(err.mean()))
    print("mean score recomputed vs stored:", score_new.mean(), score_old.mean())
    abc = core.integrals(lagged=cell.q["A0"], lagless=cell.q["Z"], actual=cell.actual)
    exact = core.exact_crps(abc=abc, w=np.full(len(cell.actual), 0.5)) * 100.0
    print("mean exact CRPS (points) at 50/50:", float(exact.mean()))


class Stash:
    """A seed-0 cell kept for fitting later months, with its integrals and window features."""

    def __init__(self, *, cell: core.Cell, series: dict[str, np.ndarray]) -> None:  # noqa: D107
        self.cell = cell
        self.abc = {
            "A0": core.integrals(lagged=cell.q["A0"], lagless=cell.q["Z"], actual=cell.actual),
            "S1": core.integrals(lagged=cell.q["S1"], lagless=cell.q["Z"], actual=cell.actual),
        }
        self.feat = core.window_features(cell=cell, series=series)
        self.arm_q: dict[str, np.ndarray] = {}


def training_rows(*, stash: list[Stash], month: str) -> list[tuple[Stash, np.ndarray]]:
    """Return earlier stashes with the rows whose target lies before the scored month."""
    start = core.month_start_index(month=month)
    return [(s, s.cell.target < start) for s in stash if s.cell.month < month]


def fit_params(*, train: list[tuple[Stash, np.ndarray]]) -> dict:
    """Fit every rule's parameters on the training rows; None means no earlier month."""
    if not train:
        return {}
    params: dict = {"b": {}, "b2": {}, "b3": {}, "T": {}, "k": {}, "n": {}}
    for pair in PAIRS:
        abc = np.concatenate([s.abc[pair][m] for s, m in train])
        grp = np.concatenate([s.cell.group[m] for s, m in train])
        band = np.concatenate([s.cell.band[m] for s, m in train])
        d = np.concatenate([s.feat[f"d_{pair}"][m] for s, m in train])
        for g in (0, 1, 2):
            sel = grp == g
            params["b2"][pair, g] = float(
                np.clip(-abc[sel, 1].sum() / max(abc[sel, 0].sum(), 1e-12), 0, 1)
            )
            for b in range(len(core.BAND_NAMES)):
                s2 = sel & (band == b)
                params["b"][pair, g, b] = float(
                    np.clip(-abc[s2, 1].sum() / max(abc[s2, 0].sum(), 1e-12), 0, 1)
                )
                params["n"][g, b] = int(s2.sum())
            loss = [
                core.exact_crps(
                    abc=abc[sel], w=core.prequential_weight(d=d[sel], temperature=t)
                ).sum()
                for t in TEMPERATURES
            ]
            params["T"][pair, g] = float(TEMPERATURES[int(np.argmin(loss))])
        for b in range(len(core.BAND_NAMES)):
            s3 = (grp < 2) & (band == b)
            params["b3"][pair, b] = float(
                np.clip(-abc[s3, 1].sum() / max(abc[s3, 0].sum(), 1e-12), 0, 1)
            )
    qz = np.concatenate([s.cell.q["Z"][m] for s, m in train])
    act = np.concatenate([s.cell.actual[m] for s, m in train])
    off = np.concatenate([s.feat["offset"][m] for s, m in train])
    grp = np.concatenate([s.cell.group[m] for s, m in train])
    band = np.concatenate([s.cell.band[m] for s, m in train])
    for g in (0, 1, 2):
        for b in range(len(core.BAND_NAMES)):
            sel = (grp == g) & (band == b)
            if sel.sum() < MIN_FIT_ROWS:
                params["k"][g, b] = 0.0
                continue
            loss = [
                core.pinball_points(q=qz[sel] + k * off[sel, None], actual=act[sel]).sum()
                for k in OFFSET_GRID
            ]
            params["k"][g, b] = float(OFFSET_GRID[int(np.argmin(loss))])
    return params


def row_weights(*, stash: Stash, params: dict) -> dict[tuple[str, str], np.ndarray]:
    """Return the per-row lagged weight of each (rule, pair); 0.5 where there are no parameters."""
    cell, n = stash.cell, len(stash.cell.actual)
    out: dict[tuple[str, str], np.ndarray] = {}
    for pair in PAIRS:
        for rule in ("b", "b2", "b3", "c"):
            w = np.full(n, 0.5)
            if params:
                if rule == "b":
                    w = np.array([[params["b"][pair, g, b] for b in range(5)] for g in range(3)])[
                        cell.group, cell.band
                    ]
                elif rule == "b2":
                    w = np.array([params["b2"][pair, g] for g in range(3)])[cell.group]
                elif rule == "b3":
                    w = np.array([params["b3"][pair, b] for b in range(5)])[cell.band]
                else:
                    t = np.array([params["T"][pair, g] for g in range(3)])[cell.group]
                    w = core.prequential_weight(d=stash.feat[f"d_{pair}"], temperature=t)
                    w = np.where(stash.feat["short_window"], 0.5, w)
            out[rule, pair] = w
    return out


def table_p(*, train: list[tuple[Stash, np.ndarray]], arm: str) -> dict:
    """Fit experiment 435's per-group, per-band tail factors on earlier no-label rows."""
    out: dict = {}
    if not train:
        return out
    q = np.concatenate([s.arm_q[arm][m & s.cell.rowset(name="nolabel")] for s, m in train])
    act = np.concatenate([s.cell.actual[m & s.cell.rowset(name="nolabel")] for s, m in train])
    grp = np.concatenate([s.cell.group[m & s.cell.rowset(name="nolabel")] for s, m in train])
    band = np.concatenate([s.cell.band[m & s.cell.rowset(name="nolabel")] for s, m in train])
    q50 = q[:, core.LEVEL_NAMES.index("q50")]
    for g in range(3):
        for b in range(5):
            sel = (grp == g) & (band == b)
            for name, tau in core.TAILS.items():
                qi = q[sel, core.LEVEL_NAMES.index(name)]
                spread = np.abs(qi - q50[sel])
                use = spread > 1e-6
                if use.sum() < MIN_FIT_ROWS:
                    continue
                if tau > 0.5:
                    ratio, level = (act[sel][use] - q50[sel][use]) / spread[use], tau
                else:
                    ratio, level = (q50[sel][use] - act[sel][use]) / spread[use], 1 - tau
                out[g, b, name] = float(np.quantile(ratio, level))
    return out


def apply_p(*, q: np.ndarray, factors: dict, group: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Scale each tail quantile's distance from the median by its factor."""
    out = q.copy()
    q50 = q[:, core.LEVEL_NAMES.index("q50")]
    for name in core.TAILS:
        i = core.LEVEL_NAMES.index(name)
        f = np.array([[factors.get((g, b, name), 1.0) for b in range(5)] for g in range(3)])
        out[:, i] = q50 + f[group, band] * (q[:, i] - q50)
    return out


def tail_stats(*, q: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return coverage and pinball loss at the four tail quantiles."""
    out: dict[str, float] = {}
    for name, tau in core.TAILS.items():
        qi = q[:, core.LEVEL_NAMES.index(name)]
        err = actual - qi
        out[f"cov_{name}"] = float((actual <= qi).mean())
        out[f"pin_{name}"] = float(np.where(err >= 0, tau * err, (tau - 1) * err).mean())
    return out


def event_clusters(*, cell: core.Cell, mask: np.ndarray) -> np.ndarray:
    """Return an event cluster id (Primary and event run) for `mask` rows."""
    ids = np.full(len(mask), -1, dtype=np.int64)
    p, t = cell.primary[mask], cell.target[mask]
    out = np.empty(len(p), dtype=np.int64)
    for prim in np.unique(p):
        sel = np.flatnonzero(p == prim)
        uniq = np.unique(t[sel])
        run = np.concatenate([[0], np.cumsum(np.diff(uniq) > 1)])
        out[sel] = prim * 10_000 + run[np.searchsorted(uniq, t[sel])]
    ids[mask] = out
    return ids


def score_cell(
    *,
    stash: Stash,
    arms: dict[str, np.ndarray],
    train: list[tuple[Stash, np.ndarray]],
    shares: dict,
) -> tuple[list[dict], list[pl.DataFrame]]:
    """Return summary rows (raw and after table P) and the event-row table of one cell."""
    cell = stash.cell
    rows: list[dict] = []
    event_mask = cell.rowset(name="event")
    cluster = event_clusters(cell=cell, mask=event_mask)
    event_frames: list[pl.DataFrame] = []
    grid = cell.target
    assert grid.max() < core.GRID_SIZE
    for arm, q in arms.items():
        factors = table_p(train=train, arm=arm)
        qp = apply_p(q=q, factors=factors, group=cell.group, band=cell.band)
        crps = core.pinball_points(q=q, actual=cell.actual)
        nra = {
            d: core.pinball_points(q=q, actual=cell.actual - shares[d][cell.primary, grid])
            for d in DEPTHS
        }
        for g, gname in enumerate(core.GROUPS):
            for setname in ("all", "nolabel", "harm", "event", "post"):
                base = (cell.group == g) & cell.rowset(name=setname)
                for bname in ("all", *core.BAND_NAMES):
                    sel = (
                        base
                        if bname == "all"
                        else base & (cell.band == core.BAND_NAMES.index(bname))
                    )
                    if not sel.any():
                        continue
                    row = {
                        "arm": arm,
                        "month": cell.month,
                        "seed": cell.seed,
                        "group": gname,
                        "rowset": setname,
                        "band": bname,
                        "n": int(sel.sum()),
                        "crps": float(crps[sel].mean()),
                    }
                    if setname in ("event", "post"):
                        for d in DEPTHS:
                            row[f"crps_nra{d}"] = float(nra[d][sel].mean())
                    for tag, qq in (("raw", q), ("p", qp)):
                        for k, v in tail_stats(q=qq[sel], actual=cell.actual[sel]).items():
                            row[f"{k}_{tag}"] = v
                    rows.append(row)
        if event_mask.any():
            event_frames.append(
                pl.DataFrame(
                    {
                        "arm": arm,
                        "month": cell.month,
                        "seed": cell.seed,
                        "group": cell.group[event_mask],
                        "cluster": cluster[event_mask],
                        "primary": cell.primary[event_mask],
                        "target": cell.target[event_mask],
                        "origin": cell.origin[event_mask],
                        "band": cell.band[event_mask],
                        "crps": crps[event_mask].astype(np.float32),
                        **{f"crps_nra{d}": nra[d][event_mask].astype(np.float32) for d in DEPTHS},
                    }
                )
            )
        if cell.seed == 0:
            stash.arm_q[arm] = q.astype(np.float32)
            if arm in SAVED_ARMS:
                folder = core.OUT_DIR / "quantiles" / arm
                folder.mkdir(parents=True, exist_ok=True)
                pl.DataFrame(
                    {
                        "primary": cell.primary.astype(np.int16),
                        "origin": cell.origin.astype(np.int32),
                        "target": cell.target.astype(np.int32),
                        "actual": cell.actual.astype(np.float32),
                        "tgt_real": cell.flags["tgt_real"],
                        "tgt_post": cell.flags["tgt_post"],
                        **{n: q[:, i].astype(np.float32) for i, n in enumerate(core.LEVEL_NAMES)},
                    }
                ).write_parquet(folder / f"{cell.month}.parquet")
    return rows, event_frames


def arms_of(*, stash: Stash, params: dict) -> dict[str, np.ndarray]:
    """Return the 11 quantiles of every arm for one cell."""
    cell = stash.cell
    weights = row_weights(stash=stash, params=params)
    arms: dict[str, np.ndarray] = {
        "B-mix": cell.q["B-mix"],
        "lagged-A0": cell.q["A0"],
        "lagged-S1": cell.q["S1"],
        "lagless": cell.q["Z"],
    }
    for pair in PAIRS:
        names = {f"{r}-{pair}": weights[r, pair] for r in ("b", "b2", "b3", "c")}
        names[f"half-{pair}"] = np.full(len(cell.actual), 0.5)
        got = core.mixture_quantiles(lagged=cell.q[pair], lagless=cell.q["Z"], weights=names)
        arms.update(got)
    k = (
        np.array([[params["k"][g, b] for b in range(5)] for g in range(3)])[cell.group, cell.band]
        if params
        else 0.0
    )
    arms["d"] = cell.q["Z"] + (k * stash.feat["offset"])[:, None] if params else cell.q["B-mix"]
    return arms


def main() -> None:
    """Run every cell in month order and write the tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_fidelity()
        return
    t0 = time.time()
    npz = np.load(core.NRA_SHARES)
    shares = {d: npz[f"f{d}"] for d in DEPTHS}
    core.OUT_DIR.mkdir(parents=True, exist_ok=True)
    series = {0: core.short_lead_series(seed=0)}
    stash: list[Stash] = []
    summary: list[dict] = []
    events: list[pl.DataFrame] = []
    weight_rows: list[dict] = []
    for month in core.MONTHS:
        seeds = (0, 1, 2) if month == "2026-06" else (0,)
        for seed in seeds:
            if seed not in series:
                series[seed] = core.short_lead_series(seed=seed)
            st = (
                Stash(cell=core.load_cell(month=month, seed=0), series=series[0])
                if seed == 0
                else Stash(cell=core.load_cell(month=month, seed=seed), series=series[seed])
            )
            train = training_rows(stash=stash, month=month)
            params = fit_params(train=train)
            arms = arms_of(stash=st, params=params)
            rows, frames = score_cell(stash=st, arms=arms, train=train, shares=shares)
            summary += rows
            events += frames
            if seed == 0:
                stash.append(st)
                for key, value in params.get("b", {}).items():
                    weight_rows.append(
                        {
                            "month": month,
                            "seed": 0,
                            "rule": f"b-{key[0]}",
                            "group": key[1],
                            "band": key[2],
                            "value": value,
                        }
                    )
                for key, value in params.get("b2", {}).items():
                    weight_rows.append(
                        {
                            "month": month,
                            "seed": 0,
                            "rule": f"b2-{key[0]}",
                            "group": key[1],
                            "band": -1,
                            "value": value,
                        }
                    )
                for key, value in params.get("T", {}).items():
                    weight_rows.append(
                        {
                            "month": month,
                            "seed": 0,
                            "rule": f"T-{key[0]}",
                            "group": key[1],
                            "band": -1,
                            "value": value,
                        }
                    )
                for key, value in params.get("k", {}).items():
                    weight_rows.append(
                        {
                            "month": month,
                            "seed": 0,
                            "rule": "k",
                            "group": key[0],
                            "band": key[1],
                            "value": value,
                        }
                    )
            w = row_weights(stash=st, params=params)
            for pair in PAIRS:
                wc = w["c", pair]
                for g in range(3):
                    sel = st.cell.group == g
                    weight_rows.append(
                        {
                            "month": month,
                            "seed": seed,
                            "rule": f"c-share-within-0.05-{pair}",
                            "group": g,
                            "band": -1,
                            "value": float((np.abs(wc[sel] - 0.5) <= 0.05).mean()),
                        }
                    )
            weight_rows.append(
                {
                    "month": month,
                    "seed": seed,
                    "rule": "short-window-share",
                    "group": -1,
                    "band": -1,
                    "value": float(st.feat["short_window"].mean()),
                }
            )
            print(f"{month} seed {seed} done at {time.time() - t0:.0f}s", flush=True)
    pl.DataFrame(summary).write_parquet(core.OUT_DIR / "summary.parquet")
    pl.concat(events).write_parquet(core.OUT_DIR / "events.parquet")
    pl.DataFrame(weight_rows).write_parquet(core.OUT_DIR / "weights.parquet")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
