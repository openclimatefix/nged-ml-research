"""Experiment 402, step 1: analytic ceilings of a gate between two stored forecasts.

For each row the CRPS of the mixture `w F1 + (1 - w) F2` is exactly `A w^2 + B w + C`, with `F1`
the forecast with power lags and `F2` the forecast without. The script gets `A`, `B`, and `C` from
the exact CRPS at `w = 0`, `0.5`, and `1`, then evaluates in closed form the per-row oracle weight,
a flag oracle, a fitted constant mixture, and experiment 400's fixed 50/50 mixture. Stored
forecasts: experiment 400's `arm_lags` (experiment 381's weekly bundle) and `arm_lagless`
(experiment 383's raw lagless arm), over experiment 395's Part F rows. Both are raw (no widening).

```bash
PYTHONPATH=ml-experiments/00400-lagged-lagless-forecast-blend-around-load-shifts-stored \
DATA_PATH_INTERNAL=... systemd-run --user --scope -p MemoryMax=5G -p OOMPolicy=continue \
    env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00402-*/e00402_ceilings.py
```
"""

import sys
from pathlib import Path
from typing import Final

import e00402_mixture as mix
import numpy as np
import polars as pl
from studies.eval_protocol.metrics import QUANTILE_COLUMNS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
W400: Final[Path] = DATA / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored" / "work"
WORK: Final[Path] = DATA / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]
CHUNK: Final[int] = 20_000
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
N_BOOT: Final[int] = 2000
AFTER_HOURS: Final[float] = 168.0
"""The 7 days after an event ends, in hours."""


def abc_table() -> pl.DataFrame:
    """Return `A`, `B`, `C` (points of effective capacity squared scale) of every scored row."""
    rows = pl.read_parquet(
        W400 / "rows_lags.parquet",
        columns=[*KEYS, "label", "lead_hours", "capacity_mw", "actual_mw", "flagged"],
    ).sort(KEYS)
    q = {}
    for name in ("lags", "lagless"):
        frame = rows.select(*KEYS).join(
            pl.read_parquet(W400 / f"arm_{name}.parquet"), on=KEYS, how="left", maintain_order="left"
        )
        q[name] = frame.select(list(QUANTILE_COLUMNS)).to_numpy().astype(np.float64)
    cap = rows["capacity_mw"].to_numpy()
    actual = rows["actual_mw"].to_numpy() / cap
    crps = {w: np.empty(rows.height) for w in (0.0, 0.5, 1.0)}
    for start in range(0, rows.height, CHUNK):
        sl = slice(start, start + CHUNK)
        c = cap[sl, None]
        pair = mix.Pair(q1=q["lags"][sl] / c, q2=q["lagless"][sl] / c, actual=actual[sl])
        for w, out in crps.items():
            out[sl] = pair.crps(w=np.full(pair.x.shape[0], w)) * 100.0
    c0, c_half, c1 = crps[0.0], crps[0.5], crps[1.0]
    a = 2.0 * (c1 + c0 - 2.0 * c_half)
    b = c1 - c0 - a
    return rows.with_columns(
        A=pl.Series(a), B=pl.Series(b), C=pl.Series(c0), crps_lags=pl.Series(c1)
    )


def lead_bin(*, hours: np.ndarray) -> np.ndarray:
    """Return the lead time in whole days (1 to 14)."""
    return np.clip(np.ceil(hours / 24.0).astype(int), 1, 14)


def fit_const(*, a: np.ndarray, b: np.ndarray, key: np.ndarray) -> dict[int, float]:
    """Return the best constant weight per `key` value: `-sum(B) / (2 sum(A))`, clipped to [0, 1]."""
    out = {}
    for k in np.unique(key):
        m = key == k
        out[int(k)] = float(np.clip(-b[m].sum() / (2.0 * max(a[m].sum(), 1e-12)), 0.0, 1.0))
    return out


def crps_at(*, t: pl.DataFrame, w: np.ndarray) -> np.ndarray:
    """Return the exact mixture CRPS of each row at weight `w`."""
    return t["A"].to_numpy() * w**2 + t["B"].to_numpy() * w + t["C"].to_numpy()


def boot(*, diff: np.ndarray, cluster: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return the mean difference and its 95% interval, resampling whole clusters."""
    ids, inv = np.unique(cluster, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(ids))
    n = np.bincount(inv, minlength=len(ids)).astype(float)
    draws = rng.integers(0, len(ids), size=(N_BOOT, len(ids)))
    est = s[draws].sum(axis=1) / n[draws].sum(axis=1)
    return float(diff.mean()), *np.quantile(est, [0.025, 0.975]).tolist()


def main() -> None:
    """Write the table of `A`, `B`, `C` and print the ceilings."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "abc.parquet"
    if path.exists():
        t = pl.read_parquet(path)
    else:
        t = abc_table()
        t.write_parquet(path)
    tags = pl.read_parquet(W400 / "tags_corrected.parquet")
    t = t.join(tags, on=KEYS, how="left", maintain_order="left", validate="1:1")
    t = t.filter(~pl.col("flagged") & pl.col("A").is_finite() & pl.col("C").is_finite())
    print("usable rows", t.height, flush=True)
    t = t.with_columns(
        lead_day=pl.Series(lead_bin(hours=t["lead_hours"].to_numpy())),
        harm=(pl.col("lag_count") >= 1) & (pl.col("target_event_id") < 0),
        clean=(pl.col("lag_count") == 0)
        & (pl.col("target_event_id") < 0)
        & ~(pl.col("hours_after_end") <= AFTER_HOURS),
    )
    a, b = t["A"].to_numpy(), t["B"].to_numpy()
    lead = t["lead_day"].to_numpy()
    harm, clean = t["harm"].to_numpy(), t["clean"].to_numpy()
    lag_count = t["lag_count"].to_numpy()
    scored = harm | clean
    const = fit_const(a=a[scored], b=b[scored], key=lead[scored])
    w_const = np.array([const[k] for k in lead])
    w_clean = fit_const(a=a[clean], b=b[clean], key=lead[clean])
    w_harm = fit_const(a=a[harm], b=b[harm], key=lead[harm])
    w_flag = np.where(harm, [w_harm.get(k, 0.5) for k in lead], [w_clean.get(k, 0.5) for k in lead])
    flag_by_count = np.empty(t.height)
    by_count = {c: fit_const(a=a[harm & (lag_count == c)], b=b[harm & (lag_count == c)], key=lead[harm & (lag_count == c)]) for c in (1, 2, 3)}
    for c in (1, 2, 3):
        m = harm & (lag_count == c)
        flag_by_count[m] = [by_count[c].get(k, 0.5) for k in lead[m]]
    flag_by_count[~harm] = w_flag[~harm]
    group = np.where(np.isin(t["label"].to_numpy(), SIX), "six", "eight")
    w_const_g = np.empty(t.height)
    w_flag_g = np.empty(t.height)
    for g in ("six", "eight"):
        gm = group == g
        c = fit_const(a=a[scored & gm], b=b[scored & gm], key=lead[scored & gm])
        wc = fit_const(a=a[clean & gm], b=b[clean & gm], key=lead[clean & gm])
        wh = fit_const(a=a[harm & gm], b=b[harm & gm], key=lead[harm & gm])
        w_const_g[gm] = [c.get(k, 0.5) for k in lead[gm]]
        w_flag_g[gm] = np.where(
            harm[gm], [wh.get(k, 0.5) for k in lead[gm]], [wc.get(k, 0.5) for k in lead[gm]]
        )
        print(g, "harm weights per lead day:", {k: round(v, 3) for k, v in wh.items()})
    w_oracle = np.clip(-b / (2.0 * np.maximum(a, 1e-12)), 0.0, 1.0)
    arms = {
        "forecast with lags (w = 1)": np.ones(t.height),
        "forecast without lags (w = 0)": np.zeros(t.height),
        "fixed 50/50 mixture": np.full(t.height, 0.5),
        "fitted constant mixture (per lead day)": w_const,
        "flag oracle (per lead day)": w_flag,
        "flag oracle by number of lags (per lead day)": flag_by_count,
        "fitted constant mixture per group (per lead day)": w_const_g,
        "flag oracle per group (per lead day)": w_flag_g,
        "per-row oracle weight": w_oracle,
    }
    score = {name: crps_at(t=t, w=w) for name, w in arms.items()}
    label = t["label"].to_numpy()
    month = t["valid_time"].dt.truncate("1mo").dt.epoch("us").to_numpy() // 10**6 + 10**9
    cluster = np.where(harm, t["lag_event_id"].to_numpy(), month)
    rng = np.random.default_rng(0)
    for set_name, mask in (("harm rows (lags in an event, target outside)", harm), ("clean rows", clean)):
        for group_name, members in (("six signed-MW", SIX), ("eight MVA-only", EIGHT), ("all 16", SIX + EIGHT)):
            m = mask & np.isin(label, members)
            if not m.any():
                continue
            print(f"\n== {set_name}, {group_name}: {int(m.sum()):,} rows, {len(np.unique(cluster[m]))} clusters (events for harm rows, months for clean rows)")
            for name, s in score.items():
                print(f"  CRPS {name}: {s[m].mean():.3f}")
            for name in list(score)[2:]:
                for ref in (
                    "forecast with lags (w = 1)",
                    "fitted constant mixture (per lead day)",
                    "fitted constant mixture per group (per lead day)",
                ):
                    if name == ref:
                        continue
                    est, lo, hi = boot(diff=(score[name] - score[ref])[m], cluster=cluster[m], rng=rng)
                    print(f"  {name} minus {ref}: {est:+.3f} (95% {lo:+.3f} to {hi:+.3f})")
    print("\n== per Primary, harm rows: CRPS lags / without / const per group / flag per group; clean rows: lags / without")
    ev = pl.read_parquet(W400 / "events_corrected.parquet")
    for lab in SIX + EIGHT:
        mh = harm & (label == lab)
        mc = clean & (label == lab)
        e = ev.filter(pl.col("label") == lab)
        print(
            f"  {lab}: harm n={int(mh.sum())} "
            + " / ".join(f"{score[k][mh].mean():.2f}" for k in list(score)[:2] + list(score)[7:9])
            + f" | clean "
            + " / ".join(f"{score[k][mc].mean():.2f}" for k in list(score)[:2])
            + f" | events {e.height}, median depth {e['depth_share'].median():.1f}% of capacity,"
            f" median length {e['length_days'].median():.1f} days"
        )
    print("\nfitted constant weights per lead day:", {k: round(v, 3) for k, v in const.items()})
    print("flag oracle harm weights:", {k: round(v, 3) for k, v in w_harm.items()})


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
