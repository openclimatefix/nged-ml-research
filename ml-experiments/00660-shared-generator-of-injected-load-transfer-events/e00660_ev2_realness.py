"""Evidence 2: can a classifier tell injected event windows from real ones?

    DATA_PATH_INTERNAL=... PYTHONPATH=<this folder> uv run --no-sync python e00660_ev2_realness.py

Real windows are the events that are visible in the telemetry (experiment 580's verified transfers
and unpaired steps of at least a day). Injected windows are positive events of default-
configuration worlds, drawn so that their size histogram in noise units equals the real one. The
features are what a step detector reads at an onset and an end of the residual (observed power minus
the lag-free forecast): the size of the step, how much of it arrives in the first 6 hours and how
much before the onset, the noise inside the event against outside, how much the step varies with
the time of day, the drift inside the event, the size of the end step against the onset step, and
the day-to-day correlation of the residual. The score is the area under the curve (AUC) of a
gradient-boosted classifier under repeated grouped cross-validation, against a label-shuffle null.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import e00660_data as data
import e00660_ev34_detector as ev
import e00660_generator as gen
import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(
    0, str(Path(__file__).parents[1] / "00653-edge-state-event-table-penalised-segmentation")
)
FEATURES = (
    "log_step_units",
    "first6h_share",
    "pre_share",
    "noise_ratio",
    "diurnal_modulation",
    "drift_share",
    "end_over_start",
    "end_first6h_share",
    "lag_day_corr",
)
W3 = 3 * gen.DAY
N_WORLDS = 10
N_REPEATS = 20
N_NULL = 200
FIRST_START = data.idx_of(when=data.GRID_START) + 6 * gen.DAY
LAST_START = ev.GRID_LENGTH - 6 * gen.DAY


def _med(x: np.ndarray, *, need: int = 6) -> float:
    ok = x[np.isfinite(x)]
    return float(np.median(ok)) if len(ok) >= need else float("nan")


def snap(*, r: np.ndarray, t: int) -> int:
    """Move a nominal step time to the half-hour (within a day) with the largest local step.

    The same rule is applied to real and injected windows, so timing error in the logged or fitted
    onset of a real event does not separate the two.
    """
    best, best_t = -1.0, t
    for u in range(t - gen.DAY, t + gen.DAY + 1):
        if u < 24 or u + 24 > len(r):
            continue
        a, b = _med(r[u - 24 : u], need=12), _med(r[u : u + 24], need=12)
        if np.isfinite(a) and np.isfinite(b) and abs(b - a) > best:
            best, best_t = abs(b - a), u
    return best_t


def _mad(x: np.ndarray) -> float:
    ok = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(ok - np.median(ok)))) if len(ok) >= 12 else float("nan")


def features(*, r: np.ndarray, sigma3: float, start: int, end: int | None) -> dict | None:
    """The detector-facing features of one event window, or None if the residual is missing."""
    start = snap(r=r, t=start)
    if end is not None:
        end = snap(r=r, t=end)
    stop = min(end if end is not None else start + W3, ev.GRID_LENGTH)
    if start < W3 or stop - start < gen.DAY:
        return None
    pre = r[start - W3 : start - 4]
    post_hi = min(stop - 4, start + W3)
    post = r[start + 4 : post_hi]
    p0, p1 = _med(pre), _med(post)
    step = p1 - p0
    if not np.isfinite(step) or abs(step) < 1e-9:
        return None
    first6 = _med(r[start : start + 12])
    before = _med(r[start - 12 : start])
    by_hod = []
    for h in range(gen.DAY):
        idx = np.arange(start + 4 + h, post_hi, gen.DAY)
        pidx = np.arange(start - W3 + h, start - 4, gen.DAY)
        a, b = _med(r[idx], need=2), _med(r[pidx], need=2)
        if np.isfinite(a) and np.isfinite(b):
            by_hod.append(a - b)
    mad_pre, mad_post = _mad(pre - p0), _mad(post - p1)
    third = max((stop - start) // 3, 1)
    drift = (_med(r[stop - third : stop - 4]) - _med(r[start + 4 : start + third])) / step
    centred = post - p1
    corr = float("nan")
    if len(centred) > 2 * gen.DAY:
        day = centred[: len(centred) - gen.DAY], centred[gen.DAY :]
        ok = np.isfinite(day[0]) & np.isfinite(day[1])
        if ok.sum() > 24:
            corr = float(np.corrcoef(day[0][ok], day[1][ok])[0, 1])
    end_ratio = end_first6 = float("nan")
    if end is not None and end + W3 <= ev.GRID_LENGTH:
        e_pre = _med(r[max(end - W3, start + 4) : end - 4])
        e_post = _med(r[end + 4 : end + W3])
        end_ratio = -(e_post - e_pre) / step
        e_first6 = _med(r[end : end + 12])
        end_first6 = (
            (e_first6 - e_pre) / (e_post - e_pre) if abs(e_post - e_pre) > 1e-9 else float("nan")
        )
    return {
        "log_step_units": float(np.log10(abs(step) / sigma3)),
        "first6h_share": float(np.clip((first6 - p0) / step, -2, 3)),
        "pre_share": float(np.clip((before - p0) / step, -2, 3)),
        "noise_ratio": float(mad_post / mad_pre) if mad_pre else float("nan"),
        "diurnal_modulation": float(np.std(by_hod) / abs(step))
        if len(by_hod) > 24
        else float("nan"),
        "drift_share": float(np.clip(drift, -3, 3)),
        "end_over_start": float(np.clip(end_ratio, -3, 3)),
        "end_first6h_share": float(np.clip(end_first6, -2, 3)),
        "lag_day_corr": corr,
    }


def real_windows(*, rig: ev.Rig) -> pl.DataFrame:
    """Feature rows of the real events visible in the telemetry (all years from 2022)."""
    off = int((data.GRID_START - data.LABEL_ORIGIN).total_seconds() // 1800)
    rows = []

    def add(site: str, on: object, end: object, source: str, key: int) -> None:
        if site not in rig.real_residual or on is None or end is None:
            return
        s, e = int(on) - off, int(end) - off  # ty: ignore[invalid-argument-type]
        if not FIRST_START <= s < LAST_START or e - s < gen.DAY:
            return
        f = features(r=rig.real_residual[site], sigma3=rig.frozen.sigma3[site], start=s, end=e)
        if f:
            rows.append({"real": 1, "group": key, "site": site, "start": s, **f})

    for i, r in enumerate(pl.read_parquet(data.E580 / "events_verified.parquet").to_dicts()):
        add(r["primary_a"], r["onset_a_idx"], r["end_a_idx"], "pair", 10_000 + i)
        add(r["primary_b"], r["onset_b_idx"], r["end_b_idx"], "pair", 10_000 + i)
    for r in pl.read_parquet(data.E580 / "events_labelled_status.parquet").to_dicts():
        if str(r["final_class"]).startswith("unpaired step"):
            add(
                r["label"], r["fit_onset_idx"], r["fit_end_idx"], "unpaired", 20_000 + r["event_id"]
            )
    return pl.DataFrame(rows)


def injected_windows(*, rig: ev.Rig, config: gen.GeneratorConfig, seed0: int) -> pl.DataFrame:
    """Feature rows of injected positive events (of at least a day) in several worlds."""
    rows = []
    for k in range(N_WORLDS):
        world = gen.inject(panel=rig.panel, config=config, seed=seed0 + k, start_times=rig.starts)
        res = rig.residual_of(world=world)
        pos = world.events.filter(pl.col("label").is_in(ev.POSITIVE))
        for r in pos.to_dicts():
            end = r["end"]
            if not FIRST_START <= r["start"] < LAST_START:
                continue
            if end is not None and end - r["start"] < gen.DAY:
                continue
            f = features(
                r=res[r["site"]], sigma3=rig.frozen.sigma3[r["site"]], start=r["start"], end=end
            )
            if f:
                rows.append(
                    {
                        "real": 0,
                        "group": 1_000_000 * (k + 1) + r["group_id"],
                        "site": r["site"],
                        "start": r["start"],
                        "units": r["units"],
                        **f,
                    }
                )
    return pl.DataFrame(rows)


def match_sizes(
    *, real: pl.DataFrame, inj: pl.DataFrame, rng: np.random.Generator, ratio: int = 8
) -> pl.DataFrame:
    """Sample injected rows so that their step-size histogram equals the real one (times ratio)."""
    edges = np.quantile(real["log_step_units"].to_numpy(), np.linspace(0, 1, 5))
    edges[0], edges[-1] = -np.inf, np.inf
    keep = []
    rb = np.digitize(real["log_step_units"].to_numpy(), edges[1:-1])
    ib = np.digitize(inj["log_step_units"].to_numpy(), edges[1:-1])
    for b in range(4):
        want = int((rb == b).sum()) * ratio
        pool = np.flatnonzero(ib == b)
        if len(pool):
            keep += list(rng.choice(pool, size=min(want, len(pool)), replace=False))
    return inj[sorted(keep)]


def cv_auc(*, table: pl.DataFrame, features_used: tuple[str, ...], seed: int) -> float:
    """Grouped, stratified 5-fold cross-validated AUC of one repeat."""
    x = table.select(features_used).to_numpy()
    med = np.nanmedian(x, axis=0)
    x = np.nan_to_num(np.where(np.isfinite(x), x, med))
    y = table["real"].to_numpy()
    g = table["group"].to_numpy()
    pred = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, g):
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.1, random_state=seed
        )
        w = np.where(y[tr] == 1, (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), 1.0)
        clf.fit(x[tr], y[tr], sample_weight=w)
        pred[te] = clf.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, pred))


def score(*, table: pl.DataFrame, features_used: tuple[str, ...]) -> dict:
    """Mean AUC over repeats, and a label-shuffle null for the same procedure."""
    aucs = [cv_auc(table=table, features_used=features_used, seed=s) for s in range(N_REPEATS)]
    rng = np.random.default_rng(0)
    null = []
    for s in range(N_NULL // 10):
        shuffled = table.with_columns(real=pl.Series(rng.permutation(table["real"].to_numpy())))
        null.append(cv_auc(table=shuffled, features_used=features_used, seed=s))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs)),
        "null_mean": float(np.mean(null)),
        "null_p95": float(np.percentile(null, 95)),
    }


def run(*, config: gen.GeneratorConfig, tag: str) -> dict:
    """AUC for one configuration with all features, then dropping each feature in turn."""
    rig = ev.Rig()
    rng = np.random.default_rng(2)
    real = real_windows(rig=rig)
    inj = match_sizes(real=real, inj=injected_windows(rig=rig, config=config, seed0=7000), rng=rng)
    table = pl.concat(
        [real.select(["real", "group", *FEATURES]), inj.select(["real", "group", *FEATURES])]
    )
    out = {"tag": tag, "n_real": real.height, "n_injected": inj.height}
    out["all"] = score(table=table, features_used=FEATURES)
    out["without"] = {
        f: score(table=table, features_used=tuple(x for x in FEATURES if x != f))["auc_mean"]
        for f in FEATURES
    }
    out["only"] = {f: score(table=table, features_used=(f,))["auc_mean"] for f in FEATURES}
    out["real_median"] = {f: float(np.nanmedian(real[f].to_numpy())) for f in FEATURES}
    out["injected_median"] = {f: float(np.nanmedian(inj[f].to_numpy())) for f in FEATURES}
    real.write_parquet(data.WORK / f"ev2_real_{tag}.parquet")
    inj.write_parquet(data.WORK / f"ev2_injected_{tag}.parquet")
    return out


if __name__ == "__main__":
    data.WORK.mkdir(parents=True, exist_ok=True)
    configs = {"default": gen.default_config()}
    if "variants" in sys.argv:
        configs |= {
            "constant_square": gen.constant_mw_square(),
            "shaped_short_heavy": gen.shaped_transfers(),
        }
    for tag, cfg in configs.items():
        res = run(config=cfg, tag=tag)
        (data.WORK / f"ev2_{tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
