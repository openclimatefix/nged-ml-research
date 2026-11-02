# ruff: noqa: E501
"""Experiment 514: injected worlds, the classifier, and its calibration.

```bash
PYTHONPATH=... DATA_PATH_INTERNAL=... OMP_NUM_THREADS=1 uv run --no-sync python \
    ml-experiments/00514-*/e00514_train.py worlds      # build all world sets (2 processes)
    ml-experiments/00514-*/e00514_train.py fit         # fit, calibrate, freeze, report
```
"""

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00514_data as d514  # ty: ignore[unresolved-import]
import e00514_detect as det  # ty: ignore[unresolved-import]
import e00514_inject as inj  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

WORK: Final[Path] = d514.WORK_DIR
TRAIN_MONTHS: Final[tuple[str, ...]] = tuple(
    f"{y}-{m:02d}" for y in range(2021, 2025) for m in range(1, 13) if f"{y}-{m:02d}" <= "2024-06"
)
VAL_MONTHS: Final[tuple[str, ...]] = tuple(f"2024-{m:02d}" for m in range(7, 13))
HELD_MONTHS: Final[tuple[str, ...]] = tuple(f"2025-{m:02d}" for m in range(1, 7))
KIND_SHARES: Final[tuple[tuple[str, float], ...]] = (
    ("pulses", 0.45), ("pulses_genuine", 0.25), ("dip", 0.10), ("null", 0.20),
)  # fmt: skip
SIZES_GRID: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0)
DURATIONS_GRID: Final[tuple[int, ...]] = (2, 12, 48, 144, 672)
GRID_REPS: Final[int] = 20
SETS: Final[dict[str, dict]] = {
    "train": {"seeds": (0, 6000), "months": TRAIN_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": None},
    "val": {"seeds": (6000, 7500), "months": VAL_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": None},
    "held_seed": {"seeds": (10_000, 11_200), "months": TRAIN_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": None},
    "held_month": {"seeds": (11_200, 12_400), "months": HELD_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": None},
    "held_shape": {"seeds": (12_400, 13_600), "months": TRAIN_MONTHS, "shapes": (inj.SHAPE_HELD_OUT,), "donors": None},
    "held_primary": {"seeds": (13_600, 14_800), "months": TRAIN_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": inj.HELD_OUT_PRIMARIES},
    "null": {"seeds": (15_000, 17_000), "months": TRAIN_MONTHS, "shapes": inj.SHAPES_TRAIN, "donors": None, "kinds": (("null", 1.0),)},
}  # fmt: skip
_STATE: dict = {}


def month_index(*, month: str) -> int:
    """Return the grid index of the first day of a month."""
    return d514.index_of(t=datetime.fromisoformat(f"{month}-01").replace(tzinfo=UTC))


def dip_windows() -> tuple[list[int], list[int]]:
    """Return D8's explained dip day indices before 2025-07 and the held-out dip day indices.

    The second list holds D8's unexplained dips and D22's dip days; no training world contains them.
    """
    root = d514.ML_EXPERIMENTS_DATA_DIR / "00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift" / "work"
    engine = pl.read_parquet(root / "partB_engine_d8.parquet")
    shape = pl.read_parquet(root / "partB_shape.parquet").filter((pl.col("kind") == "dip") & (pl.col("site") == "D22"))

    def idx(date: int) -> int:
        return d514.index_of(t=datetime(date // 10_000, date // 100 % 100, date % 100, tzinfo=UTC))

    explained = [idx(x) for x in engine.filter(pl.col("coincident"))["date"].to_list() if x < 20250701]
    unexplained = [idx(x) for x in engine.filter(~pl.col("coincident"))["date"].to_list()]
    unexplained += [idx(x) for x in shape["date"].to_list()]
    return explained, unexplained


def _init() -> None:
    """Load the cache once per worker process."""
    fold = d514.load_fold(month="2025-10")
    _STATE["fold"] = fold
    _STATE["shapes"] = d514.load_shapes().astype(np.float64)
    _STATE["events"] = d514.event_mask(n=fold.z.shape[1])
    _STATE["scales"] = {}
    explained, unexplained = dip_windows()
    _STATE["explained"] = explained
    _STATE["unexplained"] = unexplained


def _scales(*, month_start: int) -> det.Scales:
    """Return the noise scales fitted on rows below `month_start` (cached)."""
    cache = _STATE["scales"]
    if month_start not in cache:
        fold = _STATE["fold"]
        cache[month_start] = det.fit_scales(matrix=fold.partners(), mw_per_z=fold.mw_per_z, hist_end=month_start)
    return cache[month_start]


def _entry(*, world: inj.World, item: dict, meta: dict) -> dict:
    """Return the stored record of one evaluated series."""
    lo, hi = item["rows"]
    i = item["series"]
    return {
        "x": item["x"].astype(np.float32),
        "y": inj.candidate_labels(world=world, entry=item),
        "start": item["cands"].start, "duration": item["cands"].duration,
        "is_open": item["cands"].is_open, "score": item["cands"].score,
        "rows": (lo, hi), "end": world.end,
        "truth": world.truth[i, lo:hi], "genuine": world.genuine[i, lo:hi],
        "series": i, **meta,
    }  # fmt: skip


def build_chunk(args: tuple[str, int, int]) -> list[dict]:
    """Build the worlds with seeds in [lo, hi) of a set and return their records."""
    name, lo, hi = args
    if not _STATE:
        _init()
    spec = SETS[name]
    kinds = spec.get("kinds", KIND_SHARES)
    names = [k for k, _ in kinds]
    shares = np.array([s for _, s in kinds])
    out: list[dict] = []
    fold = _STATE["fold"]
    for seed in range(lo, hi):
        rng = np.random.default_rng(seed)
        month = spec["months"][int(rng.integers(len(spec["months"])))]
        ms = month_index(month=month)
        kind = str(rng.choice(names, p=shares / shares.sum()))
        forced = None
        if kind == "dip":
            pool = _STATE["explained"]
            date = int(rng.choice(pool))
            forced = date - int(rng.integers(35, 80)) * 48
            ms = max(forced - 20 * 48, 0)
            if any(abs(u - date) < 90 * 48 for u in _STATE["unexplained"]):
                continue
        world = inj.make_world(
            fold=fold, month_start=ms, rng=rng, kind=kind, shapes=spec["shapes"], donors=spec["donors"],
            shape_all=_STATE["shapes"], scales=_scales(month_start=ms), event_rows=_STATE["events"],
            forced_start=forced,
        )  # fmt: skip
        if world is None:
            continue
        meta = {"set": name, "seed": seed, "month": month, "kind": kind}
        for item in inj.world_candidates(world=world):
            rec = _entry(world=world, item=item, meta=meta)
            pulse = next((p for p in world.pulses if p.series == item["series"] and not p.genuine), None)
            rec["pulse"] = None if pulse is None else (pulse.size, pulse.duration, pulse.shape, pulse.sign, pulse.mirrored)
            out.append(rec)
    return out


def build_grid(args: tuple[int, int]) -> list[dict]:
    """Build the fixed-pulse recall-grid worlds with seeds in [lo, hi)."""
    lo, hi = args
    if not _STATE:
        _init()
    fold = _STATE["fold"]
    cells = [
        (s, dur, shape, sign, mir)
        for s in SIZES_GRID for dur in DURATIONS_GRID for shape in (*inj.SHAPES_TRAIN, inj.SHAPE_HELD_OUT)
        for sign in (-1, 1) for mir in (False, True)
    ]  # fmt: skip
    out: list[dict] = []
    for seed in range(lo, hi):
        rng = np.random.default_rng(100_000 + seed)
        s, dur, shape, sign, mir = cells[seed % len(cells)]
        month = HELD_MONTHS[int(rng.integers(len(HELD_MONTHS)))]
        ms = month_index(month=month)
        world = inj.make_world(
            fold=fold, month_start=ms, rng=rng, kind="pulses", shapes=(shape,), donors=None,
            shape_all=_STATE["shapes"], scales=_scales(month_start=ms), event_rows=_STATE["events"],
            force={"size": s, "duration": dur, "shape": shape, "sign": sign, "mirrored": mir},
        )  # fmt: skip
        if world is None:
            continue
        meta = {"set": "grid", "seed": seed, "month": month, "kind": "grid"}
        for item in inj.world_candidates(world=world):
            rec = _entry(world=world, item=item, meta=meta)
            pulse = next((p for p in world.pulses if p.series == item["series"]), None)
            rec["pulse"] = None if pulse is None else (pulse.size, pulse.duration, pulse.shape, pulse.sign, pulse.mirrored)
            out.append(rec)
    return out


def run_worlds() -> None:
    """Build every set in two processes and pickle each into the work folder."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        for name, spec in SETS.items():
            path = WORK / f"set_{name}.pkl"
            if path.exists():
                print(name, "exists", flush=True)
                continue
            t0 = time.time()
            lo, hi = spec["seeds"]
            step = max((hi - lo) // 20, 50)
            chunks = [(name, a, min(a + step, hi)) for a in range(lo, hi, step)]
            records = [r for part in pool.imap(build_chunk, chunks) for r in part]
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(pickle.dumps(records))
            tmp.rename(path)
            print(f"{name}: {len(records)} series-worlds, {time.time() - t0:.0f} s", flush=True)
        path = WORK / "set_grid.pkl"
        if not path.exists():
            n = len(SIZES_GRID) * len(DURATIONS_GRID) * 3 * 2 * 2 * GRID_REPS
            step = 120
            chunks = [(a, min(a + step, n)) for a in range(0, n, step)]
            records = [r for part in pool.imap(build_grid, chunks) for r in part]
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(pickle.dumps(records))
            tmp.rename(path)
            print(f"grid: {len(records)} series-worlds", flush=True)


def load_set(*, name: str) -> list[dict]:
    """Return a stored set of records."""
    return pickle.loads((WORK / f"set_{name}.pkl").read_bytes())  # noqa: S301


def stack(*, records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the stacked features, labels, and the record index of each candidate."""
    xs = [r["x"] for r in records if len(r["y"])]
    ys = [r["y"] for r in records if len(r["y"])]
    idx = [np.full(len(r["y"]), k) for k, r in enumerate(records) if len(r["y"])]
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(idx)


class Model:
    """The frozen classifier: a base model on candidates, a row-level Platt map, and a prior shift."""

    def __init__(self, *, kind: str, base: object, scaler: object | None) -> None:
        """Store the base model; `row_a`, `row_b`, `zero_rate`, `shift` are set by `calibrate_rows`."""
        self.kind, self.base, self.scaler = kind, base, scaler
        self.row_a, self.row_b, self.zero_rate, self.shift = 1.0, 0.0, 0.0, 0.0

    def candidate_p(self, *, x: np.ndarray) -> np.ndarray:
        """Return the base model's candidate probability."""
        if len(x) == 0:
            return np.zeros(0)
        z = self.scaler.transform(x) if self.scaler is not None else x  # type: ignore[union-attr]
        return self.base.predict_proba(z)[:, 1]  # type: ignore[union-attr]

    def row_p(self, *, record: dict, shift: float | None = None) -> np.ndarray:
        """Return the calibrated row probability over the record's evaluated rows."""
        n = record["end"]
        cands = det.Candidates(
            start=record["start"], duration=record["duration"], is_open=record["is_open"],
            score=record["score"], delta_in=np.zeros(len(record["start"])),
            delta_out=np.zeros(len(record["start"])), after_fraction=np.zeros(len(record["start"])),
        )  # fmt: skip
        raw = det.row_probability(cands=cands, p_cand=self.candidate_p(x=record["x"]), n=n)
        lo, hi = record["rows"]
        raw = raw[lo:hi]
        return self.calibrate(raw=raw, shift=self.shift if shift is None else shift)

    def calibrate(self, *, raw: np.ndarray, shift: float) -> np.ndarray:
        """Map raw row probabilities through the row-level Platt fit and the prior shift."""
        eps = 1e-6
        covered = raw > 0
        logit = np.log(np.clip(raw, eps, 1 - eps) / (1 - np.clip(raw, eps, 1 - eps)))
        p = 1.0 / (1.0 + np.exp(-(self.row_a * logit + self.row_b + shift)))
        q0 = np.clip(self.zero_rate, eps, 1 - eps)
        p0 = 1.0 / (1.0 + np.exp(-(np.log(q0 / (1 - q0)) + shift)))
        return np.where(covered, p, p0)


def brier(*, p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Return the (weighted) Brier score."""
    w = np.ones(len(p)) if w is None else w
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def reliability(*, p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, bins: int = 10) -> pl.DataFrame:
    """Return the reliability table (equal-width bins) and the weighted expected calibration error."""
    w = np.ones(len(p)) if w is None else w
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        rows.append({
            "bin": b, "weight": float(w[m].sum()), "mean_p": float(np.average(p[m], weights=w[m])),
            "observed": float(np.average(y[m], weights=w[m])),
        })  # fmt: skip
    table = pl.DataFrame(rows)
    ece = float((table["weight"] / table["weight"].sum() * (table["mean_p"] - table["observed"]).abs()).sum())
    return table.with_columns(pl.lit(ece).alias("ece"))


def row_arrays(*, model: Model, records: list[dict], shift: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row probabilities, row truth, and the record index of every evaluated row."""
    ps, ys, ks = [], [], []
    for k, r in enumerate(records):
        p = model.row_p(record=r, shift=shift)
        ps.append(p)
        ys.append(r["truth"])
        ks.append(np.full(len(p), k))
    return np.concatenate(ps), np.concatenate(ys).astype(float), np.concatenate(ks)


def prior_weights(*, y: np.ndarray, target: float) -> np.ndarray:
    """Return row weights that make the weighted share of positives equal `target`."""
    pi = float(y.mean())
    return np.where(y > 0.5, target / pi, (1 - target) / (1 - pi))


def logit(p: float) -> float:
    """Return the log-odds."""
    return float(np.log(p / (1 - p)))


def fit_and_freeze() -> dict:
    """Fit both classifiers on the training set, choose and calibrate on the validation set, and freeze."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    train, val = load_set(name="train"), load_set(name="val")
    x_tr, y_tr, _ = stack(records=train)
    x_va, y_va, _ = stack(records=val)
    print(f"train candidates {len(y_tr)} positives {y_tr.mean():.3f}; val {len(y_va)} positives {y_va.mean():.3f}", flush=True)
    scaler = StandardScaler().fit(x_tr)
    logistic = LogisticRegression(C=1.0, max_iter=2000, class_weight=None).fit(scaler.transform(x_tr), y_tr)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0).fit(x_tr, y_tr)
    scores = {
        "logistic": brier(p=logistic.predict_proba(scaler.transform(x_va))[:, 1], y=y_va),
        "tree": brier(p=tree.predict_proba(x_va)[:, 1], y=y_va),
    }
    print("validation candidate Brier", scores, flush=True)
    choice = min(scores, key=scores.get)  # type: ignore[arg-type]
    model = Model(kind=choice, base=logistic if choice == "logistic" else tree, scaler=scaler if choice == "logistic" else None)
    # row-level calibration on the validation rows: Platt on covered rows, a constant on uncovered rows
    raw_rows, y_rows = [], []
    for r in val:
        n = r["end"]
        cands = det.Candidates(
            start=r["start"], duration=r["duration"], is_open=r["is_open"], score=r["score"],
            delta_in=np.zeros(len(r["start"])), delta_out=np.zeros(len(r["start"])),
            after_fraction=np.zeros(len(r["start"])),
        )  # fmt: skip
        raw = det.row_probability(cands=cands, p_cand=model.candidate_p(x=r["x"]), n=n)
        lo, hi = r["rows"]
        raw_rows.append(raw[lo:hi])
        y_rows.append(r["truth"])
    raw_all, y_all = np.concatenate(raw_rows), np.concatenate(y_rows).astype(float)
    covered = raw_all > 0
    model.zero_rate = float(y_all[~covered].mean()) if (~covered).any() else 0.0
    eps = 1e-6
    lg = np.log(np.clip(raw_all[covered], eps, 1 - eps) / (1 - np.clip(raw_all[covered], eps, 1 - eps)))
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(lg[:, None], y_all[covered])
    model.row_a, model.row_b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    pi_val = float(y_all.mean())
    model.shift = logit(0.07) - logit(pi_val)
    info = {
        "choice": choice, "val_scores": scores, "row_a": model.row_a, "row_b": model.row_b,
        "zero_rate": model.zero_rate, "validation_row_prevalence": pi_val,
        "prior_shift_to_7_percent": model.shift,
        "train_candidates": int(len(y_tr)), "train_positive_share": float(y_tr.mean()),
    }  # fmt: skip
    (WORK / "frozen_model.pkl").write_bytes(pickle.dumps(model))
    (WORK / "frozen_model.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    return info


def main() -> None:
    """Dispatch the command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worlds", "fit"))
    args = parser.parse_args()
    if args.command == "worlds":
        run_worlds()
    else:
        fit_and_freeze()


if __name__ == "__main__":
    sys.modules.setdefault("e00514_train", sys.modules["__main__"])
    main()
