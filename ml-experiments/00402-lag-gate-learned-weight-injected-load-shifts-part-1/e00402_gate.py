"""Experiment 402, step 4: fit the gate on injected worlds and score it, leave-one-month-out.

The gate sets one weight `w` per row on the forecast with power lags, in a mixture with the
forecast without lags. The exact CRPS of the mixture is `A w^2 + B w + C`, so a squared-error
XGBoost booster on the label `w* = clip(-B / (2 A), 0, 1)` with sample weight `A` minimises the
mixture CRPS exactly (identity link, prediction clipped to [0, 1]).

Rows come from `e00402_worlds.py`: injected worlds (training seeds `TRAIN_SEEDS`, scored seed
`TEST_SEED`) and the real un-injected rows. For each scored month `M`, a gate trains on injected
rows whose forecast origin is at least `CUT_DAYS` days before the start of `M` or after the end of
`M` (so no training row shares an injected event, a lag, or a valid time with `M`); the fitted
constants are cut the same way. Arms (feature sets), each run pooled, with the metering unit as an
input, and separately per metering group: `single` (single-step surprise and residual), `own` (adds
window means and the gap between the two medians), `full` (adds neighbour surprises), `flag`
(`full` plus the injected truth flags, injected rows only), and `real` (`full`, trained on real
rows). `const` is the fitted constant weight per group and lead day; `lags` and `lagless` are the
endpoints.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00402-*/e00402_gate.py \
    --arm r1 --threads 4
```
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR
WORK: Final[Path] = DATA / "00402-lag-gate-learned-weight-injected-load-shifts-part-1" / "work"
WORLDS: Final[Path] = WORK / "worlds"
EVENTS: Final[Path] = (
    DATA
    / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
    / "work"
    / "events_corrected.parquet"
)
SIX: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
EIGHT: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
TRAIN_SEEDS: Final[tuple[int, ...]] = (0, 1)
TEST_SEED: Final[int] = 2
CUT_DAYS: Final[int] = 14
AFTER_STEPS: Final[int] = 7 * 48
ROUNDS: Final[int] = 200
MAX_TRAIN_ROWS: Final[int] = 600_000
J: Final[tuple[int, ...]] = (0, 1, 2)
SINGLE: Final[list[str]] = [f"{k}_{j}" for k in ("u", "r") for j in J] + ["lead_days"]
WINDOW: Final[list[str]] = [f"{k}_mean{h}_{j}" for k in ("u", "r") for h in (6, 24) for j in J]
GAP: Final[list[str]] = ["median_gap"]
NB: Final[list[str]] = [f"nb_{k}_{j}" for k in ("max", "min", "up", "down") for j in J]
FLAGS: Final[list[str]] = ["lags_in_event", "event_kind"]
FEATURES: Final[dict[str, list[str]]] = {
    "single": SINGLE,
    "own": SINGLE + WINDOW + GAP,
    "full": SINGLE + WINDOW + GAP + NB,
    "flag": SINGLE + WINDOW + GAP + NB + FLAGS,
    "real": SINGLE + WINDOW + GAP + NB,
}
VARIANTS: Final[tuple[str, ...]] = ("pooled", "unit", "group")


def group_of(*, site: np.ndarray) -> np.ndarray:
    """Return `six`, `eight`, or `report` (D15 and D18, which are never fitted or scored)."""
    return np.where(np.isin(site, SIX), "six", np.where(np.isin(site, EIGHT), "eight", "report"))


def load(*, pattern: str) -> pl.DataFrame:
    """Return the concatenated rows of the matching files with group, month, and unit columns."""
    frames = [pl.read_parquet(p) for p in sorted(WORLDS.glob(pattern))]
    d = pl.concat(frames)
    d = d.with_columns(
        month=pl.col("power_fcst_init_time").dt.truncate("1mo"),
        group=pl.Series(group_of(site=d["site"].to_numpy())),
    )
    return d.with_columns(unit=(pl.col("group") == "six").cast(pl.Float64))


def real_tags(*, d: pl.DataFrame) -> pl.DataFrame:
    """Add `lags_real`, `target_real`, and `after_real` from the unpadded corrected events."""
    size = common.grid_times().len()
    ev = pl.read_parquet(EVENTS)
    lags_real = np.zeros(d.height, dtype=np.int8)
    target_real = np.zeros(d.height, dtype=bool)
    after_real = np.zeros(d.height, dtype=bool)
    cluster = np.full(d.height, -1, dtype=np.int64)
    sites = d["site"].to_numpy()
    target = np.asarray(common.grid_index(times=d["valid_time"]))
    lag = np.stack([d[f"lag_{j}_idx"].to_numpy() for j in J], axis=1)
    for s in np.unique(sites):
        cell = np.full(size, -1, dtype=np.int64)
        after = np.zeros(size, dtype=bool)
        for e in ev.filter(pl.col("label") == s).iter_rows(named=True):
            cell[e["onset_idx"] : e["end_idx"] + 1] = e["event_id"]
            after[e["end_idx"] + 1 : e["end_idx"] + 1 + AFTER_STEPS] = True
        m = sites == s
        inside = cell[lag[m]] >= 0
        lags_real[m] = inside.sum(axis=1)
        first = np.where(
            inside[:, 0],
            cell[lag[m]][:, 0],
            np.where(inside[:, 1], cell[lag[m]][:, 1], cell[lag[m]][:, 2]),
        )
        cluster[m] = np.where(inside.any(axis=1), first, -1)
        target_real[m] = cell[target[m]] >= 0
        after_real[m] = after[target[m]] & ~target_real[m]
    return d.with_columns(
        lags_real=pl.Series(lags_real), target_real=pl.Series(target_real),
        after_real=pl.Series(after_real), cluster=pl.Series(cluster),
    )  # fmt: skip


def cut_train(*, d: pl.DataFrame, month) -> pl.DataFrame:  # noqa: ANN001
    """Return rows whose origin is at least `CUT_DAYS` before the scored month or after its end."""
    start = month
    end = (start + timedelta(days=32)).replace(day=1)
    origin, valid = pl.col("power_fcst_init_time"), pl.col("valid_time")
    return d.filter(
        (origin < start - timedelta(days=CUT_DAYS)) | (origin >= end + timedelta(days=CUT_DAYS))
    ).filter(((valid < start) | (valid >= end)) & (pl.col("group") != "report"))


def label_weight(*, d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the label `w*` and sample weight `A` of the gate's squared-error fit."""
    a, b = d["A"].to_numpy(), d["B"].to_numpy()
    w = np.clip(-b / (2.0 * np.maximum(a, 1e-12)), 0.0, 1.0)
    return w, a


def fit_gate(
    *, train: pl.DataFrame, columns: list[str], threads: int, seed: int = 0
) -> xgb.Booster:
    """Fit the shallow squared-error gate on the training rows."""
    if train.height > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, seed=seed)
    w, a = label_weight(d=train)
    matrix = xgb.DMatrix(
        train.select(columns).to_numpy().astype(np.float32), label=w, weight=np.maximum(a, 0.0)
    )
    params = {
        "objective": "reg:squarederror", "max_depth": 3, "eta": 0.1, "subsample": 0.8,
        "nthread": threads, "seed": seed, "tree_method": "hist",
    }  # fmt: skip
    return xgb.train(params, matrix, num_boost_round=ROUNDS)


def predict_weight(*, model: xgb.Booster, d: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the gate's weight on each row, clipped to [0, 1]."""
    x = xgb.DMatrix(d.select(columns).to_numpy().astype(np.float32))
    return np.clip(model.predict(x), 0.0, 1.0)


def const_weights(*, train: pl.DataFrame, test: pl.DataFrame, by_group: bool) -> np.ndarray:
    """Return the fitted constant weight per (group, lead day) on the test rows."""

    def key(d: pl.DataFrame) -> pl.Series:
        day = (d["lead_hours"] / 24.0).ceil().clip(1, 14).cast(pl.Int64)
        return (
            (d["group"] if by_group else pl.Series(["all"] * d.height)) + "_" + day.cast(pl.String)
        )

    table = (
        train.with_columns(k=key(train))
        .group_by("k")
        .agg(a=pl.col("A").sum(), b=pl.col("B").sum())
        .with_columns(w=(-pl.col("b") / (2 * pl.col("a").clip(1e-12))).clip(0, 1))
    )
    out = test.with_columns(k=key(test)).join(
        table.select("k", "w"), on="k", how="left", maintain_order="left"
    )
    return out["w"].fill_null(0.5).to_numpy()


def exact_crps(*, d: pl.DataFrame, w: np.ndarray) -> np.ndarray:
    """Return the exact CRPS of the mixture at weight `w` on each row."""
    return d["A"].to_numpy() * w**2 + d["B"].to_numpy() * w + d["C"].to_numpy()


def weights_for_month(
    *, train_inj: pl.DataFrame, train_real: pl.DataFrame, test: pl.DataFrame, threads: int
) -> dict[str, np.ndarray]:
    """Return every arm's weights on the test rows, for one scored month."""
    out: dict[str, np.ndarray] = {
        "lags": np.ones(test.height),
        "lagless": np.zeros(test.height),
        "fixed 50/50": np.full(test.height, 0.5),
        "const (injected, per group)": const_weights(train=train_inj, test=test, by_group=True),
        "const (injected, pooled)": const_weights(train=train_inj, test=test, by_group=False),
    }
    if train_real.height:
        out["const (real, per group)"] = const_weights(train=train_real, test=test, by_group=True)
    sources = {
        "single": train_inj,
        "own": train_inj,
        "full": train_inj,
        "flag": train_inj,
        "real": train_real,
    }
    for arm, columns in FEATURES.items():
        train = sources[arm]
        if train.height == 0:
            continue
        for variant in VARIANTS:
            name = f"gate {arm} ({variant})"
            cols = [*columns, "unit"] if variant == "unit" else columns
            if variant == "group":
                w = np.full(test.height, np.nan)
                for g in ("six", "eight"):
                    tm = (train["group"] == g).to_numpy()
                    sm = (test["group"] == g).to_numpy()
                    if tm.any() and sm.any():
                        model = fit_gate(
                            train=train.filter(pl.Series(tm)), columns=cols, threads=threads
                        )
                        w[sm] = predict_weight(
                            model=model, d=test.filter(pl.Series(sm)), columns=cols
                        )
                out[name] = np.where(np.isnan(w), 0.5, w)
            else:
                model = fit_gate(train=train, columns=cols, threads=threads)
                out[name] = predict_weight(model=model, d=test, columns=cols)
    return out


def main() -> None:
    """Run the leave-one-month-out evaluation and write the per-row weights and CRPS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("r1", "r0"), default="r1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    inj = pl.concat(
        [
            load(pattern=f"world{s}_{args.arm}_*.parquet").with_columns(seed=pl.lit(s))
            for s in (*TRAIN_SEEDS, TEST_SEED)
        ]
    )
    real = real_tags(d=load(pattern=f"real_{args.arm}_*.parquet"))
    months = sorted(inj["month"].unique().to_list())
    scored_rows = []
    model_columns = sorted({c for cols in FEATURES.values() for c in cols} | {"unit"})
    keep = [
        "site",
        "group",
        "power_fcst_init_time",
        "valid_time",
        "lead_hours",
        "month",
        "A",
        "B",
        "C",
    ]
    for month in months:
        train_inj = cut_train(d=inj.filter(pl.col("seed").is_in(list(TRAIN_SEEDS))), month=month)
        train_real = cut_train(d=real, month=month)
        test_inj = inj.filter(
            (pl.col("seed") == TEST_SEED)
            & (pl.col("month") == month)
            & (pl.col("group") != "report")
        )
        test_real = real.filter((pl.col("month") == month) & (pl.col("group") != "report"))
        if test_inj.is_empty() and test_real.is_empty():
            continue
        test_real_cols = test_real.with_columns(
            lags_in_event=pl.lit(None, dtype=pl.Int8), event_kind=pl.lit(None, dtype=pl.Int8)
        )
        both = pl.concat(
            [
                test_inj.select(*keep, *model_columns).with_columns(kind=pl.lit("injected")),
                test_real_cols.select(*keep, *model_columns).with_columns(kind=pl.lit("real")),
            ]
        )
        weights = weights_for_month(
            train_inj=train_inj, train_real=train_real, test=both, threads=args.threads
        )
        is_real = (both["kind"] == "real").to_numpy()
        for name in list(weights):
            if name.startswith("gate flag"):
                weights[name] = np.where(is_real, np.nan, weights[name])
        tags_inj = test_inj.select("lags_in_event", "event_kind", "target_in_event").with_columns(
            lags_real=pl.lit(None, dtype=pl.Int8),
            target_real=pl.lit(None, dtype=pl.Boolean),
            after_real=pl.lit(None, dtype=pl.Boolean),
            cluster=pl.lit(None, dtype=pl.Int64),
        )
        tags_real = test_real.select(
            "lags_real", "target_real", "after_real", "cluster"
        ).with_columns(
            lags_in_event=pl.lit(None, dtype=pl.Int8),
            event_kind=pl.lit(None, dtype=pl.Int8),
            target_in_event=pl.lit(None, dtype=pl.Boolean),
        )
        tags = pl.concat([tags_inj, tags_real.select(tags_inj.columns)])
        frame = pl.concat([both.select("kind", *keep[:6]), tags], how="horizontal").with_columns(
            **{f"w::{k}": pl.Series(v) for k, v in weights.items()},
            **{f"crps::{k}": pl.Series(exact_crps(d=both, w=v)) for k, v in weights.items()},
        )
        scored_rows.append(frame)
        print(month.date(), both.height, "rows", len(weights), "arms", flush=True)
    pl.concat(scored_rows, how="diagonal").write_parquet(WORK / f"gate_scores_{args.arm}.parquet")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
