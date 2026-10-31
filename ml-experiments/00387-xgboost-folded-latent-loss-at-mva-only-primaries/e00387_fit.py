"""Fit experiment 387's folded-latent XGBoost arms and write each refit's quantiles per run.

Copies experiment 381's fitter (experiment 87's `n1h` recipe: ENS control member rows, whole
history, half-life 0.5 years, 500 rounds, eta 0.05, depth 6, no column subsampling, the three
recent-power lags) and changes the loss.

Arms (a spec is `name[:k<kappa>][:m<0|1>][:q<Q0>]`):

- `plain`: `reg:quantileerror` at the 11 levels (experiment 87's recipe), used as the start.
- `f0`: the custom multi-output pinball objective with no fold, started from `plain`.
- `f2`: the same objective on the fold S(l) = sqrt(l² + Q0²) / pf, started from `plain` minus
  kappa x A x shortwave / 1000 (A is the 95th percentile of the training labels), with a
  non-increasing monotone constraint on shortwave when `m1`.

Data modes: `real` (the metered series) or `syn:<reactive>:<level>` (the gate's synthetic MVA
from a signed-MW Primary, see `e00387_data`). Output:
`work/preds/<mode>/<spec>/<site>_<refit>.parquet`
with the 11 mixture quantiles of S (points of capacity are label units, a fraction), plus
`latent_mean` (the member mean of the median latent) and `neg_share` (members with a negative
median latent).

    PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
        ml-experiments/00387-* \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00387-*/e00387_fit.py --selftest
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00387_data as data
import numpy as np
import polars as pl
import xgboost as xgb
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00387-xgboost-folded-latent-loss-at-mva-only-primaries" / "work"
)
LEVELS: Final[np.ndarray] = common.LEVELS
MEDIAN_INDEX: Final[int] = int(np.argmin(np.abs(LEVELS - 0.5)))
ROUNDS: Final[int] = common.N_ROUNDS
SMOKE_ROUNDS: Final[int] = 20
TILT_QUANTILE: Final[float] = 95.0


def parse_spec(*, spec: str) -> dict[str, Any]:
    """Return the arm name, kappa, monotone flag, and Q0 (points of capacity) of a spec."""
    name, *parts = spec.split(":")
    out: dict[str, object] = {"name": name, "kappa": 0.0, "mono": False, "q0": 0.0}
    for part in parts:
        key, value = part[0], part[1:]
        if key == "k":
            out["kappa"] = float(value)
        elif key == "m":
            out["mono"] = value == "1"
        elif key == "q":
            out["q0"] = float(value)
        else:
            msg = f"unknown spec part {part!r}"
            raise ValueError(msg)
    if name not in {"plain", "f0", "f2"}:
        msg = f"unknown arm {name!r}"
        raise ValueError(msg)
    return out


def fold(*, latent: np.ndarray, q0: float, pf: float = 1.0) -> np.ndarray:
    """Return the smooth fold S(l) = sqrt(l² + q0²) / pf; q0 is a fraction of capacity."""
    return np.sqrt(latent**2 + q0**2) / pf


def objective(*, levels: np.ndarray, q0: float, folded: bool, pf: float = 1.0):  # noqa: ANN201
    """Return an XGBoost custom objective: multi-output pinball loss on S(l), unit hessian.

    The gradient of the pinball loss with respect to S is -(alpha - 1[y < S]); dS/dl is
    l / (pf sqrt(l² + q0²)), taken as 1/pf at l = 0 (so the strict fold has a defined step).
    The hessian is the constant 1/pf² (1 without a fold). Sample weights multiply both.
    """

    def obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        label = dtrain.get_label()[:, None]
        weight = dtrain.get_weight()
        weight = weight[:, None] if weight.size else 1.0
        latent = predt.reshape(len(label), -1)
        if folded:
            apparent = fold(latent=latent, q0=q0, pf=pf)
            denominator = pf * np.sqrt(latent**2 + q0**2)
            slope = np.where(
                denominator > 0, latent / np.where(denominator > 0, denominator, 1), 1 / pf
            )
            hess = np.full_like(latent, 1 / pf**2)
        else:
            apparent, slope, hess = latent, 1.0, np.ones_like(latent)
        grad = -(levels[None, :] - (label < apparent)) * slope
        return grad * weight, hess * weight

    return obj


def selftest() -> None:
    """Check the gradient by finite differences, then the multi-output shapes on tiny data."""
    rng = np.random.default_rng(0)
    levels = np.array([0.1, 0.5, 0.9])
    y = rng.uniform(0.0, 1.0, 50).astype(np.float32)
    latent = rng.normal(0.0, 0.7, (50, 3))
    for q0, folded in ((0.0, True), (0.05, True), (0.0, False)):
        obj = objective(levels=levels, q0=q0, folded=folded)
        grad, hess = obj(latent.ravel(), xgb.DMatrix(np.zeros((50, 1)), label=y))
        step = 1e-6

        def loss(values: np.ndarray, q0: float = q0, folded: bool = folded) -> float:
            s = fold(latent=values, q0=q0) if folded else values
            u = y[:, None] - s
            return float((u * (levels[None, :] - (u < 0))).sum())

        for index in ((0, 0), (7, 1), (31, 2)):
            up, down = latent.copy(), latent.copy()
            up[index] += step
            down[index] -= step
            numeric = (loss(up) - loss(down)) / (2 * step)
            assert abs(numeric - grad[index]) < 1e-4, (q0, folded, index, numeric, grad[index])
        assert (hess > 0).all()
    x = rng.uniform(0, 1, (400, 4)).astype(np.float32)
    d = xgb.DMatrix(x, label=np.abs(x[:, 0] - 0.5), base_margin=np.full((400, 3), 0.3, np.float32))
    booster = xgb.train(
        {"tree_method": "hist", "num_target": 3, "disable_default_eval_metric": 1, "nthread": 1},
        d,
        5,
        obj=objective(levels=levels, q0=0.0, folded=True),
    )
    assert booster.predict(d, output_margin=True).shape == (400, 3)
    print("selftest passed")


def base_parameters(*, nthread: int, seed: int) -> dict[str, object]:
    """Return the XGBoost parameters shared by all arms (no objective)."""
    params = {**common.QUANTILE_PARAMETERS, "seed": seed, "nthread": nthread}
    params.pop("objective")
    params.pop("quantile_alpha")
    return params


def monotone(*, columns: list[str]) -> str:
    """Return a non-increasing constraint on the shortwave input only."""
    return "(" + ",".join("-1" if c == "shortwave" else "0" for c in columns) + ")"


def synthetic_for(*, mode: str, arrays: common.SeriesArrays) -> data.Synthetic | None:
    """Return the synthetic series of a `syn:<reactive>:<level>` mode, or `None` for `real`."""
    if mode == "real":
        return None
    _, reactive, level = mode.split(":")
    return data.synthesise(arrays=arrays, level=float(level), reactive=reactive)


def training_frame(
    *, row: dict, arrays: common.SeriesArrays, synthetic: data.Synthetic | None
) -> tuple[pl.DataFrame, common.SeriesArrays]:
    """Return a series' training rows with power and lags from the (synthetic) series."""
    fitted = arrays if synthetic is None else synthetic.arrays
    rows = fd.training_rows(row=row, arrays=fitted, frozen=False, a7=False)
    if synthetic is not None:
        index = np.asarray(common.grid_index(times=rows["valid_time"]))
        replaced = common.take(values=fitted.power, index=index) * row["effective_capacity_mw"]
        rows = rows.with_columns(power=pl.Series(replaced).cast(pl.Float32)).filter(
            pl.col("power").is_finite()
        )
    return rows, fitted


def fit_job(  # noqa: PLR0915
    *,
    mode: str,
    specs: list[str],
    site: str,
    refit: datetime,
    origins: list[datetime],
    nthread: int,
    rounds: int,
    seed: int,
    root: Path,
) -> list[str]:
    """Fit every spec at one refit of one series and forecast the refit's runs."""
    series, arrays = fd._series_and_arrays()
    row = next(r for r in series.iter_rows(named=True) if r["site"] == site)
    todo = [
        s
        for s in specs
        if not (
            root
            / mode.replace(":", "_")
            / s.replace(":", "_")
            / f"{site}_{refit:%Y%m%dT%H%M}.parquet"
        ).exists()
    ]
    if not todo:
        return []
    synthetic = synthetic_for(mode=mode, arrays=arrays[site])
    rows, fitted = training_frame(row=row, arrays=arrays[site], synthetic=synthetic)
    known = fd.known_rows(rows=rows, refit=refit, config=fd.CONFIGS["n1h"])
    columns = fd.input_columns(a7=False)
    age = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    label = (known["power"] / row["effective_capacity_mw"]).to_numpy()
    matrix = known.select(columns).to_numpy().astype(np.float32)
    weight = 0.5 ** (age / common.SHORT_HALF_LIFE_DAYS)
    started = time.monotonic()
    params = base_parameters(nthread=nthread, seed=seed)
    plain = xgb.train(
        {**params, "objective": "reg:quantileerror", "quantile_alpha": LEVELS},
        xgb.DMatrix(matrix, label=label, weight=weight),
        num_boost_round=rounds,
    )
    start_train = plain.predict(xgb.DMatrix(matrix), output_margin=True).astype(np.float32)
    amplitude = float(np.percentile(label, TILT_QUANTILE))
    shortwave_train = known["shortwave"].to_numpy().astype(np.float32)
    models: dict[str, tuple[xgb.Booster, float]] = {}
    for spec in todo:
        p = parse_spec(spec=spec)
        if p["name"] == "plain":
            models[spec] = (plain, 0.0)
            continue
        folded = p["name"] == "f2"
        q0 = float(p["q0"]) / 100.0
        kappa = float(p["kappa"]) if folded else 0.0
        tilt = kappa * amplitude * shortwave_train / 1000.0
        d = xgb.DMatrix(matrix, label=label, weight=weight, base_margin=start_train - tilt[:, None])
        arm_params = {**params, "num_target": len(LEVELS), "disable_default_eval_metric": 1}
        if p["mono"]:
            arm_params["monotone_constraints"] = monotone(columns=columns)
        models[spec] = (
            xgb.train(
                arm_params,
                d,
                num_boost_round=rounds,
                obj=objective(levels=LEVELS, q0=q0, folded=folded),
            ),
            kappa,
        )
    pieces: dict[str, list[pl.DataFrame]] = {s: [] for s in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        prepared = fd.prepared_run(
            run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
            row=row,
            arrays=fitted,
            a7=False,
        )
        keys = prepared.select(*common.KEYS).unique(maintain_order=True)
        members = prepared["ensemble_member"].n_unique()
        x = prepared.select(columns).to_numpy().astype(np.float32)
        start = plain.predict(xgb.DMatrix(x), output_margin=True).astype(np.float32)
        shortwave = prepared["shortwave"].to_numpy().astype(np.float32)
        for spec in todo:
            booster, kappa = models[spec]
            p = parse_spec(spec=spec)
            if p["name"] == "plain":
                latent = start
                apparent = start
            else:
                margin = start - (kappa * amplitude * shortwave / 1000.0)[:, None]
                latent = booster.predict(xgb.DMatrix(x, base_margin=margin), output_margin=True)
                apparent = (
                    fold(latent=latent, q0=float(p["q0"]) / 100.0) if p["name"] == "f2" else latent
                )
            apparent = np.sort(apparent, axis=1)
            piece = fd.mixture_frame(keys=keys, raw=apparent, members=members, seed=seed)
            cube = latent.reshape(-1, members, len(LEVELS))[:, :, MEDIAN_INDEX]
            piece = piece.with_columns(
                latent_mean=pl.Series(cube.mean(axis=1)).cast(pl.Float32),
                neg_share=pl.Series((cube < 0).mean(axis=1)).cast(pl.Float32),
            )
            pieces[spec].append(piece)
    lines = []
    for spec, tables in pieces.items():
        target = (
            root
            / mode.replace(":", "_")
            / spec.replace(":", "_")
            / f"{site}_{refit:%Y%m%dT%H%M}.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_write(frame=pl.concat(tables), target=target)
    lines.append(f"{mode} {site} {refit:%Y-%m-%d} {todo}: {time.monotonic() - started:.0f} s")
    return lines


def main() -> None:
    """Fit the requested specs for the requested sites and refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--mode", default="real")
    parser.add_argument("--specs", nargs="+", default=["plain"])
    parser.add_argument("--sites", nargs="+", default=list(data.SIGNED_SITES))
    parser.add_argument("--refits", nargs="+", type=int, default=None, help="indices into MONTHLY")
    parser.add_argument("--subset", default="every5d", choices=("all", "every5d", "gap"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    for spec in args.specs:
        parse_spec(spec=spec)
    served = regimes.served(name="monthly", origins=common.origins_of_subset(subset=args.subset))
    refits = sorted(served)
    if args.refits is not None:
        refits = [refits[i] for i in args.refits]
    rounds, root = ROUNDS, WORK / "preds"
    if args.smoke:
        refits, rounds, root = refits[:1], SMOKE_ROUNDS, WORK / "preds_smoke"
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                fit_job,
                mode=args.mode,
                specs=args.specs,
                site=site,
                refit=r,
                origins=served[r][:3] if args.smoke else served[r],
                nthread=1,
                rounds=rounds,
                seed=args.seed,
                root=root,
            )
            for r in refits
            for site in args.sites
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({root})")


if __name__ == "__main__":
    main()
