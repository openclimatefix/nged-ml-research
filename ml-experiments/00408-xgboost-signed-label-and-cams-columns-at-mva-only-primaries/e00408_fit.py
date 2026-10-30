"""Fit experiment 408's arms: the monthly XGBoost control, CAMS columns, and the signed label.

Every arm is experiment 381's monthly `control` recipe (experiment 87's rows, parameters, leakage
checks), with the changes of `e00408_common.install_patches` (cleaned power, the D15 and D18 cut,
and in the known-answer mode an injected farm). Arms:

- `A0`: the control.
- `A1`: adds the half-hour's own CAMS unit PV shape `cams_u` and clear-sky index `cams_k`.
- `A2_70`, `A2_50`: A1 plus the signed label at tau 0.7 or 0.5, rule refitted at each refit from the
  training window only. The quantile forecast is folded back to apparent power; the signed
  quantiles go to a second file.
- `A2_off`: the signed label with tau infinite; its forecasts must equal A1's (code check).
- `A2_shuf`: A2 at tau 0.7 with `k` permuted across days within its month and hour before the rule
  sees it (the input columns keep the real `k`).
- `A2_force`: A2 at tau 0.7 with the gate forced open (placebo).

Writes `work/<mode>/preds/<arm>/<site>_s0_<refit>.parquet`, the signed quantiles to
`work/<mode>/signed/<arm>/`, each fitted booster to `work/<mode>/boosters/<arm>/` as `.ubj`, and a
JSON line of the rule per site and refit to `work/<mode>/rules/<arm>.jsonl`.

```bash
E408_MODE=inj:2.0 PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack \
    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00408-*/e00408_fit.py \
    --mode inj:2.0 --arms A0 A2_70 --sites D5 D9 --workers 3
```
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00087_fit_demand as fd  # ty: ignore[unresolved-import]
import e00087_regimes as regimes  # ty: ignore[unresolved-import]
import e00408_common as c
import numpy as np
import polars as pl
import xgboost as xgb

SMOKE_ROUNDS: Final[int] = 20
CAMS_COLUMNS: Final[tuple[str, ...]] = ("cams_u", "cams_k")


@dataclass(frozen=True)
class Arm:
    """One arm: whether it adds the CAMS columns and how it labels the training readings."""

    name: str
    cams_columns: bool
    signed: bool
    tau: float = float("inf")
    shuffle: bool = False
    force_gate: bool = False

    def columns(self) -> list[str]:
        """Return the input columns in matrix order."""
        extra = list(CAMS_COLUMNS) if self.cams_columns else []
        return [*fd.input_columns(a7=False), *extra]


ARMS: Final[dict[str, Arm]] = {
    "A0": Arm("A0", cams_columns=False, signed=False),
    "A1": Arm("A1", cams_columns=True, signed=False),
    "A2_70": Arm("A2_70", cams_columns=True, signed=True, tau=0.7),
    "A2_50": Arm("A2_50", cams_columns=True, signed=True, tau=0.5),
    "A2_off": Arm("A2_off", cams_columns=True, signed=True, tau=float("inf")),
    "A2_shuf": Arm("A2_shuf", cams_columns=True, signed=True, tau=0.7, shuffle=True),
    "A2_force": Arm("A2_force", cams_columns=True, signed=True, tau=0.7, force_gate=True),
}


def mode_dir(*, mode: str, smoke: bool = False) -> Path:
    """Return the private folder one mode (`real` or `inj:<level>`) writes to."""
    return c.WORK_DIR / ("smoke" if smoke else "full") / mode.replace(":", "_")


def grid_arrays(*, row: dict, arrays: common.SeriesArrays) -> dict[str, np.ndarray]:
    """Return the readings in points, `u`, `k`, and the grid's UTC start hour, month, and year."""
    cams = c.cams_arrays(row=row)
    grid = common.grid_times()
    hour, month = c.start_hour_and_month(grid=grid)
    year = grid.dt.offset_by("-30m").dt.year().to_numpy()
    return {
        "y": arrays.power.astype(np.float64) * 100.0,
        "u": cams.u,
        "k": cams.k,
        "hour": hour,
        "month": month,
        "year": year,
    }


def add_cams(*, frame: pl.DataFrame, g: dict[str, np.ndarray]) -> pl.DataFrame:
    """Add the two CAMS columns, read at each row's own valid time."""
    index = np.asarray(common.grid_index(times=frame["valid_time"]))
    return frame.with_columns(
        cams_u=pl.Series(common.take(values=g["u"], index=index)),
        cams_k=pl.Series(common.take(values=g["k"], index=index)),
    )


def rule_for(
    *, arm: Arm, g: dict[str, np.ndarray], refit: datetime
) -> tuple[c.SignedRule, np.ndarray]:
    """Fit the rule on the training window and return it with the labelling index `k`."""
    k = (
        c.shuffled_k(k=g["k"], hour=g["hour"], month=g["month"], year=g["year"])
        if arm.shuffle
        else g["k"]
    )
    grid = common.grid_times()
    usable = (
        np.isfinite(g["y"])
        & np.isfinite(g["u"])
        & np.isfinite(k)
        & (grid <= c.latest_valid(refit=refit)).to_numpy()
    )
    rule = c.fit_rule(
        y_points=g["y"], u=g["u"], k=k, hour=g["hour"], month=g["month"], usable=usable
    )
    if arm.force_gate:
        rule = c.SignedRule(**{**rule.__dict__, "gate_open": True})
    return rule, k


def train_matrix(
    *, known: pl.DataFrame, row: dict, refit: datetime, arm: Arm, reverse: np.ndarray | None
) -> xgb.DMatrix:
    """Return the training matrix, weights anchored at `refit`; `reverse` flips the label's sign."""
    age_days = ((refit - known["valid_time"]).dt.total_seconds() / 86_400).to_numpy()
    label = (known["power"] / row["effective_capacity_mw"]).to_numpy()
    if reverse is not None:
        index = np.asarray(common.grid_index(times=known["valid_time"]))
        label = np.where(reverse[index], -label, label)
    return xgb.DMatrix(
        known.select(arm.columns()).to_numpy().astype(np.float32),
        label=label,
        weight=0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS),
    )


def run_job(
    *,
    arm_name: str,
    mode: str,
    refit: datetime,
    origins: list[datetime],
    sites: list[str],
    nthread: int,
    rounds: int,
    seed: int,
    root: Path,
) -> list[str]:
    """Fit one arm at one refit for each site lacking its file, and forecast the refit's runs."""
    arm = ARMS[arm_name]
    config = fd.CONFIGS["n1h"]
    series, arrays = fd._series_and_arrays()
    rows_of = {row["site"]: row for row in series.iter_rows(named=True)}
    todo = [
        s
        for s in sites
        if not (
            fd.job_path(
                root=root / "preds", config=arm.name, site=s, seed=seed, refit=refit
            ).exists()
            and (
                not arm.signed
                or fd.job_path(
                    root=root / "signed", config=arm.name, site=s, seed=seed, refit=refit
                ).exists()
            )
        )
    ]
    lines: list[str] = []
    models: dict[str, xgb.Booster] = {}
    grids: dict[str, dict[str, np.ndarray]] = {}
    for site in todo:
        started = time.monotonic()
        row = rows_of[site]
        g = grids[site] = grid_arrays(row=row, arrays=arrays[site])
        rows = fd._training_rows_cached(row=row, arrays=arrays[site], config=config)
        known = fd.known_rows(
            rows=add_cams(frame=rows, g=g) if arm.cams_columns else rows, refit=refit, config=config
        )
        reverse = None
        if arm.signed:
            rule, k_label = rule_for(arm=arm, g=g, refit=refit)
            reverse = c.reverse_mask(
                rule=rule, tau=arm.tau, y_points=g["y"], u=g["u"], k=k_label, hour=g["hour"]
            )
            share = float(reverse[np.asarray(common.grid_index(times=known["valid_time"]))].mean())
            record = {
                "site": site,
                "refit": f"{refit:%Y-%m-%d}",
                "gate_open": rule.gate_open,
                "spearman": rule.spearman,
                "p1_points": rule.p1_points,
                "n_midday": rule.n_midday,
                "capacity_points": rule.capacity,
                "d_ref": [None if np.isnan(x) else float(x) for x in rule.d_ref],
                "training_reverse_share": share,
                "training_rows": known.height,
            }
            (root / "rules").mkdir(parents=True, exist_ok=True)
            with (root / "rules" / f"{arm.name}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        model = xgb.train(
            fd.parameters(config=config, seed=seed, nthread=nthread),
            train_matrix(known=known, row=row, refit=refit, arm=arm, reverse=reverse),
            num_boost_round=rounds,
        )
        models[site] = model
        booster_path = root / "boosters" / arm.name / f"{site}_s{seed}_{refit:%Y%m%d}.ubj"
        booster_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(booster_path)
        lines.append(
            f"{arm.name} {site} refit {refit:%Y-%m-%d}: {known.height:,} rows, "
            f"fit {time.monotonic() - started:.0f} s"
        )
    pieces: dict[str, list[pl.DataFrame]] = {s: [] for s in todo}
    signed_pieces: dict[str, list[pl.DataFrame]] = {s: [] for s in todo}
    for origin in origins:
        frame = pl.read_parquet(common.demand_frame_path(run=origin - common.PUBLICATION_DELAY))
        for site in todo:
            row = rows_of[site]
            prepared = fd.prepared_run(
                run_rows=frame.filter(pl.col("time_series_id") == row["time_series_id"]),
                row=row,
                arrays=arrays[site],
                a7=False,
            )
            if arm.cams_columns:
                prepared = add_cams(frame=prepared, g=grids[site])
            keys = prepared.select(*common.KEYS).unique(maintain_order=True)
            matrix = prepared.select(arm.columns()).to_numpy().astype(np.float32)
            raw = models[site].predict(xgb.DMatrix(matrix))
            mixture = fd.mixture_frame(
                keys=keys, raw=raw, members=prepared["ensemble_member"].n_unique(), seed=seed
            )
            if arm.signed:
                signed_pieces[site].append(mixture)
                folded = c.fold_quantiles(
                    signed=mixture.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64),
                    levels=common.LEVELS,
                )
                mixture = mixture.with_columns(
                    [
                        pl.Series(name, folded[:, i]).cast(pl.Float32)
                        for i, name in enumerate(common.LEVEL_COLUMNS)
                    ]
                )
            pieces[site].append(mixture)
    for site, tables in pieces.items():
        if arm.signed:
            common.atomic_write(
                frame=pl.concat(signed_pieces[site]),
                target=fd.job_path(
                    root=root / "signed", config=arm.name, site=site, seed=seed, refit=refit
                ),
            )
        common.atomic_write(
            frame=pl.concat(tables),
            target=fd.job_path(
                root=root / "preds", config=arm.name, site=site, seed=seed, refit=refit
            ),
        )
    return lines


def selftest() -> None:
    """Check the fold, the shuffle, and the rule on small arrays."""
    levels = common.LEVELS
    rng = np.random.default_rng(1)
    positive = np.sort(rng.gamma(4.0, 1.0, (50, len(levels))), axis=1) + 0.5
    folded = c.fold_quantiles(signed=positive, levels=levels)
    assert np.abs(folded - positive).max() < 1e-9, "fold must be the identity for q01 >= 0"
    symmetric = np.tile(np.linspace(-4, 4, len(levels)), (3, 1))
    folded = c.fold_quantiles(signed=symmetric, levels=levels)
    assert (np.diff(folded, axis=1) >= -1e-12).all()
    assert abs(folded[0, 5]) < 4.0 * 0.51, "median of |x| for symmetric signed quantiles"
    k = np.tile(np.linspace(0, 1, 48), 60)
    hour = np.tile(np.arange(48) // 2, 60)
    month = np.full(k.size, 7)
    year = np.full(k.size, 2025)
    shuffled = c.shuffled_k(k=k, hour=hour, month=month, year=year)
    assert sorted(shuffled) == sorted(k), "the shuffle must keep the values"
    assert (shuffled != k).any()
    print("selftest ok")


def main() -> None:
    """Fit the requested arms at the monthly refits for the requested sites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--arms", nargs="+", default=[])
    parser.add_argument("--sites", nargs="+", default=[])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--smoke", action="store_true", help="2 refits, 3 runs, 20 rounds")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if os.environ.get(c.MODE_ENV) != args.mode:
        msg = f"set {c.MODE_ENV}={args.mode} before launching (workers read it at import)"
        raise SystemExit(msg)
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        msg = f"unknown arms {sorted(unknown)}"
        raise SystemExit(msg)
    jobs = regimes.jobs_for(regime_names=("monthly",), origins=common.all_origins())
    rounds = common.N_ROUNDS
    if args.smoke:
        jobs = {r: runs[:3] for r, runs in list(jobs.items())[:2]}
        rounds = SMOKE_ROUNDS
    root = mode_dir(mode=args.mode, smoke=args.smoke)
    nthread = max(1, args.cores // args.workers)
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                run_job,
                arm_name=a,
                mode=args.mode,
                refit=r,
                origins=runs,
                sites=args.sites,
                nthread=nthread,
                rounds=rounds,
                seed=args.seed,
                root=root,
            )
            for r, runs in sorted(jobs.items(), key=lambda kv: -len(kv[1]))
            for a in args.arms
        ]
        for done, future in enumerate(futures, start=1):
            for line in future.result():
                print(f"[{done}/{len(futures)}] {line}", flush=True)
    print(f"done in {time.monotonic() - started:.0f} s ({root})")


if __name__ == "__main__":
    main()
