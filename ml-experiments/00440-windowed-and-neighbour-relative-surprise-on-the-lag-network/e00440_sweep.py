"""Lag-encroachment sweep of S1, R-inj, S2, S3, and S3-rw (prediction only).

Experiment 392's protocol, as experiment 428's `e00428_sweep.py` ran it. A step of depth D is added
to the recent power of clean rows and the forecast distribution is compared with the unshifted one.

**Arms.** R-inj and S1 are read from experiment 428's models folder; S2, S3, and S3-rw from this
experiment's. The R-inj and S1 forecasts use experiment 428's input class unchanged; the other three
use `e00440_run.WindowInputs`.

**Rows.** Every third daily origin of the test month, world 0 (the real, clean series). A row is
clean when its target is finite, outside the padded real-event mask and the 96 half-hours after an
event, and R's three lags are finite. Every arm is swept on the same rows.

**Steps.** `c025` and `c050`: 2.5% and 5% of effective capacity. `m10`: 10% of the Primary's mean
power over the 30 days up to the newest usable reading. Both signs. MVA-only readings are clipped at
zero after the step; a row where the clip changed any shifted lag is dropped from that configuration
(for every arm alike).

**k.** The same lags are stepped in every arm: k = 1 the origin lag; k = 2 also the daily lag; k = 3
also the weekly lag. The step is added to `raw["lags"]` before any input is built.

**What recomputes and what stays fixed.** After the step every input that is a function of the lags
is rebuilt from the shifted lags: the lag values, their missing flags, and (S1, S2, S3, S3-rw) the
single-step surprise at each lag. For S2, S3, and S3-rw the window statistics are shifted by the
step. `WindowInputs.assemble` takes the step as `lags - lags_unshifted`, converts it to band units
at the lag's time, and adds it to the window's mean, sum, minimum, maximum, and tent (the standard
deviation and the two counts are unchanged, the approximation stated in the plan). The same
band-unit step shifts the lag's own surprise inside the neighbour-relative columns, so the
neighbour-relative surprise and the sign agreement respond to the step. These stay fixed: the
lagless cross-fitted median and band (which define the surprise), the neighbours' own statistics
(the step is on the Primary's lags only, so nothing is mirrored at neighbours), and every non-lag
input.

**Lag-masked control.** Every arm is evaluated with all recent power hidden, the step added before
hiding; `masked_max_gap` is the largest absolute difference of any quantile from the unshifted
masked forecast, which must be below 1e-6.

**Output.** One parquet per arm, month, and seed under `work/sweep/<arm>/`, long format, one row per
(config, Primary, lead band): counts, sums, and the histogram of the per-row share of the step.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00440-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00440-*/e00440_sweep.py sweep --months 2026-06 --seeds 0
uv run --no-sync python ml-experiments/00440-*/e00440_sweep.py report --step c050
uv run --no-sync python ml-experiments/00440-*/e00440_sweep.py selftest --arm S1
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_sweep as sw  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00440_run as run  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.slices import MVA_ONLY_LABELS

BASE_INPUTS: Final = fit.Inputs
"""Experiment 428's input class, captured before `run.install` swaps in `WindowInputs`."""
SWEEP_DIR: Final[Path] = run.WORK_DIR / "sweep"
REFERENCE_ARMS: Final[tuple[str, ...]] = ("R-inj", "S1")
ALL_ARMS: Final[tuple[str, ...]] = (*REFERENCE_ARMS, *run.ARM_NAMES)
KS: Final[tuple[int, ...]] = (1, 2, 3)
SELFTEST_ROWS: Final[int] = 200
SELFTEST_TOLERANCE: Final[float] = 1e-5
KEYS: Final[list[str]] = ["arm", "k", "step", "band", "widened"]


def use_arm(*, arm_name: str) -> type:
    """Point experiment 428's loader at the arm's models folder; return its input class."""
    if arm_name in REFERENCE_ARMS:
        fit.MODELS_DIR = data.WORK_DIR / "models"
        return BASE_INPUTS
    fit.MODELS_DIR = run.WORK_DIR / "models"
    return run.WindowInputs


def lag_shifter(*, k: int, step: torch.Tensor, mva: torch.Tensor):  # noqa: ANN201
    """Return `(modify, changed)`: `modify` adds `step` to the first `k` lags of `raw`.

    The same edit applies to every arm, so the lags stepped are identical. `changed["flag"]` marks
    rows where the zero clip of an MVA-only Primary altered a shifted lag.
    """
    changed: dict[str, torch.Tensor | None] = {"flag": None}

    def modify(raw: dict) -> None:
        lags = raw["lags"]
        before = lags[:, :k].clone()
        moved = before + step[:, None]
        clipped = torch.where(mva[:, None], moved.clamp(min=0.0), moved)
        changed["flag"] = ((clipped != moved) & torch.isfinite(before)).any(1)
        lags[:, :k] = clipped

    return modify, changed


def sweep_arm(
    *, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor, flags: pr.Flags,
    y_clean: np.ndarray,
) -> Path:  # fmt: skip
    """Sweep one saved network and write its table; return the path."""
    target = SWEEP_DIR / arm_name / f"{month}_s{seed}.parquet"
    if target.exists():
        return target
    inputs_class = use_arm(arm_name=arm_name)
    loaded = pr.load(arm_name=arm_name, month=month, seed=seed, device=device)
    arm = loaded.arm
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    keep = ((test.origin // (86_400 * 1_000_000)).astype(np.int64) % sw.ORIGIN_STRIDE) == 0
    mva_primary = torch.tensor([s in MVA_ONLY_LABELS for s in data.sites()], device=device)
    inputs = inputs_class(
        power=power, x=test.x[keep], primary=test.primary[keep], origin_idx=test.origin_idx[keep],
        target_idx=test.target_idx[keep], power_mean=loaded.power_mean,
        power_std=loaded.power_std, device=device,
        surprise=fit.surprise_for(arm=arm, month=month, seed=seed, device=device),
    )  # fmt: skip
    inputs.target_stats = fit.target_stats(
        mean=loaded.target_mean, std=loaded.target_std, device=device
    )
    n_rows = int(keep.sum())
    lead = torch.from_numpy(test.lead[keep]).to(device)
    band_of = torch.full((n_rows,), -1, dtype=torch.long, device=device)
    for i, (_, low, high) in enumerate(sw.LEAD_BANDS):
        upper = lead <= high if i == len(sw.LEAD_BANDS) - 1 else lead < high
        band_of[(lead >= low) & upper] = i
    actual = torch.from_numpy(y_clean[test.primary[keep], test.target_idx[keep]]).to(device)
    world0 = torch.zeros(n_rows, dtype=torch.long, device=device)
    everything = torch.arange(n_rows, device=device)
    flag = flags.of(inputs=inputs, rows=everything, world=world0)
    lags = inputs.raw(arm=fit.ARMS["R"], rows=everything, world=world0)["lags"]
    clean = (
        torch.isfinite(actual) & ~flag["win_real"] & ~flag["tgt_post"] & torch.isfinite(lags).all(1)
    )
    clean &= band_of >= 0
    rows_all = torch.nonzero(clean).squeeze(1)
    trailing = sw.trailing_mean(
        power=power,
        primary=inputs.primary[rows_all],
        newest=inputs.origin[rows_all] - win.NEWEST_AGE,
    )
    accumulators: dict = {}
    masked_gap = 0.0
    for start in range(0, len(rows_all), sw.CHUNK):
        rows = rows_all[start : start + sw.CHUNK]
        world = torch.zeros(len(rows), dtype=torch.long, device=device)
        q0 = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)["q"]
        primary, act, band = inputs.primary[rows], actual[rows], band_of[rows]
        mva = mva_primary[primary]
        out: list[dict] = []
        for step_name in sw.STEPS:
            size = (
                torch.full((len(rows),), 0.025 if step_name == "c025" else 0.05, device=device)
                if step_name != "m10"
                else 0.10 * trailing[start : start + sw.CHUNK].abs()
            )
            valid_step = torch.isfinite(size) & (size > 0)
            for sign in sw.SIGNS:
                signed = sign * size
                for k in KS:
                    modify, changed = lag_shifter(k=k, step=signed, mva=mva)
                    q1 = pr.forecast_rows(
                        loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify
                    )["q"]
                    ok = valid_step & ~changed["flag"]  # ty: ignore[unsupported-operator]
                    share = (q1[:, sw.Q50] - q0[:, sw.Q50]) / signed
                    event = act + signed
                    for widened in (False, True):
                        a, b = (sw.widen(q=q0), sw.widen(q=q1)) if widened else (q0, q1)
                        out.append(
                            {
                                "ok": ok, "primary": primary, "band": band, "step": step_name,
                                "sign": sign, "variant": "lag", "k": k, "widened": widened,
                                "share": share, "w0": sw.width(q=a), "w1": sw.width(q=b),
                                "cover_clean": (
                                    (act >= b[:, sw.Q05]) & (act <= b[:, sw.Q95])
                                ).float(),
                                "cover_event": (
                                    (event >= b[:, sw.Q05]) & (event <= b[:, sw.Q95])
                                ).float(),
                                "mass": None, "mass0": None,
                            }
                        )  # fmt: skip
        masked_gap = max(masked_gap, masked_control(
            loaded=loaded, inputs=inputs, rows=rows, world=world, mva=mva))  # fmt: skip
        sw.flush(out=out, accumulators=accumulators, n_days=0)
    table = sw.finish(
        accumulators=accumulators,
        arm_name=arm_name,
        month=month,
        seed=seed,
        masked_gap=masked_gap,
        n_days=0,
    ).drop("mass0_after_sum", "mass0_before_sum")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    return target


def masked_control(
    *, loaded: pr.Loaded, inputs: fit.Inputs, rows: torch.Tensor, world: torch.Tensor,
    mva: torch.Tensor,
) -> float:  # fmt: skip
    """Return the largest quantile gap between the masked forecast with and without a step."""
    signed = torch.full((len(rows),), 0.05, device=rows.device)
    ref = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world, masked=True)["q"]
    gap = 0.0
    for k in KS:
        modify, _ = lag_shifter(k=k, step=signed, mva=mva)
        moved = pr.forecast_rows(
            loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify, masked=True
        )["q"]
        gap = max(gap, float((moved - ref).abs().max()))
    return gap


def load_tables() -> pl.DataFrame:
    """Return every sweep table of this experiment."""
    frames = [pl.read_parquet(f) for f in sorted(SWEEP_DIR.glob("*/*.parquet"))]
    return pl.concat(frames, how="diagonal")


def share_report(*, table: pl.DataFrame, step: str, band: str, widened: bool) -> pl.DataFrame:
    """Return, per arm and k, the share of the step passed into the median forecast.

    Pools months, seeds, Primaries, and both signs. `median_share` is the median of the pooled
    histogram of per-row shares (+1 is the full step); `mean_share` is the mean of the same.
    """
    sub = table.filter(
        (pl.col("step") == step) & (pl.col("band") == band) & (pl.col("widened") == widened)
    )
    records = []
    for (arm, k), part in sub.group_by(["arm", "k"]):
        hist = np.sum(np.array(part["hist"].to_list()), axis=0)
        n = int(part["n"].sum())
        records.append(
            {
                "arm": arm, "k": k, "n": n,
                "median_share": sw_median(hist=hist),
                "mean_share": float(part["shift_sum"].sum()) / n,
                "width_ratio": float(part["width_after_sum"].sum())
                / float(part["width_before_sum"].sum()),
                "masked_gap": part["masked_max_gap"].max(),
            }
        )  # fmt: skip
    return pl.DataFrame(records).sort("arm", "k")


def sw_median(*, hist: np.ndarray) -> float:
    """Return the median share from histogram counts over `sw.HIST_BINS` (bins 0..61)."""
    total = hist.sum()
    if total == 0:
        return float("nan")
    i = int(np.searchsorted(np.cumsum(hist), total / 2.0))
    edges = np.concatenate([[-np.inf], sw.HIST_BINS, [np.inf]])
    low, high = edges[i], edges[i + 1]
    if not np.isfinite(low):
        return float(high)
    if not np.isfinite(high):
        return float(low)
    return float((low + high) / 2.0)


def selftest(*, arm_name: str, month: str, seed: int, device: str) -> float:
    """Return the largest median difference of an unshifted sweep forecast from the stored one.

    The sweep's own forecast path (`pr.forecast_rows` with a zero step through `lag_shifter`) is
    compared with the stored world-0 forecast on `SELFTEST_ROWS` evenly spaced test rows. If no
    stored forecast exists for the arm (S2, S3, and S3-rw have none until `predict` has run), the
    zero-step path is compared with the no-`modify` path only, and the return value says so by being
    the gap between those two.
    """
    inputs_class = use_arm(arm_name=arm_name)
    power_np, _ = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(device)
    loaded = pr.load(arm_name=arm_name, month=month, seed=seed, device=device)
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    pick = np.linspace(0, len(test.x) - 1, SELFTEST_ROWS).astype(np.int64)
    inputs = inputs_class(
        power=power, x=test.x[pick], primary=test.primary[pick], origin_idx=test.origin_idx[pick],
        target_idx=test.target_idx[pick], power_mean=loaded.power_mean,
        power_std=loaded.power_std, device=device,
        surprise=fit.surprise_for(arm=loaded.arm, month=month, seed=seed, device=device),
    )  # fmt: skip
    inputs.target_stats = fit.target_stats(
        mean=loaded.target_mean, std=loaded.target_std, device=device
    )
    rows = torch.arange(len(pick), device=device)
    world = torch.zeros(len(pick), dtype=torch.long, device=device)
    plain = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)["q"]
    modify, _ = lag_shifter(
        k=3,
        step=torch.zeros(len(pick), device=device),
        mva=torch.zeros_like(rows, dtype=torch.bool),
    )
    zero = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify)[
        "q"
    ]
    path_gap = float((zero[:, sw.Q50] - plain[:, sw.Q50]).abs().max())
    stored_file = pr.PRED_DIR / arm_name / f"{month}_s{seed}_w0.parquet"
    if arm_name in REFERENCE_ARMS:
        stored_file = data.WORK_DIR / "predictions" / arm_name / f"{month}_s{seed}_w0.parquet"
    if not stored_file.exists():
        print(f"{arm_name}: no stored forecast; zero-step vs plain median gap {path_gap:.2e}")
        return path_gap
    stored = pl.read_parquet(stored_file)["q50"].to_numpy()[pick]
    gap = float(np.nanmax(np.abs(plain[:, sw.Q50].double().cpu().numpy() - stored)))
    print(f"{arm_name}: stored vs sweep median gap {gap:.2e}; zero-step path gap {path_gap:.2e}")
    return max(gap, path_gap)


def main() -> None:
    """Run the sweep, print its report, or run the selftest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sweep", "report", "selftest"))
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    parser.add_argument("--arm", default="S1", help="selftest arm")
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", default="c050")
    parser.add_argument("--band", default="24-72 h")
    parser.add_argument("--widened", action="store_true")
    args = parser.parse_args()
    run.install()
    pr.PRED_DIR = run.WORK_DIR / "predictions"
    if args.command == "report":
        out = share_report(
            table=load_tables(), step=args.step, band=args.band, widened=args.widened
        )
        with pl.Config(tbl_rows=100, fmt_float="mixed"):
            print(out.with_columns(pl.col(pl.Float64).round(3)))
        return
    if args.command == "selftest":
        gap = selftest(
            arm_name=args.arm, month=args.months[0], seed=args.seeds[0], device=args.device
        )
        assert gap < SELFTEST_TOLERANCE, f"selftest gap {gap:.2e} exceeds {SELFTEST_TOLERANCE}"
        print("selftest passed")
        return
    power_np, code = data.load_stack(include_eval=False)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                print(
                    sweep_arm(arm_name=name, month=month, seed=seed, device=args.device,
                              power=power, flags=flags, y_clean=power_np[0]),
                    flush=True,
                )  # fmt: skip


if __name__ == "__main__":
    main()
