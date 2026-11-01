# ruff: noqa: E501, PLR0915
"""Paired lag-encroachment sweep with and without a mirror at a neighbour (prediction only).

Experiment 440's protocol (`e00440_sweep`), with a mirror added. A step of 5% of the Primary's
effective capacity, either sign, is added to the first `k` lags (k = 1, 2, 3 lags: the newest
reading, then the daily lag, then the weekly lag) of every clean row of the real clean series
(world 0), and the share of the step that the median forecast follows is
`(q50 with the step - q50 without) / step`. Every arm is swept on the same rows.

- **No mirror:** nothing else changes, as in experiment 440.
- **Mirror:** the Primary's first neighbour in the adjacency file is the recipient. Its window
  statistics and surprise move by `-step * capacity_primary / capacity_recipient`, converted to its
  own band units at each stepped lag's time (the load-conserving signature of experiment 402). The
  recipient's own lags are not inputs of any arm, so the mirror reaches only S3-amb (real
  neighbours). S3-rw-amb has random neighbours that do not hold the recipient, and U1n-2 and S1 see no
  neighbour columns, so they are controls that the mirror cannot move.

The window statistics shift as in experiment 440 (mean, sum, minimum, maximum, tent by the step in
band units; standard deviation and counts unchanged, an approximation). Rows where the zero clip of
an MVA-only Primary changed a stepped lag are dropped for every arm. The step is on the lags only,
so a row follows the step only through the lag inputs, and the share measures reliance on the lags,
not the Bayes-optimal posterior of the training worlds.

**Output.** One parquet per arm, month, and seed under `work/sweep/<arm>/` with one row per
(k, sign, mirror, clean row): `primary`, `lead`, `share`. `report` pools them.

```bash
source /tmp/pp462.sh; export DATA_PATH_INTERNAL=...
PYTHONPATH=$PP uv run --no-sync python ml-experiments/00462-*/e00462_sweep.py sweep --seeds 0 1
uv run --no-sync python ml-experiments/00462-*/e00462_sweep.py report
```
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_sweep as sw  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import e00440_sweep as sweep440  # ty: ignore[unresolved-import]
import e00462_run as run
import e00462_worlds as worlds
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.slices import MVA_ONLY_LABELS

SWEEP_DIR: Final[Path] = run.WORK_DIR / "sweep"
ARMS: Final[tuple[str, ...]] = ("S3-amb", "S3-rw-amb", "U1n-2", "S1")
STEP: Final[float] = 0.05
KS: Final[tuple[int, ...]] = (1, 2, 3)
SIGNS: Final[tuple[int, ...]] = (1, -1)
MAX_LEAD_HOURS: Final[float] = 240.0
FIRST_NEIGHBOUR: Final[int] = 0


def recipient_tables(*, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(recipient, capacity)`: each Primary's first neighbour index and every capacity in MW."""
    _, capacity, neighbours, _, _ = worlds.context()
    recipient = torch.tensor([n[FIRST_NEIGHBOUR] if n else -1 for n in neighbours], device=device)
    return recipient, torch.tensor(capacity, dtype=torch.float32, device=device)


def sweep_arm(
    *, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor, flags: pr.Flags,
    y_clean: np.ndarray,
) -> Path:  # fmt: skip
    """Sweep one saved network and write its table; return the path."""
    target = SWEEP_DIR / arm_name / f"{month}_s{seed}.parquet"
    if target.exists():
        return target
    mine = arm_name in run.ARM_NAMES
    run.use_models(arm_name=arm_name)
    loaded = pr.load(arm_name=arm_name, month=month, seed=seed, device=device)
    arm = loaded.arm
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    keep = ((test.origin // (86_400 * 1_000_000)).astype(np.int64) % sw.ORIGIN_STRIDE) == 0
    mva_primary = torch.tensor([s in MVA_ONLY_LABELS for s in data.sites()], device=device)
    inputs_class = run.AmbInputs if mine else fit.Inputs
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
    actual = torch.from_numpy(y_clean[test.primary[keep], test.target_idx[keep]]).to(device)
    world0 = torch.zeros(n_rows, dtype=torch.long, device=device)
    everything = torch.arange(n_rows, device=device)
    flag = flags.of(inputs=inputs, rows=everything, world=world0)
    lags = inputs.raw(arm=fit.ARMS["R"], rows=everything, world=world0)["lags"]
    clean = (
        torch.isfinite(actual) & ~flag["win_real"] & ~flag["tgt_post"] & torch.isfinite(lags).all(1)
        & (lead <= MAX_LEAD_HOURS)
    )  # fmt: skip
    rows_all = torch.nonzero(clean).squeeze(1)
    recipient_of, capacity = recipient_tables(device=device)
    half = inputs.surprise[1]
    records = []
    for start in range(0, len(rows_all), sw.CHUNK):
        rows = rows_all[start : start + sw.CHUNK]
        world = torch.zeros(len(rows), dtype=torch.long, device=device)
        primary = inputs.primary[rows]
        q0 = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)["q"]
        mva = mva_primary[primary]
        recipient = recipient_of[primary]
        has_recipient = recipient >= 0
        safe_recipient = recipient.clamp(min=0)
        lag_grid = win.lag_indices(origin=inputs.origin[rows], target=inputs.target[rows])
        half_recipient = win.take(
            power=half[None], world=world, primary=safe_recipient, index=lag_grid
        )  # fmt: skip
        for sign in SIGNS:
            signed = torch.full((len(rows),), sign * STEP, device=device)
            for k in KS:
                modify_lags, changed = sweep440.lag_shifter(k=k, step=signed, mva=mva)
                ok = torch.ones(len(rows), dtype=torch.bool, device=device)
                for mirror in (False, True):
                    if mirror and not mine:
                        continue
                    if mirror:
                        ratio = capacity[primary] / capacity[safe_recipient]
                        step_band = -(signed * ratio)[:, None] / half_recipient
                        step_band = torch.nan_to_num(step_band, nan=0.0)
                        step_band[:, k:] = 0.0

                    modify = with_mirror(
                        modify_lags=modify_lags,
                        step_band=step_band if mirror else None,
                        recipient=recipient,
                    )

                    q1 = pr.forecast_rows(
                        loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify
                    )["q"]
                    keep_row = ~changed["flag"] & (has_recipient if mirror else ok)  # ty: ignore[unsupported-operator]
                    share = (q1[:, sw.Q50] - q0[:, sw.Q50]) / signed
                    records.append(
                        pl.DataFrame(
                            {
                                "k": k,
                                "sign": sign,
                                "mirror": mirror,
                                "primary": primary[keep_row].cpu().numpy(),
                                "lead": inputs_lead(lead=lead, rows=rows)[keep_row].cpu().numpy(),
                                "share": share[keep_row].float().cpu().numpy(),
                            }
                        ).with_columns(pl.col("k").cast(pl.Int8), pl.col("sign").cast(pl.Int8))
                    )
    table = pl.concat(records).with_columns(
        arm=pl.lit(arm_name), month=pl.lit(month), seed=pl.lit(seed)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    return target


def with_mirror(
    *, modify_lags: Callable[[dict], None], step_band: torch.Tensor | None, recipient: torch.Tensor
) -> Callable[[dict], None]:
    """Return an edit that steps the lags and, if `step_band` is given, the recipient's columns."""

    def modify(raw: dict) -> None:
        modify_lags(raw)
        if step_band is not None:
            raw["mirror_shift"] = step_band
            raw["recipient"] = recipient

    return modify


def inputs_lead(*, lead: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Return the lead hours of the swept rows."""
    return lead[rows]


def report() -> None:
    """Pool the sweep tables: mean share followed per arm, k, and mirror, and the gap."""
    frames = [pl.read_parquet(f) for f in sorted(SWEEP_DIR.glob("*/*.parquet"))]
    table = pl.concat(frames, how="diagonal")
    pooled = (
        table.group_by("arm", "k", "mirror")
        .agg(pl.len().alias("n"), pl.col("share").mean().alias("mean_share"),
             pl.col("share").median().alias("median_share"))
        .sort("arm", "k", "mirror")
    )  # fmt: skip
    with pl.Config(tbl_rows=100):
        print(pooled)
        wide = pooled.pivot(on="mirror", index=["arm", "k"], values="mean_share").rename(
            {"false": "without_mirror", "true": "with_mirror"}
        )
        if "with_mirror" in wide.columns:
            wide = wide.with_columns(gap=pl.col("without_mirror") - pl.col("with_mirror"))
        print(wide.sort("arm", "k"))
    out = run.WORK_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pooled.write_parquet(out / "sweep_pooled.parquet")
    table.group_by("arm", "k", "mirror", "month", "seed").agg(
        pl.len().alias("n"), pl.col("share").sum().alias("share_sum")
    ).write_parquet(out / "sweep_blocks.parquet")


def main() -> None:
    """Sweep the saved networks or print the pooled report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sweep", "report"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run.install()
    pr.PRED_DIR = run.PRED_DIR
    if args.command == "report":
        report()
        return
    power_np, code = data.load_stack(include_eval=True)
    power = torch.from_numpy(power_np).to(args.device)
    flags = pr.Flags(code=code, real=data.real_event_mask(), power=power_np, device=args.device)
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                print(
                    sweep_arm(
                        arm_name=name, month=month, seed=seed, device=args.device, power=power,
                        flags=flags, y_clean=power_np[0],
                    ),
                    flush=True,
                )  # fmt: skip


if __name__ == "__main__":
    main()
