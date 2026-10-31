# ruff: noqa: ANN201, PLR0915, SIM401
"""Lag-encroachment sweep of experiment 428's saved networks (prediction only).

Follows experiment 392's `e00392_sweep.py`. A step of depth D is added to the recent power of clean
rows and the forecast distribution is compared with the unshifted one.

**Rows.** Every third daily origin of the test month, world 0 (the real, clean series). A row is
clean when its target is finite, outside the padded real-event mask and the 96 half-hours after an
event, no observed half-hour of its three windows is inside the padded mask, and R's three lags are
finite. Every arm is swept on the same rows.

**Steps.** `c025` and `c050`: 2.5% and 5% of effective capacity. `m10`: 10% of the Primary's mean
power over the 30 days up to the newest usable reading. Both signs. MVA-only readings are clipped at
zero after the step; a row where the clip changed any shifted value is dropped from that
configuration (for every arm alike).

**k.** R and R-inj shift the lags as experiment 392 does: k = 1 `lag_origin`; k = 2 also the daily
lag; k = 3 also the weekly lag. Encoders, variant `point` (variant 1 of the plan): k = 1 the newest
half-hour of D1; k = 2 also the half-hour of D1 at the target's time of day; k = 3 also the centre
half-hour of D2; k = 4 (E3 only) also the centre half-hour of D3. Variant `day` (variant 2): k = 1
shifts every observed half-hour of D1; k = 2 also D2; k = 3 also D3. L0 has no recent power and is
swept as a control (share 0).

**Lag-masked control.** Every arm is evaluated in evaluation mode with all recent power hidden, the
step added before hiding; `masked_max_gap` is the largest absolute difference of any quantile from
the unshifted masked forecast, which must be below 1e-6.

**Output.** One parquet per arm, month, and seed under `sweep/`, long format, one row per
(config, Primary, lead band) with counts, sums, the histogram of the per-row share of the step, and
the attention mass sums.

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00428-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00428-*/e00428_sweep.py --arms R E3 --months 2026-01
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_fit as fit  # ty: ignore[unresolved-import]
import e00428_predict as pr  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.eval_protocol.slices import MVA_ONLY_LABELS

SWEEP_DIR: Final[Path] = data.WORK_DIR / "sweep"
ORIGIN_STRIDE: Final[int] = 3
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("24-72 h", 24.0, 72.0),
    ("72-168 h", 72.0, 168.0),
    ("168-240 h", 168.0, 240.0),
)
STEPS: Final[tuple[str, ...]] = ("c025", "c050", "m10")
SIGNS: Final[tuple[int, ...]] = (1, -1)
WIDENING: Final[float] = 1.15
TRAILING_STEPS: Final[int] = 30 * 48
Q05, Q50, Q95 = 1, 5, 9
HIST_BINS: Final[np.ndarray] = np.linspace(-1.0, 2.0, 61)
"""Edges of the histogram of the share of the step followed (+1 is the full step)."""
CHUNK: Final[int] = 4_096


def configs(*, arm: fit.Arm) -> list[tuple[str, int]]:
    """Return the (variant, k) pairs swept for an arm."""
    if arm.history:
        return [("hist", k) for k in (1, 2, 3)]
    if arm.kind == "lags":
        return [("lag", k) for k in (1, 2, 3)]
    if arm.kind == "none":
        return [("none", 1)]
    point = [("point", k) for k in range(1, arm.days + 2)]
    day = [("day", k) for k in range(1, arm.days + 1)]
    return point + day


def trailing_mean(
    *, power: torch.Tensor, primary: torch.Tensor, newest: torch.Tensor
) -> torch.Tensor:
    """Return each row's mean clean power over the 30 days up to `newest` (NaN if none observed)."""
    clean = power[0]
    finite = torch.isfinite(clean)
    csum = torch.cat(
        [
            torch.zeros(clean.shape[0], 1, device=clean.device),
            torch.cumsum(torch.where(finite, clean, torch.zeros_like(clean)), 1),
        ],
        1,
    )
    ccount = torch.cat(
        [
            torch.zeros(clean.shape[0], 1, device=clean.device),
            torch.cumsum(finite.to(torch.float32), 1),
        ],
        1,
    )
    hi = newest.clamp(0, clean.shape[1] - 1) + 1
    lo = (newest - TRAILING_STEPS).clamp(0, clean.shape[1] - 1) + 1
    total = csum[primary, hi] - csum[primary, lo]
    count = ccount[primary, hi] - ccount[primary, lo]
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, float("nan")))


def shift_history(*, raw: dict, k: int, step: torch.Tensor, mva: torch.Tensor, changed: dict):
    """Add `step` to the last `k` whole days of the history window and to the lags inside them.

    A lag whose grid index falls inside the shifted days moves with them, so the step is one event
    seen consistently by every input. Rows where a negative step is clipped at zero (MVA-only
    Primaries) are flagged and dropped from the sweep.
    """
    newest = raw["origin"] - win.NEWEST_AGE
    flag = torch.zeros(len(newest), dtype=torch.bool, device=newest.device)

    def move(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        inside = index > (newest - win.STEPS_PER_DAY * k)[:, None] if values.dim() == 2 else None
        inside = inside & (index <= newest[:, None]) & torch.isfinite(values)  # ty: ignore[unsupported-operator]
        new = values + step[:, None]
        clipped = torch.where(mva[:, None], new.clamp(min=0.0), new)
        nonlocal flag
        flag = flag | ((clipped != new) & inside).any(1)
        return torch.where(inside, clipped, values)

    raw["hist"] = move(raw["hist"], raw["hindex"])
    if "lags" in raw:
        raw["lags"] = move(raw["lags"], win.lag_indices(origin=raw["origin"], target=raw["target"]))
    changed["flag"] = flag


def shifter(*, arm: fit.Arm, variant: str, k: int, step: torch.Tensor, mva: torch.Tensor):
    """Return a function that adds `step` (rows,) to the recent power of `raw` and the clip flag."""
    changed = {"flag": None}

    def modify(raw: dict) -> None:
        if arm.history:
            shift_history(raw=raw, k=k, step=step, mva=mva, changed=changed)
            return
        if arm.kind == "none":
            changed["flag"] = None
            return
        if arm.kind == "lags":
            target = raw["lags"]
            before = target.clone()
            target[:, :k] = target[:, :k] + step[:, None]
            clipped = torch.where(mva[:, None], target[:, :k].clamp(min=0.0), target[:, :k])
            changed["flag"] = (clipped != target[:, :k]).any(1) & torch.isfinite(before[:, :k]).any(
                1
            )
            target[:, :k] = clipped
            return
        values = raw["raw"]
        rows = values.shape[0]
        if variant == "point":
            newest_pos, day_pos = 47, None
            gap = raw["target"] - (raw["origin"] - win.NEWEST_AGE)
            day_idx = raw["target"] - win.STEPS_PER_DAY * win.ceil_div(a=gap, b=win.STEPS_PER_DAY)
            day_pos = 47 - ((raw["origin"] - win.NEWEST_AGE) - day_idx)
            positions = [
                (0, torch.full((rows,), newest_pos, device=values.device)),
                (0, day_pos.clamp(0, 47)),
                (1, torch.full((rows,), 24, device=values.device)),
                (2, torch.full((rows,), 24, device=values.device)),
            ][:k]
            flag = torch.zeros(rows, dtype=torch.bool, device=values.device)
            sel = torch.arange(rows, device=values.device)
            done: list[tuple[int, torch.Tensor]] = []
            for window, pos in positions:
                if window >= values.shape[1]:
                    continue
                repeated = torch.zeros(rows, dtype=torch.bool, device=values.device)
                for earlier_window, earlier_pos in done:
                    if earlier_window == window:
                        repeated |= earlier_pos == pos
                done.append((window, pos))
                old = values[sel, window, pos]
                new = old + step
                clipped = torch.where(mva, new.clamp(min=0.0), new)
                # A position that is unobserved cannot carry the step, and one already shifted
                # in this call is not shifted twice.
                flag |= ((clipped != new) & ~repeated) | ~torch.isfinite(old)
                values[sel, window, pos] = torch.where(repeated, old, clipped)
            changed["flag"] = flag
        else:
            new = values[:, :k, :] + step[:, None, None]
            clipped = torch.where(mva[:, None, None], new.clamp(min=0.0), new)
            visible = raw["index"][:, :k, :] <= (raw["origin"] - win.NEWEST_AGE)[:, None, None]
            changed["flag"] = (
                ((clipped != new) & torch.isfinite(values[:, :k, :]) & visible).flatten(1).any(1)
            )
            values[:, :k, :] = clipped

    return modify, changed


def widen(*, q: torch.Tensor) -> torch.Tensor:
    """Return `q` with every spread about the median multiplied by `WIDENING`."""
    median = q[:, [Q50]]
    return median + WIDENING * (q - median)


def width(*, q: torch.Tensor) -> torch.Tensor:
    """Return the 5%-95% band width per row."""
    return q[:, Q95] - q[:, Q05]


def sweep_arm(*, arm_name: str, month: str, seed: int, device: str, power: torch.Tensor,
              flags: pr.Flags, y_clean: np.ndarray) -> Path:  # fmt: skip
    """Sweep one saved network and write its table; return the path."""
    target = SWEEP_DIR / arm_name / f"{month}_s{seed}.parquet"
    if target.exists():
        return target
    loaded = pr.load(arm_name=arm_name, month=month, seed=seed, device=device)
    arm = loaded.arm
    test = data.build_test(month=month, standardiser=loaded.standardiser)
    day_number = (test.origin // (86_400 * 1_000_000)).astype(np.int64)
    keep = (day_number % ORIGIN_STRIDE) == 0
    sites = data.sites()
    mva_primary = torch.tensor([s in MVA_ONLY_LABELS for s in sites], device=device)
    inputs = fit.Inputs(
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
    for i, (_, low, high) in enumerate(LEAD_BANDS):
        upper = lead <= high if i == len(LEAD_BANDS) - 1 else lead < high
        band_of[(lead >= low) & upper] = i
    actual = torch.from_numpy(y_clean[test.primary[keep], test.target_idx[keep]]).to(device)
    world0 = torch.zeros(n_rows, dtype=torch.long, device=device)
    everything = torch.arange(n_rows, device=device)
    flag = flags.of(inputs=inputs, rows=everything, world=world0)
    ref_arm = fit.ARMS["R"]
    lags = inputs.raw(arm=ref_arm, rows=everything, world=world0)["lags"]
    clean = (
        torch.isfinite(actual) & ~flag["win_real"] & ~flag["tgt_post"] & torch.isfinite(lags).all(1)
    )
    clean &= band_of >= 0
    rows_all = torch.nonzero(clean).squeeze(1)
    trailing = trailing_mean(
        power=power,
        primary=inputs.primary[rows_all],
        newest=inputs.origin[rows_all] - win.NEWEST_AGE,
    )
    out: list[dict] = []
    ACCUMULATORS.clear()
    masked_gap = 0.0
    cfgs = configs(arm=arm)
    n_days = arm.days if arm.kind == "encoder" else 3 if arm.history in ("h14s", "h14s_ow") else 0
    for start in range(0, len(rows_all), CHUNK):
        rows = rows_all[start : start + CHUNK]
        world = torch.zeros(len(rows), dtype=torch.long, device=device)
        base = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world)
        q0 = base["q"]
        mass0 = base.get("mass")
        primary = inputs.primary[rows]
        act = actual[rows]
        band = band_of[rows]
        for step_name in STEPS:
            size = (
                torch.full((len(rows),), 0.025 if step_name == "c025" else 0.05, device=device)
                if step_name != "m10"
                else 0.10 * trailing[start : start + CHUNK].abs()
            )
            valid_step = torch.isfinite(size) & (size > 0)
            for sign in SIGNS:
                signed = sign * size
                for variant, k in cfgs:
                    modify, changed = shifter(
                        arm=arm, variant=variant, k=k, step=signed, mva=mva_primary[primary]
                    )
                    res = pr.forecast_rows(
                        loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify
                    )
                    ok = (
                        valid_step & ~changed["flag"] if changed["flag"] is not None else valid_step
                    )
                    if arm.kind == "none":
                        ok = valid_step
                    q1 = res["q"]
                    share = (q1[:, Q50] - q0[:, Q50]) / signed
                    event = act + signed
                    for widened in (False, True):
                        a, b = (widen(q=q0), widen(q=q1)) if widened else (q0, q1)
                        out.append(
                            {
                                "ok": ok,
                                "primary": primary,
                                "band": band,
                                "step": step_name,
                                "sign": sign,
                                "variant": variant,
                                "k": k,
                                "widened": widened,
                                "share": share,
                                "w0": width(q=a),
                                "w1": width(q=b),
                                "cover_clean": ((act >= b[:, Q05]) & (act <= b[:, Q95])).float(),
                                "cover_event": (
                                    (event >= b[:, Q05]) & (event <= b[:, Q95])
                                ).float(),
                                "mass": res["mass"] if "mass" in res else None,
                                "mass0": mass0,
                            }
                        )
        # lag-masked control: step added, then everything hidden
        signed = torch.full((len(rows),), 0.05, device=device)
        ref = pr.forecast_rows(loaded=loaded, inputs=inputs, rows=rows, world=world, masked=True)[
            "q"
        ]
        for variant, k in cfgs:
            modify, _ = shifter(
                arm=arm, variant=variant, k=k, step=signed, mva=mva_primary[primary]
            )
            moved = pr.forecast_rows(
                loaded=loaded, inputs=inputs, rows=rows, world=world, modify=modify, masked=True
            )["q"]
            masked_gap = max(masked_gap, float((moved - ref).abs().max()))
        flush(out=out, accumulators=ACCUMULATORS, n_days=n_days)
        out = []
    table = finish(accumulators=ACCUMULATORS, arm_name=arm_name, month=month, seed=seed,
                   masked_gap=masked_gap, n_days=n_days)  # fmt: skip
    ACCUMULATORS.clear()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    table.write_parquet(temporary)
    temporary.rename(target)
    return target


ACCUMULATORS: dict = {}


def flush(*, out: list[dict], accumulators: dict, n_days: int) -> None:
    """Add a chunk's per-row results to the accumulators keyed by config and Primary and band."""
    for item in out:
        ok = item["ok"]
        if not bool(ok.any()):
            continue
        key_base = (item["step"], item["sign"], item["variant"], item["k"], item["widened"])
        primary, band = item["primary"][ok], item["band"][ok]
        cell = (primary * len(LEAD_BANDS) + band).cpu().numpy()
        n_cells = len(data.sites()) * len(LEAD_BANDS)
        acc = accumulators.setdefault(
            key_base,
            {
                "n": np.zeros(n_cells), "shift": np.zeros(n_cells), "w0": np.zeros(n_cells),
                "w1": np.zeros(n_cells), "cc": np.zeros(n_cells), "ce": np.zeros(n_cells),
                "hist": np.zeros((n_cells, len(HIST_BINS) + 1)),
                "mass": np.zeros((n_cells, n_days + 1)), "mass0": np.zeros((n_cells, n_days + 1)),
            },
        )  # fmt: skip
        for name, source in (("n", None), ("shift", "share"), ("w0", "w0"), ("w1", "w1"),
                             ("cc", "cover_clean"), ("ce", "cover_event")):  # fmt: skip
            values = (
                np.ones(len(cell)) if source is None else item[source][ok].double().cpu().numpy()
            )
            np.add.at(acc[name], cell, values)
        bins = np.digitize(item["share"][ok].double().cpu().numpy(), HIST_BINS)
        np.add.at(acc["hist"], (cell, bins), 1)
        if item["mass"] is not None:
            np.add.at(acc["mass"], cell, item["mass"][ok].double().cpu().numpy())
            np.add.at(acc["mass0"], cell, item["mass0"][ok].double().cpu().numpy())


def finish(*, accumulators: dict, arm_name: str, month: str, seed: int, masked_gap: float,
           n_days: int) -> pl.DataFrame:  # fmt: skip
    """Return the long table of one network's sweep."""
    sites = data.sites()
    records = []
    for (step, sign, variant, k, widened), acc in accumulators.items():
        for p, site in enumerate(sites):
            for b, (band_name, _, _) in enumerate(LEAD_BANDS):
                c = p * len(LEAD_BANDS) + b
                if acc["n"][c] == 0:
                    continue
                record = {
                    "arm": arm_name, "month": month, "seed": seed, "site": site,
                    "band": band_name, "step": step, "sign": sign, "variant": variant, "k": k,
                    "widened": widened, "n": acc["n"][c], "shift_sum": acc["shift"][c],
                    "width_after_sum": acc["w1"][c], "width_before_sum": acc["w0"][c],
                    "cover_clean_sum": acc["cc"][c], "cover_event_sum": acc["ce"][c],
                    "hist": acc["hist"][c].tolist(), "masked_max_gap": masked_gap,
                }  # fmt: skip
                for d in range(n_days + 1):
                    record[f"mass{d}_after_sum"] = acc["mass"][c, d]
                    record[f"mass{d}_before_sum"] = acc["mass0"][c, d]
                records.append(record)
    return pl.DataFrame(records)


def main() -> None:
    """Sweep the requested arms, months, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(fit.ARMS))
    parser.add_argument("--months", nargs="+", default=list(fit.SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(fit.SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
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
