# ruff: noqa: E501, PLR0915
r"""Fit the ICON-EU adaptors (N1, N2) and forecast N0, N1, N2 for experiment 552, month by month.

N0 feeds the ECMWF ensemble mean (the latest run usable at the issue time) straight into experiment
524's frozen stage-1 network. N1 replaces the seven mean-weather inputs by an adaptor's output from
the latest usable ICON-EU run. N2 gives the adaptor the ECMWF means and spreads as well. An adaptor
is fitted in two phases on rows with `valid <= month start - 3.5 h`: phase A regresses the seven
standardised ECMWF means, phase B trains the adaptor through the frozen stage 1 on the power
negative log-likelihood. Early stopping uses every 5th issue day.

```bash
cd <worktree>; source ml-experiments/00552-*/env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00552-*/e00552_run.py --months 2026-03 2026-04 2026-05 2026-06
```
"""

import argparse
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00524_stage1 as s1  # ty: ignore[unresolved-import]
import e00524_stage2 as s2  # ty: ignore[unresolved-import]
import e00552_data as d  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PRED_DIR: Final[Path] = d.WORK_DIR / "preds"
K: Final[int] = c.COMPONENTS
FIRST_ISSUE_DAY: Final[datetime] = datetime(2026, 2, 10, tzinfo=UTC)
ARMS: Final[tuple[str, ...]] = ("N0", "N1", "N2")
PHASE_A_EPOCHS, PHASE_B_EPOCHS, PATIENCE = 40, 25, 5
BATCH: Final[int] = 4_096
STAGE1_KEEP: Final[list[int]] = [*range(7), *range(14, 35)]
"""Stage 1's inputs: the seven means, then everything from `cos_zenith` to `nearest_inside`."""


def issue_times(*, start: datetime, stop: datetime) -> np.ndarray:
    """Return every 00, 06, 12, and 18 UTC issue time in [start, stop), in microseconds."""
    step = 6 * d.H
    first = int(start.timestamp() * 1e6)
    return np.arange(first, int(stop.timestamp() * 1e6), step, dtype=np.int64)


def fold_stats(*, month: str, clean: np.ndarray) -> dict:
    """Return (and cache) the fold's standardiser and per-Primary target statistics."""
    path = d.WORK_DIR / f"stats_{month}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    rows = data.build_fold(refit=c.month_start(month=month), clean=clean)
    assert rows.columns[:7] == [
        "mean_shortwave", "mean_longwave", "mean_temperature_2m", "mean_dew_point_temperature_2m",
        "mean_pressure_surface", "mean_precipitation", "mean_wind_speed_10m",
    ]  # fmt: skip
    assert s1.stage1_columns(columns=rows.columns) == STAGE1_KEEP
    out = {
        "standardiser": rows.standardiser,
        "target_mean": rows.target_mean,
        "target_std": rows.target_std,
        "columns": rows.columns,
    }
    path.write_bytes(pickle.dumps(out))
    return out


def adaptor_inputs(
    *, x: np.ndarray, icon: np.ndarray, lead: np.ndarray, arm: str, icon_stats: tuple
) -> np.ndarray:
    """Return the adaptor's inputs: standardised ICON-EU features, cos_zenith, hour terms, lead, and for N2 the ECMWF means and spreads."""
    mu, sd = icon_stats
    parts = [
        np.clip((d.icon_features(icon) - mu) / sd, -5, 5),
        x[:, [14, 17, 18]],
        (lead / 78.0)[:, None],
    ]
    if arm == "N2":
        parts.append(x[:, :14])
    return np.column_stack(parts).astype(np.float32)


def new_adaptor(n_in: int) -> torch.nn.Module:
    """Return a 2 x 32 multilayer perceptron from adaptor inputs to the seven standardised means."""
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 7),
    )


def stage_input(*, adapted: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """Return stage 1's input row, with the seven means replaced by `adapted` when given."""
    rest = x[:, 14:35]
    means = x[:, :7] if adapted is None else adapted
    return torch.cat([means, rest], dim=1)


def fit_adaptor(
    *,
    arm: str,
    a: np.ndarray,
    x: np.ndarray,
    p: np.ndarray,
    z: np.ndarray,
    val: np.ndarray,
    stage1: torch.nn.Module,
    seed: int,
    dev: str,
) -> torch.nn.Module:
    """Fit an adaptor in two phases and return it in eval mode."""
    torch.manual_seed(seed)
    net = new_adaptor(a.shape[1]).to(dev)
    t = {k: torch.from_numpy(v).to(dev) for k, v in {"a": a, "x": x, "p": p, "z": z}.items()}
    tr, va = np.nonzero(~val)[0], torch.from_numpy(np.nonzero(val)[0]).to(dev)
    rng = np.random.default_rng(seed)

    def loss_fn(idx: torch.Tensor, phase: str) -> torch.Tensor:
        out = net(t["a"][idx])
        if phase == "A":
            return ((out - t["x"][idx][:, :7]) ** 2).mean()
        raw = stage1(stage_input(adapted=out, x=t["x"][idx]), t["p"][idx])
        return base.mixture_nll(out=raw, z=t["z"][idx], components=K).mean()

    for phase, epochs, lr in (("A", PHASE_A_EPOCHS, 3e-3), ("B", PHASE_B_EPOCHS, 1e-3)):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
        best, stale, state = float("inf"), 0, None
        for _ in range(epochs):
            net.train()
            order = torch.from_numpy(rng.permutation(tr)).to(dev)
            for s in range(0, len(order), BATCH):
                opt.zero_grad()
                loss_fn(order[s : s + BATCH], phase).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(va, phase))
            if v < best - 1e-5:
                best, stale, state = v, 0, {k: q.clone() for k, q in net.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        assert state is not None
        net.load_state_dict(state)
    net.eval()
    return net


def quantiles(*, raw: np.ndarray, p: np.ndarray, stats: dict) -> np.ndarray:
    """Return the sorted 11 quantiles as a fraction of capacity from raw mixture outputs."""
    out = []
    with torch.no_grad():
        for s in range(0, len(raw), 65_536):
            log_w, mean, std = base.mixture_parts(
                out=torch.from_numpy(raw[s : s + 65_536]).double(), components=K
            )
            q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
            pp = p[s : s + 65_536]
            q = (
                q * torch.from_numpy(stats["target_std"][pp])[:, None]
                + torch.from_numpy(stats["target_mean"][pp])[:, None]
            )
            out.append(torch.sort(q, dim=1).values.float().numpy())
    return np.concatenate(out)


def run_month(*, month: str, seeds: list[int], src: d.Sources, clean: np.ndarray, dev: str) -> None:
    """Fit and forecast one month's three arms for each seed."""
    refit = c.month_start(month=month)
    stats = fold_stats(month=month, clean=clean)
    std = stats["standardiser"]
    cutoff_us = int(refit.timestamp() * 1e6)
    train = src.make_rows(issues=issue_times(start=FIRST_ISSUE_DAY, stop=refit))
    train_ok = train.valid <= cutoff_us - c.CUTOFF_AGE_US
    y = clean[train.primary, d.target_idx(train.valid)]
    train_ok &= np.isfinite(y)
    sel = np.nonzero(train_ok)[0]
    print(
        f"{month}: {len(sel):,} adaptor training rows from {len(np.unique(train.issue[sel]))} issue times",
        flush=True,
    )
    xs_tr = std.apply(x=np.pad(train.x_ecm[sel], ((0, 0), (0, 0))))  # standardised, 37 columns
    xs_tr = xs_tr.astype(np.float32)
    lead_tr = ((train.valid[sel] - train.issue[sel]) / d.H).astype(np.float32)
    icon_tr = d.icon_features(train.icon[sel])
    icon_stats = (icon_tr.mean(0), icon_tr.std(0) + 1e-6)
    z_tr = (
        (y[sel] - stats["target_mean"][train.primary[sel]])
        / stats["target_std"][train.primary[sel]]
    ).astype(np.float32)
    issue_day = ((train.issue[sel] - int(FIRST_ISSUE_DAY.timestamp() * 1e6)) // d.DAY).astype(int)
    val = issue_day % 5 == 4
    test = src.make_rows(issues=issue_times(start=refit, stop=c.month_stop(month=month)))
    d.print_example(test)
    xs_te = std.apply(x=test.x_ecm).astype(np.float32)
    lead_te = ((test.valid - test.issue) / d.H).astype(np.float32)
    n_prim = len(stats["target_mean"])
    for seed in seeds:
        todo = [a for a in ARMS if not (PRED_DIR / a / f"{month}_s{seed}.parquet").exists()]
        if not todo:
            continue
        stage1, keep = s2.load_stage1(path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_prim)
        assert keep == STAGE1_KEEP
        stage1 = stage1.to(dev)
        for q in stage1.parameters():
            q.requires_grad_(False)
        for arm in todo:
            started = time.monotonic()
            if arm == "N0":
                adapted = None
            else:
                a_tr = adaptor_inputs(
                    x=xs_tr, icon=train.icon[sel], lead=lead_tr, arm=arm, icon_stats=icon_stats
                )
                net = fit_adaptor(
                    arm=arm,
                    a=a_tr,
                    x=xs_tr,
                    p=train.primary[sel],
                    z=z_tr,
                    val=val,
                    stage1=stage1,
                    seed=seed,
                    dev=dev,
                )
                a_te = adaptor_inputs(
                    x=xs_te, icon=test.icon, lead=lead_te, arm=arm, icon_stats=icon_stats
                )
            raws = []
            with torch.no_grad():
                for s in range(0, len(xs_te), 131_072):
                    sl = slice(s, s + 131_072)
                    xb = torch.from_numpy(xs_te[sl]).to(dev)
                    adapted = None if arm == "N0" else net(torch.from_numpy(a_te[sl]).to(dev))
                    raws.append(
                        stage1(
                            stage_input(adapted=adapted, x=xb),
                            torch.from_numpy(test.primary[sl]).to(dev),
                        )
                        .cpu()
                        .numpy()
                    )
            q = quantiles(raw=np.concatenate(raws), p=test.primary, stats=stats)
            frame = pl.DataFrame(
                {
                    "primary": test.primary,
                    "issue": test.issue,
                    "valid": test.valid,
                    "ecm_origin": test.ecm_origin,
                    "icon_init": test.icon_init,
                }
            ).hstack(pl.DataFrame(q, schema=list(c.common.LEVEL_COLUMNS), orient="row"))
            path = PRED_DIR / arm / f"{month}_s{seed}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
            print(
                f"  {month} s{seed} {arm}: {len(frame):,} rows, {time.monotonic() - started:.0f} s",
                flush=True,
            )


def main() -> None:
    """Run the months."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", default=["2026-03", "2026-04", "2026-05", "2026-06"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    clean = c.clean_stack()
    src = d.Sources()
    for month in args.months:
        run_month(month=month, seeds=args.seeds, src=src, clean=clean, dev=args.device)


if __name__ == "__main__":
    main()
