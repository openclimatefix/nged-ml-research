# ruff: noqa: E501
r"""Stage 2 of experiment 524: the ensemble calibrator on the mixture of a frozen stage 1.

For each fold month and seed, `cache` runs the frozen stage-1 networks (`e00524_stage1.py`) over the
ENS rows of the 730-day window with a cross-fit (each row uses a stage-1 network that left out the
power around its valid time, which removes the stacking leak) and over the month's test rows, and stores the raw 9 mixture outputs with the stage-2 inputs.
`fit` then trains, at each of the five weekly cutoffs, a stage-2 calibrator on the cross-fitted
rows that satisfy experiment 510's row rule (valid time at most cutoff - 3.5 h, origin at most the
cutoff, within 730 days, weight 0.5 ** (age / 90)), and forecasts the cutoff's week.

Forms: `A` is a per-(group, lead bin) shift of the three means and shift of the three log standard
deviations (identity at start); `B` is a 2 x 32 multilayer perceptron on the 9 mixture outputs, the
lead, two spread terms, two hour terms, and the group, returning a residual on the 9 outputs.
Losses: `nll` and the closed-form `crps` of the Gaussian mixture. The arm `S2-0F` is stage 1 alone.

```bash
cd <worktree>; source ml-experiments/00524-*/env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00524-*/e00524_stage2.py cache
```
"""

import argparse
import math
import time
from pathlib import Path
from typing import Final

import e00397_models as base  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00448_refs as refs448  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00510_fit as fit510  # ty: ignore[unresolved-import]
import e00524_stage1 as s1  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

W: Final[Path] = s1.WORK_DIR
CACHE: Final[Path] = W / "cache"
LEAD_BINS: Final[int] = 8
LEAD_BIN_HOURS: Final[float] = 30.0
HALF_LIFE: Final[float] = c.HALF_LIFE_90
STAGE2_LR: Final[float] = 3e-3
STAGE2_EPOCHS: Final[int] = 20
STAGE2_PATIENCE: Final[int] = 4
BATCH: Final[int] = 8_192
K: Final[int] = c.COMPONENTS


def group_flag() -> np.ndarray:
    """Return 1 for the signed-MW six Primaries and 0 otherwise, by Primary index."""
    table = refs448.sites_table().sort("primary")
    return (table["group"].to_numpy() == "signed_six").astype(np.float32)


def aux_inputs(
    *, x: np.ndarray, columns: list[str], lead: np.ndarray, primary: np.ndarray
) -> np.ndarray:
    """Return stage 2's side inputs: lead / 240, two spread terms, two hour terms, and the group flag."""
    pick = [
        columns.index(n) for n in ("std_temperature_2m", "std_shortwave", "sin_hour", "cos_hour")
    ]
    side = x[:, pick].astype(np.float32)
    return np.column_stack([lead.astype(np.float32) / 240.0, side, group_flag()[primary]]).astype(
        np.float32
    )


def load_stage1(*, path: Path, n_primaries: int) -> tuple[torch.nn.Module, list[int]]:
    """Load a frozen stage-1 network and its kept column positions."""
    meta = s1.json.loads(path.with_suffix(".json").read_text())
    model = c.new_model(outlier=False, n_inputs=len(meta["keep"]), n_primaries=n_primaries)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, meta["keep"]


def raw_outputs(*, model: torch.nn.Module, x: np.ndarray, primary: np.ndarray) -> np.ndarray:
    """Return the network's 9 raw outputs (logits, means, log standard deviations) per row."""
    out = []
    with torch.no_grad():
        for a in range(0, len(x), 262_144):
            sl = slice(a, a + 262_144)
            out.append(model(torch.from_numpy(x[sl]), torch.from_numpy(primary[sl])).numpy())
    return np.concatenate(out).astype(np.float32)


def cross_fitted_outputs(
    *,
    month: str,
    seed: int,
    x: np.ndarray,
    primary: np.ndarray,
    valid_us: np.ndarray,
    n_primaries: int,
) -> np.ndarray:
    """Return stage-1 outputs that never saw the power at the row's valid time.

    A row whose valid time lies inside block `k` of the ERA5 and CAMS history uses the network that
    left block `k` out (with a 3-day pad). A row after the last block (valid time after the
    pretraining rows end) uses the fold's full stage-1 network, which has not seen that power.
    """
    meta = s1.json.loads((s1.CROSSFIT / f"{month}_s{seed}_b0.json").read_text())
    edges = np.array(meta["edges"])
    idx = (valid_us - c.prep422.GRID_START_US) // c.prep422.STEP_US
    block = np.clip(np.searchsorted(edges, idx, side="right") - 1, 0, s1.BLOCKS)
    block = np.where(idx >= edges[-1], s1.BLOCKS, block)
    out = np.zeros((len(x), 3 * K), dtype=np.float32)
    for b in range(s1.BLOCKS + 1):
        sel = np.nonzero(block == b)[0]
        if len(sel) == 0:
            continue
        path = (
            s1.STAGE1 / f"{month}_s{seed}.pt"
            if b == s1.BLOCKS
            else s1.CROSSFIT / f"{month}_s{seed}_b{b}.pt"
        )
        model, keep = load_stage1(path=path, n_primaries=n_primaries)
        out[sel] = raw_outputs(
            model=model, x=np.ascontiguousarray(x[sel][:, keep]), primary=primary[sel]
        )
    return out


def cache(*, months: list[str], seeds: list[int]) -> None:
    """Write the stage-1 outputs of the cross-fitted ENS rows and the test rows of each month and seed."""
    torch.set_num_threads(2)
    clean = c.clean_stack()
    for month in months:
        todo = [s for s in seeds if not (CACHE / f"{month}_s{s}.npz").exists()]
        if not todo:
            continue
        mr = c.build_month(month=month, clean=clean)
        test = data.build_test(month=month, standardiser=mr.base.standardiser)
        columns = list(mr.base.columns)
        n_primaries = len(mr.base.target_mean)
        window_start = c.us(t=c.month_start(month=month)) - c.WINDOW_DAYS * c.DAY_US
        rows = np.nonzero(mr.valid_us > window_start)[0]
        z = (mr.y[rows] - mr.base.target_mean[mr.primary[rows]]) / mr.base.target_std[
            mr.primary[rows]
        ]
        aux_train = aux_inputs(
            x=mr.x[rows], columns=columns, lead=mr.lead[rows], primary=mr.primary[rows]
        )
        aux_test = aux_inputs(x=test.x, columns=columns, lead=test.lead, primary=test.primary)
        for seed in todo:
            started = time.monotonic()
            out_train = cross_fitted_outputs(
                month=month, seed=seed, x=mr.x[rows], primary=mr.primary[rows],
                valid_us=mr.valid_us[rows], n_primaries=n_primaries,
            )  # fmt: skip
            model, keep = load_stage1(
                path=s1.STAGE1 / f"{month}_s{seed}.pt", n_primaries=n_primaries
            )
            out_test = raw_outputs(
                model=model, x=np.ascontiguousarray(test.x[:, keep]), primary=test.primary
            )
            CACHE.mkdir(parents=True, exist_ok=True)
            tmp = CACHE / f"{month}_s{seed}.tmp.npz"
            np.savez(
                tmp, out_train=out_train, aux_train=aux_train, z_train=z.astype(np.float32),
                primary_train=mr.primary[rows], valid_train=mr.valid_us[rows], origin_train=mr.origin_us[rows],
                lead_train=mr.lead[rows], out_test=out_test, aux_test=aux_test, primary_test=test.primary,
                valid_test=test.valid, origin_test=test.origin, lead_test=test.lead,
                target_mean=mr.base.target_mean, target_std=mr.base.target_std,
            )  # fmt: skip
            tmp.rename(CACHE / f"{month}_s{seed}.npz")
            print(
                f"cache {month} s{seed}: {len(rows):,} ENS rows, {len(test.primary):,} test rows, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del mr, test


class FormA(torch.nn.Module):
    """Per-(group, lead bin) shift of the means and of the log standard deviations (identity at start)."""

    def __init__(self) -> None:
        """Create zero-initialised parameters."""
        super().__init__()
        self.mean_shift = torch.nn.Parameter(torch.zeros(2, LEAD_BINS))
        self.log_std_shift = torch.nn.Parameter(torch.zeros(2, LEAD_BINS))

    def forward(self, out: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Return the calibrated raw outputs."""
        lead_bin = (aux[:, 0] * 240.0 / LEAD_BIN_HOURS).long().clamp(0, LEAD_BINS - 1)
        group = aux[:, -1].long()
        delta = torch.zeros_like(out)
        delta[:, K : 2 * K] = self.mean_shift[group, lead_bin][:, None]
        delta[:, 2 * K :] = self.log_std_shift[group, lead_bin][:, None]
        return out + delta


class FormB(torch.nn.Module):
    """Two hidden layers of 32 on the mixture outputs and the side inputs; residual on the outputs."""

    def __init__(self) -> None:
        """Create the network with a zero last layer."""
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3 * K + 6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
            torch.nn.ReLU(), torch.nn.Linear(32, 3 * K),
        )  # fmt: skip
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, out: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Return the calibrated raw outputs."""
        return out + self.net(torch.cat([out, aux], dim=1))


def _a(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Return E|X| - style term A(m, s) = m (2 Phi(m / s) - 1) + 2 s phi(m / s) of the mixture CRPS."""
    t = m / s
    cdf = 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * t**2) / math.sqrt(2.0 * math.pi)
    return m * (2.0 * cdf - 1.0) + 2.0 * s * pdf


def mixture_crps(*, out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Return the closed-form CRPS of each row's Gaussian mixture at `z` (standardised units)."""
    log_w, mean, std = base.mixture_parts(out=out, components=K)
    w = log_w.exp()
    first = (w * _a(z[:, None] - mean, std)).sum(-1)
    diff = mean[:, :, None] - mean[:, None, :]
    scale = torch.sqrt(std[:, :, None] ** 2 + std[:, None, :] ** 2)
    second = 0.5 * (w[:, :, None] * w[:, None, :] * _a(diff, scale)).sum((1, 2))
    return first - second


def loss_rows(
    *, model: torch.nn.Module, out: torch.Tensor, aux: torch.Tensor, z: torch.Tensor, loss: str
) -> torch.Tensor:
    """Return the per-row training loss of the calibrated mixture."""
    calibrated = model(out, aux)
    if loss == "nll":
        return base.mixture_nll(out=calibrated, z=z, components=K)
    return mixture_crps(out=calibrated, z=z)


def train_stage2(
    *, form: str, loss: str, out: np.ndarray, aux: np.ndarray, z: np.ndarray, weight: np.ndarray,
    origin_day: np.ndarray, seed: int,
) -> torch.nn.Module:  # fmt: skip
    """Train a calibrator; every 5th origin day is the early-stopping set."""
    torch.manual_seed(seed)
    model = FormA() if form == "A" else FormB()
    val = origin_day % 5 == 4
    t = {k: torch.from_numpy(v) for k, v in {"out": out, "aux": aux, "z": z, "w": weight}.items()}
    train_idx, val_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    val_idx = np.random.default_rng(0).permutation(val_idx)[: c.VALIDATION_ROWS]
    opt = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=c.WEIGHT_DECAY)
    best, best_state, stale = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for _ in range(STAGE2_EPOCHS):
        model.train()
        order = rng.permutation(train_idx)[: int(c.ROW_FRACTION * len(train_idx))]
        for a in range(0, len(order), BATCH):
            b = torch.from_numpy(order[a : a + BATCH])
            rows = loss_rows(model=model, out=t["out"][b], aux=t["aux"][b], z=t["z"][b], loss=loss)
            value = (t["w"][b] * rows).sum() / t["w"][b].sum()
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(
                loss_rows(
                    model=model,
                    out=t["out"][val_idx],
                    aux=t["aux"][val_idx],
                    z=t["z"][val_idx],
                    loss=loss,
                ).mean()
            )
        if v < best - 1e-5:
            best, stale = v, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            stale += 1
            if stale >= STAGE2_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


def quantile_frame(*, out: np.ndarray, d: dict, select: np.ndarray) -> pl.DataFrame:
    """Return the sorted, un-standardised 11 quantiles of raw mixture outputs for the selected test rows."""
    idx = np.nonzero(select)[0]
    parts = []
    with torch.no_grad():
        for a in range(0, len(idx), 65_536):
            sl = idx[a : a + 65_536]
            log_w, mean, std = base.mixture_parts(
                out=torch.from_numpy(out[sl]).double(), components=K
            )
            q = base.mixture_quantiles(log_w=log_w, mean=mean, std=std)
            p = d["primary_test"][sl]
            q = (
                q * torch.from_numpy(d["target_std"][p])[:, None]
                + torch.from_numpy(d["target_mean"][p])[:, None]
            )
            parts.append(torch.sort(q, dim=1).values.float().numpy())
    q = np.concatenate(parts)
    table = c.prep.primaries().select("primary", "time_series_id")
    frame = (
        pl.DataFrame(
            {
                "primary": d["primary_test"][idx],
                "power_fcst_init_time": d["origin_test"][idx],
                "valid_time": d["valid_test"][idx],
            }
        )
        .with_columns(
            pl.col("power_fcst_init_time").cast(pl.Datetime("us", "UTC")),
            pl.col("valid_time").cast(pl.Datetime("us", "UTC")),
        )
        .join(table, on="primary", how="left", maintain_order="left")
        .drop("primary")
    )
    levels = pl.DataFrame(q, schema=list(c.common.LEVEL_COLUMNS), orient="row")
    return frame.select(
        pl.col("time_series_id").cast(pl.Int32), "power_fcst_init_time", "valid_time"
    ).hstack(levels)


def write_arm(*, arm: str, month: str, seed: int, frame: pl.DataFrame) -> None:
    """Write one arm's month forecasts atomically in experiment 510's layout."""
    target = W / "seed_preds" / arm / f"{month}_s{seed}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.rename(target)


ARMS: Final[dict[str, tuple[str, str]]] = {
    "S2-F-A-nll": ("A", "nll"), "S2-F-A-crps": ("A", "crps"),
    "S2-F-B-nll": ("B", "nll"), "S2-F-B-crps": ("B", "crps"),
}  # fmt: skip


def fit(*, months: list[str], seeds: list[int], arms: list[str]) -> None:
    """Fit stage 2 at each weekly cutoff and write each arm's month forecasts (plus S2-0F)."""
    torch.set_num_threads(2)
    for month in months:
        mr_cutoffs = c.cutoffs(month=month)
        for seed in seeds:
            todo = [
                a
                for a in ["S2-0F", *arms]
                if not (W / "seed_preds" / a / f"{month}_s{seed}.parquet").exists()
            ]
            if not todo:
                continue
            d = dict(np.load(CACHE / f"{month}_s{seed}.npz"))
            weeks: dict[str, list[pl.DataFrame]] = {a: [] for a in todo}
            for week, cut in enumerate(mr_cutoffs):
                lo, hi = fit510.window_of(month=month, week=week)
                sel = (d["origin_test"] >= lo) & (d["origin_test"] < hi)
                if "S2-0F" in todo:
                    weeks["S2-0F"].append(quantile_frame(out=d["out_test"], d=d, select=sel))
                cu = c.us(t=cut)
                keep = (
                    (d["valid_train"] <= cu - c.CUTOFF_AGE_US)
                    & (d["origin_train"] <= cu)
                    & (d["valid_train"] > cu - c.WINDOW_DAYS * c.DAY_US)
                )
                assert not (d["valid_train"][keep] > cu - c.CUTOFF_AGE_US).any(), (
                    "valid time past the cutoff"
                )
                assert not (d["origin_train"][keep] > cu).any(), "origin past the cutoff"
                age = (cu - d["valid_train"][keep]) / c.DAY_US
                weight = (0.5 ** (age / HALF_LIFE)).astype(np.float32)
                origin_day = (
                    (d["origin_train"][keep] - c.us(t=c.prep.FIRST_RUN)) // c.DAY_US
                ).astype(np.int64)
                for arm in [a for a in todo if a != "S2-0F"]:
                    started = time.monotonic()
                    form, loss = ARMS[arm]
                    model = train_stage2(
                        form=form, loss=loss, out=d["out_train"][keep], aux=d["aux_train"][keep],
                        z=d["z_train"][keep], weight=weight, origin_day=origin_day, seed=seed,
                    )  # fmt: skip
                    with torch.no_grad():
                        out = model(
                            torch.from_numpy(d["out_test"][sel]),
                            torch.from_numpy(d["aux_test"][sel]),
                        ).numpy()
                    full = np.zeros_like(d["out_test"])
                    full[sel] = out
                    weeks[arm].append(quantile_frame(out=full, d=d, select=sel))
                    print(
                        f"  {month} s{seed} w{week + 1} {arm}: {int(keep.sum()):,} rows, {time.monotonic() - started:.0f} s",
                        flush=True,
                    )
            for arm, frames in weeks.items():
                write_arm(arm=arm, month=month, seed=seed, frame=pl.concat(frames))


def main() -> None:
    """Run a command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "fit"))
    parser.add_argument("--months", nargs="+", default=list(c.MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    if args.command == "cache":
        cache(months=args.months, seeds=args.seeds)
    else:
        fit(months=args.months, seeds=args.seeds, arms=args.arms)


if __name__ == "__main__":
    main()
