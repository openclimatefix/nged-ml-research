# ruff: noqa: E501
"""Fit the arms of experiment 428 on a fold and save the networks.

Arms (see the plan): `R` (three origin lags, clean power), `R-inj` (three origin lags, clean and
injected power), `L0` (no recent power), `E1`, `E2`, `E3` (encoder over one, two, three day windows,
clean and injected power). Training is experiment 397's loop (weighted mixture negative
log-likelihood, AdamW, early stopping on held-out weeks) with setting g09. For a mixed arm each
training row draws, every batch, world 0 (clean) or one of worlds 1 to 3 (injected) with equal
probability for the power in its windows or lags; the target is always the clean series. The
held-out rows use a fixed world per row (row number modulo 4).

```bash
source <paths>; PYTHONPATH=$PP:ml-experiments/00428-* OMP_NUM_THREADS=1 \
  systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
  uv run --no-sync python ml-experiments/00428-*/e00428_fit.py --arms R E3 --months 2026-01 --seeds 0
```
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00397_fit as fit397  # ty: ignore[unresolved-import]
import e00397_models as base  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00428_models as models  # ty: ignore[unresolved-import]
import e00428_windows as win  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

BATCH: Final[int] = 8_192
MAX_EPOCHS: Final[int] = 60
PATIENCE: Final[int] = 8
ROW_FRACTION: Final[float] = 0.2
"""Share of the training rows drawn at random for each epoch (every arm alike)."""
VALIDATION_ROWS: Final[int] = 200_000
"""A fixed random subset of the held-out rows scores each epoch."""
GRAD_CLIP: Final[float] = 1.0
SETTING: Final[fit397.Setting] = fit397.setting_by_name(name="g09")
MODELS_DIR: Final[Path] = data.WORK_DIR / "models"
LAGLESS_DIR: Final[Path] = (
    data.WORK_DIR.parents[1] / "00404-mdn-lagless-and-calibration-module-trust-weights/work/lagless"
)
"""Experiment 404's cross-fitted lagless tables (one parquet per month and seed)."""
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SCREEN_MONTHS: Final[tuple[str, ...]] = ("2026-06", "2026-01", "2025-10", "2026-03")
FIFTH_MONTH: Final[str] = "2025-12"


@dataclass(frozen=True)
class Arm:
    """How an arm sees recent power and what power it trains on."""

    name: str
    kind: str
    days: int
    mixed: bool
    surprise: bool = False
    widen: bool = False
    event_target: bool = False
    """Train on the injected world's power where the target half-hour lies in an injected event."""
    """Append the surprise channel (experiment 404's cross-fitted lagless forecast)."""


ARMS: Final[dict[str, Arm]] = {
    "R": Arm("R", "lags", 0, False),
    "R-inj": Arm("R-inj", "lags", 0, True),
    "L0": Arm("L0", "none", 0, False),
    "E1": Arm("E1", "encoder", 1, True),
    "E2": Arm("E2", "encoder", 2, True),
    "E3": Arm("E3", "encoder", 3, True),
    "E3-surp": Arm("E3-surp", "encoder", 3, True, surprise=True),
    "E3-surp-w": Arm("E3-surp-w", "encoder", 3, True, surprise=True, widen=True),
}


def make_model(*, arm: Arm, n_inputs: int, n_primaries: int) -> torch.nn.Module:
    """Return an untrained network for an arm (`n_inputs` counts the non-lag columns)."""
    if arm.kind == "encoder":
        return models.DayEncoderMdn(
            n_inputs=n_inputs,
            n_primaries=n_primaries,
            days=arm.days,
            hidden=SETTING.hidden,
            layers=SETTING.layers,
            components=SETTING.components,
            dropout=SETTING.dropout,
            channels=models.CHANNELS + int(arm.surprise),
            widen_path=arm.widen,
        )
    extra = 9 if arm.kind == "lags" else 0
    return models.LaggedMdn(
        n_inputs=n_inputs + extra,
        n_primaries=n_primaries,
        hidden=SETTING.hidden,
        layers=SETTING.layers,
        components=SETTING.components,
        dropout=SETTING.dropout,
    )


class Inputs:
    """The GPU-resident inputs of a set of rows, gathering recent power per batch."""

    def __init__(
        self, *, power: torch.Tensor, x: np.ndarray, primary: np.ndarray, origin_idx: np.ndarray,
        target_idx: np.ndarray, power_mean: float, power_std: float, device: str,
        surprise: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:  # fmt: skip
        """Move the row arrays to `device`; `power` is (worlds, primaries, grid)."""
        self.power = power
        self.x = torch.from_numpy(x).to(device)
        self.primary = torch.from_numpy(primary).to(device)
        self.origin = torch.from_numpy(origin_idx).to(device)
        self.target = torch.from_numpy(target_idx).to(device)
        self.power_mean, self.power_std = power_mean, power_std
        self.surprise = surprise
        self.target_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    def raw(self, *, arm: Arm, rows: torch.Tensor, world: torch.Tensor) -> dict:
        """Return the raw (unstandardised) recent power of `rows` read from `world`.

        The dict holds `x`, `primary`, `origin`, `target`, and, for a lag arm, `lags` (rows, 3)
        or, for an encoder arm, `index` and `raw` (rows, days, 48). A caller may modify `lags` or
        `raw` (the encroachment sweep does) and pass the dict to `assemble`.
        """
        x, primary = self.x[rows], self.primary[rows]
        origin, target = self.origin[rows], self.target[rows]
        out = {"x": x, "primary": primary, "origin": origin, "target": target}
        if arm.kind == "lags":
            out["lags"] = win.origin_lags(
                power=self.power, world=world, primary=primary, origin=origin, target=target
            )
        elif arm.kind == "encoder":
            index = win.window_indices(origin=origin, target=target, days=arm.days)
            out["index"] = index
            out["raw"] = win.take(power=self.power, world=world, primary=primary, index=index)
        return out

    def assemble(self, *, arm: Arm, raw: dict, masked: bool = False) -> tuple:
        """Return `(x_in, primary, windows)` from `raw`; `masked` hides every recent-power input."""
        x, primary = raw["x"], raw["primary"]
        if arm.kind == "none":
            return x, primary, None
        if arm.kind == "lags":
            lags = raw["lags"]
            if masked:
                lags = torch.full_like(lags, float("nan"))
            columns = win.lag_columns(
                lags=lags,
                ages=win.lag_ages(origin=raw["origin"], target=raw["target"]),
                power_mean=self.power_mean,
                power_std=self.power_std,
            )
            return torch.cat([x, columns], dim=1), primary, None
        values = torch.full_like(raw["raw"], float("nan")) if masked else raw["raw"]
        windows = win.build_windows(
            raw=values, index=raw["index"], origin=raw["origin"], newest_limit=True
        )
        windows = win.standardise_power(
            windows=windows, power_mean=self.power_mean, power_std=self.power_std
        )
        if self.surprise is not None:
            windows = self.add_surprise(windows=windows, values=values, raw=raw)
        return x, primary, windows

    def add_surprise(
        self, *, windows: torch.Tensor, values: torch.Tensor, raw: dict
    ) -> torch.Tensor:
        """Append the surprise channel: (power - lagless median) / half band width, clipped.

        The channel is 0 where the power or the lagless forecast is unobserved.
        """
        assert self.surprise is not None
        median, half = self.surprise
        index = raw["index"]
        primary = raw["primary"]
        zero = torch.zeros_like(primary)
        centre = win.take(power=median[None], world=zero, primary=primary, index=index)
        width = win.take(power=half[None], world=zero, primary=primary, index=index)
        z = ((values - centre) / width).clamp(-win.CLIP_SD, win.CLIP_SD)
        usable = (windows[:, :, 1, :] > 0.5) & torch.isfinite(z)
        z = torch.where(usable, z, torch.zeros_like(z))
        return torch.cat([windows, z[:, :, None, :]], dim=2)

    def event_targets(
        self, *, rows: torch.Tensor, world: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Return `z` with the injected world's value wherever the target lies in an injected event."""
        assert self.target_stats is not None
        mean, std = self.target_stats
        primary, target = self.primary[rows], self.target[rows]
        zero = torch.zeros_like(world)
        injected = win.take(power=self.power, world=world, primary=primary, index=target)
        clean = win.take(power=self.power, world=zero, primary=primary, index=target)
        hit = (world > 0) & torch.isfinite(injected) & torch.isfinite(clean) & (injected != clean)
        return torch.where(hit, (injected - mean[primary]) / std[primary], z)

    def batch(self, *, arm: Arm, rows: torch.Tensor, world: torch.Tensor) -> tuple:
        """Return `(x_in, primary, windows)` for `rows` read from `world` (one id per row)."""
        return self.assemble(arm=arm, raw=self.raw(arm=arm, rows=rows, world=world))


_SURPRISE_CACHE: dict = {}


def surprise_for(
    *, arm: Arm, month: str, seed: int, device: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return experiment 404's lagless median and half band width (primaries, grid), or None.

    The table is the cross-fitted forecast for the refit of `month` and the same seed (frozen at
    experiment 404 commit 63d3799f). NaN marks a half-hour without a forecast.
    """
    if not arm.surprise:
        return None
    key = (month, seed, device)
    if key not in _SURPRISE_CACHE:
        _SURPRISE_CACHE.clear()
        table = pl.read_parquet(LAGLESS_DIR / f"month={month}" / f"seed={seed}.parquet")
        grid = data.grid_length()
        median = np.full((len(data.sites()), grid), np.nan, dtype=np.float32)
        half = np.full_like(median, np.nan)
        p, g = table["primary"].to_numpy(), table["grid"].to_numpy()
        median[p, g] = table["q50"].to_numpy()
        half[p, g] = np.maximum((table["q95"] - table["q05"]).to_numpy() / 2.0, 1e-3)
        _SURPRISE_CACHE[key] = (
            torch.from_numpy(median).to(device),
            torch.from_numpy(half).to(device),
        )
    return _SURPRISE_CACHE[key]


def forward(*, model: torch.nn.Module, x, primary, windows):  # noqa: ANN001, ANN201
    """Return the raw mixture outputs of any network."""
    if isinstance(model, models.DayEncoderMdn):
        return model(x, primary, windows)
    return model(x, primary)


def train(
    *, arm: Arm, model: torch.nn.Module, inputs: Inputs, z: torch.Tensor, w: torch.Tensor,
    is_val: torch.Tensor, seed: int, device: str,
) -> tuple[int, float]:  # fmt: skip
    """Train with weighted negative log-likelihood and early stopping; return epochs and best loss."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if len(val_idx) > VALIDATION_ROWS:
        pick = torch.randperm(len(val_idx), generator=generator)[:VALIDATION_ROWS]
        val_idx = val_idx[pick.to(device)]
    val_world = (val_idx % (data.N_TRAIN_WORLDS + 1)) if arm.mixed else torch.zeros_like(val_idx)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=SETTING.lr, weight_decay=SETTING.weight_decay
    )
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    k = SETTING.components
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device)]
        order = order[: int(ROW_FRACTION * len(order))]
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            world = (
                torch.randint(0, data.N_TRAIN_WORLDS + 1, (len(b),), device=device)
                if arm.mixed
                else torch.zeros(len(b), dtype=torch.long, device=device)
            )
            x, primary, windows = inputs.batch(arm=arm, rows=b, world=world)
            nll = base.mixture_nll(
                out=forward(model=model, x=x, primary=primary, windows=windows),
                z=inputs.event_targets(rows=b, world=world, z=z[b]) if arm.event_target else z[b],
                components=k,
            )
            loss = (w[b] * nll).sum() / w[b].sum()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            losses = []
            for chunk, chunk_world in zip(
                val_idx.split(BATCH * 2), val_world.split(BATCH * 2), strict=True
            ):
                x, primary, windows = inputs.batch(arm=arm, rows=chunk, world=chunk_world)
                out = forward(model=model, x=x, primary=primary, windows=windows)
                losses.append(base.mixture_nll(out=out, z=z[chunk], components=k))
            val = float(torch.cat(losses).mean())
        epochs = epoch + 1
        print(f"  epoch {epochs}: val nll {val:.4f}", flush=True)
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {a: v.detach().clone() for a, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def power_tensor(*, device: str) -> torch.Tensor:
    """Return the training worlds' power (worlds 0 to 3, primaries, grid) on `device`."""
    power, _ = data.load_stack(include_eval=False)
    return torch.from_numpy(power).to(device)


def fit_one(
    *, arm_name: str, month: str, seed: int, power: torch.Tensor, stack: np.ndarray, device: str,
    rows_cache: dict,
) -> str:  # fmt: skip
    """Fit and save one arm, fold, and seed; return a progress line."""
    target = MODELS_DIR / arm_name / month / f"s{seed}"
    if (target / "meta.json").exists():
        return f"{arm_name} {month} s{seed}: exists"
    arm = ARMS[arm_name]
    started = time.monotonic()
    if month not in rows_cache:
        rows_cache.clear()
        rows_cache[month] = data.build_fold(refit=prep.refit_of(month=month), clean=stack[0])
    rows: data.Rows = rows_cache[month]
    inputs = Inputs(
        power=power, x=rows.x, primary=rows.primary, origin_idx=rows.origin_idx,
        target_idx=rows.target_idx, power_mean=rows.power_mean, power_std=rows.power_std,
        device=device, surprise=surprise_for(arm=arm, month=month, seed=seed, device=device),
    )  # fmt: skip
    inputs.target_stats = (
        torch.from_numpy(rows.target_mean.astype(np.float32)).to(device),
        torch.from_numpy(rows.target_std.astype(np.float32)).to(device),
    )
    z = torch.from_numpy(
        ((rows.y - rows.target_mean[rows.primary]) / rows.target_std[rows.primary]).astype(
            np.float32
        )
    ).to(device)
    w = torch.from_numpy(rows.weight).to(device)
    is_val = torch.from_numpy(rows.is_val).to(device)
    torch.manual_seed(seed)
    model = make_model(arm=arm, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)).to(
        device
    )
    epochs, best = train(
        arm=arm, model=model, inputs=inputs, z=z, w=w, is_val=is_val, seed=seed, device=device
    )
    target.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target / "model.pt")
    np.savez(
        target / "standardiser.npz",
        mean=rows.standardiser.mean, std=rows.standardiser.std, mask=rows.standardiser.mask,
        target_mean=rows.target_mean, target_std=rows.target_std,
    )  # fmt: skip
    (target / "meta.json").write_text(
        json.dumps(
            {
                "arm": arm_name, "month": month, "seed": seed, "epochs": epochs, "val_nll": best,
                "power_mean": rows.power_mean, "power_std": rows.power_std,
                "n_rows": len(rows.x), "unusable_targets": rows.dropped,
                "n_inputs": int(rows.x.shape[1]),
                "parameters": models.count_parameters(model=model),
            }
        )
    )  # fmt: skip
    return (
        f"{arm_name} {month} s{seed}: {epochs} epochs, val nll {best:.4f}, "
        f"{time.monotonic() - started:.0f} s"
    )


def main() -> None:
    """Fit the requested arms, months, and seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--months", nargs="+", default=list(SCREEN_MONTHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    power = power_tensor(device=args.device)
    stack, _ = data.load_stack(include_eval=False)
    cache: dict = {}
    for month in args.months:
        for seed in args.seeds:
            for name in args.arms:
                print(
                    fit_one(
                        arm_name=name, month=month, seed=seed, power=power, stack=stack,
                        device=args.device, rows_cache=cache,
                    ),
                    flush=True,
                )  # fmt: skip


if __name__ == "__main__":
    main()
