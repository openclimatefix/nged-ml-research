"""Training rows, the training loop, and test-month forecasts of experiment 404.

A row is one (Primary, forecast origin, valid time) of experiment 397's summary-scheme rows. The
base inputs and the target are the same for every world. What a world changes is the three origin
lags (read from the world's power with `origin_lags`) and the trust inputs (recomputed from the
world's surprise series). The lagless forecast of a row comes from the frozen cross-fitted table,
converted from the table's standardised units to the network's.

Every arm trains for a fixed number of epochs with experiment 397's recency weights. The saved file
holds the network, the input scalers, and the arm's configuration, so the later evaluation
experiment can reload it.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators:\
ml-experiments/00404-mdn-lagless-and-calibration-module-trust-weights \
DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data OMP_NUM_THREADS=1 \
    uv run --no-sync python ml-experiments/00404-*/e00404_fit.py fit --arm b1 --month 2026-06 \
    --seed 0 --device cuda
```
"""

import argparse
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00404-mdn-lagless-and-calibration-module-trust-weights" / "work"
)
MODEL_DIR: Final[Path] = WORK_DIR / "models"
PRED_DIR: Final[Path] = WORK_DIR / "predictions"
MVA_ONLY: Final[frozenset[str]] = frozenset({"D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"})
"""The MVA-only eight; the other eight Primaries report signed MW (D15 and D18 are report-only)."""
ARMS: Final[tuple[str, ...]] = (
    "A0",
    "A0-inj",
    "B2",
    "B0",
    "B1",
    "B1m",
    "B1g",
    "B1o",
    "B3",
    "B4a",
    "B4b",
    "B4c",
    "B5",
    "S1",
    "B6",
    "B7",
)
EPOCHS: Final[int] = 6
BATCH: Final[int] = 4096
LEARNING_RATE: Final[float] = 2e-3
LAG_DROPOUT: Final[float] = 0.2
TRUST_SAMPLE: Final[int] = 100_000
GRID_START_US: Final[int] = int(common.GRID_START.timestamp() * 1_000_000)
STEP_US: Final[int] = 1_800_000_000
CLIP_SD: Final[float] = 5.0


def trust_arm(*, arm: str) -> str | None:
    """Return the `TrustMdn` arm name for a named arm, or None for the plain networks."""
    return {
        "B0": "b0",
        "B1": "b1",
        "B1m": "b1m",
        "B1g": "b1g",
        "B1o": "b1o",
        "B6": "b6",
        "B7": "b7",
        "B4a": "b1",
        "B4b": "b1",
        "B4c": "b1",
        "B3": "b3",
        "B5": "b1",
    }.get(arm)


def world_power(*, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return a world's power over capacity (16, grid) and its event codes.

    `seed=None` is the clean world: the observed power (with the short-history cut) and no codes.
    """
    if seed is None:
        sites = prep.primaries()["site"].to_list()
        arrays = lagless.power_arrays()
        power = np.stack([arrays[s] for s in sites])
        return power, np.zeros(power.shape, dtype=np.int8)
    folder = next(ML_EXPERIMENTS_DATA_DIR.glob("00428-*")) / "work" / "worlds"
    path = folder / f"seed{seed}.npz"
    z = np.load(path)
    return z["power"].astype(np.float32), z["code"]


def metered_indicator() -> np.ndarray:
    """Return 1 for the Primaries that report signed MW, 0 for the MVA-only eight."""
    return np.array(
        [0.0 if s in MVA_ONLY else 1.0 for s in prep.primaries()["site"].to_list()],
        dtype=np.float32,
    )


@dataclass
class TableArrays:
    """The lagless table as dense arrays indexed by (Primary, grid - g0)."""

    g0: int
    raw: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    present: np.ndarray


def table_arrays(*, table: pl.DataFrame) -> TableArrays:
    """Lay the table out densely."""
    g0, g1 = int(table["grid"].min()), int(table["grid"].max())  # ty: ignore[invalid-argument-type]
    n = g1 - g0 + 1
    primary = table["primary"].to_numpy().astype(np.int64)
    pos = table["grid"].to_numpy().astype(np.int64) - g0
    raw = np.zeros((16, n, 9), dtype=np.float32)
    raw[primary, pos] = table.select([f"raw_{i}" for i in range(9)]).to_numpy()
    y_mean, y_std = np.zeros((16, n), np.float32), np.ones((16, n), np.float32)
    y_mean[primary, pos], y_std[primary, pos] = (
        table["y_mean"].to_numpy(),
        table["y_std"].to_numpy(),
    )
    present = np.zeros((16, n), dtype=bool)
    present[primary, pos] = True
    return TableArrays(g0=g0, raw=raw, y_mean=y_mean, y_std=y_std, present=present)


def lagless_in_network_units(
    *, arrays: TableArrays, primary: np.ndarray, grid: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the table's mixture (rows, 9) in the network's z units and a presence mask.

    Args:
        arrays: The dense table.
        primary: Primary index per row.
        grid: Grid index of the row's valid time.
        mean: The network's per-Primary target mean (fraction of capacity).
        std: The network's per-Primary target standard deviation.
    """
    pos = grid - arrays.g0
    inside = (pos >= 0) & (pos < arrays.raw.shape[1])
    pos = np.clip(pos, 0, arrays.raw.shape[1] - 1)
    present = inside & arrays.present[primary, pos]
    raw = arrays.raw[primary, pos].copy()
    t_mean, t_std = arrays.y_mean[primary, pos], arrays.y_std[primary, pos]
    net_mean, net_std = mean[primary], std[primary]
    comp_mean = (raw[:, 3:6] * t_std[:, None] + t_mean[:, None] - net_mean[:, None]) / net_std[
        :, None
    ]
    comp_std = np.exp(np.clip(raw[:, 6:9], -3.0, 3.0)) * (t_std / net_std)[:, None]
    raw[:, 3:6] = comp_mean
    raw[:, 6:9] = np.log(np.maximum(comp_std, 1e-4))
    return raw, present


@dataclass
class Pool:
    """Rows on the device, ready to batch.

    Attributes:
        primary: Primary index.
        origin: Grid index of the forecast origin.
        target: Grid index of the valid time.
        lead_hours: Lead in hours.
        base: Standardised base inputs (weather, solar, calendar, holidays, lead).
        lead: The two lead features (lead, square root of lead) for the trust heads.
        y: Observed power over capacity (the clean target).
        ll: Lagless mixture in network z units.
        weight: Recency weight.
    """

    primary: torch.Tensor
    origin: torch.Tensor
    target: torch.Tensor
    lead_hours: torch.Tensor
    base: torch.Tensor
    lead: torch.Tensor
    y: torch.Tensor
    ll: torch.Tensor
    weight: torch.Tensor

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.primary)


@dataclass
class Scalers:
    """What a saved network needs to rebuild its inputs."""

    standardiser: prep.Standardiser
    target_mean: np.ndarray
    target_std: np.ndarray
    trust_mean: torch.Tensor | None = None
    trust_std: torch.Tensor | None = None


def _gather_rows(*, month: str, test: bool, stride: int) -> dict[str, np.ndarray]:
    """Return the raw summary-scheme rows of the training window or of the scored month."""
    refit = prep.refit_of(month=month)
    names, _, layout = prep.feature_names(scheme="summary")
    cut = int(common.grid_index(times=lagless.SHORT_DROP_DATE))
    parts = []
    origins_us = (
        {int(o.timestamp() * 1_000_000) for o in prep.test_origins(month=month)} if test else None
    )
    for row in prep.primaries().iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        if test:
            keep = np.isin(z["origin"], list(origins_us))  # ty: ignore[invalid-argument-type]
        else:
            keep = (
                (z["valid"] <= prep._us(t=refit - timedelta(hours=3.5)))
                & (z["origin"] <= prep._us(t=refit))
                & (z["valid"] > prep._us(t=refit - timedelta(days=prep.WINDOW_DAYS)))
            )
            day = ((z["origin"] - prep._us(t=prep.FIRST_RUN)) // (86_400 * 1_000_000)).astype(
                np.int64
            )
            keep &= (day % stride) == 0
        target = (z["valid"] - GRID_START_US) // STEP_US
        if row["site"] in lagless.SHORT_HISTORY_SITES:
            keep &= target >= cut
        keep &= np.isfinite(z["y"])
        parts.append(
            {
                "x": z["x"][keep],
                "y": z["y"][keep],
                "primary": np.full(int(keep.sum()), row["primary"], dtype=np.int64),
                "origin": ((z["origin"][keep] - GRID_START_US) // STEP_US).astype(np.int64),
                "target": target[keep].astype(np.int64),
                "lead_hours": z["lead"][keep],
            }
        )
    del names, layout
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def build_pools(
    *, month: str, seed: int, device: str, stride: int, saved: dict | None = None
) -> tuple[Pool | None, Pool, Scalers, TableArrays, np.ndarray]:
    """Return the training pool, the scored-month pool, the scalers, and the lagless table.

    The last item is the standardiser's mean and standard deviation at the 3 lag columns. With
    `saved` (a saved network's file), the scalers come from it and no training pool is built, so
    a forecast uses exactly the scalers the network trained with.
    """
    names, mask, layout = prep.feature_names(scheme="summary")
    refit = prep.refit_of(month=month)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    arrays = table_arrays(table=table)
    base_columns = layout.columns_of(groups=(*lagless.data.LAGLESS_GROUPS, "lead"))
    lag_columns = layout.columns_of(groups=("lags",))[:3]
    lead_columns = [names.index("lead"), names.index("lead_sqrt")]
    pools = []
    if saved is None:
        train_raw = _gather_rows(month=month, test=False, stride=stride)
        standardiser = prep.Standardiser.fit(x=train_raw["x"], mask=mask)
        target_mean, target_std = np.zeros(16), np.ones(16)
        for p in range(16):
            sel = train_raw["primary"] == p
            if sel.any():
                target_mean[p] = train_raw["y"][sel].mean()
                target_std[p] = max(float(train_raw["y"][sel].std()), 1e-3)
        sources = ((train_raw, False), (_gather_rows(month=month, test=True, stride=1), True))
    else:
        standardiser = prep.Standardiser(
            mean=saved["standardiser_mean"],
            std=saved["standardiser_std"],
            mask=saved["standardiser_mask"],
        )
        target_mean, target_std = saved["target_mean"], saved["target_std"]
        sources = ((_gather_rows(month=month, test=True, stride=1), True),)
    scalers = Scalers(standardiser=standardiser, target_mean=target_mean, target_std=target_std)
    for raw, test in sources:
        ll, present = lagless_in_network_units(
            arrays=arrays,
            primary=raw["primary"],
            grid=raw["target"],
            mean=target_mean,
            std=target_std,
        )
        keep = present
        standardised = standardiser.apply(x=raw["x"][keep])
        age_days = (prep._us(t=refit) - (GRID_START_US + raw["target"][keep] * STEP_US)) / (
            86_400 * 1e6
        )
        weight = 0.5 ** (age_days / common.SHORT_HALF_LIFE_DAYS)
        pools.append(
            Pool(
                primary=torch.tensor(raw["primary"][keep], device=device),
                origin=torch.tensor(raw["origin"][keep], device=device),
                target=torch.tensor(raw["target"][keep], device=device),
                lead_hours=torch.tensor(raw["lead_hours"][keep], device=device),
                base=torch.tensor(standardised[:, base_columns], device=device),
                lead=torch.tensor(raw["x"][keep][:, lead_columns], device=device),
                y=torch.tensor(raw["y"][keep], device=device),
                ll=torch.tensor(ll[keep], device=device),
                weight=torch.tensor(
                    weight if not test else np.ones_like(weight), dtype=torch.float32, device=device
                ),
            )
        )
    lag_stats = np.stack([standardiser.mean[lag_columns], standardiser.std[lag_columns]])
    return (pools[0] if saved is None else None), pools[-1], scalers, arrays, lag_stats


WEEKS: Final[dict[str, int]] = {"B4a": 1, "B4b": 3, "B4c": 5}
"""Arms that read the same weekday and time from the last N weeks, and each arm's N."""


SURPRISE_ARM: Final[str] = "S1"
"""The plain network on injected data plus the surprise at each of the three lags."""


def surprise_array(*, power: np.ndarray, table: pl.DataFrame) -> torch.Tensor:
    """Return the world's power minus the cross-fitted lagless median (NaN where unknown)."""
    median = np.full(power.shape, np.nan, dtype=np.float32)
    median[
        table["primary"].to_numpy().astype(np.int64), table["grid"].to_numpy().astype(np.int64)
    ] = table["q50"].to_numpy()
    return torch.from_numpy(power - median)


def n_extra_features(*, arm: str) -> int:
    """Return the columns of extra weekly-lag features an arm reads (0 for arms without any)."""
    if arm == SURPRISE_ARM:
        return 3
    weeks = WEEKS.get(arm, 0)
    return 2 * (weeks - 1) + 2 if weeks else 0


def lag_features(
    *,
    pool: Pool,
    power: torch.Tensor,
    lag_stats: np.ndarray,
    device: str,
    weeks: int = 0,
    surprise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lag tensor (rows, 9 + extra) and the lag grid indices (rows, 3).

    With `weeks` > 0, the extra columns hold the value and missing flag of the same weekday and
    time in weeks 2 to `weeks` before the newest usable week, then the median of the standardised
    values over the weeks that exist and that median minus the latest week.
    """
    lag_grid = torch.from_numpy(
        rows_mod.lag_grids(target=pool.target.cpu().numpy(), origin=pool.origin.cpu().numpy())
    ).to(device)
    primary = pool.primary
    size = power.shape[-1]
    inside = (lag_grid >= 0) & (lag_grid < size)
    values = power[primary[:, None], lag_grid.clamp(0, size - 1)]
    values = torch.where(inside, values, torch.full_like(values, float("nan")))
    missing = ~torch.isfinite(values)
    mean = torch.tensor(lag_stats[0], dtype=torch.float32, device=device)
    std = torch.tensor(lag_stats[1], dtype=torch.float32, device=device)
    z = ((values - mean) / std).clamp(-CLIP_SD, CLIP_SD)
    z = torch.nan_to_num(z, nan=0.0)
    gap = torch.round(pool.lead_hours * 2.0)[:, None] + common.NEWEST_AGE
    age = (
        torch.stack(
            [
                gap[:, 0] / 2.0,
                24.0 * torch.ceil(gap[:, 0] / common.STEPS_PER_DAY),
                168.0 * torch.ceil(gap[:, 0] / (7 * common.STEPS_PER_DAY)),
            ],
            dim=1,
        )
        / prep.LEAD_SCALE_HOURS
    )
    lags = torch.stack([z, missing.float(), age], dim=2).reshape(len(primary), 9)
    if surprise is not None:
        s = surprise.to(device)[primary[:, None], lag_grid.clamp(0, size - 1)]
        s = torch.where(inside, s, torch.full_like(s, float("nan")))
        s = torch.nan_to_num((s / std).clamp(-CLIP_SD, CLIP_SD), nan=0.0)
        lags = torch.cat([lags, s], dim=1)
    if weeks:
        week_values = [values[:, 2]]
        extra = []
        for k in range(1, weeks):
            grid_k = lag_grid[:, 2] - k * 7 * common.STEPS_PER_DAY
            ok = (grid_k >= 0) & (grid_k < size)
            v = torch.where(
                ok,
                power[primary, grid_k.clamp(0, size - 1)],
                torch.full_like(values[:, 2], float("nan")),
            )
            week_values.append(v)
            zk = torch.nan_to_num(((v - mean[2]) / std[2]).clamp(-CLIP_SD, CLIP_SD), nan=0.0)
            extra += [zk, (~torch.isfinite(v)).float()]
        zs = ((torch.stack(week_values, dim=1) - mean[2]) / std[2]).clamp(-CLIP_SD, CLIP_SD)
        median = torch.nan_to_num(torch.nanmedian(zs, dim=1).values, nan=0.0)
        extra += [median, median - z[:, 2]]
        lags = torch.cat([lags, torch.stack(extra, dim=1)], dim=1)
    return lags, lag_grid


def build_network(*, arm: str, n_base: int, n_trust: int) -> torch.nn.Module:
    """Return the network of an arm."""
    kind = trust_arm(arm=arm)
    if kind is None:
        return models.PlainLagMdn(
            n_base=n_base,
            n_primaries=16,
            lag_dropout=LAG_DROPOUT if arm == "B2" else 0.0,
            n_extra=n_extra_features(arm=arm),
        )
    return models.TrustMdn(
        arm=kind,
        n_base=n_base,
        n_trust=n_trust,
        n_primaries=16,
        n_extra=n_extra_features(arm=arm),
    )


def _oracle_flags(
    *, event_mask: torch.Tensor | None, primary: torch.Tensor, lag_grid: torch.Tensor
) -> torch.Tensor | None:
    """Return the true event flag (rows, 3) at each lag's valid time, or None without a mask."""
    if event_mask is None:
        return None
    safe = lag_grid.clamp(0, event_mask.shape[1] - 1)
    return event_mask[primary[:, None], safe].float()


def event_mask_of(*, code: np.ndarray, real: np.ndarray, device: str) -> torch.Tensor:
    """Return the (primaries, grid) mask of injected events and padded real events."""
    return torch.from_numpy((code != 0) | real).to(device)


def _history_input(
    *, pool: Pool, idx: torch.Tensor, power: torch.Tensor, scalers: Scalers
) -> torch.Tensor:
    """Return the latest 7 days of actual power before each row's origin, (rows, 2, 336).

    Channel 0 is the power standardised by the Primary's target mean and standard deviation (0 where
    missing); channel 1 is 1 where observed. The window ends at the newest usable reading.
    """
    primary = pool.primary[idx]
    end = pool.origin[idx] - common.NEWEST_AGE
    offsets = torch.arange(-(models.HISTORY_STEPS - 1), 1, device=primary.device)
    pos = end[:, None] + offsets[None, :]
    size = power.shape[-1]
    ok = (pos >= 0) & (pos < size)
    values = power[primary[:, None], pos.clamp(0, size - 1)]
    observed = ok & torch.isfinite(values)
    mean = torch.tensor(scalers.target_mean, dtype=torch.float32, device=primary.device)
    std = torch.tensor(scalers.target_std, dtype=torch.float32, device=primary.device)
    z = ((values - mean[primary][:, None]) / std[primary][:, None]).clamp(-CLIP_SD, CLIP_SD)
    return torch.stack([torch.where(observed, z, torch.zeros_like(z)), observed.float()], dim=1)


def forward(
    *,
    net: torch.nn.Module,
    pool: Pool,
    idx: torch.Tensor,
    lags: torch.Tensor,
    stack: rows_mod.Stack | None,
    lag_grid: torch.Tensor,
    scalers: Scalers,
    metered: torch.Tensor,
    details: bool = False,
    masked: tuple[int, ...] = (),
    event_mask: torch.Tensor | None = None,
    power: torch.Tensor | None = None,
) -> tuple:
    """Return the final mixture (log weights, means, stds) for the rows `idx`, with details."""
    primary = pool.primary[idx]
    lag_rows = lags[idx]
    if masked:
        # Masked lags are hidden: value 0, flagged missing (arm B5 hides the newest lag; the
        # lag-masked control of the encroachment sweep hides all three).
        lag_rows = lag_rows.clone()
        for lag in masked:
            lag_rows[:, 3 * lag], lag_rows[:, 3 * lag + 1] = 0.0, 1.0
    if isinstance(net, models.PlainLagMdn):
        return net(pool.base[idx], lag_rows, primary)
    trust = torch.stack(
        [
            rows_mod.trust_inputs(
                stack=stack,  # ty: ignore[invalid-argument-type]
                primary=primary,
                lag_grid=lag_grid[idx],
                origin=pool.origin[idx],
                step=s,
            )
            for s in range(rows_mod.SMOOTHING_STEPS)
        ],
        dim=1,
    )
    trust = ((trust - scalers.trust_mean) / scalers.trust_std).clamp(-CLIP_SD, CLIP_SD)  # ty: ignore[unsupported-operator]
    for lag in masked:
        trust[:, :, lag * rows_mod.N_PER_LAG : (lag + 1) * rows_mod.N_PER_LAG] = 0.0
    window = None
    if net.arm == "b6":
        window = torch.stack(
            [
                rows_mod.window_input(
                    stack=stack,  # ty: ignore[invalid-argument-type]
                    primary=primary,
                    lag_grid=lag_grid[idx],
                    origin=pool.origin[idx],
                    step=s,
                )
                for s in range(rows_mod.SMOOTHING_STEPS)
            ],
            dim=1,
        )
    history = None
    if net.arm == "b7":
        assert power is not None
        history = _history_input(pool=pool, idx=idx, power=power, scalers=scalers)
    band = torch.from_numpy(models.lead_band_of(lead_hours=pool.lead_hours[idx].cpu().numpy())).to(
        primary.device
    )
    return net(
        base=pool.base[idx],
        lags=lag_rows,
        ll=pool.ll[idx],
        trust=trust,
        lead=pool.lead[idx],
        primary=primary,
        metered=metered[primary][:, None],
        lead_band=band,
        details=details,
        window=window,
        history=history,
        oracle=_oracle_flags(event_mask=event_mask, primary=primary, lag_grid=lag_grid[idx]),
    )


def fit_arm(
    *, arm: str, month: str, seed: int, device: str, stride: int, epochs: int = EPOCHS
) -> Path:
    """Fit one arm on one scored month and seed; save the network and scalers; return the path."""
    started = time.monotonic()
    train, _test, scalers, arrays, lag_stats = build_pools(
        month=month, seed=seed, device=device, stride=stride
    )
    inject_world = arm != "A0"
    power_np, code = world_power(seed=100 + seed if inject_world else None)
    import e00404_score  # ty: ignore[unresolved-import]  (score imports fit)

    event_mask = event_mask_of(code=code, real=e00404_score.real_event_mask(), device=device)
    power = torch.tensor(power_np, device=device)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    stack = rows_mod.build_stack(
        power=power_np,
        table=table,
        neighbours=rows_mod.neighbour_matrix(
            sites=prep.primaries()["site"].to_list(), path=_neighbour_path()
        ),
        device=device,
    )
    lags, lag_grid = lag_features(
        pool=train,  # ty: ignore[invalid-argument-type]
        power=power,
        lag_stats=lag_stats,
        device=device,
        weeks=WEEKS.get(arm, 0),
        surprise=surprise_array(power=power_np, table=table) if arm == SURPRISE_ARM else None,
    )
    metered = torch.tensor(metered_indicator(), device=device)
    torch.manual_seed(seed)
    net = build_network(arm=arm, n_base=train.base.shape[1], n_trust=rows_mod.N_TRUST).to(device)  # ty: ignore[unresolved-attribute]
    if trust_arm(arm=arm) is not None:
        sample = torch.randperm(len(train), device=device)[:TRUST_SAMPLE]  # ty: ignore[invalid-argument-type]
        stats = torch.stack(
            [
                rows_mod.trust_inputs(
                    stack=stack,
                    primary=train.primary[sample],  # ty: ignore[unresolved-attribute]
                    lag_grid=lag_grid[sample],
                    origin=train.origin[sample],  # ty: ignore[unresolved-attribute]
                    step=0,
                )
            ]
        )[0]
        scalers.trust_mean = stats.mean(0)
        scalers.trust_std = stats.std(0).clamp(min=1e-3)
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(train)  # ty: ignore[invalid-argument-type]
    losses = []
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            out = forward(
                net=net, pool=train, idx=idx, lags=lags, stack=stack, lag_grid=lag_grid,  # ty: ignore[invalid-argument-type]
                scalers=scalers, metered=metered, masked=(0,) if arm == "B5" else (),
                event_mask=event_mask,
                power=power,
            )  # fmt: skip
            log_w, mean, std = out[0] if isinstance(out[0], tuple) else out
            z = (train.y[idx] - tm[train.primary[idx]]) / ts[train.primary[idx]]  # ty: ignore[unresolved-attribute]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = train.weight[idx]  # ty: ignore[unresolved-attribute]
            loss = (nll * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(idx)
        losses.append(total / n)
        print(f"{arm} {month} s{seed} epoch {epoch}: loss {losses[-1]:.4f}", flush=True)
    path = MODEL_DIR / arm / f"{month}_s{seed}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": net.state_dict(),
            "arm": arm,
            "month": month,
            "seed": seed,
            "n_base": train.base.shape[1],  # ty: ignore[unresolved-attribute]
            "target_mean": scalers.target_mean,
            "target_std": scalers.target_std,
            "standardiser_mean": scalers.standardiser.mean,
            "standardiser_std": scalers.standardiser.std,
            "standardiser_mask": scalers.standardiser.mask,
            "lag_stats": lag_stats,
            "trust_mean": scalers.trust_mean,
            "trust_std": scalers.trust_std,
            "losses": losses,
            "rows": n,
            "seconds": time.monotonic() - started,
            "g0": arrays.g0,
        },
        path,
    )
    print(f"{arm} {month} s{seed}: {n:,} rows, {time.monotonic() - started:.0f} s -> {path}")
    return path


def _neighbour_path() -> str:
    """Return the path of experiment 383's adjacency file."""
    folder = next(ML_EXPERIMENTS_DATA_DIR.glob("00383-*"))
    return str(folder / "work" / "neighbours_nged.json")


def main() -> None:
    """Fit an arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit",))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--origin-stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    fit_arm(
        arm=args.arm, month=args.month, seed=args.seed, device=args.device,
        stride=args.origin_stride, epochs=args.epochs,
    )  # fmt: skip


if __name__ == "__main__":
    main()
