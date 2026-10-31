"""Forecast a scored month with saved networks of experiment 404, flag the rows, and score them.

For each arm, month, seed, and scoring world (experiment 428's seeds 200 to 202), `predict` writes a
parquet with the 11 quantiles q01 to q99 of the final forecast (fractions of capacity), the row
class flags, and (for the trust arms) the 3 trust weights. B-mix (a fixed 50/50 mixture of the
A0-inj forecast and the frozen lagless table's forecast) is written as the arm `B-mix`.

The row classes follow experiment 428's `Flags`: a window is inside an event when any observed
half-hour of the first 3 day windows is, `win_inj` for injected events and `win_real` for
padded real events. A harm row has its target outside every event and a window inside one. A
second set of flags, `lag_inj` and `lag_real`, uses only the 3 origin lags.

```bash
PYTHONPATH=<as e00404_fit.py shows> \
    uv run --no-sync python ml-experiments/00404-*/e00404_score.py predict --arm B1 \
    --month 2026-06 --seed 0 --device cuda
```
"""

import argparse
from pathlib import Path
from typing import Final

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_lagless as lagless  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SCORING_SEEDS: Final[tuple[int, ...]] = (200, 201, 202)
POST_EVENT_STEPS: Final[int] = 96
EVENT_PADDING: Final[int] = 48
WINDOW: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 336
LEVEL_NAMES: Final[tuple[str, ...]] = tuple(f"q{round(level * 100):02d}" for level in models.LEVELS)
"""Quantile column names `q01` to `q99`."""
SIX_SIGNED: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")


def post_event_mask(*, padded: np.ndarray) -> np.ndarray:
    """Return the padded real-event mask extended by `POST_EVENT_STEPS` after each event's end."""
    mask = padded.copy()
    edges = np.diff(np.concatenate([[0], padded.astype(np.int8), [0]]))
    for end in np.nonzero(edges == -1)[0]:
        mask[end : end + POST_EVENT_STEPS] = True
    return mask[: padded.size]


def window_indices(*, origin: torch.Tensor, target: torch.Tensor, days: int = 3) -> torch.Tensor:
    """Return the grid index of every position of the first `days` windows: (rows, days, 48).

    Day window 1 is the 48 half-hours ending at the newest usable reading. Day windows 2 and 3 are
    centred on the weekly lag and on the weekly lag a week earlier.
    """
    newest = origin - common.NEWEST_AGE
    gap = target - newest
    week = target - STEPS_PER_WEEK * torch.div(
        gap + STEPS_PER_WEEK - 1, STEPS_PER_WEEK, rounding_mode="floor"
    )
    offsets = torch.arange(WINDOW, device=origin.device)
    starts = [newest - (WINDOW - 1), week - WINDOW // 2, week - WINDOW // 2 - STEPS_PER_WEEK]
    return torch.stack([s[:, None] + offsets[None, :] for s in starts[:days]], dim=1)


class Flags:
    """Row-class flags from the injected event codes and the padded real-event mask."""

    def __init__(
        self, *, code: np.ndarray, real: np.ndarray, power: np.ndarray, device: str
    ) -> None:
        """Hold the masks on `device`: code (primaries, grid), real (primaries, grid)."""
        self.code = torch.from_numpy(code).to(device)
        self.power = torch.from_numpy(power).to(device)
        self.real = torch.from_numpy(real).to(device)
        self.post = torch.from_numpy(np.stack([post_event_mask(padded=r) for r in real])).to(device)

    def of(self, *, primary: torch.Tensor, origin: torch.Tensor, target: torch.Tensor) -> dict:
        """Return boolean flags per row."""
        index = window_indices(origin=origin, target=target)
        size = self.real.shape[-1]
        newest = origin - common.NEWEST_AGE
        inside = (index >= 0) & (index < size) & (index <= newest[:, None, None])
        safe = index.clamp(0, size - 1)
        p = primary[:, None, None]
        observed = torch.isfinite(self.power[p, safe])
        inside = inside & observed
        lag_grid = torch.from_numpy(
            rows_mod.lag_grids(target=target.cpu().numpy(), origin=origin.cpu().numpy())
        ).to(origin.device)
        lag_safe = lag_grid.clamp(0, size - 1)
        pl_ = primary[:, None]
        return {
            "win_inj": ((self.code[p, safe] != 0) & inside).flatten(1).any(1),
            "win_real": (self.real[p, safe] & inside).flatten(1).any(1),
            "lag_inj": (self.code[pl_, lag_safe] != 0).any(1),
            "lag_real": self.real[pl_, lag_safe].any(1),
            "tgt_inj": self.code[primary, target] != 0,
            "tgt_real": self.real[primary, target],
            "tgt_post": self.post[primary, target],
        }


def real_event_mask() -> np.ndarray:
    """Return the padded real-event mask (primaries, grid) from the corrected events.

    This matches experiment 402's `load_context().event_mask`: each corrected event covers its onset
    to its end, padded by `EVENT_PADDING` half-hours on both sides. Primaries with no event get an
    all-false row.
    """
    events = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00400-lagged-lagless-forecast-blend-around-load-shifts-stored"
        / "work"
        / "events_corrected.parquet"
    )
    size = common.grid_times().len()
    out = []
    for site in prep.primaries()["site"].to_list():
        mask = np.zeros(size, dtype=bool)
        for e in events.filter(pl.col("label") == site).iter_rows(named=True):
            mask[max(0, e["onset_idx"] - EVENT_PADDING) : e["end_idx"] + 1 + EVENT_PADDING] = True
        out.append(mask)
    return np.stack(out)


def load_network(*, arm: str, month: str, seed: int, device: str) -> tuple[torch.nn.Module, dict]:
    """Load a saved network and its scalers, in evaluation mode (dropout off)."""
    saved = torch.load(fit.MODEL_DIR / arm / f"{month}_s{seed}.pt", weights_only=False)
    net = fit.build_network(arm=arm, n_base=saved["n_base"], n_trust=rows_mod.N_TRUST).to(device)
    net.load_state_dict(saved["state"])
    net.eval()
    return net, saved


@torch.no_grad()
def mixture_quantiles_capacity(
    *,
    log_w: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> torch.Tensor:
    """Return sorted quantiles in capacity fractions for mixtures given in z units."""
    q = models.mixture_quantiles(log_w=log_w, mean=mean, std=std)
    return torch.sort(q * sd[:, None] + mu[:, None], dim=1).values


@torch.no_grad()
def forecast(
    *, arm: str, month: str, seed: int, world: int, device: str, real: np.ndarray
) -> pl.DataFrame:
    """Forecast the scored month's rows on a scoring world with a saved network (or B-mix)."""
    net, saved = load_network(
        arm="A0-inj" if arm == "B-mix" else arm, month=month, seed=seed, device=device
    )
    _train, test, scalers, _arrays, lag_stats = fit.build_pools(
        month=month, seed=seed, device=device, stride=1, saved=saved
    )
    scalers.trust_mean = saved["trust_mean"]
    scalers.trust_std = saved["trust_std"]
    power_np, code = fit.world_power(seed=None if world < 0 else world)
    power = torch.tensor(power_np, device=device)
    table = pl.read_parquet(lagless.table_path(month=month, seed=seed))
    stack = rows_mod.build_stack(
        power=power_np,
        table=table,
        neighbours=rows_mod.neighbour_matrix(
            sites=prep.primaries()["site"].to_list(), path=fit._neighbour_path()
        ),
        device=device,
    )
    lags, lag_grid = fit.lag_features(pool=test, power=power, lag_stats=lag_stats, device=device)
    metered = torch.tensor(fit.metered_indicator(), device=device)
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    quantiles, weights = [], []
    for start in range(0, len(test), 16384):
        idx = torch.arange(start, min(start + 16384, len(test)), device=device)
        out = fit.forward(
            net=net, pool=test, idx=idx, lags=lags, stack=stack, lag_grid=lag_grid,
            scalers=scalers, metered=metered, details=True, masked=(0,) if arm == "B5" else (),
        )  # fmt: skip
        mixture, detail = out if isinstance(out[0], tuple) else (out, None)
        log_w, mean, std = mixture
        if arm == "B-mix":
            ll_w, ll_mean, ll_std = models.mixture_parts(
                out=test.ll[idx], components=models.COMPONENTS
            )
            half = float(np.log(2.0))
            log_w = torch.cat([log_w - half, ll_w - half], dim=1)
            mean, std = torch.cat([mean, ll_mean], dim=1), torch.cat([std, ll_std], dim=1)
        primary = test.primary[idx]
        quantiles.append(
            mixture_quantiles_capacity(
                log_w=log_w, mean=mean, std=std, mu=mu[primary], sd=sd[primary]
            ).cpu()
        )
        if detail is not None and arm in ("B0", "B1", "B1m"):
            weights.append(detail.cpu())
    q = torch.cat(quantiles).numpy()
    flags = Flags(code=code, real=real, power=power_np, device=device).of(
        primary=test.primary, origin=test.origin, target=test.target
    )
    columns: dict[str, np.ndarray] = {
        "primary": test.primary.cpu().numpy().astype(np.int16),
        "origin": test.origin.cpu().numpy().astype(np.int32),
        "target": test.target.cpu().numpy().astype(np.int32),
        "lead_hours": test.lead_hours.cpu().numpy(),
        "actual": test.y.cpu().numpy(),
        **{name: q[:, i] for i, name in enumerate(LEVEL_NAMES)},
        **{k: v.cpu().numpy() for k, v in flags.items()},
    }
    if weights:
        w = torch.cat(weights).numpy()
        columns.update({f"trust_{i}": w[:, i] for i in range(3)})
    return pl.DataFrame(columns)


def prediction_path(*, arm: str, month: str, seed: int, world: int) -> Path:
    """Return where a forecast frame is stored."""
    return fit.PRED_DIR / arm / f"{month}_s{seed}_w{world}.parquet"


def pinball_points(*, frame: pl.DataFrame) -> pl.Expr:
    """Return the CRPS-like score in points of capacity (twice the mass-weighted pinball loss).

    The levels are weighted by `common.ATOM_MASSES`, as `common.pinball_quantile_score` does.
    """
    terms = []
    for level, name, mass in zip(models.LEVELS, LEVEL_NAMES, common.ATOM_MASSES, strict=True):
        diff = pl.col("actual") - pl.col(name)
        terms.append(float(mass) * pl.max_horizontal(level * diff, (level - 1.0) * diff))
    return 200.0 * pl.sum_horizontal(terms)


def main() -> None:
    """Forecast a scored month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict",))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worlds", type=int, nargs="*", default=list(SCORING_SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    real = real_event_mask()
    for world in args.worlds:
        frame = forecast(
            arm=args.arm, month=args.month, seed=args.seed, world=world, device=args.device,
            real=real,
        )  # fmt: skip
        target = prediction_path(arm=args.arm, month=args.month, seed=args.seed, world=world)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        print(f"{args.arm} {args.month} s{args.seed} w{world}: {frame.height:,} rows -> {target}")


if __name__ == "__main__":
    main()
