"""Experiment 507: the lag correction on the ERA5 lab anchor, with exact, repaired, or shifted lags.

The correction is experiment 472's 32-unit `AnchorCorrection` on a single-seed 3-component anchor.
What differs between arms is where the three origin lags come from (`LagSource`) and, on the
new-generator worlds, the target:

- `GridLags`: a world stored as a grid (experiment 428's seeds 100 to 102 and 200 to 202, or the
  stored reading as world -1).
- `GenLags`: a window-level world of `e00507_generator` in one of the modes `shifted` (families A
  and B), `exact` (family A removed, family B kept), or `const` (a constant per A event removed).

A fit trains one network on one lag source and forecasts the scored month on any other, so the
oracle re-prediction (trained shifted, scored exact) needs no second fit.

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_arms.py old --month 2026-01
```
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00404_fit as fit  # ty: ignore[unresolved-import]
import e00404_models as models  # ty: ignore[unresolved-import]
import e00404_rows as rows_mod  # ty: ignore[unresolved-import]
import e00404_score as score  # ty: ignore[unresolved-import]
import e00472_arm as arm472  # ty: ignore[unresolved-import]
import e00507_common as c  # ty: ignore[unresolved-import]
import e00507_generator as gen  # ty: ignore[unresolved-import]
import e00507_lagless as lab  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

PRED_DIR: Final[Path] = c.WORK_DIR / "predictions"
SCORE_DIR: Final[Path] = c.WORK_DIR / "scores"
PENALTY: Final[float] = 1e-2
CLIP_SD: Final[float] = fit.CLIP_SD


@dataclass
class Rows:
    """The pool's rows as numpy arrays, with each row's three lag positions."""

    primary: np.ndarray
    origin: np.ndarray
    target: np.ndarray
    lag_grid: np.ndarray

    @classmethod
    def of(cls, *, pool: fit.Pool) -> "Rows":
        """Read the rows from a pool on the device."""
        primary = pool.primary.cpu().numpy().astype(np.int64)
        origin = pool.origin.cpu().numpy().astype(np.int64)
        target = pool.target.cpu().numpy().astype(np.int64)
        return cls(primary, origin, target, rows_mod.lag_grids(target=target, origin=origin))

    @property
    def day(self) -> np.ndarray:
        """Return the day index of each row's origin."""
        day = (self.origin - gen.ORIGIN_PHASE) // gen.STEPS_PER_DAY
        assert ((self.origin - gen.ORIGIN_PHASE) % gen.STEPS_PER_DAY == 0).all(), "odd origin"
        return day


class LagSource(Protocol):
    """Where the three lags (and, optionally, a different target) of each row come from."""

    def values(self, *, rows: Rows) -> np.ndarray:
        """Return the (rows, 3) lag readings, as a fraction of capacity, NaN where unobserved."""
        ...

    def target(self, *, rows: Rows, clean: np.ndarray) -> np.ndarray:
        """Return the target for each row; `clean` is the stored reading."""
        ...


class GridLags:
    """A world stored as `power[primary, grid]`; the target stays the stored reading."""

    def __init__(self, *, power: np.ndarray) -> None:
        """Hold the world's power."""
        self.power = power

    def values(self, *, rows: Rows) -> np.ndarray:
        """Return the world's power at each lag position."""
        size = self.power.shape[1]
        inside = (rows.lag_grid >= 0) & (rows.lag_grid < size)
        v = self.power[rows.primary[:, None], np.clip(rows.lag_grid, 0, size - 1)]
        return np.where(inside, v, np.nan).astype(np.float32)

    def target(self, *, rows: Rows, clean: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return the stored reading."""
        return clean


class GenLags:
    """A window-level world of the generator, read in one mode."""

    def __init__(self, *, spec: gen.Spec, ev: gen.Events, mode: str) -> None:
        """Hold the events; `mode` is `shifted`, `exact`, or `const`."""
        assert mode in ("shifted", "exact", "const")
        self.spec, self.ev, self.mode = spec, ev, mode
        self._const: np.ndarray | None = None

    def values(self, *, rows: Rows) -> np.ndarray:
        """Return the lag readings in this mode."""
        p = np.repeat(rows.primary, 3)
        d = np.repeat(rows.day, 3)
        g = rows.lag_grid.reshape(-1)
        if self.mode == "exact":
            v = gen.reading(spec=self.spec, ev=self.ev, p=p, d=d, g=g, mode="true")
        elif self.mode == "shifted":
            v = gen.reading(spec=self.spec, ev=self.ev, p=p, d=d, g=g, mode="shifted")
        else:
            v = self._constant_repair(p=p, d=d, g=g)
        return v.reshape(-1, 3)

    def _constant_repair(self, *, p: np.ndarray, d: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return the shifted reading with each family A event's history-mean step removed."""
        const = self.constants()
        adjust = np.zeros(len(g))
        for k in range(gen.SLOTS):
            is_a = self.ev.family[p, d, k] == gen.FAMILY_A
            inside = (g >= self.ev.start[p, d, k]) & (g < self.ev.end[p, d, k])
            adjust += np.where(is_a & inside, const[p, d, k], 0.0)
        a = np.clip(g, 0, self.spec.real.shape[1] - 1)
        base = self.spec.real[p, a]
        total = (
            base
            + gen.step_values(
                spec=self.spec, ev=self.ev, p=p, d=d, g=g, families=(gen.FAMILY_A, gen.FAMILY_B)
            )
            - adjust
        )
        return np.where(self.spec.mva_only[p], np.abs(total), total).astype(np.float32)

    def constants(self) -> np.ndarray:
        """Return, per event, its mean step over the part of its support inside the history."""
        if self._const is None:
            n_p, days, _ = self.ev.family.shape
            out = np.zeros(self.ev.family.shape)
            newest = gen.newest_of(day=np.arange(days))
            offsets = np.arange(-(gen.HISTORY - 1), 1)
            for k in range(gen.SLOTS):
                for p in range(n_p):
                    g = newest[:, None] + offsets[None, :]
                    dd = np.repeat(np.arange(days), gen.HISTORY)
                    pp = np.full(dd.shape, p)
                    one = np.zeros_like(self.ev.family)
                    one[p, :, k] = self.ev.family[p, :, k]
                    sub = gen.Events(
                        family=one, start=self.ev.start, end=self.ev.end, size=self.ev.size,
                        source=self.ev.source, ramp=self.ev.ramp, stair=self.ev.stair,
                    )  # fmt: skip
                    step = gen.step_values(
                        spec=self.spec,
                        ev=sub,
                        p=pp,
                        d=dd,
                        g=g.reshape(-1),
                        families=(gen.FAMILY_A,),
                    ).reshape(days, gen.HISTORY)
                    inside = (g >= self.ev.start[p, :, k, None]) & (g < self.ev.end[p, :, k, None])
                    count = inside.sum(axis=1)
                    out[p, :, k] = np.where(
                        count > 0, (step * inside).sum(axis=1) / np.maximum(count, 1), 0.0
                    )
            self._const = out
        return self._const

    def target(self, *, rows: Rows, clean: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return the stored reading plus the family B step at the target time."""
        return gen.reading(
            spec=self.spec, ev=self.ev, p=rows.primary, d=rows.day, g=rows.target, mode="true"
        )


class ZeroLags:
    """Experiment 472's lag-free control: every lag input is zero."""

    def values(self, *, rows: Rows) -> np.ndarray:  # noqa: ARG002
        """Never used: the lag tensor is zeroed by `lag_tensor`."""
        raise NotImplementedError

    def target(self, *, rows: Rows, clean: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """Return the stored reading."""
        return clean


def median_grid(*, month: str) -> np.ndarray:
    """Return the lab anchor's median (primaries, grid) in capacity fractions, NaN where absent."""
    table = pl.read_parquet(lab.table_path(month=month, seed=0))
    size = common.grid_times().len()
    out = np.full((len(c.sites()), size), np.nan, dtype=np.float32)
    out[table["primary"].to_numpy().astype(np.int64), table["grid"].to_numpy().astype(np.int64)] = (
        table["q50"].to_numpy()
    )
    return out


def lag_tensor(
    *, pool: fit.Pool, values: np.ndarray | None, med: np.ndarray, rows: Rows,
    lag_stats: np.ndarray, device: str,
) -> torch.Tensor:  # fmt: skip
    """Return the (rows, 12) lag input: value, missing flag, age per lag, then the three surprises.

    With `values=None` the lag-free control gets zeros. The formulas follow
    `e00404_fit.lag_features` with the surprise option on.
    """
    if values is None:
        return torch.zeros(len(pool), 12, device=device)
    v = torch.from_numpy(values).to(device)
    m = torch.from_numpy(
        med[rows.primary[:, None], np.clip(rows.lag_grid, 0, med.shape[1] - 1)]
    ).to(device)
    mean = torch.tensor(lag_stats[0], dtype=torch.float32, device=device)
    std = torch.tensor(lag_stats[1], dtype=torch.float32, device=device)
    z = torch.nan_to_num(((v - mean) / std).clamp(-CLIP_SD, CLIP_SD), nan=0.0)
    missing = ~torch.isfinite(v)
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
    lags = torch.stack([z, missing.float(), age], dim=2).reshape(len(pool), 9)
    s = torch.nan_to_num(((v - m) / std).clamp(-CLIP_SD, CLIP_SD), nan=0.0)
    return torch.cat([lags, s], dim=1)


def train_net(
    *, pool: fit.Pool, lags: torch.Tensor, y: torch.Tensor, scalers: fit.Scalers, seed: int,
    device: str,
) -> tuple[arm472.AnchorCorrection, list[float]]:  # fmt: skip
    """Fit one correction network (experiment 472's loop) and return it with its epoch losses."""
    torch.manual_seed(seed)
    net = arm472.AnchorCorrection(components=3).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=fit.LEARNING_RATE)
    tm = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    ts = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    n = len(pool)
    losses = []
    for _ in range(fit.EPOCHS):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, fit.BATCH):
            idx = order[start : start + fit.BATCH]
            (log_w, mean, std), out = net(lags[idx], pool.lead[idx], pool.ll[idx])
            z = (y[idx] - tm[pool.primary[idx]]) / ts[pool.primary[idx]]
            nll = models.mixture_nll_parts(log_w=log_w, mean=mean, std=std, z=z)
            w = pool.weight[idx]
            loss = (nll * w).sum() / w.sum() + PENALTY * (out**2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / n)
    return net, losses


@torch.no_grad()
def predict_quantiles(
    *, net: arm472.AnchorCorrection, pool: fit.Pool, lags: torch.Tensor, scalers: fit.Scalers,
    device: str,
) -> np.ndarray:  # fmt: skip
    """Return the 11 quantiles (rows, 11), as capacity fractions."""
    net.eval()
    mu = torch.tensor(scalers.target_mean, dtype=torch.float32, device=device)
    sd = torch.tensor(scalers.target_std, dtype=torch.float32, device=device)
    out_q = []
    for start in range(0, len(pool), 16384):
        idx = torch.arange(start, min(start + 16384, len(pool)), device=device)
        (log_w, mean, std), _ = net(lags[idx], pool.lead[idx], pool.ll[idx])
        p = pool.primary[idx]
        out_q.append(
            score.mixture_quantiles_capacity(
                log_w=log_w, mean=mean, std=std, mu=mu[p], sd=sd[p]
            ).cpu()
        )
    return torch.cat(out_q).numpy()


def pinball_single(*, actual: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Return the pinball loss at one level in points of capacity (capacity fraction times 100)."""
    diff = actual - q
    return 100.0 * np.maximum(level * diff, (level - 1.0) * diff)


def crps_points(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return experiment 404's score: twice the mass-weighted pinball over 11 levels, times 100."""
    total = np.zeros(len(actual))
    for i, (level, mass) in enumerate(zip(models.LEVELS, common.ATOM_MASSES, strict=True)):
        total += float(mass) * np.maximum(
            level * (actual - q[:, i]), (level - 1.0) * (actual - q[:, i])
        )
    return 200.0 * total


def aggregate(
    *, frame: dict[str, np.ndarray], q: np.ndarray, classes: np.ndarray, groups: np.ndarray,
    meta: dict[str, object],
) -> pl.DataFrame:  # fmt: skip
    """Return mean scores per (group, class, band): CRPS, q99 and q95 pinball and coverage."""
    actual = frame["actual"]
    lead = frame["lead_hours"]
    crps = crps_points(actual=actual, q=q)
    cols = {
        "crps": crps,
        "pin_q01": pinball_single(actual=actual, q=q[:, 0], level=0.01),
        "pin_q05": pinball_single(actual=actual, q=q[:, 1], level=0.05),
        "pin_q95": pinball_single(actual=actual, q=q[:, 9], level=0.95),
        "pin_q99": pinball_single(actual=actual, q=q[:, 10], level=0.99),
        "cov_q01": (actual <= q[:, 0]).astype(float),
        "cov_q05": (actual <= q[:, 1]).astype(float),
        "cov_q95": (actual <= q[:, 9]).astype(float),
        "cov_q99": (actual <= q[:, 10]).astype(float),
    }
    df = pl.DataFrame(
        {
            "group": groups,
            "cls": classes,
            "band": np.where((lead >= 12) & (lead <= 36), "12-36h", "all"),
            **cols,
        }
    )
    both = (
        pl.concat([df, df.with_columns(band=pl.lit("all"))]).filter(pl.col("band") == "all")
        if False
        else pl.concat([df.with_columns(band=pl.lit("all")), df.filter(pl.col("band") == "12-36h")])
    )
    out = both.group_by("group", "cls", "band").agg(
        pl.len().alias("n"), pl.all().exclude("group", "cls", "band").mean()
    )
    return out.with_columns(**{k: pl.lit(v) for k, v in meta.items()})


def group_array(*, primary: np.ndarray) -> np.ndarray:
    """Return 'six', 'eight', or 'other' for each row's Primary."""
    return c.group_of_primary()[primary]


def old_world_classes(
    *, pool: fit.Pool, code: np.ndarray, real: np.ndarray, power: np.ndarray, device: str,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return experiment 404's row flags for a grid world."""
    flags = score.Flags(code=code, real=real, power=power, device=device).of(
        primary=pool.primary, origin=pool.origin, target=pool.target
    )
    return {k: v.cpu().numpy() for k, v in flags.items()}


def cls_old(*, flags: dict[str, np.ndarray]) -> np.ndarray:
    """Return `tgt_inj`, `harm_inj`, or `clean_inj` per row (experiment 472's classes)."""
    return np.where(
        flags["tgt_inj"], "tgt_inj", np.where(flags["win_inj"], "harm_inj", "clean_inj")
    )


def cls_real(*, flags: dict[str, np.ndarray]) -> np.ndarray:
    """Return `event` for rows near a labelled event and `clean` (no labelled event) otherwise."""
    near = flags["win_real"] | flags["tgt_real"] | flags["tgt_post"]
    return np.where(near, "event", "clean")


@dataclass
class MonthContext:
    """Everything one scored month's fits share: pools, medians, scalers, and the real world."""

    month: str
    device: str
    train: fit.Pool
    test: fit.Pool
    scalers: fit.Scalers
    lag_stats: np.ndarray
    med: np.ndarray
    train_rows: Rows
    test_rows: Rows
    clean_train: np.ndarray
    clean_test: np.ndarray
    real_mask: np.ndarray

    @classmethod
    def build(cls, *, month: str, device: str) -> "MonthContext":
        """Load the pools once for the month."""
        c.use_lab_tables()
        train, test, scalers, _arrays, lag_stats = fit.build_pools(
            month=month, seed=0, device=device, stride=2
        )
        assert train is not None
        return cls(
            month=month, device=device, train=train, test=test, scalers=scalers,
            lag_stats=lag_stats, med=median_grid(month=month), train_rows=Rows.of(pool=train),
            test_rows=Rows.of(pool=test), clean_train=train.y.cpu().numpy(),
            clean_test=test.y.cpu().numpy(), real_mask=score.real_event_mask(),
        )  # fmt: skip

    def lags(self, *, source: LagSource | None, train: bool) -> torch.Tensor:
        """Return the lag tensor of the train or test rows from a source (None gives zeros)."""
        pool, rows = (self.train, self.train_rows) if train else (self.test, self.test_rows)
        values = (
            None if source is None or isinstance(source, ZeroLags) else source.values(rows=rows)
        )
        return lag_tensor(
            pool=pool,
            values=values,
            med=self.med,
            rows=rows,
            lag_stats=self.lag_stats,
            device=self.device,
        )

    def targets(self, *, source: LagSource, train: bool) -> torch.Tensor:
        """Return the target tensor for the train or test rows."""
        rows, clean = (
            (self.train_rows, self.clean_train) if train else (self.test_rows, self.clean_test)
        )
        return torch.tensor(source.target(rows=rows, clean=clean), device=self.device)


def old_world_sources() -> tuple[dict[int, GridLags], GridLags, dict[int, np.ndarray]]:
    """Return the training worlds, the stored reading as a world, and the scoring worlds' codes."""
    train = {s: GridLags(power=fit.world_power(seed=100 + s)[0]) for s in range(3)}
    real_world = GridLags(power=c.clean_world())
    return train, real_world, {w: fit.world_power(seed=w)[1] for w in c.SCORE_WORLDS}


def run_old(*, month: str, device: str, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Fit plain, oracle-exact, and zero-lag corrections on the old worlds and score them.

    Scoring worlds 200 to 202 give the injected classes; world -1 gives the clean rows (no
    labelled event) and the real-event rows. The oracle arm reads the stored reading on every
    world. The re-prediction arm is the plain network scored on the stored reading.
    """
    started = time.monotonic()
    ctx = MonthContext.build(month=month, device=device)
    print(f"{month}: {len(ctx.test):,} scored rows, {len(ctx.train):,} training rows", flush=True)
    train_worlds, real_world, codes = old_world_sources()
    score_worlds = {w: GridLags(power=fit.world_power(seed=w)[0]) for w in c.SCORE_WORLDS}
    out: list[pl.DataFrame] = []
    groups = group_array(primary=ctx.test_rows.primary)
    frame = {"actual": ctx.clean_test, "lead_hours": ctx.test.lead_hours.cpu().numpy()}
    # Row classes: injected classes per scoring world; real-event classes from world -1.
    flags_w = {
        w: old_world_classes(
            pool=ctx.test, code=codes[w], real=ctx.real_mask,
            power=score_worlds[w].power, device=device,
        )
        for w in c.SCORE_WORLDS
    }  # fmt: skip
    flags_real = old_world_classes(
        pool=ctx.test, code=np.zeros_like(codes[200]), real=ctx.real_mask,
        power=real_world.power, device=device,
    )  # fmt: skip
    for seed in seeds:
        fits = {
            "plain": (train_worlds[seed], "shifted"),
            "oracle": (real_world, "exact"),
            "zero": (ZeroLags(), "zero"),
        }
        for arm, (train_source, _) in fits.items():
            net, losses = train_net(
                pool=ctx.train,
                lags=ctx.lags(source=train_source, train=True),
                y=ctx.targets(source=train_source, train=True),
                scalers=ctx.scalers,
                seed=seed,
                device=device,
            )
            meta = {"arm": arm, "month": month, "seed": seed, "final_loss": losses[-1]}
            # Scoring reads: (label, lag source, flag world).
            reads: list[tuple[str, LagSource | None, int]] = []
            if arm == "plain":
                reads += [("plain", score_worlds[w], w) for w in c.SCORE_WORLDS]
                reads += [("oracle_reprediction", real_world, w) for w in c.SCORE_WORLDS]
                reads += [("plain", real_world, -1)]
            elif arm == "oracle":
                reads += [("oracle", real_world, w) for w in c.SCORE_WORLDS]
                reads += [("oracle", real_world, -1)]
            else:
                reads += [("zero", ZeroLags(), w) for w in (*c.SCORE_WORLDS, -1)]
            for label, source, world in reads:
                lags = ctx.lags(source=source, train=False)
                q = predict_quantiles(net=net, pool=ctx.test, lags=lags, scalers=ctx.scalers,
                                      device=device)  # fmt: skip
                classes = cls_old(flags=flags_w[world]) if world > 0 else cls_real(flags=flags_real)
                out.append(
                    aggregate(
                        frame=frame,
                        q=q,
                        classes=classes,
                        groups=groups,
                        meta={**meta, "label": label, "world": world},
                    )  # fmt: skip
                )
            print(f"{month} s{seed} {arm}: loss {losses[-1]:.4f}, "
                  f"{time.monotonic() - started:.0f} s", flush=True)  # fmt: skip
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    target = SCORE_DIR / f"old_{month}.parquet"
    pl.concat(out).write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").replace(target)
    print(f"wrote {target}")


def main() -> None:
    """Run one step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("old",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.step == "old":
        run_old(month=args.month, device=args.device)


if __name__ == "__main__":
    main()
