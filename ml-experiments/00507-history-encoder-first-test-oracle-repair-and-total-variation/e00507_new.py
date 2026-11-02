"""Experiment 507: the correction arms on the new-generator worlds (families A and B).

Training worlds G100 to G102 (one per fit seed) and held-out worlds G200 to G202 come from
`e00507_generator.sample_world`. Every arm trains on its own lag source with the target "stored
reading plus the family B step at the target time", and is scored on worlds G200 to G202 with the
same target. Row classes: `temp` (a family A event covers one of the row's three lag positions),
`gen` (family B does), and `none`. Arms: `plain` (shifted lags), `oracle` (exact: family A removed,
family B kept), `oracle_const` (a constant per family A event removed), `zero` (lags zeroed), and
`tv` (total-variation repair with the settings given).

```bash
ml-experiments/00507-*/run.sh ml-experiments/00507-*/e00507_new.py run --month 2026-01 \
    --weight 4 --tau 1.5
```
"""

import argparse
import time
from typing import Final

import e00507_arms as arms  # ty: ignore[unresolved-import]
import e00507_common as c  # ty: ignore[unresolved-import]
import e00507_generator as gen  # ty: ignore[unresolved-import]
import e00507_tv as tv  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

TRAIN_SEEDS = (100, 101, 102)
SCORE_SEEDS = (200, 201, 202)


class WithTarget:
    """Lags from one source, target from another."""

    def __init__(self, *, lags: arms.LagSource, target: arms.LagSource) -> None:
        """Hold both sources."""
        self.lags, self.target_source = lags, target

    def values(self, *, rows: arms.Rows) -> np.ndarray:
        """Return the lag source's values."""
        return self.lags.values(rows=rows)

    def target(self, *, rows: arms.Rows, clean: np.ndarray) -> np.ndarray:
        """Return the target source's target."""
        return self.target_source.target(rows=rows, clean=clean)


def row_classes(*, spec: gen.Spec, ev: gen.Events, rows: arms.Rows) -> np.ndarray:
    """Return `temp`, `gen`, or `none` per row, from the events covering its three lag positions."""
    p = np.repeat(rows.primary, 3)
    d = np.repeat(rows.day, 3)
    g = rows.lag_grid.reshape(-1)
    hit_a = np.zeros(len(g), dtype=bool)
    hit_b = np.zeros(len(g), dtype=bool)
    for k in range(gen.SLOTS):
        covered = (g >= ev.start[p, d, k]) & (g < ev.end[p, d, k])
        hit_a |= covered & (ev.family[p, d, k] == gen.FAMILY_A)
        hit_b |= covered & (ev.family[p, d, k] == gen.FAMILY_B)
    a = hit_a.reshape(-1, 3).any(axis=1)
    b = hit_b.reshape(-1, 3).any(axis=1)
    return np.where(a, "temp", np.where(b, "gen", "none"))


def sources(
    *, spec: gen.Spec, ev: gen.Events, med: np.ndarray, tv_params: tuple[float, float] | None,
    device: str,
) -> dict[str, arms.LagSource]:  # fmt: skip
    """Return the lag sources of one world, keyed by arm."""
    exact = arms.GenLags(spec=spec, ev=ev, mode="exact")
    out: dict[str, arms.LagSource] = {
        "plain": arms.GenLags(spec=spec, ev=ev, mode="shifted"),
        "oracle": exact,
        "oracle_const": WithTarget(lags=arms.GenLags(spec=spec, ev=ev, mode="const"), target=exact),
        "zero": WithTarget(lags=arms.ZeroLags(), target=exact),
    }
    if tv_params is not None:
        out["tv"] = tv.TvLags(
            reader=gen_reader(spec=spec, ev=ev), med=med, spread=spec.spread,
            weight=tv_params[0], tau=tv_params[1], target_source=exact, device=device,
        )  # fmt: skip
    return out


def gen_reader(*, spec: gen.Spec, ev: gen.Events) -> tv.Reader:
    """Return a reader of shifted histories (windows, 672) for the total-variation repair."""

    def read(p: np.ndarray, d: np.ndarray) -> np.ndarray:
        newest = gen.newest_of(day=d)
        g = newest[:, None] + np.arange(-(gen.HISTORY - 1), 1)[None, :]
        out = gen.reading(
            spec=spec, ev=ev, p=np.repeat(p, gen.HISTORY), d=np.repeat(d, gen.HISTORY),
            g=g.reshape(-1), mode="shifted",
        )  # fmt: skip
        return out.reshape(len(p), gen.HISTORY)

    return read


def run(
    *, month: str, device: str, tv_params: tuple[float, float] | None, arm_names: tuple[str, ...],
    seeds: tuple[int, ...] = (0, 1, 2),
) -> None:  # fmt: skip
    """Fit and score the arms for one month; write `new_<month>.parquet`."""
    started = time.monotonic()
    ctx = arms.MonthContext.build(month=month, device=device)
    spec = c.load_spec()
    groups = arms.group_array(primary=ctx.test_rows.primary)
    worlds_train = {s: gen.sample_world(spec=spec, seed=TRAIN_SEEDS[s]) for s in seeds}
    worlds_test = {w: gen.sample_world(spec=spec, seed=w) for w in SCORE_SEEDS}
    train_src = {
        s: sources(spec=spec, ev=worlds_train[s], med=ctx.med, tv_params=tv_params, device=device)
        for s in seeds
    }
    test_src = {
        w: sources(spec=spec, ev=worlds_test[w], med=ctx.med, tv_params=tv_params, device=device)
        for w in SCORE_SEEDS
    }
    classes = {
        w: row_classes(spec=spec, ev=worlds_test[w], rows=ctx.test_rows) for w in SCORE_SEEDS
    }
    out: list[pl.DataFrame] = []
    for seed in seeds:
        for arm in arm_names:
            source = train_src[seed][arm]
            net, losses = arms.train_net(
                pool=ctx.train, lags=ctx.lags(source=source, train=True),
                y=ctx.targets(source=source, train=True), scalers=ctx.scalers, seed=seed,
                device=device,
            )  # fmt: skip
            for w in SCORE_SEEDS:
                read = test_src[w][arm]
                lags = ctx.lags(source=read, train=False)
                q = arms.predict_quantiles(net=net, pool=ctx.test, lags=lags, scalers=ctx.scalers,
                                           device=device)  # fmt: skip
                frame = {
                    "actual": test_src[w]["oracle"].target(
                        rows=ctx.test_rows, clean=ctx.clean_test
                    ),
                    "lead_hours": ctx.test.lead_hours.cpu().numpy(),
                }
                out.append(
                    arms.aggregate(
                        frame=frame,
                        q=q,
                        classes=classes[w],
                        groups=groups,
                        meta={
                            "label": arm,
                            "month": month,
                            "seed": seed,
                            "world": w,
                            "final_loss": losses[-1],
                        },
                    )
                )
            print(f"{month} s{seed} {arm}: loss {losses[-1]:.4f}, "
                  f"{time.monotonic() - started:.0f} s", flush=True)  # fmt: skip
    arms.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    target = arms.SCORE_DIR / f"new_{month}.parquet"
    pl.concat(out).write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").replace(target)
    print(f"wrote {target}")


WEIGHTS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TAUS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0, 3.0, 5.0)
TUNING_SEED: Final[int] = 110
TUNING_WINDOWS: Final[int] = 6000


def tune(*, device: str) -> None:
    """Pick the total-variation weight and tau on world 110 with origins in 2024.

    The criterion is the mean absolute error, in spreads, of the three repaired lag readings
    against the exact readings (family A and family B steps removed), over windows with a finite
    lab median. Scoring months never enter.
    """
    spec = c.load_spec()
    ctx = arms.MonthContext.build(month="2025-10", device=device)
    ev = gen.sample_world(spec=spec, seed=TUNING_SEED)
    rng = np.random.default_rng(0)
    day_all = np.arange(spec.days)
    origin_us = c.GRID_START_US + gen.origin_of(day=day_all) * 1_800_000_000
    year = origin_us.astype("datetime64[us]").astype("datetime64[Y]").astype(int) + 1970
    days_2024 = day_all[year == 2024]
    assert len(days_2024) > 300, len(days_2024)
    p = rng.integers(0, spec.real.shape[0], size=TUNING_WINDOWS)
    d = rng.choice(days_2024, size=TUNING_WINDOWS)
    newest = gen.newest_of(day=d)
    g = newest[:, None] + np.arange(-(gen.HISTORY - 1), 1)[None, :]
    median = ctx.med[p[:, None], np.clip(g, 0, ctx.med.shape[1] - 1)]
    keep = np.isfinite(median).mean(axis=1) > 0.9
    p, d, median = p[keep], d[keep], median[keep]
    exact = gen_reader_mode(spec=spec, ev=ev, mode="true")(p, d)
    shifted = gen_reader(spec=spec, ev=ev)(p, d)
    positions = np.array([gen.HISTORY - 1, gen.HISTORY - 1 - 48, gen.HISTORY - 1 - 336])
    spread = spec.spread[p]
    best = None
    print(f"tuning on {len(p)} windows of world {TUNING_SEED}, origins in 2024")
    plain_err = np.nanmean(np.abs(shifted[:, positions] - exact[:, positions]) / spread[:, None])
    print(f"unrepaired error {plain_err:.4f} spreads")
    for weight in WEIGHTS:
        for tau in TAUS:
            fixed = tv.repair(reading=shifted, median=median, spread=spread, weight=weight,
                              tau=tau, device=device)  # fmt: skip
            err = float(np.nanmean(np.abs(fixed[:, positions] - exact[:, positions])
                                   / spread[:, None]))  # fmt: skip
            print(f"weight {weight:5.1f} tau {tau:3.1f}: error {err:.4f} spreads", flush=True)
            if best is None or err < best[0]:
                best = (err, weight, tau)
    assert best is not None
    print(f"best: weight {best[1]} tau {best[2]} (error {best[0]:.4f} spreads)")


def gen_reader_mode(*, spec: gen.Spec, ev: gen.Events, mode: str) -> tv.Reader:
    """Return a reader of histories (windows, 672) in the given generator mode."""

    def read(p: np.ndarray, d: np.ndarray) -> np.ndarray:
        newest = gen.newest_of(day=d)
        g = newest[:, None] + np.arange(-(gen.HISTORY - 1), 1)[None, :]
        out = gen.reading(spec=spec, ev=ev, p=np.repeat(p, gen.HISTORY),
                          d=np.repeat(d, gen.HISTORY), g=g.reshape(-1), mode=mode)  # fmt: skip
        return out.reshape(len(p), gen.HISTORY)

    return read


def main() -> None:
    """Run the arms for one month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("run", "tune"))
    parser.add_argument("--month", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--arms", nargs="*", default=["plain", "oracle", "oracle_const", "zero"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.step == "tune":
        tune(device=args.device)
        return
    assert args.month is not None, "--month is required for run"
    tv_params = (args.weight, args.tau) if args.weight is not None else None
    run(month=args.month, device=args.device, tv_params=tv_params, arm_names=tuple(args.arms))


if __name__ == "__main__":
    main()
