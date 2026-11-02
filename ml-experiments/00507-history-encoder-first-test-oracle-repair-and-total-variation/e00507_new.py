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

    def __init__(self, *, lags: "arms.LagSource", target: "arms.LagSource") -> None:
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
                    )  # fmt: skip
                )
            print(f"{month} s{seed} {arm}: loss {losses[-1]:.4f}, "
                  f"{time.monotonic() - started:.0f} s", flush=True)  # fmt: skip
    arms.SCORE_DIR.mkdir(parents=True, exist_ok=True)
    target = arms.SCORE_DIR / f"new_{month}.parquet"
    pl.concat(out).write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").replace(target)
    print(f"wrote {target}")


def main() -> None:
    """Run the arms for one month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("run",))
    parser.add_argument("--month", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--arms", nargs="*", default=["plain", "oracle", "oracle_const", "zero"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    tv_params = (args.weight, args.tau) if args.weight is not None else None
    run(month=args.month, device=args.device, tv_params=tv_params, arm_names=tuple(args.arms))


if __name__ == "__main__":
    main()
