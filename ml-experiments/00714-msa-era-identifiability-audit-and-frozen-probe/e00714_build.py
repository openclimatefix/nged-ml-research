"""Experiment 714: build the second generator's worlds and the feature table of every world.

    E=ml-experiments
    export PYTHONPATH=$E/00660-shared-generator-of-injected-load-transfer-events:\
$E/00709-era-signal-in-the-candidate-part-1:\
$E/00714-msa-era-identifiability-audit-and-frozen-probe
    .venv/bin/python ml-experiments/00714-*/e00714_build.py generate --world g2 [--limit 2]
    .venv/bin/python ml-experiments/00714-*/e00714_build.py features --world bank|g2|gas [--limit 2]

The `features` command writes `features/<world>.parquet` under this experiment's private folder:
one row per event and origin, with the features, the rewired-neighbour features
(`rw_mirror_*`), the known-answer features computed on the noiseless change (`ka_*`), the
no-event placebo features computed on the real panel (`np_*`), and the event's truth columns.
`--limit n` builds only the first `n` passes of each fold month (a smoke run).
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Final

import e00709_features as f709  # ty: ignore[unresolved-import]
import e00714_common as common
import e00714_features as feats
import e00714_generator2 as g2
import numpy as np
import polars as pl

FEATURE_DIR: Final[Path] = common.DATA_DIR / "features"
KA_FEATURES: Final[tuple[str, ...]] = (*feats.HOST_FEATURES, *feats.NEIGHBOUR_FEATURES)


def rewired_neighbours(
    *, ctx: common.Context, host: int, touched: set[int], rng: np.random.Generator
) -> list[int]:
    """Return the neighbours of another Primary that is not electrically linked to the host.

    The other Primary is neither the host nor a neighbour of the host, and neither it nor its
    neighbours is touched by an event of the pass, so its neighbours carry no injected change.

    Args:
        ctx: The context.
        host: The host Primary.
        touched: Indices of every host, donor, and recipient of the pass.
        rng: The random generator.

    Returns:
        The neighbour indices of the chosen Primary, without the host.
    """
    panel = ctx.panel
    linked = {host, *panel.neighbours[host]}
    options = [
        u
        for u in range(len(panel.sites))
        if u not in linked
        and panel.neighbours[u]
        and u not in touched
        and not touched & set(panel.neighbours[u])
    ]
    if not options:
        options = [u for u in range(len(panel.sites)) if u not in linked and panel.neighbours[u]]
    chosen = int(rng.choice(options))
    return [j for j in panel.neighbours[chosen] if j != host]


def _pass_rows(
    *, ctx: common.Context, truth: pl.DataFrame, power: common.PassPower, touched: list[int],
    world: str, month: str, pass_index: int,
) -> tuple[list[dict], dict[str, int]]:  # fmt: skip
    """Return the feature rows of every event and origin of one pass, and the drop counts."""
    month_day = common.month_start_day(month=month)
    reference = common.reference_for(ctx=ctx, month_day=month_day)
    lo, hi = common.window_bounds(onset_day=month_day)
    real = common.PassPower(start_hh=lo, power=ctx.panel.power[:, lo:hi])
    rng = np.random.default_rng(714_300_000 + common.MONTHS.index(month) * 1000 + pass_index)
    rows: list[dict] = []
    dropped = {"missing": 0}
    for e, ev in enumerate(truth.iter_rows(named=True)):
        host = ev["host"]
        neighbours = list(ctx.panel.neighbours[host])
        rewired = rewired_neighbours(ctx=ctx, host=host, touched=set(touched), rng=rng)
        for k in common.ORIGIN_DAYS:
            base = feats.EventInput(
                host=host, onset_hh=ev["onset_hh"], onset_day=ev["onset_day"], neighbours=neighbours
            )

            def compute(
                event: feats.EventInput,
                src: common.PassPower,
                ref: common.PassPower,
                k: int = k,
                host: int = host,
            ) -> dict[str, float] | None:
                return feats.build_features(
                    event=event, k=k, power=src, reference=ref,
                    shortwave=ctx.weather.shortwave[host], capacity_mw=ctx.panel.capacity_mw,
                )  # fmt: skip

            main = compute(base, power, reference)
            if main is None:
                dropped["missing"] += 1
                continue
            rw = compute(dataclasses.replace(base, neighbours=rewired), power, reference)
            ka = compute(base, power, real)
            nop = compute(base, real, reference)
            row = {
                "world": world, "month": month, "pass_index": pass_index,
                "pass_id": f"{world}_{month}_p{pass_index}", "event": e, "k": k,
                "host": host, "truth": ev["truth"], "kind": ev["kind"],
                "label": int(ev["truth"] == "era"),
                "size_true": ev["size"], "onset_day": ev["onset_day"], "onset_hh": ev["onset_hh"],
                "duration_days": ev["duration_days"], "band": ev["band"],
                "mirror_state": ev["mirror_state"], **main,
            }  # fmt: skip
            row["rw_mirror_share"] = rw["mirror_share"] if rw else None
            row["rw_mirror_corr"] = rw["mirror_corr"] if rw else None
            for name in KA_FEATURES:
                row[f"ka_{name}"] = ka[name] if ka else None
                row[f"np_{name}"] = nop[name] if nop else None
            rows.append(row)
    return rows, dropped


def bank_truth(*, parts: dict, month: str) -> pl.DataFrame:
    """Return experiment 709's truth table with the columns this experiment reads."""
    truth = parts["truth"]
    return truth.with_columns(
        onset_hh=pl.col("onset_day") * common.DAY,
        mirror_state=pl.when(pl.col("truth") == "era")
        .then(pl.lit("none"))
        .when(pl.col("hidden_mirror"))
        .then(pl.lit("hidden"))
        .otherwise(pl.lit("full")),
    )


def generate(*, world: g2.WorldType, limit: int | None) -> None:
    """Generate and store the passes of the second generator."""
    ctx = common.load_context()
    for month in common.MONTHS:
        for index in range(g2.PASSES_PER_MONTH[world] if limit is None else limit):
            started = time.monotonic()
            built = g2.build_pass(ctx=ctx, world=world, month=month, index=index)
            g2.write_pass(world=world, month=month, index=index, generated=built)
            print(f"{world} {month} p{index}: {time.monotonic() - started:.1f} s", flush=True)


def build_features(*, world: str, limit: int | None) -> None:
    """Build and store the feature table of a world (`bank`, `g2`, or `gas`)."""
    ctx = common.load_context()
    all_rows: list[dict] = []
    drops: dict[str, int] = {}
    for month in common.MONTHS:
        n_passes = (
            common.BANK_PASSES_PER_MONTH if world == "bank" else g2.PASSES_PER_MONTH[world]  # ty: ignore[invalid-argument-type]
        )
        for index in range(n_passes if limit is None else min(limit, n_passes)):
            started = time.monotonic()
            if world == "bank":
                parts = common.load_bank_pass(month=month, index=index)
                truth, power, touched = (
                    bank_truth(parts=parts, month=month),
                    parts["power"],
                    parts["touched"],
                )
            else:
                built = g2.read_pass(world=world, month=month, index=index)  # ty: ignore[invalid-argument-type]
                truth, power, touched = built.truth, built.power, built.touched
            rows, dropped = _pass_rows(
                ctx=ctx, truth=truth, power=power, touched=touched, world=world, month=month,
                pass_index=index,
            )  # fmt: skip
            all_rows.extend(rows)
            drops[month] = drops.get(month, 0) + dropped["missing"]
            print(
                f"{world} {month} p{index}: {len(rows)} rows, {time.monotonic() - started:.1f} s",
                flush=True,
            )
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(all_rows, infer_schema_length=None)
    table.write_parquet(FEATURE_DIR / f"{world}.parquet")
    pl.DataFrame({"month": list(drops), "dropped": list(drops.values())}).write_csv(
        FEATURE_DIR / f"{world}_dropped.csv"
    )
    print(f"wrote {table.height} rows; events dropped for missing data by month: {drops}")


def main() -> None:
    """Run the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "features"))
    parser.add_argument("--world", required=True, choices=("bank", "g2", "gas"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.command == "generate":
        if args.world == "bank":
            sys.exit("the bank is experiment 709's and is not generated here")
        generate(world=args.world, limit=args.limit)
    else:
        build_features(world=args.world, limit=args.limit)


if __name__ == "__main__":
    assert f709.N_HALF_HOURS > 0
    main()
