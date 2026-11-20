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
`--limit n` builds only the first `n` passes of each fold month (a smoke run) and writes
`features_smoke/` instead of `features/`.
"""

import e00714_threads  # noqa: F401  (must run before numpy)

# isort: split

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Final

import e00714_common as common
import e00714_features as feats
import e00714_generator2 as g2
import e00714_model as model
import numpy as np
import polars as pl

FEATURE_DIR: Final[Path] = common.DATA_DIR / "features"
KA_FEATURES: Final[tuple[str, ...]] = (*feats.HOST_FEATURES, *feats.NEIGHBOUR_FEATURES)


def rewired_neighbours(
    *, ctx: common.Context, host: int, touched: set[int], rng: np.random.Generator
) -> list[int] | None:
    """Return the neighbours of another Primary that is not electrically linked to the host.

    The other Primary is neither the host nor a neighbour of the host. The returned neighbours
    exclude the host and every site that carries an injected change in the pass (a host, a donor,
    or a recipient), so the rewired features never read an injected step.

    Args:
        ctx: The context.
        host: The host Primary.
        touched: Indices of every host, donor, and recipient of the pass.
        rng: The random generator.

    Returns:
        The clean neighbour indices of a random valid Primary, or None if no Primary has a
        neighbour that is both unlinked to the host and untouched.
    """
    panel = ctx.panel
    linked = {host, *panel.neighbours[host]}
    options = []
    for u in range(len(panel.sites)):
        if u in linked or u in touched:
            continue
        clean = [j for j in panel.neighbours[u] if j != host and j not in touched]
        if clean:
            options.append(clean)
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


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
    dropped = {"missing": 0, "events": 0, "no_rewired_set": 0}
    hosts = truth["host"].to_list()
    all_recipients = set(touched) - set(hosts)
    for e, ev in enumerate(truth.iter_rows(named=True)):
        host = ev["host"]
        neighbours, n_foreign = own_neighbours(
            ctx=ctx, truth=truth, index=e, world=world, all_recipients=all_recipients
        )
        rewired = rewired_neighbours(ctx=ctx, host=host, touched=set(touched), rng=rng)
        dropped["no_rewired_set"] += int(rewired is None)
        any_dropped = False
        for k in common.ORIGIN_DAYS:
            base = feats.EventInput(
                host=host, onset_hh=ev["onset_hh"], onset_day=ev["onset_day"], neighbours=neighbours
            )

            def compute(
                *, event: feats.EventInput, src: common.PassPower, ref: common.PassPower,
                k: int = k, host: int = host,
            ) -> dict[str, float] | None:  # fmt: skip
                return feats.build_features(
                    event=event,
                    k=k,
                    power=src,
                    reference=ref,
                    shortwave=ctx.weather.shortwave[host],
                    capacity_mw=ctx.panel.capacity_mw,
                )

            main = compute(event=base, src=power, ref=reference)
            if main is None:
                dropped["missing"] += 1
                any_dropped = True
                continue
            rw = (
                None
                if rewired is None
                else compute(
                    event=dataclasses.replace(base, neighbours=rewired), src=power, ref=reference
                )
            )
            ka = compute(event=base, src=power, ref=real)
            nop = compute(event=base, src=real, ref=reference)
            cut = (ev["onset_day"] + k) * common.DAY + common.ORIGIN_HOUR_HH - common.LATENCY_HH
            duration = ev["duration_days"]
            in_force = duration is None or ev["onset_hh"] + duration * common.DAY > cut
            row = {
                "world": world, "month": month, "pass_index": pass_index,
                "pass_id": f"{world}_{month}_p{pass_index}", "event": e, "k": k,
                "host": host, "truth": ev["truth"], "kind": ev["kind"],
                "label": int(ev["truth"] == "era"),
                "size_true": ev["size"], "onset_day": ev["onset_day"], "onset_hh": ev["onset_hh"],
                "duration_days": duration, "band": ev["band"], "in_force": in_force,
                "mirror_state": ev["mirror_state"], "n_foreign_neighbours": n_foreign,
                "receiving_host": bool(ev.get("receiving_host", False)), **main,
            }  # fmt: skip
            row["rw_mirror_share"] = rw["mirror_share"] if rw else None
            row["rw_mirror_z"] = rw["mirror_z"] if rw else None
            for name in KA_FEATURES:
                row[f"ka_{name}"] = ka[name] if ka else None
                row[f"np_{name}"] = nop[name] if nop else None
            rows.append(row)
        dropped["events"] += int(any_dropped)
    return rows, dropped


def own_neighbours(
    *, ctx: common.Context, truth: pl.DataFrame, index: int, world: str, all_recipients: set[int]
) -> tuple[list[int], int]:
    """Return the host's neighbours without the sites of other events of the pass.

    A neighbour that hosts another event of the pass (or is that event's recipient) would carry a
    change that is not this event's mirror, so it is dropped. The second generator keeps such sites
    apart by construction, so nothing is dropped there. The bank does not store which recipient
    belongs to which transfer, so a metered-mirror transfer keeps every neighbour recipient, and the
    other events drop all recipients.

    Args:
        ctx: The context.
        truth: The pass's truth table.
        index: Row of the event.
        world: The world.
        all_recipients: Touched sites that host no event (the bank's recipients).

    Returns:
        The kept neighbours and the number dropped.
    """
    ev = truth.row(index, named=True)
    other_hosts = {h for j, h in enumerate(truth["host"].to_list()) if j != index}
    foreign = set(other_hosts)
    if world == "bank":
        if ev["mirror_state"] in ("none", "hidden"):
            foreign |= all_recipients
    else:
        for j, row in enumerate(truth.iter_rows(named=True)):
            if j != index and row["recipient"] >= 0:
                foreign.add(row["recipient"])
    neighbours = [j for j in ctx.panel.neighbours[ev["host"]] if j not in foreign]
    return neighbours, len(ctx.panel.neighbours[ev["host"]]) - len(neighbours)


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
    """Generate and store the passes of the second generator.

    A full run generates `CANDIDATE_FACTOR` times as many candidate passes as the world keeps and
    stores the balanced ones (`select_balanced`), then prints the site-prior AUROC of the
    candidates and of the kept passes, computed from the truth tables alone. A smoke run
    (`limit`) stores the first passes of each fold month, unbalanced, in `worlds_smoke`.
    """
    ctx = common.load_context()
    n_keep = g2.PASSES_PER_MONTH[world]
    if limit is not None:
        g2.use_folder(name="worlds_smoke")
        for month in common.MONTHS:
            for index in range(limit):
                built = g2.build_pass(ctx=ctx, world=world, month=month, index=index)
                g2.write_pass(world=world, month=month, index=index, generated=built)
        return
    folder = g2.world_dir(world=world)
    for stale in [*folder.glob("*.npz"), *folder.glob("*.parquet")]:
        stale.unlink()
    candidates: dict[tuple[str, int], g2.GeneratedPass] = {}
    for month in common.MONTHS:
        for index in range(g2.CANDIDATE_FACTOR * n_keep):
            started = time.monotonic()
            candidates[(month, index)] = g2.build_pass(
                ctx=ctx, world=world, month=month, index=index
            )
            print(
                f"{world} {month} candidate {index}: {time.monotonic() - started:.1f} s", flush=True
            )
    n_sites = len(ctx.panel.sites)
    vectors = {
        key: g2.balance_vector(truth=built.truth, n_sites=n_sites)
        for key, built in candidates.items()
    }
    kept = g2.select_balanced(
        vectors=vectors, months=common.MONTHS, per_month=n_keep, modulus=g2.CELL_MODULUS[world]
    )
    rows = []
    for month, indices in kept.items():
        for index in indices:
            built = candidates[(month, index)]
            g2.write_pass(world=world, month=month, index=index, generated=built)
            rows.append(built.truth)
    chosen = pl.concat(rows)
    everyone = pl.concat([built.truth for built in candidates.values()])
    for name, truth in (("candidates", everyone), ("kept", chosen)):
        print(f"{world} {name}: site-prior AUROC {truth_site_prior(truth=truth):.3f}")
    chosen.write_parquet(folder.parent / f"{world}_kept_truth.parquet")


def truth_site_prior(*, truth: pl.DataFrame) -> float:
    """Return the leave-one-month-out site-prior AUROC of a world's truth tables."""
    label = (truth["truth"] == "era").to_numpy().astype(float)
    score = model.site_prior_scores(
        sites=truth["host"].to_numpy(), label=label, months=truth["month"].to_numpy()
    )
    return model.auroc(label=label, score=score)


def build_features(*, world: str, limit: int | None) -> None:
    """Build and store the feature table of a world (`bank`, `g2`, or `gas`)."""
    if limit is not None:
        g2.use_folder(name="worlds_smoke")
    ctx = common.load_context()
    all_rows: list[dict] = []
    drops = {m: {"events": 0, "no_rewired_set": 0} for m in common.MONTHS}
    for month in common.MONTHS:
        indices = (
            list(range(common.BANK_PASSES_PER_MONTH))
            if world == "bank"
            else g2.stored_indices(world=world, month=month)  # ty: ignore[invalid-argument-type]
        )
        for index in indices if limit is None else indices[:limit]:
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
            for key in ("events", "no_rewired_set"):
                drops[month][key] += dropped[key]
            print(
                f"{world} {month} p{index}: {len(rows)} rows, {time.monotonic() - started:.1f} s",
                flush=True,
            )
    folder = FEATURE_DIR if limit is None else FEATURE_DIR.with_name("features_smoke")
    folder.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(all_rows, infer_schema_length=None)
    table.write_parquet(folder / f"{world}.parquet")
    pl.DataFrame(
        {
            "month": list(drops),
            "events_dropped": [d["events"] for d in drops.values()],
            "events_without_rewired_set": [d["no_rewired_set"] for d in drops.values()],
        }
    ).write_csv(folder / f"{world}_dropped.csv")
    print(f"wrote {table.height} rows; per fold month: {drops}")
    report = panel_report(ctx=ctx)
    report.write_csv(folder / "panel_missing.csv")
    print(report)


def panel_report(*, ctx: common.Context) -> pl.DataFrame:
    """Count, per fold month's data window, the half-hours of the panel that are missing or zero.

    The panel comes from the cleaned power table, in which the `substation_zero` rule has already
    set its flagged half-hours to missing, so no exact zero should remain.
    """
    rows = []
    for month in common.MONTHS:
        lo, hi = common.window_bounds(onset_day=common.month_start_day(month=month))
        window = ctx.panel.power[:, lo:hi]
        rows.append(
            {
                "month": month, "half_hours_read": int(np.isfinite(window).sum()),
                "missing_half_hours": int(np.isnan(window).sum()),
                "exact_zero_half_hours": int((window == 0).sum()),
            }
        )  # fmt: skip
    return pl.DataFrame(rows)


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
    main()
