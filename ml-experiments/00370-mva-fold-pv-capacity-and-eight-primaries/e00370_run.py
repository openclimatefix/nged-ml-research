r"""Experiment 370: the MVA fold at D9 (signed-MW) and at the 8 MVA-only Primaries.

Two kinds of job, each writing its own JSON file atomically:

* `signed`: experiment 358's job for D9, written to this experiment's folder (the five other
  signed-MW Primaries reuse 358's stored cells);
* `census`: at an MVA-only Primary and a cut, fit the naive fit (N) and the fit through the meter
  (L0) to the raw readings, record the rule of experiment 366's statistics, the fitted capacity, the
  share of sunny-midday training half-hours with fitted reverse flow, and the readings' mean absolute
  error on the following year.

    PYTHONPATH=ml-experiments/00110-differentiable-physics-5-embedded-generation-and-capacity:ml-experiments/00358-mva-fold-known-answer-at-folded-mw-primaries:ml-experiments/00366-mva-fold-irradiance-correlation-reverse-flow-rule \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00370-*/e00370_run.py --workers 4
"""

import argparse
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import e00366_rule as rule  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_gam import fit_gam, predict_gam
from studies.dp_meter import fit_gam_through_meter, predict_meter
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-pv-capacity-and-eight-primaries"
OUT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00370-{SLUG}" / "work"
SIGNED_SITES: Final[tuple[str, ...]] = ("D9",)
CENSUS_SITES: Final[tuple[str, ...]] = common.MVA_SITES


def _write(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=float) + "\n")
    temporary.replace(path)
    return path


def run_job(job: tuple[str, str, str, float, str]) -> Path:
    """Run one `(kind, site, cut, level, reactive)` job and return its result file."""
    kind, site, cut_name, level, reactive = job
    if kind == "signed":
        known.OUT_DIR = OUT_DIR / "signed"
        return known.run_job((site, cut_name, level, reactive))
    path = OUT_DIR / "census" / f"{site}_{cut_name}.json"
    if path.exists():
        return path
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(2)
    cut = known.CUTS[cut_name]
    cache = common.load_cache()
    part_4 = arms.read_part_4_tuned()
    spec = replace(arms.base_specs(part_4=part_4)["b"], capacity="fitted")
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    features = cache.sites[site]
    reading = features.y
    chosen = rule.statistics(cache=cache, site=site, reading=reading, cut=cut)
    fit_n = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
    )
    meter = fit_gam_through_meter(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=window,
        season_years=common.SEASON_YEARS,
        qmode="zero",
    )
    train = np.flatnonzero(
        (cache.end_times <= common.to_64(moment=cut))
        & np.isfinite(reading)
        & np.isfinite(features.pv_unit_shape)
        & ~features.switching_mask
    )
    train_sunny = train[known._sunny(cache=cache, site=site, index=train)]  # noqa: SLF001
    pred_train = predict_meter(cache=cache, meter=meter, index=train_sunny)
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    index = index[np.isfinite(reading[index]) & ~features.switching_mask[index]]
    sunny = known._sunny(cache=cache, site=site, index=index)  # noqa: SLF001
    pred_l0 = predict_meter(cache=cache, meter=meter, index=index)
    pred_n = predict_gam(cache=cache, fit=fit_n, index=index)
    result: dict[str, Any] = {
        "site": site,
        "cut": cut_name,
        **chosen,
        "capacity_n": float(fit_n.capacity_points()[fit_n.last_season_year]),
        "capacity_l0": float(meter.fit.capacity_points()[meter.fit.last_season_year]),
        "converged": bool(meter.converged),
        "starts_capacity_spread": float(meter.capacity_spread),
        "n_train_sunny": len(train_sunny),
        "reverse_share_sunny_train_l0": float((pred_train.real_power < 0).mean()),
        "n_rows": len(index),
        "n_sunny_rows": int(sunny.sum()),
    }
    for label, mask in (("all", np.ones(len(index), bool)), ("sunny", sunny)):
        if mask.sum() == 0:
            continue
        r = reading[index][mask]
        result[f"mae_l0_{label}"] = float(np.abs(pred_l0.apparent_power[mask] - r).mean())
        result[f"mae_n_{label}"] = float(np.abs(pred_n[mask] - r).mean())
    return _write(path, result)


def main() -> None:
    """Run every job not yet done, in a spawned pool."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    jobs = [
        ("signed", site, cut, level, reactive)
        for cut in known.CUTS
        for level in known.LEVELS
        for reactive in known.REACTIVE
        for site in SIGNED_SITES
    ] + [("census", site, cut, 0.0, "Q0") for cut in known.CUTS for site in CENSUS_SITES]
    context = multiprocessing.get_context("spawn")
    with context.Pool(arguments.workers) as pool:
        for finished in pool.imap_unordered(run_job, jobs):
            print("done", finished.name, flush=True)


if __name__ == "__main__":
    main()
