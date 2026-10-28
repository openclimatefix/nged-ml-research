r"""Experiment 366: a rule on the meter readings and irradiance that chooses between two fits.

For each cell of experiment 358 (site, cut, injection level, reactive setting), compute two
statistics on the training sunny-midday readings: their Pearson correlation with irradiance, and
their 1st percentile. The rule chooses the fit through the meter (L0) if the correlation exceeds
`CORRELATION_THRESHOLD` or the percentile is below `PERCENTILE_THRESHOLD_POINTS`, otherwise the
naive fit (N). The rule's score is the stored score of the chosen arm. A census applies the same
rule to the 8 MVA-only Primaries. The reading construction is copied from experiment 365's `cell`.

    PYTHONPATH=ml-experiments/00110-*:ml-experiments/00358-* \
        uv run --no-sync python ml-experiments/00366-*/e00366_rule.py
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import e00110_arms as arms  # ty: ignore[unresolved-import]
import e00110_common as common  # ty: ignore[unresolved-import]
import e00110_inject as inject  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, pinned_solar_points, predict_gam
from studies.dp_operator import tan_phi_from_power_factor
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

CORRELATION_THRESHOLD: Final[float] = 0.3
PERCENTILE_THRESHOLD_POINTS: Final[float] = 1.0
PERCENTILE: Final[float] = 1.0
SLUG: Final[str] = "mva-fold-irradiance-correlation-reverse-flow-rule"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00366-{SLUG}" / "work"
KNOWN_DIR: Final[Path] = known.OUT_DIR


def statistics(
    *, cache: FeatureCache, site: str, reading: np.ndarray, cut: Any
) -> dict[str, float | int | str]:
    """Return the correlation, 1st percentile, count, and chosen arm for one set of readings."""
    features = cache.sites[site]
    rows = np.flatnonzero(
        (cache.end_times <= common.to_64(moment=cut))
        & np.isfinite(reading)
        & ~features.switching_mask
    )
    sunny = known._sunny(cache=cache, site=site, index=rows)
    values = reading[rows][sunny]
    irradiance = np.nan_to_num(features.ghi_w_m2[rows][sunny], nan=0.0)
    if len(values) < 3:
        return {
            "correlation": float("nan"),
            "p1_training_sunny": float("nan"),
            "n_training_sunny": len(values),
            "chosen": "n",
        }
    correlation = float(np.corrcoef(values, irradiance)[0, 1])
    p1 = float(np.percentile(values, PERCENTILE))
    use_l0 = correlation > CORRELATION_THRESHOLD or p1 < PERCENTILE_THRESHOLD_POINTS
    return {
        "correlation": correlation,
        "p1_training_sunny": p1,
        "n_training_sunny": len(values),
        "chosen": "l0" if use_l0 else "n",
    }


def cell(job: tuple[str, str, float, str]) -> dict[str, Any]:
    """Return the rule's statistics for one cell of experiment 358."""
    site, cut_name, level, reactive = job
    cut = known.CUTS[cut_name]
    cache = common.load_cache()
    part_4 = arms.read_part_4_tuned()
    spec_b = arms.base_specs(part_4=part_4)["b"]
    window = arms.arm_definitions(part_4=part_4, tuned=arms.read_tuned())["lq"].window
    features = cache.sites[site]
    pinned = pinned_capacity(vintages=read_vintages(), table=features.table, cut=cut)
    scale = inject.midday_median_absolute(cache=cache, site=site, before=cut)
    signed_cache = (
        cache
        if level == 0.0
        else inject.inject_farm(
            cache=cache, site=site, farm=known.FARM, capacity_points=level * scale
        )
    )
    signed = signed_cache.sites[site].y
    if reactive == "Q0":
        reading = np.abs(signed)
    else:
        fit_b = fit_gam(
            cache=cache,
            site=site,
            specs=[replace(spec_b)],
            cut=cut,
            window=window,
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        rows = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
        demand = np.full(len(cache.end_times), np.nan)
        demand[rows] = predict_gam(cache=cache, fit=fit_b, index=rows) + pinned_solar_points(
            cache=cache, features=features, pinned=pinned, index=rows
        )
        tan_phi = tan_phi_from_power_factor(power_factor=known.POWER_FACTOR)
        reading = (
            inject.inject_meter(
                cache=signed_cache, site=site, signed=signed, demand=demand, tan_phi=tan_phi
            )
            .sites[site]
            .y
        )
    return {
        "site": site,
        "cut": cut_name,
        "level": level,
        "reactive": reactive,
        **statistics(cache=cache, site=site, reading=reading, cut=cut),
    }


def census() -> list[dict[str, Any]]:
    """Return the statistics and the chosen arm at the 8 MVA-only Primaries at the scored cut."""
    cache = common.load_cache()
    return [
        {
            "site": site,
            **statistics(
                cache=cache, site=site, reading=cache.sites[site].y, cut=known.CUTS["scored"]
            ),
        }
        for site in common.MVA_SITES
    ]


def main() -> None:
    """Compute the statistics, join the stored scores, and print the numbers."""
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = [
        (site, cut, level, reactive)
        for cut in known.CUTS
        for level in known.LEVELS
        for reactive in known.REACTIVE
        for site in known.SITES
    ]
    stats = pl.DataFrame([cell(job) for job in jobs])
    stored = pl.DataFrame(
        [json.loads(path.read_text()) for path in sorted(KNOWN_DIR.glob("*.json"))],
        infer_schema_length=None,
    )
    joined = (
        stats.join(stored, on=["site", "cut", "level", "reactive"])
        .with_columns(
            acc_chosen=pl.when(pl.col("chosen") == "l0")
            .then(pl.col("acc_l0_sunny"))
            .otherwise(pl.col("acc_n_sunny")),
            acc_reverse_chosen=pl.when(pl.col("chosen") == "l0")
            .then(pl.col("acc_l0_sunny_reverse"))
            .otherwise(pl.col("acc_n_sunny_reverse")),
        )
        .with_columns(gap_to_signed=pl.col("acc_s_sunny") - pl.col("acc_chosen"))
    )
    joined.write_parquet(WORK / "rule.parquet")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(240)
    for cut in known.CUTS:
        for reactive in known.REACTIVE:
            part = joined.filter(pl.col("cut") == cut, pl.col("reactive") == reactive).sort(
                "level", "site"
            )
            print(f"\n== {cut} cut, {reactive} ==")
            print(
                part.select(
                    "level",
                    "site",
                    pl.col("correlation").round(2),
                    pl.col("p1_training_sunny").round(2),
                    "n_training_sunny",
                    "chosen",
                    pl.col("reverse_flow_share_sunny").round(3),
                    pl.col("acc_s_sunny").round(3),
                    pl.col("acc_l0_sunny").round(3),
                    pl.col("acc_n_sunny").round(3),
                    pl.col("acc_chosen").round(3),
                    pl.col("gap_to_signed").round(3),
                    pl.col("acc_s_sunny_reverse").round(3),
                    pl.col("acc_reverse_chosen").round(3),
                )
            )
            for level in known.LEVELS:
                sub = part.filter(pl.col("level") == level)
                print(
                    f"level {level}: sites within 0.02 of the signed fit:",
                    int((sub["gap_to_signed"].abs() <= 0.02).sum()),
                    "of",
                    len(sub),
                    "| L0 chosen at:",
                    sub.filter(pl.col("chosen") == "l0")["site"].to_list(),
                )
    print("\n== census at the 8 MVA-only Primaries (scored cut, not scored) ==")
    census_table = pl.DataFrame(census())
    census_table.write_parquet(WORK / "census.parquet")
    print(
        census_table.with_columns(
            pl.col("correlation").round(2), pl.col("p1_training_sunny").round(2)
        )
    )


if __name__ == "__main__":
    main()
