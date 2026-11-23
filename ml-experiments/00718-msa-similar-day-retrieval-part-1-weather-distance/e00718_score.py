# ruff: noqa: E501
"""Experiment 718, WP7a: scoring on stored forecasts, KR9, and the per-Primary table.

Usage:
    e00718_score.py kr9          # KR9 on experiment 713's stored screen forecasts (reads data, CPU only)
    e00718_score.py self-test    # the per-Primary table and the loader's routing on synthetic frames

F9 scores with experiment 713's functions: the conversion to the contract origin, the losses, the seed
average, and the contrast with its guards are imported from `e00713_score` and never copied. This
module adds the routing of an arm to its forecast file (F9's folder for an F9 arm, experiment 713's
folder for `C`, `M1`, and `P`) and the table of every contrast by Primary. Every loss is in points of
capacity, as in experiment 713. KR9 requires that F9's loader and experiment 713's loader give the
same `M*` minus `P` contrast on the stored screen forecasts to 3 decimals, and that the Candidate table
of experiment 713's section 1.5 is reproduced by `e00713_score.py candidate`, which F9 runs unchanged.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import e00713_arms as a713  # ty: ignore[unresolved-import]
import e00713_score as s713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

KEYS = s713.KEYS
QCOLS = s713.QCOLS
KR9_TOLERANCE = 5e-4
"""Three decimals: the contrast must agree to half a unit of the third decimal."""


def is_f9_arm(*, arm: str) -> bool:
    """Return whether `arm` is one of F9's arms (including its `-nohist` and `-swap` read-outs)."""
    return arm.split("-nohist", maxsplit=1)[0].split("-swap", maxsplit=1)[0] in arms.PART_OF_ARM


def fit_path(*, arm: str, month: str, seed: int, data_dir: Path) -> Path:
    """Return the per-seed forecast file of `arm`: F9's folder for an F9 arm, else the file that experiment 713's scorer reads."""
    if is_f9_arm(arm=arm):
        return arms.fit_file(data_dir=data_dir, arm=arm, month=month, seed=seed)
    if arm == "C":
        return s713.candidate_files(month=month, seed=seed)
    return a713.fit_file(data_dir=data_dir, arm=arm, month=month, seed=seed)


def load_arm(
    *,
    arm: str,
    seeds: Sequence[int],
    months: Sequence[str],
    data_dir: Path = ML_EXPERIMENTS_DATA_DIR,
) -> pl.DataFrame:
    """Return an arm's per-seed rows in contract origin and capacity shares, with a `seed` column."""
    parts = []
    for month in months:
        for seed in seeds:
            frame = pl.read_parquet(
                fit_path(arm=arm, month=month, seed=seed, data_dir=data_dir),
                columns=[*KEYS, *QCOLS],
            ).with_columns(seed=pl.lit(seed, dtype=pl.Int64))
            parts.append(s713.to_contract_shares(frame=frame))
    frame = pl.concat(parts)
    assert frame.select(*KEYS, "seed").is_duplicated().sum() == 0, (
        f"arm {arm!r} holds a duplicated (key, seed) row"
    )
    assert frame["seed"].n_unique() == len(seeds), (
        f"arm {arm!r} does not hold every seed of {tuple(seeds)}"
    )
    return frame


def load_all(
    *, names: Sequence[str], months: Sequence[str], seeds: Sequence[int]
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, np.ndarray]]:
    """Return every arm's per-seed and seed-averaged scored rows and the Primary groups; assert equal keys."""
    groups, truth = s713.group_ids(), s713.truth_stack()
    per_seed, averaged = {}, {}
    for n in names:
        raw = load_arm(arm=n, seeds=seeds, months=months)
        per_seed[n] = s713.add_losses(frame=raw, truth=truth)
        averaged[n] = s713.add_losses(frame=s713.seed_average(frame=raw), truth=truth)
    s713.assert_same_keys(frames=averaged)
    return per_seed, averaged, groups


def per_primary_table(
    *, scored: dict[str, pl.DataFrame], reference: str, labels: dict[int, str],
    band: tuple[float, float] = s713.HEADLINE_BAND_HOURS,
) -> pl.DataFrame:  # fmt: skip
    """Return one row per (arm, Primary): mean CRPS and MAE in the lead band, and the CRPS minus the reference's.

    `scored` holds seed-averaged frames with the `crps`, `abs_err`, and `lead_hours` columns that
    experiment 713's `add_losses` adds. Every arm must hold the same keys as the reference, and the
    difference is taken on the joined keys, never by row position.
    """
    rows = []
    ref = scored[reference].filter(
        (pl.col("lead_hours") >= band[0]) & (pl.col("lead_hours") < band[1])
    )
    for name, frame in scored.items():
        sel = frame.filter((pl.col("lead_hours") >= band[0]) & (pl.col("lead_hours") < band[1]))
        joined = sel.select(*KEYS, "crps", "abs_err").join(
            ref.select(*KEYS, pl.col("crps").alias("crps_ref")), on=list(KEYS), how="inner"
        )
        assert joined.height == sel.height == ref.height, (
            f"arm {name!r} and {reference!r} hold different keys"
        )
        by = (
            joined.group_by("time_series_id")
            .agg(
                crps=pl.col("crps").mean(),
                mae=pl.col("abs_err").mean(),
                crps_minus_reference=(pl.col("crps") - pl.col("crps_ref")).mean(),
                n=pl.len(),
            )
            .with_columns(
                arm=pl.lit(name),
                primary=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String),
            )
            .sort("primary")
        )
        rows.append(by)
    return pl.concat(rows).select("arm", "primary", "crps", "mae", "crps_minus_reference", "n")


def kr9(*, months: Sequence[str] = a713.SCREEN_FOLDS, seeds: Sequence[int] = (0, 1)) -> int:
    """KR9: F9's loader and experiment 713's loader give the same `M*` minus `P` on the stored screen forecasts."""
    star = a713.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    mine = load_all(names=[star, "P"], months=months, seeds=seeds)
    theirs = s713.load_all(names=[star, "P"], months=tuple(months), seeds=tuple(seeds))
    lines = []
    for group in ("six", "eight"):
        a = s713.contrast(
            per_seed=mine[0],
            averaged=mine[1],
            treatment=star,
            reference="P",
            group=group,
            groups=mine[2],
            level=95.0,
        )
        b = s713.contrast(
            per_seed=theirs[0],
            averaged=theirs[1],
            treatment=star,
            reference="P",
            group=group,
            groups=theirs[2],
            level=95.0,
        )
        assert abs(a["point"] - b["point"]) < KR9_TOLERANCE, (group, a["point"], b["point"])
        lines.append(
            f"KR9: {star} minus P, {group}: {a['point']:+.3f} (F9's loader) and {b['point']:+.3f} (experiment 713's loader) points of CRPS"
        )
    print("\n".join(lines))
    print(
        "The Candidate table of design.md section 1.5 is checked by `e00713_score.py candidate`, which F9 runs unchanged."
    )
    return 0


def self_test() -> int:
    """Check the arm routing and the per-Primary table on synthetic scored frames."""
    data_dir = Path("/nonexistent-data-dir")
    assert fit_path(arm="Z", month="2025-11", seed=0, data_dir=data_dir).parts[-4:-1] == (
        "work",
        "fits",
        "Z",
    )
    assert "00718" in str(fit_path(arm="R-nohist", month="2025-11", seed=0, data_dir=data_dir))
    assert "00719" in str(fit_path(arm="U27-A-swap", month="2025-11", seed=0, data_dir=data_dir))
    assert "00713" in str(fit_path(arm="M1", month="2025-11", seed=0, data_dir=data_dir))
    rng = np.random.default_rng(0)
    n = 600
    ids = np.repeat([10, 11, 12], n // 3)
    keys = pl.DataFrame(
        {
            "time_series_id": ids,
            "power_fcst_init_time": np.tile(np.arange(n // 3), 3),
            "valid_time": np.tile(np.arange(n // 3), 3) + 1,
        }
    )
    base = rng.uniform(1.0, 2.0, size=n)
    frames = {
        name: keys.with_columns(
            crps=pl.Series(base + delta),
            abs_err=pl.Series(2 * base + delta),
            lead_hours=pl.Series(np.full(n, 48.0)),
        )
        for name, delta in (("ref", 0.0), ("better", -0.25), ("worse", 0.1))
    }
    table = per_primary_table(scored=frames, reference="ref", labels={10: "D1", 11: "D2", 12: "D3"})
    assert table.height == 9
    for name, delta in (("ref", 0.0), ("better", -0.25), ("worse", 0.1)):
        got = table.filter(pl.col("arm") == name)["crps_minus_reference"].to_numpy()
        assert np.allclose(got, delta), (name, got)
    d1 = table.filter((pl.col("arm") == "ref") & (pl.col("primary") == "D1"))["crps"].item()
    assert abs(d1 - float(base[ids == 10].mean())) < 1e-12
    broken = {**frames, "worse": frames["worse"].head(n - 1)}
    try:
        per_primary_table(scored=broken, reference="ref", labels={10: "D1", 11: "D2", 12: "D3"})
    except AssertionError:
        pass
    else:
        raise SystemExit("an arm with different keys was accepted")
    print(
        "per-Primary table: the CRPS, the MAE, and the CRPS minus the reference are right on synthetic frames, and an arm with other keys is refused"
    )
    print(
        "routing: F9 arms read F9's folders (their -nohist and -swap read-outs too), and C, M1, and P read experiment 713's"
    )
    print("e00718_score: all checks passed")
    return 0


def main() -> int:
    """Run a command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["kr9", "self-test"])
    args = parser.parse_args()
    return kr9() if args.command == "kr9" else self_test()


if __name__ == "__main__":
    sys.exit(main())
