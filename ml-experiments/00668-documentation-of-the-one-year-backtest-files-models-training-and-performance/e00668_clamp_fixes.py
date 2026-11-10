"""Clamp night-time solar quantiles and the lower quantiles of the MVA Primaries (private files).

uv run --no-sync python ml-experiments/00668-*/e00668_clamp_fixes.py dry-run
uv run --no-sync python ml-experiments/00668-*/e00668_clamp_fixes.py apply

`dry-run` computes the clamped frames in memory and writes a report to
`<backtest folder>/clamp_fixes/` without touching the forecast files. `apply` also copies each
changed file to `pre_clamp/` (with its SHA-256), writes the clamped file to a temporary name, and
renames it over the original. Only quantile values change: the schema, the dtypes, the column order
and the row order stay the same.

Solar: every quantile is set to zero on the half-hours where the sun is at or below the horizon at
the middle of the half-hour (the daylight rule of `e00668_score_generators.py`). Primaries: every
quantile of an MVA series is raised to at least zero. The report also lists every series whose lower
quantile is negative, with the minimum observed power.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Final

import e00668_score_lib as lib
import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites
from studies.solar import zenith

REPORT_DIR: Final[Path] = lib.FOLDER / "clamp_fixes"
PRE_CLAMP_DIR: Final[Path] = lib.FOLDER / "pre_clamp"
FILES: Final[tuple[str, ...]] = ("primaries", "bsps", "gsps", "wind", "solar")


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def solar_night_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return True on rows where the sun is not above the horizon at the middle of the half-hour."""
    sites = pv_sites()
    night = np.zeros(frame.height, dtype=bool)
    ids = frame["time_series_id"].to_numpy()
    for rec in sites.iter_rows(named=True):
        sel = np.flatnonzero(ids == rec["time_series_id"])
        if len(sel) == 0:
            continue
        mid = frame["valid_time"].gather(sel.tolist()).dt.offset_by("-15m")
        z = np.asarray(zenith(stamps=mid, latitude=rec["latitude"], longitude=rec["longitude"]))
        night[sel] = ~(z < 90.0)
    return night


def clamp_solar(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the solar frame with every night-time quantile set to zero, and the night mask."""
    night = solar_night_mask(frame=frame)
    keep = pl.Series("night", night)
    out = frame.with_columns(
        [
            pl.when(keep).then(pl.lit(0.0, dtype=pl.Float32)).otherwise(pl.col(c)).alias(c)
            for c in lib.QCOLS
        ]
    )
    return out, night


def mva_primary_ids() -> list[int]:
    """Return the original ids of the Primaries whose series are apparent power (MVA)."""
    capacity = pl.read_parquet(lib.FOLDER / "series_capacity.parquet")
    primaries = pl.read_parquet(lib.FOLDER / "primaries.parquet", columns=["time_series_id"])
    ids = primaries["time_series_id"].unique().to_list()
    return sorted(
        capacity.filter((pl.col("power_unit") == "MVA") & pl.col("time_series_id").is_in(ids))[
            "time_series_id"
        ].to_list()
    )


def clamp_primaries(*, frame: pl.DataFrame, mva_ids: list[int]) -> pl.DataFrame:
    """Return the Primaries frame with every quantile of an MVA series raised to at least zero."""
    is_mva = pl.col("time_series_id").is_in(mva_ids)
    return frame.with_columns(
        [
            pl.when(is_mva).then(pl.col(c).clip(lower_bound=0.0)).otherwise(pl.col(c)).alias(c)
            for c in lib.QCOLS
        ]
    )


def change_stats(*, before: pl.DataFrame, after: pl.DataFrame, by: str = "time_series_id") -> dict:
    """Return rows changed, their share, and the largest absolute change, overall and per series."""
    paired = before.select(by, *lib.QCOLS).hstack(
        after.select([pl.col(c).alias(f"{c}_after") for c in lib.QCOLS])
    )
    diffs = paired.select(
        by,
        changed=pl.any_horizontal([pl.col(c) != pl.col(f"{c}_after") for c in lib.QCOLS]),
        max_change=pl.max_horizontal([(pl.col(c) - pl.col(f"{c}_after")).abs() for c in lib.QCOLS]),
    )
    per = (
        diffs.group_by(by)
        .agg(rows=pl.len(), changed=pl.col("changed").sum(), max_change=pl.col("max_change").max())
        .sort(by)
    )
    return {
        "rows": before.height,
        "rows_changed": int(diffs["changed"].sum()),
        "share_changed": float(diffs["changed"].mean()),
        "largest_change": float(diffs["max_change"].max()),
        "per_series": per.to_dicts(),
    }


def crossing_rows(*, frame: pl.DataFrame) -> int:
    """Return the number of rows whose quantiles are not non-decreasing."""
    bad = pl.any_horizontal(
        [pl.col(b) < pl.col(a) for a, b in zip(lib.QCOLS[:-1], lib.QCOLS[1:], strict=True)]
    )
    return int(frame.select(bad.sum()).item())


def negative_lower_quantile_flags(*, mva_ids: list[int]) -> list[dict]:
    """List series in all five files whose q01 is negative while the observed power never is.

    Series in `mva_ids` are listed too, so the table covers every series; they are clamped.
    """
    rows: list[dict] = []
    for name in FILES:
        truth = pl.read_parquet(lib.FOLDER / f"ground_truth_{name}.parquet")
        obs_min = truth.group_by("time_series_id").agg(
            observed_min=pl.col("power").min(), observed_min_raw=pl.col("power_raw").min()
        )
        scan = pl.scan_parquet(lib.FOLDER / f"{name}.parquet")
        keys = ["time_series_id"] + (["NRA_or_prevailing"] if name == "primaries" else [])
        low = (
            scan.group_by(keys)
            .agg(
                rows=pl.len(),
                rows_q01_negative=(pl.col("q01") < 0).sum(),
                rows_q05_negative=(pl.col("q05") < 0).sum(),
                rows_q50_negative=(pl.col("q50") < 0).sum(),
                min_q01=pl.col("q01").min(),
            )
            .collect()
            .join(obs_min, on="time_series_id", how="left")
        )
        for rec in low.filter(pl.col("rows_q01_negative") > 0).sort(keys).iter_rows(named=True):
            rec["file"] = name
            rec["observed_never_negative"] = bool(
                rec["observed_min"] is not None and rec["observed_min"] >= 0
            )
            rec["is_mva_primary_clamped"] = name == "primaries" and rec["time_series_id"] in mva_ids
            rows.append(rec)
    return rows


def main(mode: str) -> None:
    """Compute the clamps, write the report, and (in `apply` mode) replace the two files."""
    REPORT_DIR.mkdir(exist_ok=True)
    mva_ids = mva_primary_ids()
    report: dict = {"mva_primary_ids_clamped": mva_ids}

    solar = pl.read_parquet(lib.FOLDER / "solar.parquet")
    solar_after, night = clamp_solar(frame=solar)
    report["solar"] = {
        "night_rows": int(night.sum()),
        **change_stats(before=solar, after=solar_after),
        "crossing_rows_after": crossing_rows(frame=solar_after),
        "daylight_rows_bit_identical": bool(
            all(
                solar.filter(pl.Series(~night))[c].equals(solar_after.filter(pl.Series(~night))[c])
                for c in lib.QCOLS
            )
        ),
        "schema_same": solar.schema == solar_after.schema,
        "null_or_nan_after": int(
            solar_after.select(
                pl.sum_horizontal(
                    [pl.col(c).is_null() | pl.col(c).is_nan() for c in lib.QCOLS]
                ).sum()
            ).item()
        ),
    }
    prim = pl.read_parquet(lib.FOLDER / "primaries.parquet")
    prim_after = clamp_primaries(frame=prim, mva_ids=mva_ids)
    report["primaries"] = {
        **change_stats(before=prim, after=prim_after),
        "crossing_rows_before": crossing_rows(frame=prim),
        "crossing_rows_after": crossing_rows(frame=prim_after),
        "schema_same": prim.schema == prim_after.schema,
        "non_mva_rows_identical": bool(
            prim.filter(~pl.col("time_series_id").is_in(mva_ids)).equals(
                prim_after.filter(~pl.col("time_series_id").is_in(mva_ids))
            )
        ),
    }
    report["negative_lower_quantile_flags"] = negative_lower_quantile_flags(mva_ids=mva_ids)
    (REPORT_DIR / f"clamp_report_{mode}.json").write_text(json.dumps(report, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "negative_lower_quantile_flags"},
            indent=1,
            default=str,
        )[:6000]
    )
    if mode != "apply":
        return
    PRE_CLAMP_DIR.mkdir(exist_ok=True)
    hashes: dict[str, dict[str, str]] = {}
    for name, frame in (("solar", solar_after), ("primaries", prim_after)):
        original = lib.FOLDER / f"{name}.parquet"
        backup = PRE_CLAMP_DIR / f"{name}.parquet"
        if backup.exists():
            msg = f"{backup} exists; refusing to overwrite the saved original"
            raise FileExistsError(msg)
        shutil.copy2(original, backup)
        before_hash = sha256(original)
        if sha256(backup) != before_hash:
            msg = f"copy of {name}.parquet differs from the original"
            raise RuntimeError(msg)
        tmp = lib.FOLDER / f"{name}.parquet.tmp"
        row_group = 91980 if name == "solar" else 122640
        frame.write_parquet(tmp, compression="zstd", row_group_size=row_group)
        tmp.rename(original)
        hashes[name] = {"before": before_hash, "after": sha256(original)}
    (REPORT_DIR / "sha256.json").write_text(json.dumps(hashes, indent=1))
    print(json.dumps(hashes, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
