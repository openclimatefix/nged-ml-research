# ruff: noqa: D103, E501
"""Replace the `prevailing` rows of the one-year backtest's `primaries.parquet` with the extra-lag head.

The head is experiment 681's L3 head (14 extra lagged power inputs) on experiment 667's frozen
Candidate quantiles, banks, seeds and monthly folds. The NRA rows stay as they are. The folds
2025-11 and 2025-12 use the fallback hazard head, which has no inputs to widen, so their prevailing
rows stay as they are.

    uv run --no-sync python e00668_extra_lags_prevailing.py reproduce   # stored 667 head -> current prevailing rows
    uv run --no-sync python e00668_extra_lags_prevailing.py export      # L3 head -> exports_extra_lags/
    uv run --no-sync python e00668_extra_lags_prevailing.py replace     # copy aside, assemble, check, write atomically

`reproduce` is the gate: the export code path, fed experiment 667's stored heads, must reproduce the
current file's prevailing rows bit for bit (after the MVA clamp) for the folds 2026-01 to 2026-10.
"""

import hashlib
import shutil
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(next(Path(__file__).resolve().parents[1].glob("00667-*/prevailing"))))

import e00667_export as x667
import e00668_clamp_fixes as clamp
import e00668_score_lib as lib
import numpy as np
import p667_export as p667  # ty: ignore[unresolved-import]
import polars as pl

FOLDER: Final[Path] = lib.FOLDER
PRIMARIES: Final[Path] = FOLDER / "primaries.parquet"
BACKUP_DIR: Final[Path] = FOLDER / "pre_extra_lags"
L3_FITS: Final[Path] = (
    x667.ROOT.parent / "00681-extra-lagged-power-inputs-for-the-prevailing-forecast" / "fits" / "L3"
)
EXPORTS: Final[Path] = x667.ROOT / "exports_extra_lags"
KEY: Final = ["time_series_id", "power_fcst_init_time", "valid_time", "NRA_or_prevailing"]
CHANGED_MONTHS: Final = tuple(m for m in p667.MONTHS if m >= "2026-01")
"""Folds that have a trained survival head (experiment 681's L3 head applies)."""
FALLBACK_MONTHS: Final = tuple(m for m in p667.MONTHS if m < "2026-01")
N_ROWS: Final[int] = 7_848_960
ROW_GROUPS: Final[int] = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def clamped(*, frame: pl.DataFrame) -> pl.DataFrame:
    return clamp.clamp_primaries(frame=frame, mva_ids=clamp.mva_primary_ids())


def month_rows(*, frame: pl.DataFrame, month: str) -> pl.DataFrame:
    stamp = pl.col("power_fcst_init_time").dt.strftime("%Y-%m")
    return frame.filter(stamp == month)


def build_months(*, fits: Path) -> dict[str, pl.DataFrame]:
    """Return the clamped contract rows of the prevailing output of each trained fold, from `fits`."""
    p667.FITS = fits
    out = {}
    for month in CHANGED_MONTHS:
        frame, method = p667.prevailing_month(month=month)
        assert method == "Rhaz", (month, method)
        out[month] = clamped(frame=frame)
        print(month, method, out[month].height, "rows", flush=True)
    return out


def current_prevailing() -> pl.DataFrame:
    return pl.read_parquet(PRIMARIES).filter(pl.col("NRA_or_prevailing") == "prevailing")


def reproduce() -> None:
    cur = current_prevailing()
    mine = build_months(fits=p667.FITS)
    for month, frame in mine.items():
        old = month_rows(frame=cur, month=month).sort(KEY)
        assert frame.sort(KEY).equals(old), f"{month}: the export does not reproduce the file"
        print(month, "reproduces the current prevailing rows exactly", flush=True)


def export() -> None:
    EXPORTS.mkdir(exist_ok=True)
    for month, frame in build_months(fits=L3_FITS).items():
        tmp = EXPORTS / f"prevailing_{month}.parquet.tmp"
        frame.write_parquet(tmp)
        tmp.rename(EXPORTS / f"prevailing_{month}.parquet")


def structural_checks(*, df: pl.DataFrame) -> None:
    assert df.height == N_ROWS, df.height
    assert df.height == df.unique(KEY).height, "duplicate keys"
    q = df.select(x667.QCOLS).to_numpy()
    assert np.isfinite(q).all(), "non-finite quantile"
    assert (np.diff(q, axis=1) >= 0).all(), "quantiles cross"
    assert df.schema == pl.Schema({**x667.CONTRACT_SCHEMA, "NRA_or_prevailing": pl.String})
    assert df["time_series_id"].n_unique() == 16
    assert df["power_fcst_init_time"].n_unique() == 365
    per = df.group_by("time_series_id", "power_fcst_init_time", "NRA_or_prevailing").len()
    assert (per["len"] == 672).all()
    mva = clamp.mva_primary_ids()
    low = df.filter(pl.col("time_series_id").is_in(mva)).select(pl.col("q01").min()).item()
    assert low >= 0.0, f"an MVA Primary has a negative lower quantile: {low}"


def replace() -> None:
    old_path = PRIMARIES
    old = pl.read_parquet(old_path)
    BACKUP_DIR.mkdir(exist_ok=True)
    backup = BACKUP_DIR / "primaries.parquet"
    assert not backup.exists(), "the pre-extra-lags copy exists already; refusing to overwrite it"
    shutil.copy2(old_path, backup)
    digest = sha256(backup)
    assert digest == sha256(old_path)
    (BACKUP_DIR / "primaries.parquet.sha256").write_text(f"{digest}  primaries.parquet\n")
    nra = old.filter(pl.col("NRA_or_prevailing") == "NRA")
    prev_old = old.filter(pl.col("NRA_or_prevailing") == "prevailing")
    new_months = [pl.read_parquet(EXPORTS / f"prevailing_{m}.parquet") for m in CHANGED_MONTHS]
    kept = [month_rows(frame=prev_old, month=m) for m in FALLBACK_MONTHS]
    prev_new = pl.concat(
        [*kept, *[f.cast(dict(prev_old.drop("NRA_or_prevailing").schema)) for f in new_months]]  # ty: ignore[no-matching-overload]
    ).with_columns(NRA_or_prevailing=pl.lit("prevailing"))
    prev_new = prev_new.select(old.columns).cast(dict(old.schema))  # ty: ignore[no-matching-overload]
    df = pl.concat([nra, prev_new]).sort(KEY)
    structural_checks(df=df)
    key_cols = [*KEY, "nwp_init_time"]
    assert prev_new.select(key_cols).sort(KEY).equals(prev_old.select(key_cols).sort(KEY)), (
        "the new prevailing rows have different keys or nwp_init_time from the old ones"
    )
    # The NRA rows must be bit-identical to the current file.
    nra_new = df.filter(pl.col("NRA_or_prevailing") == "NRA")
    assert nra_new.equals(nra.sort(KEY)), "NRA rows changed"
    # The fallback folds' prevailing rows must be bit-identical too.
    fb_new = df.filter(
        (pl.col("NRA_or_prevailing") == "prevailing")
        & pl.col("power_fcst_init_time").dt.strftime("%Y-%m").is_in(FALLBACK_MONTHS)
    )
    assert fb_new.equals(pl.concat(kept).sort(KEY)), "fallback fold rows changed"
    tmp = old_path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd", row_group_size=N_ROWS // ROW_GROUPS, statistics=True)
    check = pl.read_parquet(tmp)
    assert check.equals(df)
    tmp.rename(old_path)
    print("wrote", old_path, "sha256", sha256(old_path)[:20], "old", digest[:20], flush=True)


if __name__ == "__main__":
    {"reproduce": reproduce, "export": export, "replace": replace}[sys.argv[1]]()
