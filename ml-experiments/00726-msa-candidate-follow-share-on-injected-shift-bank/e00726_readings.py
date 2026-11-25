# ruff: noqa: D103, E501
"""Experiment 726: the report-only readings 7 (coverage), 8 (CRPS), and 10 (experiment 709's level-step eras).

Usage (CPU only, stored files only; run through `run.sh cpu`):
    e00726_readings.py level10     # reading 10: M1 minus C and K minus C on experiment 709's true level steps
    e00726_readings.py coverage    # reading 7 (the share of unshifted readings on the safe side) and reading 8 (CRPS)

None of these readings changes the verdict (design 5.4); each is read at 95%.

**Reading 10.** Experiment 709's bank holds, per pass, eras that start at the cutoff and never end,
and that shift the truth as well as the input. The scorer takes the follow share
`(q50 on the bank - q50 unshifted) * sign / |size|` at the host's rows whose origin lies 7 to 28
days after the onset, at lead 24 to 120 hours, and averages the seeds 0 and 1 before scoring.

**Reading 7, safe side.** For a pass that raises the power the unshifted reading is wrongly high
if the forecast follows, so the safe side is "the unshifted power is at or above the shifted q05";
for a pass that lowers the power it is "at or below the shifted q95". The reading prints the share
of rows on the safe side for the Candidate and for `M1`, in each counted slice.

**Reading 8, CRPS.** The CRPS of a forecast is approximated from its 11 quantiles as twice the
pinball loss averaged over the levels with trapezoid weights. It is computed against the
unshifted power for both counted slices, and against the shifted power for target half-hours before
the shift ends in the lasting-step slice.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import e00726_common as com
import e00726_score as sc
import numpy as np
import polars as pl

SEEDS_10: Final = (0, 1)
N_PASSES_10: Final = 16
ONSET_DAYS: Final = (7, 28)
TAUS: Final = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
CRPS_WEIGHTS: Final = np.gradient(TAUS) / np.gradient(TAUS).sum()


def crps_from_quantiles(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the CRPS approximated by twice the weighted pinball loss over the 11 levels, per row."""
    diff = actual[:, None] - quantiles
    pinball = np.where(diff >= 0, TAUS * diff, (TAUS - 1.0) * diff)
    return 2.0 * (pinball * CRPS_WEIGHTS).sum(axis=1)


def bank_files(*, net: str, month: str, seed: int, r: int) -> tuple[Path, Path]:
    """Return (bank forecast, unshifted forecast) of `net` (`C`, `M1`, or `K`) for experiment 709's bank."""
    if net == "C":
        bank = (
            com.folder_of(prefix="00709")
            / "bank"
            / f"none_{month}_s{seed}"
            / f"pass{r:02d}.parquet"
        )
        return bank, com.root_of_seed(seed=seed).stored_fit(month=month, seed=seed)
    bank = (
        com.folder_of(prefix="00713")
        / "bank713"
        / f"{net}_{month}_s{seed}"
        / f"pass{r:02d}.parquet"
    )
    return bank, com.msa_fit_file(arm=net, month=month, seed=seed)


def level_step_events(*, month: str, r: int) -> pl.DataFrame:
    """Return the level-step eras of one pass of experiment 709's bank: host label, signed size."""
    truth = pl.read_parquet(
        com.folder_of(prefix="00709") / "bank" / f"none_{month}_s0" / f"pass{r:02d}_truth.parquet"
    )
    return truth.filter(pl.col("kind").str.starts_with("level_")).select("site", "kind", "size")


def follow_on_host(
    *, frame_bank: pl.DataFrame, frame_base: pl.DataFrame, tsid: int, size: float, month: str
) -> float:
    """Return the host's mean follow share on rows 7 to 28 days after the onset, lead 24 to 120 hours."""
    both = frame_bank.filter(pl.col("time_series_id") == tsid).join(
        frame_base.filter(pl.col("time_series_id") == tsid),
        on=list(com.KEYS),
        how="inner",
        suffix="_b",
    )
    ms = com.month_start_us(month=month)
    origin = both["power_fcst_init_time"].dt.epoch("us").to_numpy()
    lead = (both["valid_time"].dt.epoch("us").to_numpy() - origin) / 3_600_000_000
    day = (origin - ms) / com.DAY_US
    keep = (day >= ONSET_DAYS[0]) & (day < ONSET_DAYS[1]) & (lead >= 24) & (lead < 120)
    if not keep.any():
        return float("nan")
    share = com.follow_share(
        shifted=both["q50"].to_numpy()[keep].astype(np.float64),
        base=both["q50_b"].to_numpy()[keep].astype(np.float64),
        sign=np.full(int(keep.sum()), np.sign(size)),
        depth=np.full(int(keep.sum()), abs(size)),
    )
    return float(share.mean())


def run_level10() -> int:
    primary_of_id, labels = sc.site_table()
    tsid_of_label = {labels[p]: t for t, p in primary_of_id.items()}
    rows = []
    for month in com.BANK_FOLDS:
        for r in range(N_PASSES_10):
            for ev in level_step_events(month=month, r=r).iter_rows(named=True):
                shares = {}
                for net in ("C", "M1", "K"):
                    per_seed = []
                    for seed in SEEDS_10:
                        bank, base = bank_files(net=net, month=month, seed=seed, r=r)
                        per_seed.append(
                            follow_on_host(
                                frame_bank=com.read_forecast(path=bank, columns=("q50",)),
                                frame_base=com.read_forecast(path=base, columns=("q50",)),
                                tsid=tsid_of_label[ev["site"]], size=ev["size"], month=month,
                            )
                        )  # fmt: skip
                    shares[f"{net}|q50"] = float(np.mean(per_seed))
                rows.append({"month": month, "pass_r": r, "kind": ev["kind"], **shares})
    events = pl.DataFrame(rows)
    out = []
    for a in ("M1", "K"):
        for kind in (None, "level_m10", "level_m25", "level_p15"):
            sel = events if kind is None else events.filter(pl.col("kind") == kind)
            sel = (
                sel.with_columns(diff=pl.col(f"{a}|q50") - pl.col("C|q50"))
                .drop_nulls("diff")
                .filter(pl.col("diff").is_finite())
            )
            sums, counts = sc.month_pass_arrays(
                rows=sel, value="diff", months=com.BANK_FOLDS, n_passes=N_PASSES_10
            )
            low, high = sc.bootstrap_interval(
                sums=sums, counts=counts, draws=sc.REPORT_DRAWS, level=sc.REPORT_LEVEL
            )
            out.append({"contrast": f"{a} minus C", "kind": kind or "all level steps", "n_events": sel.height,
                        "D": sc.point_estimate(sums=sums, counts=counts), "low": low, "high": high})  # fmt: skip
    sc.print_table(
        title="Reading 10: follow share at 7 to 28 days after the onset of experiment 709's level steps (95%)",
        rows=out,
    )
    com.write_json_atomic(obj=out, path=com.data_root() / "scores" / "reading10.json")
    return 0


def make_extra(
    *, clean: np.ndarray, grid: int, step: int, primary_of_id: dict[int, int], nets: tuple[str, ...]
) -> Callable[[pl.DataFrame, str, int], pl.DataFrame]:
    """Return the row hook that adds the safe-side flag and the CRPS columns of each network."""

    def extra(rows: pl.DataFrame, month: str, r: int) -> pl.DataFrame:
        prim = rows["primary"].to_numpy()
        valid_idx = ((rows["valid_time"].dt.epoch("us").to_numpy() - grid) // step).astype(np.int64)
        actual = clean[prim, valid_idx].astype(np.float64)
        sign = rows["sign"].to_numpy()
        depth = rows["depth"].to_numpy()
        end_idx = (com.month_start_us(month=month) - grid) // step + com.bank.END_DAY * com.PER_DAY
        before_end = valid_idx < end_idx
        shifted_actual = actual + sign * depth * before_end
        new = {}
        for net in nets:
            q_s = np.column_stack([rows[f"{net}:{q}:s"].to_numpy() for q in com.QCOLS]).astype(
                np.float64
            )
            q05, q95 = q_s[:, 1], q_s[:, 9]
            safe = np.where(sign > 0, actual >= q05, actual <= q95).astype(np.float64)
            safe[~np.isfinite(actual)] = np.nan
            new[f"{net}|safe"] = safe
            new[f"{net}|crps_clean"] = crps_from_quantiles(quantiles=q_s, actual=actual)
            vs_shifted = crps_from_quantiles(quantiles=q_s, actual=shifted_actual)
            vs_shifted[~before_end] = np.nan
            new[f"{net}|crps_shifted"] = vs_shifted
        return rows.with_columns([pl.Series(k, v).fill_nan(None) for k, v in new.items()])

    return extra


def run_coverage() -> int:
    primary_of_id, labels = sc.site_table()
    e = com.prime_pipeline(month="2026-01")
    clean = e.c.clean_stack()
    grid, step = e.c.prep422.GRID_START_US, e.c.prep422.STEP_US
    nets = ("C_host", "M1")
    cols = [f"{n}|{m}" for n in nets for m in ("safe", "crps_clean", "crps_shifted")]
    got = sc.collect(
        nets=(sc.C_HOST, sc.M1), seeds=com.SEEDS_ALL, months=com.FOLD_MONTHS, passes=range(com.N_PASSES),
        primary_of_id=primary_of_id, n_sets=len(com.read_sets()), levels=com.QCOLS,
        extra=make_extra(clean=clean, grid=grid, step=step, primary_of_id=primary_of_id, nets=nets),
        extra_columns=cols,
    )  # fmt: skip
    groups = sc.group_primaries(labels=labels)
    out = []
    for slice_name in (com.SLICE_AFTER, com.SLICE_LASTING):
        for group in ("six_signed_mw", "eight_mva"):
            for metric in ("safe", "crps_clean", "crps_shifted"):
                stats = sc.contrast(events=got.events, a="M1", b="C_host", slice_name=slice_name, primaries=groups[group],
                                    months=com.FOLD_MONTHS, level=metric, draws=sc.REPORT_DRAWS, ci_level=sc.REPORT_LEVEL)  # fmt: skip
                levels = {n: sc.level_of(events=got.events, net=n, slice_name=slice_name, primaries=groups[group],
                                         months=com.FOLD_MONTHS, level=metric) for n in nets}  # fmt: skip
                out.append({"slice": slice_name, "group": group, "metric": metric, **levels,
                            **{f"M1 minus C {k}": stats[k] for k in ("D", "low", "high", "n_events")}})  # fmt: skip
    sc.print_table(
        title="Readings 7 (safe side) and 8 (CRPS, share of capacity), M1 and C host-only, 95%",
        rows=out,
    )
    com.write_json_atomic(obj=out, path=com.data_root() / "scores" / "reading7_8.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("level10", "coverage"))
    args = parser.parse_args()
    return run_level10() if args.command == "level10" else run_coverage()


if __name__ == "__main__":
    sys.exit(main())
