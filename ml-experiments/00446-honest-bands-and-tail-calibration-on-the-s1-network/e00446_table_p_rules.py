"""Experiment 446 addendum: the rules that waited for table P, read from the stored P scores.

No forecast is read or fitted. The stored score tables of experiment 446 (`scores_s012`, `scores_C`)
already hold the "P" variant, which refits experiment 435's table P recipe (one factor per metering
group, lead band, tail, and class, on the arm's own earlier folds) for every arm. This script reads
those tables and applies each rule of the 446 pre-registration that said "pending table P":

- Brief B (a): per class, the experiment 435 tolerance (q95 within 0.03 and q99 within 0.015 of the
  class's own clean-row coverage), on every class where R-inj with P fails.
- Brief B (b): clean-row CRPS at most 0.03 and each tail pinball at most 0.02 above R-inj with P.
- Brief B (c): 90% band-width ratio to R-inj at most 1.05 on classes where R-inj with P passes.
- Brief B (d): real-harm and real-event q95 coverage not below R-inj's, with an event-cluster
  bootstrap interval, with and without the event with the most scored rows.
- Brief B (e): cold winter nights q95 at least 0.92 and q99 at least 0.975, and not below R-inj's.
- Brief C: the stored `brief_c.parquet` flags, and predictions P6 and P7 read after P.

```bash
OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00446-*/e00446_table_p_rules.py
```
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00446-honest-bands-and-tail-calibration-on-the-s1-network" / "work"
)
OUT_DIR: Final[Path] = WORK_DIR / "table_p_rules"
GROUPS: Final[tuple[str, ...]] = ("signed-MW six", "MVA-only eight")
U_ARMS: Final[tuple[str, ...]] = ("U1n-2", "U1n-5", "U2", "U2i", "U2c", "U2-clean", "U3")
T_ARMS: Final[tuple[str, ...]] = ("T1", "T2", "T3", "T4", "T5")
HARM_CLASSES: Final[tuple[str, ...]] = ("inj_harm", "gen_jump", "real_harm", "real_event_nra100")
Q95_TOLERANCE: Final[float] = 0.03
Q99_TOLERANCE: Final[float] = 0.015
CLEAN_RANGES: Final[dict[str, tuple[float, float]]] = {
    "coverage_q95": (0.93, 0.97),
    "coverage_q99": (0.98, 0.995),
    "coverage_q05": (0.03, 0.07),
    "coverage_q01": (0.005, 0.02),
}
CRPS_LIMIT: Final[float] = 0.03
TAIL_LIMIT: Final[float] = 0.02
WIDTH_LIMIT: Final[float] = 1.05
COLD_Q95: Final[float] = 0.92
COLD_Q99: Final[float] = 0.975
RESAMPLES: Final[int] = 4000
SEED: Final[int] = 446


def read(*, folder: str, variant: str, name: str) -> pl.DataFrame:
    """Return one stored score table of the given variant."""
    return pl.read_parquet(WORK_DIR / folder / variant / f"{name}.parquet")


def class_passes(*, absolute: pl.DataFrame, arm: str, group: str) -> dict[str, bool]:
    """Return, per harm class, whether the arm meets experiment 435's coverage tolerance."""
    rows = absolute.filter((pl.col("arm") == arm) & (pl.col("group") == group))
    clean = rows.filter(pl.col("row_set") == "clean").row(0, named=True)
    out: dict[str, bool] = {}
    for cls in HARM_CLASSES:
        part = rows.filter(pl.col("row_set") == cls)
        if part.is_empty():
            continue
        row = part.row(0, named=True)
        out[cls] = (
            abs(row["coverage_q95"] - clean["coverage_q95"]) <= Q95_TOLERANCE
            and abs(row["coverage_q99"] - clean["coverage_q99"]) <= Q99_TOLERANCE
        )
    return out


def clean_in_range(*, absolute: pl.DataFrame, arm: str, group: str) -> dict[str, bool]:
    """Return, per tail, whether clean-row coverage lies in the target range."""
    row = absolute.filter(
        (pl.col("arm") == arm) & (pl.col("group") == group) & (pl.col("row_set") == "clean")
    ).row(0, named=True)
    return {k: lo <= row[k] <= hi for k, (lo, hi) in CLEAN_RANGES.items()}


def event_blocks(
    *, blocks: pl.DataFrame, row_set: str, group: str, drop_largest: bool
) -> pl.DataFrame:
    """Return per-event scored rows and q95-covered rows per arm for one row set and group."""
    part = (
        blocks.filter((pl.col("row_set") == row_set) & (pl.col("group") == group))
        .group_by("arm", "block")
        .agg(pl.col("n").sum(), pl.col("cov_q95").sum())
    )
    if drop_largest:
        largest = (
            part.filter(pl.col("arm") == "R-inj")
            .sort("n", descending=True)
            .row(0, named=True)["block"]
        )
        part = part.filter(pl.col("block") != largest)
    return part


def coverage_difference(
    *, events: pl.DataFrame, arm: str, base: str
) -> tuple[float, float, float, int]:
    """Return arm minus base q95 coverage with an event-cluster bootstrap 95% interval."""
    wide = (
        events.filter(pl.col("arm").is_in([arm, base]))
        .pivot(on="arm", index="block", values=["n", "cov_q95"])
        .fill_null(0)
    )
    n_a, n_b = wide[f"n_{arm}"].to_numpy(), wide[f"n_{base}"].to_numpy()
    c_a, c_b = wide[f"cov_q95_{arm}"].to_numpy(), wide[f"cov_q95_{base}"].to_numpy()

    def diff(idx: np.ndarray) -> float:
        return c_a[idx].sum() / n_a[idx].sum() - c_b[idx].sum() / n_b[idx].sum()

    everything = np.arange(len(wide))
    rng = np.random.default_rng(SEED)
    draws = np.array([diff(rng.integers(0, len(wide), len(wide))) for _ in range(RESAMPLES)])
    return (
        float(diff(everything)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        len(wide),
    )


def pooled_cold(*, cold: pl.DataFrame, arm: str, group: str) -> dict[str, float]:
    """Return cold-winter-night coverage pooled over folds, weighted by rows, and the row count."""
    part = cold.filter((pl.col("arm") == arm) & (pl.col("group") == group))
    n = part["n"].sum()
    return {
        "n": float(n),
        "q95": float((part["coverage_q95"] * part["n"]).sum() / n),
        "q99": float((part["coverage_q99"] * part["n"]).sum() / n),
    }


def brief_b(*, variant: str) -> pl.DataFrame:
    """Return one row per (arm, group) with every Brief B rule read on the given variant."""
    absolute = read(folder="scores_s012", variant=variant, name="absolute")
    contrasts = read(folder="scores_s012", variant=variant, name="contrasts")
    cold = read(folder="scores_s012", variant=variant, name="cold_nights")
    blocks = read(folder="scores_s012", variant=variant, name="blocks")
    rows: list[dict] = []
    for group in GROUPS:
        base_pass = class_passes(absolute=absolute, arm="R-inj", group=group)
        base_cold = pooled_cold(cold=cold, arm="R-inj", group=group)
        for arm in ("R-inj", "S1", *U_ARMS):
            passes = class_passes(absolute=absolute, arm=arm, group=group)
            ranges = clean_in_range(absolute=absolute, arm=arm, group=group)
            record: dict = {"variant": variant, "arm": arm, "group": group}
            for cls, ok in passes.items():
                record[f"tolerance_{cls}"] = ok
            record |= {f"in_range_{k}": v for k, v in ranges.items()}
            record["rule_a"] = all(passes[c] for c in passes if not base_pass.get(c, True))
            if arm != "R-inj":
                ct = contrasts.filter(
                    (pl.col("contrast") == f"{arm} - R-inj")
                    & (pl.col("group") == group)
                    & (pl.col("row_set") == "clean")
                )
                diffs = {r["metric"]: r["difference"] for r in ct.iter_rows(named=True)}
                record["crps_vs_R_inj"] = diffs["crps"]
                record["worst_tail_pinball_vs_R_inj"] = max(
                    diffs[f"pinball_q{q}"] for q in ("01", "05", "95", "99")
                )
                record["rule_b"] = (
                    diffs["crps"] <= CRPS_LIMIT
                    and record["worst_tail_pinball_vs_R_inj"] <= TAIL_LIMIT
                )
                rh = contrasts.filter(
                    (pl.col("contrast") == f"{arm} - R-inj")
                    & (pl.col("group") == group)
                    & (pl.col("row_set") == "real_harm")
                    & (pl.col("metric") == "pinball_q99")
                )
                record["real_harm_q99_pinball_vs_R_inj"] = (
                    rh["difference"][0] if not rh.is_empty() else None
                )
                widths = {}
                for cls in HARM_CLASSES + ("clean",):
                    a = absolute.filter(
                        (pl.col("arm") == arm)
                        & (pl.col("group") == group)
                        & (pl.col("row_set") == cls)
                    )
                    b = absolute.filter(
                        (pl.col("arm") == "R-inj")
                        & (pl.col("group") == group)
                        & (pl.col("row_set") == cls)
                    )
                    if not a.is_empty() and not b.is_empty():
                        widths[cls] = a["width90_mean"][0] / b["width90_mean"][0]
                passing = [c for c, ok in base_pass.items() if ok] + ["clean"]
                record["max_width_ratio_on_r_inj_passing_classes"] = max(
                    widths[c] for c in passing if c in widths
                )
                record["rule_c"] = record["max_width_ratio_on_r_inj_passing_classes"] <= WIDTH_LIMIT
                for row_set, tag in (
                    ("real_harm", "harm"),
                    ("real_harm_wo_largest", "harm_wo_largest"),
                    ("real_event_nra100", "event_nra"),
                    ("real_event_nra100_wo_largest", "event_nra_wo_largest"),
                    ("real_event", "event_metered"),
                ):
                    drop = row_set.endswith("_wo_largest")
                    source = row_set.removesuffix("_wo_largest")
                    events = event_blocks(
                        blocks=blocks, row_set=source, group=group, drop_largest=drop
                    )
                    d, lo, hi, k = coverage_difference(events=events, arm=arm, base="R-inj")
                    record[f"q95_diff_{tag}"] = d
                    record[f"q95_diff_{tag}_low"] = lo
                    record[f"q95_diff_{tag}_high"] = hi
                    record[f"events_{tag}"] = k
                record["rule_d_veto"] = (
                    record["q95_diff_harm"] < 0 or record["q95_diff_event_nra"] < 0
                )
            cold_arm = pooled_cold(cold=cold, arm=arm, group=group)
            record |= {f"cold_{k}": v for k, v in cold_arm.items()}
            record["rule_e"] = (
                cold_arm["q95"] >= COLD_Q95
                and cold_arm["q99"] >= COLD_Q99
                and cold_arm["q95"] >= base_cold["q95"]
                and cold_arm["q99"] >= base_cold["q99"]
            )
            for key in ("crps", "coverage_q95", "coverage_q99", "width90_mean"):
                clean = absolute.filter(
                    (pl.col("arm") == arm)
                    & (pl.col("group") == group)
                    & (pl.col("row_set") == "clean")
                )
                record[f"clean_{key}"] = clean[key][0]
            rows.append(record)
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    """Write the Brief B rule table for each variant and print the P-variant verdicts."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    for variant in ("P", "w1_15", "raw"):
        table = brief_b(variant=variant)
        table.write_parquet(OUT_DIR / f"brief_b_{variant}.parquet")
        print("== Brief B", variant)
        print(
            table.select(
                "arm",
                "group",
                "rule_a",
                "rule_b",
                "rule_c",
                "rule_e",
                *(c for c in ("rule_d_veto",) if c in table.columns),
                "cold_q95",
                "cold_q99",
                "clean_coverage_q95",
                "clean_coverage_q99",
                "clean_crps",
            )  # fmt: skip
        )
    brief_c = read(folder="scores_C", variant="P", name="brief_c")
    brief_c.write_parquet(OUT_DIR / "brief_c_P.parquet")
    print(
        brief_c.select(
            "arm",
            "group",
            "cells_met",
            "cells_met_by_base",
            "limits_ok",
            "cold_ok",
            "primary_ok",
            "pass_brief_c",
        )
    )
    cold_c = read(folder="scores_C", variant="P", name="cold_nights")
    cold_c.write_parquet(OUT_DIR / "cold_c_P.parquet")


if __name__ == "__main__":
    main()
