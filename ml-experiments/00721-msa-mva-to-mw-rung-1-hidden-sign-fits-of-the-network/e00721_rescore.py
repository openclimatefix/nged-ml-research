# ruff: noqa: E501
"""Experiment 721: a read-only re-score of the stored quick look (`W2`, fold 2026-05, seed 0).

The quick look stopped at the harm stop (`SG-A` minus `P0` on import rows above 0.20 points). This
script re-reads the stored predictions of `P0`, `SG-0`, and `SG-A` and gives, for `SG-0` minus `P0`,
`SG-A` minus `P0`, and `SG-A` minus `SG-0`: the signed CRPS difference on import rows, reverse-flow
rows, and all rows, at each hidden Primary and pooled, with the plan's two-stage block-bootstrap
interval from `e00721_score` (unchanged, 20,000 resamples, generator seed 721). It also splits the
import harm by lead band (24 to 48 h, 48 to 120 h, and 120 to 240 h) and by hour of day.

One fold means the fold stage of the two-stage resampler is degenerate (the one fold is drawn every
time), so every interval is a block-only interval. The fold has fewer than 20 blocks of 10 valid
dates, so each interval is also widened by the page's rule (`lower` and `upper`; the raw interval is
kept as `raw_lower` and `raw_upper`). Every reading is interim, post hoc (chosen after the harm
stop), and has no bar. With 4 resampling units each interval is a rough guide only, and a widened
interval that spans zero is no evidence of no harm. The script writes no prediction and no fit. It
prints aggregates only.

Usage: `CUDA_VISIBLE_DEVICES= SLOT_CPU=<n> run.sh --go e00721_rescore.py [--n-resamples 20000]`
Output: `<data root>/rescore/rescore.json` and `<data root>/rescore/rescore.md`.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

import e00721_paths

e00721_paths.pin_threads()

# isort: split
import e00721_core as core  # noqa: E402
import e00721_driver as drv  # noqa: E402
import e00721_fit as fit  # noqa: E402
import e00721_score as score  # noqa: E402
import polars as pl  # noqa: E402

FOLD: Final = core.MDE_FOLD
"""The quick look's fold, 2026-05."""
PAIRS: Final = (("SG-0", "P0"), ("SG-A", "P0"), ("SG-A", "SG-0"))
"""The contrasts `a` minus `b`."""
LEAD_BANDS: Final = ((24.0, 48.0), (48.0, 120.0), (120.0, 240.0))
"""Lead bands in hours since the 12:00 UTC origin: the lower bound is included, the upper is not."""
ROW_CLASSES: Final = {
    "import": pl.col("actual_a") >= 0.0,
    "reverse": pl.col("actual_a") < 0.0,
    "all": pl.lit(True),
}
"""Row restrictions on the signed truth, as the quick look defines import and reverse-flow rows."""
NOTE: Final = (
    "interim, post hoc (chosen after the harm stop), no bar; one fold, so the fold stage of the "
    "two-stage resampler is degenerate and the interval is block-only; widened because the fold has "
    "fewer than 20 blocks; with so few resampling units the interval is a rough guide only, and a "
    "widened interval that spans zero is no evidence of no harm"
)


def load_arms(*, data_dir: Path, ids: list[int]) -> dict[str, pl.DataFrame]:
    """Return the stored scored rows of `P0`, `SG-0`, and `SG-A` at the Primaries `ids`.

    The rows are those of `W2`, fold 2026-05, seed 0.
    """
    out = {}
    for arm in ("P0", "SG-0", "SG-A"):
        df = drv.load_frame(data_dir=data_dir, job=fit.Job(arm=arm, world="W2", fold=FOLD, seed=0))
        if df is None:
            msg = (
                f"the re-score needs the stored predictions of {arm} in W2 on fold {FOLD} at seed 0"
            )
            raise FileNotFoundError(msg)
        out[arm] = df.filter(pl.col("time_series_id").is_in(ids))
    return out


def lead_hours() -> pl.Expr:
    """Return the lead in hours of a paired row, from the 12:00 UTC origin."""
    return (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0


def reading(*, paired: pl.DataFrame, restrict: pl.Expr, n_resamples: int) -> dict:
    """Return the point, the interval, and the counts of one paired contrast on the rows of `restrict`."""
    table = score.block_table(paired=paired, seeds=(0,), restrict=restrict)
    return score.read_contrast(table=table, n_resamples=n_resamples).as_dict() | {"note": NOTE}


def hour_points(*, paired: pl.DataFrame, restrict: pl.Expr) -> list[dict]:
    """Return the mean difference and the row count of each valid hour of the day (UTC), without an interval."""
    rows = (
        paired.filter(restrict)
        .with_columns(hour=pl.col("valid_time").dt.hour(), diff=pl.col("crps_a") - pl.col("crps_b"))
        .group_by("hour")
        .agg(point=pl.col("diff").mean(), n_rows=pl.len())
        .sort("hour")
    )
    return rows.to_dicts()


def rescore(
    *, data_dir: Path, hidden_ids: dict[str, int], n_resamples: int = core.N_RESAMPLES
) -> dict:
    """Return every reading of the re-score (the arms are read once and paired once per contrast)."""
    ids = list(hidden_ids.values())
    arms = load_arms(data_dir=data_dir, ids=ids)
    scopes = {"pooled": ids} | {site: [i] for site, i in hidden_ids.items()}
    out: dict = {"fold": FOLD, "seed": 0, "note": NOTE, "n_resamples": n_resamples, "contrasts": {}}
    for a, b in PAIRS:
        paired = score.pair_arms(a=arms[a], b=arms[b])
        body: dict = {"rows": {}, "lead_bands": {}, "hours": {}}
        for scope, scope_ids in scopes.items():
            in_scope = pl.col("time_series_id").is_in(scope_ids)
            for name, rows in ROW_CLASSES.items():
                body["rows"][f"{scope}/{name}"] = reading(
                    paired=paired, restrict=in_scope & rows, n_resamples=n_resamples
                )
            for lo, hi in LEAD_BANDS:
                band = (lead_hours() >= lo) & (lead_hours() < hi)
                body["lead_bands"][f"{scope}/import/{lo:g}-{hi:g}h"] = reading(
                    paired=paired, restrict=in_scope & ROW_CLASSES["import"] & band, n_resamples=n_resamples
                )  # fmt: skip
            body["hours"][f"{scope}/import"] = hour_points(
                paired=paired, restrict=in_scope & ROW_CLASSES["import"]
            )
        out["contrasts"][f"{a}_minus_{b}"] = body
    return out


def format_markdown(*, result: dict) -> str:
    """Return the re-score as text (aggregates only: points, intervals, and row counts)."""
    lines = [
        f"Re-score of the quick look, fold {result['fold']}, seed {result['seed']}: {result['note']}.",
        "",
    ]
    for name, body in result["contrasts"].items():
        lines.append(f"## {name} (signed CRPS difference, points)")
        for section in ("rows", "lead_bands"):
            for key, r in body[section].items():
                lines.append(
                    f"- {key}: {r['point']:+.3f} [{r['lower']:+.3f}, {r['upper']:+.3f}] (raw [{r['raw_lower']:+.3f}, {r['raw_upper']:+.3f}]), {r['n_rows']} rows, {r['n_blocks']} blocks, {r['n_units']} units, kish_b {r['kish_b']:.1f}"
                )
        for key, hours in body["hours"].items():
            cells = ", ".join(
                f"{h['hour']:02d}h {h['point']:+.2f} (n={h['n_rows']})" for h in hours
            )
            lines.append(f"- by valid hour (UTC), {key}: {cells}")
        lines.append("")
    return "\n".join(lines)


def run(*, data_dir: Path, hidden_ids: dict[str, int], n_resamples: int) -> dict:
    """Run the re-score, write `rescore.json` and `rescore.md` under the data root, and print the text."""
    result = rescore(data_dir=data_dir, hidden_ids=hidden_ids, n_resamples=n_resamples)
    folder = fit.data_root(data_dir=data_dir) / "rescore"
    folder.mkdir(parents=True, exist_ok=True)
    text = format_markdown(result=result)
    fit.atomic_write_bytes(path=folder / "rescore.md", payload=text.encode())
    fit.atomic_write_bytes(
        path=folder / "rescore.json", payload=json.dumps(result, indent=1, default=float).encode()
    )
    print(text)
    return result


def main() -> int:
    """Read the stored quick-look fits and write the re-score."""
    import e00721_cli as cli

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-resamples", type=int, default=core.N_RESAMPLES)
    args = parser.parse_args()
    run(
        data_dir=Path(os.environ["DATA_PATH_INTERNAL"]),
        hidden_ids=cli.hidden_series_ids(),
        n_resamples=args.n_resamples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
