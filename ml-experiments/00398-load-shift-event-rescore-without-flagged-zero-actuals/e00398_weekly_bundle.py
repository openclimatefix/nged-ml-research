"""Rescore experiment 381's weekly bundle and its references with and without the flagged actuals.

The scorers need forecasts, which the stored row files of the weekly bundle do not hold, so this
script calls experiment 381's own scoring functions twice: once with the scoring actuals as
published, and once with the flagged (Primary, valid time) keys removed from the actuals.

    PYTHONPATH=ml-experiments/00377-demand-v1-first-run-feasibility:\
ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00381-simple-weekly-xgboost-demand-against-gam-stack \
    uv run --no-sync python ml-experiments/00398-*/e00398_weekly_bundle.py
"""

import e00377_score as base  # ty: ignore[unresolved-import]
import e00381_score as e381  # ty: ignore[unresolved-import]
import polars as pl
from e00398_flagged_keys import OUT
from studies.eval_protocol import metrics

BUNDLE = "bundle_solar+sunny+hl1"
REFS = ("stack", "weekly_baseline", "bg2", "manual_heuristic", "climatology")


def main() -> None:
    """Score both versions and write the tables."""
    series = e381.adapter.series_table()
    ctx = e381.build_context(series=series)
    flagged = pl.read_parquet(OUT / "flagged_keys.parquet").select("time_series_id", "valid_time")
    stack = base.stack_frames(series=series)["stack"]
    arms = {
        "bundle": e381.arm_frame(regime="weekly", arm=BUNDLE, series=series),
        "weekly_baseline": e381.stored_frame(name="only_W", series=series),
        "stack": stack,
        **base.reference_frames(series=series),
    }
    results, contrasts = {}, []
    for version, drop in (("old", False), ("new", True)):
        c = dict(ctx)
        if drop:
            c["actuals"] = ctx["actuals"].join(
                flagged, on=["time_series_id", "valid_time"], how="anti"
            )
            print(f"actuals {ctx['actuals'].height} -> {c['actuals'].height}")
        table, scored, rows = e381.score_arms(arms=arms, series=series, ctx=c)
        results[version] = pl.concat([t.with_columns(arm=pl.lit(n)) for n, t in scored.items()])
        for ref in REFS:
            for gname, where in e381.GROUPS.items():
                r = metrics.month_bootstrap_contrast(
                    treatment=rows["bundle"], reference=rows[ref], metric="crps",
                    where=where & pl.col(e381.HEADLINE), slice_table=table,
                    level_percent=99.0, n_resamples=5000,
                )  # fmt: skip
                contrasts.append({"version": version, "reference": ref, "group": gname, **r})
                print(version, ref, gname, r, flush=True)
    both = pl.concat([results["old"].with_columns(version=pl.lit("old")),
                      results["new"].with_columns(version=pl.lit("new"))])  # fmt: skip
    both.write_parquet(OUT / "weekly_bundle_scores.parquet")
    pl.DataFrame(contrasts).write_parquet(OUT / "weekly_bundle_contrasts.parquet")


if __name__ == "__main__":
    main()
