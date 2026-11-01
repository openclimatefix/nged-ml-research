r"""Print the experiment 465 sweep tables from `work/sweep/sweep.parquet`.

Shares are the mean shift of the median forecast divided by the injected step (signs pooled by
dividing each sign's mean shift by its signed step). Group views: signed-MW six and MVA-only eight
(experiment 392's `GROUP_OF`). Rows are pooled over refits and Primaries.

```bash
DATA_PATH_INTERNAL=... uv run --no-sync python ml-experiments/00465-*/e00465_sweep_report.py
```
"""

import polars as pl
from e00465_fit import WORK_DIR  # ty: ignore[unresolved-import]
from studies.eval_protocol.slices import MVA_ONLY_LABELS, SIGNED_MW_LABELS

GROUP_OF = {
    **dict.fromkeys(SIGNED_MW_LABELS, "signed-MW six"),
    **dict.fromkeys(MVA_ONLY_LABELS, "MVA-only eight"),
}
BAND_ORDER = ["0-24 h", "12-36 h", "24-72 h", "72-168 h", "168-240 h", "0-240 h"]
L_ORDER = ["6h", "24h", "48h", "3d", "5d", "7d"]


def pooled(*, table: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Sum the metric sums over everything not in `by`."""
    sums = ["n", "shift_sum", "width_sum", "width_base_sum", "cover_clean_sum", "cover_event_sum"]
    return table.group_by(by).agg(pl.col(c).sum() for c in sums)


def show(*, frame: pl.DataFrame, index: list[str], on: str, values: str, order: list[str]) -> None:
    """Print `frame` pivoted with `on` as columns in `order`."""
    wide = frame.pivot(on=on, index=index, values=values)
    print(wide.select(*index, *[c for c in order if c in wide.columns]).sort(index))


def main() -> None:
    """Print the tables."""
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(3)
    sweep = pl.read_parquet(WORK_DIR / "sweep" / "sweep.parquet").with_columns(
        group=pl.col("site").replace_strict(GROUP_OF, default="short-history")
    )
    main_view = sweep.filter(pl.col("group").is_in(["signed-MW six", "MVA-only eight"]))
    steps = main_view.filter(pl.col("kind") == "step", pl.col("widening") == 1.0)
    for share in (0.05, 0.025):
        by = ["group", "arm", "variant", "sign", "band"]
        table = pooled(table=steps.filter(pl.col("share") == share), by=by).with_columns(
            share_of_step=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * share)
        )
        both = table.group_by("group", "arm", "variant", "band").agg(
            pl.col("share_of_step").mean(), pl.col("n").sum()
        )
        for band in ("24-72 h", "12-36 h", "0-24 h", "72-168 h", "0-240 h"):
            print(f"\n== step {share:.1%} of capacity: median shift / step, lead {band} ==")
            show(
                frame=both.filter(pl.col("band") == band),
                index=["group", "arm"],
                on="variant",
                values="share_of_step",
                order=L_ORDER,
            )
        print(f"\n== step {share:.1%}: rows per cell (lead 24-72 h, pooled signs) ==")
        show(
            frame=both.filter(pl.col("band") == "24-72 h"),
            index=["group", "arm"],
            on="variant",
            values="n",
            order=L_ORDER,
        )
    column = main_view.filter(
        pl.col("kind") == "column_k1", pl.col("widening") == 1.0, pl.col("share") == 0.05
    )
    table = pooled(table=column, by=["group", "arm", "sign", "band"]).with_columns(
        s=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * 0.05)
    )
    print("\n== lag_origin alone moved by 5% (column level, k = 1): shift / step ==")
    show(
        frame=table.group_by("group", "arm", "band").agg(pl.col("s").mean()),
        index=["group", "arm"],
        on="band",
        values="s",
        order=BAND_ORDER,
    )
    print("\n== band-width ratio (5%-95% band, modified / baseline), step 5%, by widening ==")
    width = pooled(
        table=steps.filter(pl.col("share") == 0.05, pl.col("band") == "24-72 h"),
        by=["group", "arm", "variant", "widening"],
    )
    wide_all = main_view.filter(
        pl.col("kind") == "step", pl.col("share") == 0.05, pl.col("band") == "24-72 h"
    )
    width = pooled(table=wide_all, by=["group", "arm", "variant", "widening"]).with_columns(
        ratio=pl.col("width_sum") / pl.col("width_base_sum"),
        cover_clean=pl.col("cover_clean_sum") / pl.col("n"),
        cover_event=pl.col("cover_event_sum") / pl.col("n"),
    )
    for metric in ("ratio", "cover_clean", "cover_event"):
        print(f"-- {metric} (lead 24-72 h, L = 48 h and 7 d)")
        show(
            frame=width.filter(pl.col("variant").is_in(["48h", "7d"])).with_columns(
                key=pl.col("arm") + " " + pl.col("variant")
            ),
            index=["group", "widening"],
            on="key",
            values=metric,
            order=sorted(width.select(pl.col("arm") + " " + pl.col("variant"))["arm"].unique()),
        )
    bad = main_view.filter(
        pl.col("kind").is_in(["stuck", "dropout", "spike"]), pl.col("widening") == 1.0
    )
    table = pooled(table=bad, by=["group", "arm", "kind", "variant", "band"]).with_columns(
        shift_points=100 * pl.col("shift_sum") / pl.col("n")
    )
    print("\n== bad data: mean shift of the median, points of capacity (lead 24-72 h) ==")
    show(
        frame=table.filter(pl.col("band") == "24-72 h").with_columns(
            key=pl.col("kind") + " " + pl.col("variant")
        ),
        index=["group", "arm"],
        on="key",
        values="shift_points",
        order=sorted((table["kind"] + " " + table["variant"]).unique().to_list()),
    )
    level = main_view.filter(
        pl.col("kind") == "level", pl.col("widening") == 1.0, pl.col("band") == "24-72 h"
    )
    table = pooled(table=level, by=["group", "arm", "sign", "day"]).with_columns(
        s=pl.col("shift_sum") / pl.col("n") / (pl.col("sign") * 0.05)
    )
    print("\n== level change (5% for the refit month): median shift / step by days since start ==")
    both = table.group_by("group", "arm", "day").agg(pl.col("s").mean())
    show(
        frame=both.with_columns(day=pl.col("day").cast(pl.String)),
        index=["group", "arm"],
        on="day",
        values="s",
        order=[str(d) for d in range(40)],
    )


if __name__ == "__main__":
    main()
