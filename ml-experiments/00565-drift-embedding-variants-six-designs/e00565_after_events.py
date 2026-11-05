# ruff: noqa: E501
"""Experiment 565: forecast error in the weeks after a labelled load-shift event ends, D5 against M0 (daily harness).

For each labelled event at one of the sixteen Primaries, the rows whose valid time lies in the first
14 days after the event's end, or in days 14 to 28, at that Primary, with a lead of 24 to 120 h.
The contrast is the mean over seeds 0 and 1 of D5's CRPS minus M0's. The same contrast over all other
rows of the same Primaries and folds (valid time not within 28 days after any event) is the reference.
"""

import e00510_common as c  # ty: ignore[unresolved-import]
import e00565_d5d_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    """Print the contrast after events and elsewhere."""
    t, _ = sc.build_table()
    t = t.with_columns(
        d=pl.mean_horizontal([pl.col(f"D5d_s{x}") - pl.col(f"M0_s{x}") for x in sc.SEEDS]),
    ).filter(pl.col("band").is_in(sc.BANDS) & pl.col("d").is_finite())
    ev = c.labelled_events()
    tg = t["target"].to_numpy()
    prim = t["primary"].to_numpy()
    after1 = np.zeros(len(t), bool)
    after2 = np.zeros(len(t), bool)
    in_ev = np.zeros(len(t), bool)
    day = 48  # half-hourly grid steps per day
    for p, spans in ev.items():
        sel = prim == p
        for onset, end in spans:
            after1 |= sel & (tg > end) & (tg <= end + 14 * day)
            after2 |= sel & (tg > end + 14 * day) & (tg <= end + 28 * day)
            in_ev |= sel & (tg >= onset) & (tg <= end)
    other = ~(after1 | after2 | in_ev)
    for name, mask in (
        ("days 0-14 after an event", after1),
        ("days 14-28 after an event", after2),
        ("all other rows", other),
    ):
        for grp in ("eight", "six"):
            sub = t.filter(pl.Series(mask))
            part = sc.s.row_set(table=sub, name=grp) if sub.height else sub
            if part.height:
                m = float(part["d"].mean())
                n = part.height
                days = part.select(pl.col("month"), pl.col("day")).unique().height
                print(
                    f"{name:28s} {grp:6s} rows {n:7d} day-blocks {days:4d} D5 - M0 mean CRPS {m:+.3f}"
                )


if __name__ == "__main__":
    main()
