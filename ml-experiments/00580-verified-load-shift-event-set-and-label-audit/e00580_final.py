# ruff: noqa: D103, E501, FURB110, PLR0915, RUF059
"""Experiment 580, step 3: write `events_verified.parquet`, `events_labelled_status.parquet` and the README.

`events_verified.parquet` holds one row per transfer found in the telemetry alone (a mirrored pair of
sustained steps, 7 to 60 days, see `e00580_verify.scan_pairs`), whether or not a label covers it.
`events_labelled_status.parquet` holds one row per labelled event with its final class and reasons.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import e00580_verify as v  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LABEL_FIRST: Final[int] = 0


def iso(i: int) -> str:
    return (START + timedelta(minutes=30 * int(i))).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ev, labels, _, cap, series, thr, thr95 = v.load_series()
    status = pl.read_parquet(v.WORK / "events_verified_all.parquet")
    cand = pl.read_parquet(v.WORK / "candidate_pairs.parquet")
    first_label, last_label = (
        int(ev["onset_idx"].min()),  # ty: ignore[invalid-argument-type]
        int(ev["end_idx"].max()),  # ty: ignore[invalid-argument-type]
    )
    w = 3 * v.DAY
    rows = []
    for k, r in enumerate(cand.iter_rows(named=True)):
        a, b = r["primary_a"], r["primary_b"]
        p99 = all(
            abs(r[c]) >= thr[(lab, w)]
            for lab, cols in (
                (a, ("onset_step_a_share", "end_step_a_share")),
                (b, ("onset_step_b_share", "end_step_b_share")),
            )
            for c in cols
        )
        # depth: mean of the onset step and the sign-flipped end step, signed MW (positive = more load during the event)
        da = 0.5 * (r["onset_step_a_share"] - r["end_step_a_share"]) * cap[a]
        db = 0.5 * (r["onset_step_b_share"] - r["end_step_b_share"]) * cap[b]
        lab_ids = r["labelled_event_ids"]
        rows.append(
            {
                "transfer_id": k,
                "primary_a": a,
                "primary_b": b,
                "onset_a": iso(r["onset_a"]),
                "end_a": iso(r["end_a"]),
                "onset_b": iso(r["onset_b"]),
                "end_b": iso(r["end_b"]),
                "onset_a_idx": r["onset_a"],
                "end_a_idx": r["end_a"],
                "onset_b_idx": r["onset_b"],
                "end_b_idx": r["end_b"],
                "length_days": r["length_days"],
                "depth_a_mw": da,
                "depth_b_mw": db,
                "onset_step_a_share": r["onset_step_a_share"],
                "end_step_a_share": r["end_step_a_share"],
                "onset_step_b_share": r["onset_step_b_share"],
                "end_step_b_share": r["end_step_b_share"],
                "noise_p95_a_share": thr95[(a, w)],
                "noise_p95_b_share": thr95[(b, w)],
                "noise_p99_a_share": thr[(a, w)],
                "noise_p99_b_share": thr[(b, w)],
                "passes_p99_at_both": bool(p99),
                "load_mismatch_share": r["mismatch"],
                "labelled_event_ids": lab_ids,
                "in_label_period": bool(first_label <= r["onset_a"] <= last_label),
                "source": "labelled" if lab_ids else "telemetry only",
            }
        )
    tr = pl.DataFrame(rows, infer_schema_length=None)
    # Hand check against the plots (telemetry_transfers_*.svg), made by the worker on 2026-11-05.
    hand = {
        22: "clear",
        23: "clear",
        10: "plausible (D4 steps, D14 unclear)",
        17: "data artefact (D9 record starts)",
    }
    checked = set(tr.filter(pl.col("in_label_period"))["transfer_id"].to_list())
    tr = tr.with_columns(
        hand_check=pl.col("transfer_id").map_elements(
            lambda t: hand.get(
                t,
                "not confirmed (no clear step at both Primaries)"
                if t in checked
                else "not hand-checked (before the labelled period)",
            ),
            return_dtype=pl.String,
        )
    )
    tr.write_parquet(v.WORK / "events_verified.parquet")
    # final class per labelled event
    final, why, tid = [], [], []
    for r in status.iter_rows(named=True):
        match = tr.filter(pl.col("labelled_event_ids").list.contains(r["event_id"]))
        if match.height:
            m = match.row(0, named=True)
            lab_sign = np.sign(r["label_depth_mw"])
            fit = m["depth_a_mw"] if m["primary_a"] == r["label"] else m["depth_b_mw"]
            if np.sign(fit) == lab_sign:
                final.append("verified transfer")
                why.append("")
            else:
                final.append("wrong sign")
                why.append("telemetry transfer of the opposite sign at the labelled dates")
            tid.append(m["transfer_id"])
            continue
        over = tr.filter(
            ((pl.col("primary_a") == r["label"]) | (pl.col("primary_b") == r["label"]))
            & (pl.col("onset_a_idx") <= r["label_end_idx"])
            & (pl.col("end_a_idx") >= r["label_onset_idx"])
        )
        if over.height:
            m = over.row(0, named=True)
            fit = m["depth_a_mw"] if m["primary_a"] == r["label"] else m["depth_b_mw"]
            final.append("wrong dates or sign")
            why.append(
                f"the telemetry shows a transfer at this Primary from {m['onset_a'][:10]} to {m['end_a'][:10]} "
                f"with fitted depth {fit:+.2f} MW (partner {m['primary_b'] if m['primary_a'] == r['label'] else m['primary_a']}); the label says {r['label_depth_mw']:+.2f} MW from {iso(r['label_onset_idx'])[:10]}"
            )
            tid.append(m["transfer_id"])
            continue
        if r["status"] == "too short to check":
            final.append("too short to check")
        elif r["pair_status"] == "unpaired step" or r["status"] == "unpaired step":
            final.append("unpaired step (possible level change)")
        else:
            final.append("rejected")
        why.append(r["reason"] if r["reason"] else r["pair_reasons"])
        tid.append(None)
    st = status.with_columns(
        final_class=pl.Series(final),
        final_reason=pl.Series(why),
        transfer_id=pl.Series(tid, dtype=pl.Int64),
        fit_onset=pl.Series(
            [iso(i) if i >= 0 else None for i in status["fit_onset_idx"].to_list()]
        ),
        fit_end=pl.Series([iso(i) if i >= 0 else None for i in status["fit_end_idx"].to_list()]),
    )
    st.write_parquet(v.WORK / "events_labelled_status.parquet")
    print(st["final_class"].value_counts().sort("final_class"))
    long = st.filter(pl.col("label_length_days") >= 7)
    print("long", long["final_class"].value_counts().sort("final_class"))
    print(
        "transfers",
        tr.height,
        "labelled",
        tr.filter(pl.col("source") == "labelled").height,
        "p99",
        tr["passes_p99_at_both"].sum(),
    )


if __name__ == "__main__":
    main()
