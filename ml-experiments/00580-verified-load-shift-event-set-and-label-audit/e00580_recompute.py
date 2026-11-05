# ruff: noqa: D103, E501, PLR0915
"""Experiment 580, step 6: recompute what depends on labels, for the verified D5 and D16 pair.

Reads experiment 578's stored v3+D5 forecasts (seed-mean median, lead 24 to 120 h). For each
Primary of the verified transfer it prints the follow ratio against the lags-masked twin (and, at
D16, against the corrected-lag twin, whose correction used the label that matches the telemetry),
and the level share against the telemetry counterfactual. It also prints the label-based level share
of event 146 for contrast, and how many labelled (month, event) overlaps in the five fold months
belong to a verified event.
"""

import datetime as dt

import e00578_common as k  # ty: ignore[unresolved-import]
import e00578_follow2 as f2  # ty: ignore[unresolved-import]
import e00580_verify as v  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl


def main() -> None:
    ev, labels, x, cap, *_ = v.load_series()
    tr = (
        pl.read_parquet(v.WORK / "events_verified.parquet")
        .filter(pl.col("transfer_id") == 23)
        .row(0, named=True)
    )
    sm = k.sites()
    prim_of = dict(zip(sm["time_series_id"].to_list(), sm["primary"].to_list(), strict=True))
    allm = (*k.MONTHS, *f2.NEW)
    arms = {"v": "V3D5", "c": "V3D5-corrected", "r": "V3D5-removed"}
    j = None
    for n, tag in arms.items():
        fr = f2.load(root=f2._Root(), tag=tag, months=allm).rename({"q50": n})
        j = fr if j is None else j.join(fr, on=f2.KEYS)
    assert j is not None
    j = j.with_columns(
        lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours(),
        prim=pl.col("time_series_id").replace_strict(prim_of),
        idx=((pl.col("valid_time").dt.epoch("us") - k.prep.GRID_START_US) // k.prep.STEP_US).cast(
            pl.Int64
        ),
    ).filter((pl.col("lead") >= 24) & (pl.col("lead") < 120))
    # primary index in the site map's numbering equals the row order of `labels`
    assert [r["label"] for r in sm.iter_rows(named=True)] == labels
    rows = []
    for lab, depth_mw, on, en in (
        ("D5", tr["depth_a_mw"], tr["onset_a_idx"], tr["end_a_idx"]),
        ("D16", tr["depth_b_mw"], tr["onset_b_idx"], tr["end_b_idx"]),
    ):
        p = labels.index(lab)
        d = depth_mw / cap[lab]
        sign = np.sign(d)
        sub = j.filter(pl.col("prim") == p)
        obs = x[p][sub["idx"].to_numpy()]
        cf = obs - np.where((sub["idx"].to_numpy() >= on) & (sub["idx"].to_numpy() < en), d, 0.0)
        sub = sub.with_columns(obs=pl.Series(obs), cf=pl.Series(cf)).filter(
            pl.col("obs").is_finite()
        )
        idx = sub["idx"].to_numpy()
        for phase, lo, hi in (
            ("whole event", on, en),
            ("after day 4", on + 4 * 48, en),
            ("7 days after the end", en, en + 7 * 48),
        ):
            m = (idx >= lo) & (idx < hi)
            if m.sum() < 100:
                continue
            for ref in ("r", "c"):
                if ref == "c" and lab != "D16":
                    continue
                follow = float(
                    np.mean(sign * (sub["v"].to_numpy()[m] - sub[ref].to_numpy()[m]) / abs(d))
                )
                rows.append(
                    {
                        "primary": lab,
                        "phase": phase,
                        "reference": {"r": "lags-masked twin", "c": "corrected-lag twin"}[ref],
                        "follow_ratio": follow,
                        "n_rows": int(m.sum()),
                    }
                )
        m = (idx >= on) & (idx < en)
        level = float(
            np.sum(sub["v"].to_numpy()[m] - sub["cf"].to_numpy()[m])
            / np.sum(sub["obs"].to_numpy()[m] - sub["cf"].to_numpy()[m])
        )
        rows.append(
            {
                "primary": lab,
                "phase": "whole event",
                "reference": "level share (telemetry counterfactual)",
                "follow_ratio": level,
                "n_rows": int(m.sum()),
            }
        )
    # label-based level share of event 146 as experiment 578 computed it (corrected = observed minus label)
    e146 = ev.filter(pl.col("event_id") == 146).row(0, named=True)
    p = labels.index("D5")
    sub = j.filter(pl.col("prim") == p)
    idx = sub["idx"].to_numpy()
    obs = x[p][idx]
    m = (idx >= e146["onset_idx"]) & (idx <= e146["end_idx"]) & np.isfinite(obs)
    cf = obs - e146["depth_mw"] / cap["D5"]
    level146 = float(np.sum(sub["v"].to_numpy()[m] - cf[m]) / np.sum(obs[m] - cf[m]))
    rows.append(
        {
            "primary": "D5",
            "phase": "label span of event 146",
            "reference": "level share (label counterfactual, as in 578)",
            "follow_ratio": level146,
            "n_rows": int(m.sum()),
        }
    )
    out = pl.DataFrame(rows)
    out.write_csv(v.WORK / "recompute_verified_pair.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    print(out)
    # how many (month, event) pairs of 565/566 belong to a verified event
    first = dict.fromkeys((0,), k.MONTHS)[0]
    del first
    n_pairs, n_ver = 0, 0
    for month in k.MONTHS:
        y, mo = int(month[:4]), int(month[5:])
        lo = int((dt.datetime(y, mo, 1, tzinfo=dt.UTC) - k.GRID_START).total_seconds() // 1800)
        hi = int(
            (
                dt.datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=dt.UTC) - k.GRID_START
            ).total_seconds()
            // 1800
        )
        hit = ev.filter((pl.col("onset_idx") < hi) & (pl.col("end_idx") >= lo))
        n_pairs += hit.height
        n_ver += hit.filter(pl.col("event_id") == 55).height
    print(
        "labelled (month, event) overlaps in the five fold months:",
        n_pairs,
        "of which verified (event 55):",
        n_ver,
    )


if __name__ == "__main__":
    main()
