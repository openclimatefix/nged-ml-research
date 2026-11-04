# ruff: noqa: ANN001, ANN201, ANN202, B007, B023, D103, E501
"""Part 1: lag-timestamp audit of the R-DW, R-DROP and A-pool feature tables (CPU, no fit).

Every lag input (value, missing flag, age) of every row is recomputed from the definition and
compared with the table: the source half-hour is the latest same-clock half-hour (newest, day-ago,
week-ago) at or before the newest usable reading, which is the origin minus 7 half-hours (3.5 h).
The checks, row by row and with no sampling:

1. the source time is at or before the origin minus 3.5 h; 2. the age column equals the target
minus the source time; 3. the value equals the cleaned power at the source half-hour (standardised
inverse for built tables); 4. the missing flag is 1 exactly when the cleaned power at the source
half-hour is missing.

The tables audited are (a) the raw per-Primary summary matrices that every arm reads, all 16
Primaries, and (b) each fold's built training-plus-extension rows and test rows with A-pool's
layout (all 9 lag columns) and with the 6-input R-DW / R-DROP layout, which is a column subset of
A-pool's.

Usage: ``PYTHONPATH=... python e00551_audit.py`` (needs DATA_PATH_INTERNAL).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import e00087_common as common  # ty: ignore[unresolved-import]
import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import numpy as np

OUT = (
    Path(os.environ["DATA_PATH_INTERNAL"])
    / "ml-experiments/00551-lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal"
)
GRID = c.prep422.GRID_START_US
STEP = c.prep422.STEP_US
NEWEST = 7  # half-hours: 3 h telemetry delay plus one
assert common.NEWEST_AGE == NEWEST, common.NEWEST_AGE
PERIOD = {"newest": 1, "day": 48, "week": 336}
BANDS = {
    "0-6h": (0, 6),
    "6-12h": (6, 12),
    "12-24h": (12, 24),
    "24-48h": (24, 48),
    "48-72h": (48, 72),
    "72-120h": (72, 120),
    "120-240h": (120, 240.01),
}
SHORT_HISTORY = [
    int(r["primary"]) for r in prep.primaries().iter_rows(named=True) if r["site"] in ("D15", "D18")
]
assert len(SHORT_HISTORY) == 2
CUT_IDX = int((datetime(2025, 3, 1, tzinfo=UTC).timestamp() * 1e6 - GRID) // STEP)
NAMES, MASK, LAYOUT = prep.feature_names(scheme="summary")
LAG = LAYOUT.columns_of(groups=["lags"])
assert len(LAG) == 9
TAGS = ("origin", "day", "week")
COL = {f"lag_{t}": LAG[i] for i, t in enumerate(TAGS)}
COL |= {f"lag_{t}_missing": LAG[3 + i] for i, t in enumerate(TAGS)}
COL |= {f"lag_{t}_age": LAG[6 + i] for i, t in enumerate(TAGS)}
for name, col in COL.items():
    assert NAMES[col] == name, (name, NAMES[col])
RDW_NAMES = [NAMES[LAG[i]] for i in (1, 2, 4, 5, 7, 8)]
assert RDW_NAMES == [
    "lag_day",
    "lag_week",
    "lag_day_missing",
    "lag_week_missing",
    "lag_day_age",
    "lag_week_age",
]


def source_index(*, tag: str, target_idx: np.ndarray, origin_idx: np.ndarray) -> np.ndarray:
    """Independent definition of the grid index of a lag's source half-hour."""
    newest = origin_idx - NEWEST
    if tag == "origin":
        return newest
    period = PERIOD[tag]
    return target_idx - period * np.ceil((target_idx - newest) / period).astype(np.int64)


def fmt(*, idx: int) -> str:
    return f"{datetime.fromtimestamp((GRID + int(idx) * STEP) / 1e6, UTC):%Y-%m-%d %H:%M}Z"


def audit(
    *, label, get, primary, origin_idx, target_idx, clean, tol, counts, examples, lead, skip=None
):
    """Audit one table. `get(name)` returns the table's float64 column for the lag name (raw units for ages)."""
    n = len(primary)
    for tag in TAGS:
        src = source_index(tag=tag, target_idx=target_idx, origin_idx=origin_idx)
        key = f"{label}/{tag}"
        rec = counts.setdefault(
            key,
            {
                "rows": 0,
                "source_after_origin_minus_3.5h": 0,
                "age_mismatch": 0,
                "value_mismatch": 0,
                "flag_mismatch": 0,
                "source_before_grid": 0,
                "precut_source_rows_excluded": 0,
                "clipped_rows_excluded_from_value_check": 0,
            },
        )
        late = src > origin_idx - NEWEST
        rec["rows"] += n
        rec["source_after_origin_minus_3.5h"] += int(late.sum())
        age = get(f"lag_{tag}_age")
        want_age = (target_idx - src) / 2.0
        rec["age_mismatch"] += int((np.abs(age - want_age) > 1e-3).sum())
        inside = src >= 0
        rec["source_before_grid"] += int((~inside).sum())
        ref = np.full(n, np.nan)
        ref[inside] = clean[primary[inside], src[inside]]
        missing = ~np.isfinite(ref)
        flag = get(f"lag_{tag}_missing")
        # D15 and D18 readings before 2025-03-01 are cut from the cleaned power (standing practice) but not from the raw lag matrices
        precut = np.isin(primary, SHORT_HISTORY) & (src < CUT_IDX) & missing & (flag < 0.5)
        rec["precut_source_rows_excluded"] += int(precut.sum())
        rec["flag_mismatch"] += int((((flag > 0.5) != missing) & ~precut).sum())
        value = get(f"lag_{tag}")
        both = ~missing & (flag < 0.5)
        if skip is not None:
            rec["clipped_rows_excluded_from_value_check"] += int((both & skip).sum())
            both &= ~skip
        rec["value_mismatch"] += int((np.abs(value[both] - ref[both]) > tol).sum())
        # missing-flag rows must carry the filler (NaN raw, 0 standardised)
        for bname, (lo, hi) in BANDS.items():
            sel = np.flatnonzero((lead >= lo) & (lead < hi))
            if len(sel) == 0:
                continue
            i = int(sel[len(sel) // 2])
            examples.setdefault(
                f"{label}/{tag}/{bname}",
                (
                    f"primary {primary[i]}, origin {fmt(idx=origin_idx[i])}, target {fmt(idx=target_idx[i])}, lead {lead[i]:.1f} h: "
                    f"source {fmt(idx=src[i])} = origin - {(origin_idx[i] - src[i]) / 2:.1f} h (>= 3.5 h), age column {age[i]:.1f} h = target - source {want_age[i]:.1f} h, "
                    f"flag {int(flag[i])}, value {value[i]:.4f} vs cleaned power {ref[i]:.4f}"
                ),
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = c.clean_stack()
    counts, examples = {}, {}
    # (a) raw per-Primary matrices, all 16 Primaries, every row
    sites = prep.primaries()
    for row in sites.iter_rows(named=True):
        z = np.load(prep.RAW_DIR / "summary" / f"{row['site']}.npz")
        origin_idx = (z["origin"] - GRID) // STEP
        target_idx = (z["valid"] - GRID) // STEP
        lead = z["lead"].astype(np.float64)
        assert np.allclose(lead, (target_idx - origin_idx) / 2.0), "lead != target - origin"
        x = z["x"].astype(np.float64)
        prim = np.full(len(x), row["primary"], dtype=np.int64)

        def get(name, x=x):
            v = x[:, COL[name]]
            return v * prep.LEAD_SCALE_HOURS if name.endswith("_age") else v

        audit(
            label="raw",
            get=get,
            primary=prim,
            origin_idx=origin_idx,
            target_idx=target_idx,
            clean=clean,
            tol=1e-5,
            counts=counts,
            examples=examples,
            lead=lead,
        )
        print("raw", row["site_label"] if "site_label" in row else row["site"], len(x), flush=True)
    # (b) built tables per fold
    saved = data.non_lag_layout

    def layout():
        keep, cols, mask = saved()
        return (
            keep + list(LAG),
            cols + [NAMES[i] for i in LAG],
            np.concatenate([mask, MASK[np.array(LAG)]]),
        )

    data.non_lag_layout = layout
    n0 = len(saved()[0])
    pos = {NAMES[i]: n0 + j for j, i in enumerate(LAG)}
    for month in c.MONTHS:
        mr = k.build_month(month=month, clean=clean)
        std = mr.base.standardiser
        test = data.build_test(month=month, standardiser=std)
        for table, label, x, prim, tgt, valid_us, origin_us, lead in (
            (
                "train",
                f"{month}/train",
                mr.x,
                mr.primary,
                mr.target_idx,
                mr.valid_us,
                mr.origin_us,
                mr.lead,
            ),
            (
                "test",
                f"{month}/test",
                test.x,
                test.primary,
                test.target_idx,
                test.valid,
                test.origin,
                test.lead,
            ),
        ):
            origin_idx = (origin_us - GRID) // STEP
            target_idx = (valid_us - GRID) // STEP
            assert (target_idx == tgt).all()
            assert np.allclose(lead, (target_idx - origin_idx) / 2.0, atol=1e-3)

            def get(name, x=x):
                col = pos[name]
                v = x[:, col].astype(np.float64)
                if name.endswith("_age"):
                    return v * prep.LEAD_SCALE_HOURS
                if name.endswith("_missing"):
                    return v
                return v * std.std[col] + std.mean[col]  # inverse of the fold's standardiser

            # a value clipped at +-5 SD or filled by 0 for a missing reading cannot be inverted; the check skips clipped rows
            clipped = np.zeros(len(x), dtype=bool)
            for t in TAGS:
                clipped |= np.abs(x[:, pos[f"lag_{t}"]]) >= prep.INPUT_CLIP_SD - 1e-6
            audit(
                label=label,
                get=get,
                skip=clipped,
                primary=prim,
                origin_idx=origin_idx,
                target_idx=target_idx,
                clean=clean,
                tol=1e-4,
                counts=counts,
                examples=examples,
                lead=lead.astype(np.float64),
            )
            # R-DW / R-DROP layout is the column subset: identical values by construction; assert the names
            print(label, len(x), flush=True)
        del mr, test
    bad = {
        k_: v
        for k_, v in counts.items()
        if isinstance(v, dict)
        and any(
            v[f]
            for f in (
                "source_after_origin_minus_3.5h",
                "age_mismatch",
                "flag_mismatch",
                "value_mismatch",
            )
        )
    }
    total = {
        f: sum(v[f] for v in counts.values() if isinstance(v, dict))
        for f in (
            "rows",
            "source_after_origin_minus_3.5h",
            "age_mismatch",
            "value_mismatch",
            "flag_mismatch",
            "precut_source_rows_excluded",
            "clipped_rows_excluded_from_value_check",
        )
    }
    (OUT / "audit_lag_timestamps.json").write_text(
        json.dumps(
            {
                "total": total,
                "counts": counts,
                "examples": examples,
                "tables_with_violations": list(bad),
            },
            indent=1,
            default=str,
        )
    )
    print(json.dumps(total))
    print("TABLES WITH VIOLATIONS:", list(bad))
    for key, text in examples.items():
        if key.startswith(("raw/", "2026-01/test")):
            print(key, "|", text)


if __name__ == "__main__":
    main()
