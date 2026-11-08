# ruff: noqa: D103, E501, E402
"""Experiment 647, step 3: rebuild K1 on the injected bank and assert experiment 613's K1 scores (CPU).

The stored forecasts on the bank are raw per-seed networks, so K1 has to be rebuilt: the seed mean of
five seeds, experiment 613's tail constants (fitted on earlier months' event-free rows, so none for
2025-10), and the G4 gated band from the gate state recomputed on the injected lags. This script
runs experiment 613's own scorer (`e00613_score`, imported and not edited) with its output
redirected to
this experiment's folder, spies on the two functions that carry the rebuilt quantiles, and saves per
bank row the rebuilt K1 for the later steps. It then asserts three things:

1. the G4 band-width mask rebuilt on the real rows equals experiment 613's stored `mode_g4.npy`
   (the comparison with experiment 592's stored state is printed too, but it is `False` in
   experiment 613's own `gate_check.json` as well, because experiment 613 recomputes the gate from
   its own lag-free twin);
2. the event-free cells of K1 and K1+t (coverage, pinball losses, width, CRPS) equal experiment 613's
   stored `event_free_cells.csv`;
3. the injected-bank table (every K1 and K1+t row: coverage, CRPS, follow, safe-side shares) equals
   experiment 613's stored `bank_event_phase.csv`.

Writes `k1_bank/` in the data folder: `rows_<month>_r<pass>.npz` and `reproduction.json`.

Usage: python e00647_k1_rebuild.py [--months 2025-10 ...]
"""

import json
import sys
import time

import e00647_common as c647  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = c647.WORK / "k1_bank"
OUT.mkdir(parents=True, exist_ok=True)
STORED = (
    ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds/score/k1_d5"
)
TOL = 1e-5
extra = sys.argv[1:]
sys.argv = ["e00613_score.py", str(OUT / "scratch"), "--arm", "k1_d5", *extra]
import e00613_score as S  # ty: ignore[unresolved-import]  # noqa: N812  (parses sys.argv at import)

pending: dict = {"calls": []}
saved: list[str] = []


def flush() -> None:
    """Write the last bank pass's rebuilt rows, if a pass is pending."""
    calls = pending["calls"]
    if "rows" not in pending or len(calls) < 2:
        return
    t = pending["t"]
    rows = pending["rows"]
    q_inj, q_twin = calls[0], calls[1]
    name = f"rows_{pending['month']}_r{pending['r']:02d}.npz"
    np.savez_compressed(
        OUT / name,
        prim=t.prim[rows].astype(np.int8), origin_idx=t.origin_idx[rows].astype(np.int32),
        idx=t.idx[rows].astype(np.int32), lead=t.lead[rows].astype(np.float32),
        obs=t.obs[rows], vin=pending["vin"], k1_inj=q_inj["K1"], k1t_inj=q_inj["K1+t"],
        k1t_clean=q_twin["K1+t"], k2_inj=q_inj["K2"],
    )  # fmt: skip
    saved.append(name)


orig_load = S.s.load_injected
orig_systems = S.systems


def load_spy(*, t, month, r, rows, order, sorted_keys):  # noqa: ANN001, ANN201
    flush()
    vin = orig_load(t=t, month=month, r=r, rows=rows, order=order, sorted_keys=sorted_keys)
    ok = np.isfinite(vin).all(axis=1)
    pending.clear()
    pending.update(calls=[], rows=rows[ok], vin=vin[ok], t=t, month=month, r=r)
    return vin


def systems_spy(**kw):  # noqa: ANN003, ANN201
    out = orig_systems(**kw)
    if "rows" in pending and len(pending["calls"]) < 2:
        pending["calls"].append(out)
    return out


S.s.load_injected = load_spy
S.systems = systems_spy


def max_diff(*, new: pl.DataFrame, old: pl.DataFrame, keys: list[str], what: str) -> dict:
    """Join two tables on `keys` and return the largest absolute difference of each shared numeric column."""
    num = [
        c for c in new.columns
        if c in old.columns and c not in keys and new[c].dtype.is_float()
    ]  # fmt: skip
    j = new.join(old, on=keys, how="full", suffix="_old", coalesce=True)
    only_new = int(
        j.select(pl.col(f"{num[0]}_old").is_null() & pl.col(num[0]).is_not_null()).sum().item()
    )
    only_old = int(
        j.select(pl.col(num[0]).is_null() & pl.col(f"{num[0]}_old").is_not_null()).sum().item()
    )
    diffs = {}
    for c in num:
        d = (j[c] - j[f"{c}_old"]).abs().drop_nulls()
        diffs[c] = float(d.max()) if len(d) else 0.0  # ty: ignore[invalid-argument-type]
    worst = max(diffs.values())
    print(f"{what}: {new.height} rebuilt rows vs {old.height} stored; only in rebuild {only_new}, only in stored {only_old}; largest column difference {worst:.2e}", flush=True)  # fmt: skip
    return {"rows_new": new.height, "rows_old": old.height, "only_new": only_new, "only_old": only_old,
            "max_diff": worst, "worst_column": max(diffs, key=diffs.get)}  # ty: ignore[no-matching-overload]  # fmt: skip


def main() -> None:
    t0 = time.time()
    S.k600.WORK = S.OUT
    inp = S.k600.Inputs(arm="k1_d5")
    print(f"inputs built in {time.time() - t0:.0f}s; {inp.t.n} rows", flush=True)
    result: dict = {"months": list(S.MONTHS)}
    chk = inp.check_gate_state()
    result["gate_state_equals_592"] = chk
    print(
        "G4 state equals experiment 592's stored state (False in experiment 613's own run too):",
        chk,
        flush=True,
    )
    result["g4_mode_equals_613"] = bool(np.array_equal(inp.mode, np.load(STORED / "mode_g4.npy")))
    print(
        "G4 mask equals experiment 613's stored mode_g4.npy:",
        result["g4_mode_equals_613"],
        flush=True,
    )
    lower, upper, consts = S.row_factors(t=inp.t)
    consts.write_csv(OUT / "tail_constants.csv")
    cells, _cost, _rows = S.event_free(inp=inp, lower=lower, upper=upper)
    old_cells = pl.read_csv(STORED / "event_free_cells.csv").filter(
        pl.col("system").is_in(["K1", "K1+t", "K2", "K2+t"])
    )
    cells = cells.filter(pl.col("system").is_in(["K1", "K1+t", "K2", "K2+t"]))
    keys = ["prim", "month", "band", "system"]
    if list(S.MONTHS) != ["2025-10", "2025-12", "2026-01", "2026-03", "2026-06"]:
        old_cells = old_cells.filter(pl.col("month").is_in(list(S.MONTHS)))
    result["event_free_cells"] = max_diff(
        new=cells, old=old_cells, keys=keys, what="event-free cells (K1, K1+t, K2, K2+t)"
    )
    tab, _trig = S.score_bank(inp=inp, lower=lower, upper=upper)
    flush()
    tab.write_csv(OUT / "bank_event_phase_rebuild.csv")
    old_bank = pl.read_csv(STORED / "bank_event_phase.csv", infer_schema_length=None).filter(
        pl.col("system").is_in(["K1", "K1+t", "K2", "K2+t"]) & pl.col("month").is_in(list(S.MONTHS))
    )
    new_bank = tab.filter(pl.col("system").is_in(["K1", "K1+t", "K2", "K2+t"]))
    bkeys = ["month", "pass", "primary", "phase", "band", "system"]
    result["bank_event_phase"] = max_diff(
        new=new_bank, old=old_bank, keys=bkeys, what="injected-bank table (K1, K1+t, K2, K2+t)"
    )
    summ = S.summarise_bank(tab=tab)
    summ.write_csv(OUT / "bank_summary_rebuild.csv")
    head = summ.filter(
        (pl.col("group") == "both groups") & (pl.col("length_class") == "15 and 30 days")
        & (pl.col("phase") == "days4-end") & (pl.col("band") == S.BANDS[1]) & (pl.col("system") == "K1")
        & (pl.col("month_set") == "all")
    )  # fmt: skip
    cov = float(head["cover90"][0]) if head.height else float("nan")
    result["k1_day4_to_end_cover90_15_30d_24_120h"] = cov
    print(
        f"K1 day-4-to-end q05-q95 coverage, 15 and 30 day events, 24-120 h, both groups: {cov:.4f} (experiment 613's K1-ref gate: 0.925 +- 0.003)",
        flush=True,
    )
    result["saved_files"] = len(saved)
    result["seconds"] = round(time.time() - t0)
    (OUT / "reproduction.json").write_text(json.dumps(result, indent=1))
    assert result["g4_mode_equals_613"], (
        "the G4 mask differs from experiment 613's stored mode_g4.npy"
    )
    for k in ("event_free_cells", "bank_event_phase"):
        r = result[k]
        assert r["only_new"] == 0, (k, r)
        assert r["only_old"] == 0, (k, r)
        assert r["max_diff"] <= TOL, (k, r)
    print("K1 REBUILD PASS", flush=True)


if __name__ == "__main__":
    main()
