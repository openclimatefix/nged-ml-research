# ruff: noqa: PLR0915
"""Lead-swap test on experiment 511's stored L-1d weights (fold 2026-01, seed 0, CPU, no refit).

For each daily cutoff `d` the stored model `00511-*/work/models/L-1d/2026-01_d{d}_s0.pt` forecasts
the origins from that cutoff to the next, as in experiment 511. The same test rows are scored four
times with the `lead` and `lead_sqrt` input columns replaced (`lead` is the lead in hours divided by
336 and `lead_sqrt` is its square root, neither standardised):

- `true`: the real lead (reproduces experiment 511's stored seed-0 quantiles; checked);
- `day1`: 24 hours for every row;
- `day10`: 240 hours for every row;
- `shuffled`: the leads shuffled among the rows of each (Primary, origin) run;
- `samehour_d2`, `samehour_d10`: the lead moved to the second or the tenth day while keeping its
  hour of day (lead modulo 24 hours). Every origin is at 09:00 UTC, so lead fixes the hour of day,
  and only these two conditions change lead without changing the clock.

Quantiles are raw (unwidened). Widths are the q05 to q95 distance in points of capacity. Changes are
reported by the row's true lead band. K2-S1's lag-free heads are not covered: experiment 454 stored
those heads' forecasts, not their weights (`work/heads/*/` hold `{"head", "keys"}` forecast
tuples), so no swap is possible there.

```bash
source /tmp/e523_env.sh
systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00523-*/e00523_lead_swap.py
```
"""

from typing import Final

import e00428_data as data  # ty: ignore[unresolved-import]
import e00446_models as models446  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import e00521_refs as refs  # ty: ignore[unresolved-import]
import e00521_stats as st  # ty: ignore[unresolved-import]
import e00523_panel as panel  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

MONTH: Final[str] = "2026-01"
SEED: Final[int] = 0
CONDITIONS: Final[tuple[str, ...]] = (
    "true", "day1", "day10", "shuffled", "samehour_d2", "samehour_d10"
)  # fmt: skip
LEAD_SCALE_HOURS: Final[float] = 336.0


def lead_inputs(*, lead_hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the `lead` and `lead_sqrt` input columns for a lead in hours.

    Both columns are unstandardised (mask false in the fold's standardiser): `lead` is the lead in
    hours divided by 336 (14 days), and `lead_sqrt` is its square root. The reconstruction check in
    `main` confirms this against the stored test rows.
    """
    scaled = np.clip(lead_hours.astype(np.float64), 0, None) / LEAD_SCALE_HOURS
    return scaled.astype(np.float32), np.sqrt(scaled).astype(np.float32)


def swapped_lead(*, test: data.TestRows, condition: str, rng: np.random.Generator) -> np.ndarray:
    """Return the lead in hours each row is given under a condition."""
    lead = test.lead.astype(np.float64)
    if condition == "true":
        return lead
    if condition == "day1":
        return np.full_like(lead, 24.0)
    if condition == "day10":
        return np.full_like(lead, 240.0)
    if condition in ("samehour_d2", "samehour_d10"):
        # Every origin is at 09:00 UTC, so lead fixes the hour of day. These keep the hour of day
        # (lead modulo 24 hours) and move the lead to the second or the tenth day.
        within = np.where(lead % 24.0 == 0.0, 24.0, lead % 24.0)
        return within + (24.0 if condition == "samehour_d2" else 216.0)
    run = (test.primary.astype(np.int64) * 10_000_000) + (
        test.origin_idx.astype(np.int64) % 10_000_000
    )
    order = np.lexsort((rng.random(len(lead)), run))
    out = lead.copy()
    out[order] = lead[np.lexsort((lead, run))]
    return out


def main() -> None:
    """Run the swap and write `lead_swap.csv` and `lead_swap_check.csv`."""
    clean = c.clean_stack()
    rows = data.build_fold(refit=c.month_start(month=MONTH), clean=clean)
    test = data.build_test(month=MONTH, standardiser=rows.standardiser)
    i_lead, i_sqrt = models446.lead_columns()
    recon, recon_sqrt = lead_inputs(lead_hours=test.lead)
    check_x = float(
        max(np.abs(recon - test.x[:, i_lead]).max(), np.abs(recon_sqrt - test.x[:, i_sqrt]).max())
    )
    print(f"input reconstruction check, max abs difference on the input scale: {check_x:.2e}")
    cuts = k.cutoffs(month=MONTH, cadence="d")
    rng = np.random.default_rng(0)
    frames: dict[str, list[pl.DataFrame]] = {cond: [] for cond in CONDITIONS}
    leads = {cond: swapped_lead(test=test, condition=cond, rng=rng) for cond in CONDITIONS}
    for i, cut in enumerate(cuts):
        day = f511.day_of(month=MONTH, cut=cut)
        path = f511.model_file(arm=f511.ARMS["L-1d"], month=MONTH, day=day, seed=SEED)
        lo, hi = k.window_of(month=MONTH, cadence="d", index=i)
        sel = (test.origin >= lo) & (test.origin < hi)
        if not path.exists() or not sel.any():
            continue
        model = c.new_model(
            outlier=False, n_inputs=rows.x.shape[1], n_primaries=len(rows.target_mean)
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        for cond in CONDITIONS:
            x = test.x.copy() if cond != "true" else test.x
            if cond != "true":
                x[:, i_lead], x[:, i_sqrt] = lead_inputs(lead_hours=leads[cond])
            swapped = data.TestRows(
                x=x, primary=test.primary, origin_idx=test.origin_idx, target_idx=test.target_idx,
                lead=test.lead, origin=test.origin, valid=test.valid,
            )  # fmt: skip
            frame = c.forecast_frame(
                model=model,
                outlier=False,
                test=swapped,
                select=sel,
                rows=rows,
                widen=1.0,
                device="cpu",
            )
            frames[cond].append(
                frame.with_columns(lead_hours=pl.Series(test.lead[sel].astype(np.float64)))
            )
        print(f"d{day}: {int(sel.sum())} rows", flush=True)
    levels = list(refs.LEVELS)
    out = {cond: pl.concat(parts) for cond, parts in frames.items()}
    stored = pl.concat(
        [pl.read_parquet(refs.E511 / "seed_preds" / "L-1d" / f"{MONTH}_s{SEED}.parquet")]
    )
    joined = out["true"].join(
        stored, on=["time_series_id", "power_fcst_init_time", "valid_time"], suffix="_stored"
    )
    diff = float(
        max((joined[q] - joined[f"{q}_stored"]).abs().max() for q in levels)  # ty: ignore[invalid-argument-type, no-matching-overload]
    )
    pl.DataFrame(
        {"check": ["max abs quantile difference to experiment 511 seed 0 (fraction of capacity)",
                   "input reconstruction max abs difference", "stored rows matched", "rows scored"],
         "value": [diff, check_x, float(joined.height), float(out["true"].height)]}
    ).write_csv(panel.DATA / "scores" / "lead_swap_check.csv")  # fmt: skip
    print(f"reproduction check: max abs quantile difference {diff:.2e}")
    base = out["true"]
    records = []
    bands = base["lead_hours"]
    band_of = st.band_expr  # noqa: F841
    edges = [(name, lo, hi) for name, (lo, hi) in st.BANDS.items()]
    for cond in CONDITIONS:
        frame = out[cond]
        for name, lo, hi in (*edges, ("all", 0.0, 1e9)):
            sel = ((bands >= lo) & (bands < hi if name != "120-240h" else bands <= hi)).to_numpy()
            q_true = base.filter(pl.Series(sel))
            q_cond = frame.filter(pl.Series(sel))
            w_true = 100.0 * (q_true["q95"] - q_true["q05"])
            w_cond = 100.0 * (q_cond["q95"] - q_cond["q05"])
            shift = 100.0 * (q_cond["q50"] - q_true["q50"])
            records.append(
                {"condition": cond, "band": name, "n": int(sel.sum()),
                 "w90_true": float(w_true.mean()), "w90": float(w_cond.mean()),  # ty: ignore[invalid-argument-type]
                 "w90_ratio": float(w_cond.mean() / w_true.mean()),  # ty: ignore[unsupported-operator]
                 "median_shift_mean": float(shift.mean()),  # ty: ignore[invalid-argument-type]
                 "median_shift_mean_abs": float(shift.abs().mean())}  # ty: ignore[invalid-argument-type]
            )  # fmt: skip
    pl.DataFrame(records).write_csv(panel.DATA / "scores" / "lead_swap.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        print(pl.DataFrame(records).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
