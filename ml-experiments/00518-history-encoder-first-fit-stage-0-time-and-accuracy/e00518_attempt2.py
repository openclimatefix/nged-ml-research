"""Experiment 518, attempt 2: three seeds, a larger validation block, a validation-chosen threshold.

The maintainer-approved second attempt changes the fit of attempt 1 in six ways (all fixed before
this attempt ran): 3 seeds (0 to 2, each with its own network initialisation and training worlds);
a validation block of 130 days (attempt 1 used 30); two further validation worlds for the flag
threshold, 160 to 162, distinct from the early-stopping worlds 150 to 152 and from every training
and test world; a learning rate of 5e-4 (attempt 1: 2e-3); epochs of 32,768 windows (attempt 1:
16,384); and early stopping after 10 epochs without a 0.5% improvement (attempt 1: 8). The flag
threshold is chosen per seed on the validation block alone, as the grid value that maximises
`planted precision at 2% * recall within 2 hours`, and written to `v2_thresholds.json` before any
test window is scored. The four checks P1 to P4 of the plan are read unchanged, at the frozen
threshold, per seed, with the spread over seeds. The cap is 0.5 GPU-hours for all three fits.

```bash
ml-experiments/00518-*/run.sh ml-experiments/00518-*/e00518_attempt2.py run
```
"""

import argparse
import json
import time
from typing import Any, Final

import e00518_data as data  # ty: ignore[unresolved-import]
import e00518_fit as fit  # ty: ignore[unresolved-import]
import e00518_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

TAG: Final[str] = "v2"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
LEARNING_RATE: Final[float] = 5e-4
EPOCH_WINDOWS: Final[int] = 32768
PATIENCE: Final[int] = 10
VALIDATION_DAYS: Final[int] = 130
EARLY_STOP_WORLDS: Final[tuple[int, ...]] = (150, 151, 152)
THRESHOLD_WORLDS: Final[tuple[int, ...]] = (160, 161, 162)
THRESHOLD_GRID: Final[tuple[float, ...]] = (0.5, 0.75, 1.0, 1.5)
CAP_SECONDS: Final[float] = 0.5 * 3600.0
"""The cap for all three fits together."""
P2_RATIO: Final[float] = 0.5
P2_CLEAN: Final[float] = 0.1
P3_SHARE: Final[float] = 0.7
P4_PRECISION: Final[float] = 0.3
P1_EPOCH: Final[int] = 8


def choose_threshold(*, ctx: fit.Context, predictor: fit.Predictor) -> dict[str, Any]:
    """Return the grid threshold that maximises precision at 2% times recall, on validation only."""
    windows = ctx.split.validation
    empty = fit.empty_events(spec=ctx.spec)
    real = data.arrays_of(spec=ctx.spec, ev=empty, med=ctx.med, windows=windows)
    real_level, _ = predictor(real)
    injected = []
    for w in THRESHOLD_WORLDS:
        ev = fit.gen.sample_world(spec=ctx.spec, seed=w)
        arrays = data.arrays_of(spec=ctx.spec, ev=ev, med=ctx.med, windows=windows)
        level, _ = predictor(arrays)
        events = sc.true_events(ev=ev, p=windows.primary, d=windows.day, step_true=arrays.step_true)
        injected.append((level, events, arrays.step_true))
    table = []
    for th in THRESHOLD_GRID:
        err = np.concatenate(
            [
                sc.score_events(level=lv, events=e, step_true=st, threshold=th)[
                    "onset_error_hours"
                ].to_numpy()
                for lv, e, st in injected
            ]
        )
        recall = float((err <= 2.0).mean())
        false_rate = float(sc.flagged(level=real_level, threshold=th).mean())
        precision = sc.precision_at_rate(recall=recall, false_rate=false_rate)
        table.append(
            {"threshold": th, "recall_2h": recall, "false_flag_rate": false_rate,
             "precision_2pc": precision, "objective": precision * recall}
        )  # fmt: skip
    best = max(table, key=lambda r: float(r["objective"]))
    return {"chosen": best["threshold"], "table": table}


def p_checks(*, result: dict, best_epoch: int) -> dict[str, Any]:
    """Return the four pre-registered checks from one seed's scores at its frozen threshold."""
    a = result["all"]
    ratio = a["lag_error_repaired"] / a["lag_error_unrepaired"]
    found6 = float(a["found_6h"].split()[0])
    out = {
        "P1_best_epoch": best_epoch,
        "P1_pass": best_epoch >= P1_EPOCH,
        "P2_repaired_over_unrepaired": ratio,
        "P2_clean_rows_over_injected_step": a["clean_rows_error_over_injected_step"],
        "P2_pass": bool(ratio < P2_RATIO and a["clean_rows_error_over_injected_step"] < P2_CLEAN),
        "P3_found_6h": found6,
        "P3_ceiling_6h": a["ceiling_found_6h"],
        "P3_share_of_ceiling": found6 / a["ceiling_found_6h"],
        "P3_pass": bool(found6 / a["ceiling_found_6h"] >= P3_SHARE),
        "P4_planted_precision_2pc": a["planted_precision_at_2pc"],
        "P4_pass": bool(a["planted_precision_at_2pc"] >= P4_PRECISION),
    }
    out["all_pass"] = bool(out["P1_pass"] and out["P2_pass"] and out["P3_pass"] and out["P4_pass"])
    return out


def run(*, device: str) -> None:
    """Fit three seeds, freeze the thresholds, then score the test windows."""
    ctx = fit.build_context(validation_days=VALIDATION_DAYS)
    s = ctx.split
    print(
        f"windows: train {len(s.train):,}, validation {len(s.validation):,} "
        f"(x{len(THRESHOLD_WORLDS)} worlds), test {len(s.test):,}",
        flush=True,
    )
    out_dir = fit.EXPERIMENT_DIR
    started = time.monotonic()
    summaries = {}
    for seed in SEEDS:
        remaining = CAP_SECONDS - (time.monotonic() - started)
        summaries[seed] = fit.fit(
            ctx=ctx, device=device, out_dir=out_dir, seed=seed, lr=LEARNING_RATE,
            epoch_windows=EPOCH_WINDOWS, patience=PATIENCE, validation_worlds=EARLY_STOP_WORLDS,
            tag=TAG, limit=remaining / (len(SEEDS) - seed),
        )  # fmt: skip
        print(f"seed {seed}: {summaries[seed]}", flush=True)
    nets = {s: fit.load_net(out_dir=out_dir, device=device, tag=TAG, seed=s) for s in SEEDS}

    def predictor(seed: int) -> fit.Predictor:
        return lambda a: fit.predict(net=nets[seed], arrays=a, device=device)

    thresholds = {s: choose_threshold(ctx=ctx, predictor=predictor(s)) for s in SEEDS}
    (out_dir / "v2_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print("thresholds frozen:", {s: t["chosen"] for s, t in thresholds.items()}, flush=True)
    checks, scores = {}, {}
    for s in SEEDS:
        th = float(thresholds[s]["chosen"])
        scores[s] = fit.score(ctx=ctx, predictor=predictor(s), threshold=th)
        scores[s]["at_threshold_0.5"] = fit.score(ctx=ctx, predictor=predictor(s))["all"]
        checks[s] = p_checks(result=scores[s], best_epoch=int(summaries[s]["best_epoch"]))
        print(f"seed {s} (threshold {th}):", json.dumps(checks[s]), flush=True)
    frame = pl.DataFrame([{"seed": s, **checks[s]} for s in SEEDS])
    print(frame)
    agree = {k: frame[k].n_unique() == 1 for k in ("P1_pass", "P2_pass", "P3_pass", "P4_pass")}
    print("seeds agree on pass or fail:", agree)
    (out_dir / "v2_score.json").write_text(
        json.dumps({"checks": checks, "scores": scores, "agree": agree}, indent=2, default=str)
    )


def main() -> None:
    """Run attempt 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "counts"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "counts":
        ctx = fit.build_context(validation_days=VALIDATION_DAYS)
        s = ctx.split
        print(f"train {len(s.train):,}, validation {len(s.validation):,} windows "
              f"({len(s.validation) * len(THRESHOLD_WORLDS):,} window-world rows per threshold "
              f"world set), test {len(s.test):,}")  # fmt: skip
        return
    run(device=args.device)


if __name__ == "__main__":
    main()
