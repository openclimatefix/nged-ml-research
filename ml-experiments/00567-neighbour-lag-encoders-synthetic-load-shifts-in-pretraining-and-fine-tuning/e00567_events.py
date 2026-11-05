# ruff: noqa: E501, E402, ANN001, PLR0915, D103, N812, PLR0912
r"""Real-event leak harness (experiment 556's, as ported in experiment 566) for the neighbour arms.

For every labelled event with a refit cutoff at least `--min-age` days after its end, the arm is
fine-tuned from its pretrained weights once on the metered targets and once on targets with that
event's labelled step removed (experiment 435's correction, depth factor 1.0); the lag inputs stay as
observed (the arm's own training world, then the real lags at forecast time). The absorbed share is
the event's sign times the difference of the two median forecasts over the 7 days after the cutoff at
the event's Primary, divided by the labelled depth. Same events, cutoffs, and output columns as
experiment 566's `real_events`, so the tables join.

```bash
OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
    uv run --no-sync python ml-experiments/00567-*/e00567_events.py --arms N1-r7 N1aux-r7-w0.1
```
"""

import argparse
import sys
from datetime import datetime

_argv = sys.argv
parser = argparse.ArgumentParser()
parser.add_argument("--arms", nargs="+", required=True)
parser.add_argument("--months", nargs="+", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--min-age", type=float, default=1.0)
parser.add_argument("--limit", type=int, default=None)
ARGS = parser.parse_args(_argv[1:])
sys.argv = [_argv[0], "coarse"]

import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00567_fit as F  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

sys.argv = _argv
b = F.b
DAY_US = c.DAY_US
WINDOW_DAYS = 7


def fit_at(
    *, cell: F.Cell, arm: F.Arm, cut: datetime, seed: int, shift: np.ndarray | None
) -> torch.nn.Module:
    """Fine-tune the arm at `cut`; `shift` (capacity fraction per selected row) corrects the targets."""
    cell.apply_world(arm=arm, seed=seed)
    refit = cell.select(cut=cut)
    shift_z = (
        None if shift is None else shift / cell.mr.base.target_std[cell.mr.primary[refit.index]]
    )
    b.CTX.update(on=False, shift=None, sign=None, block=None, seed=seed, cut=cut)
    t = cell.g.refit_tensors(refit=refit, shift_z=shift_z)
    model = F.new_net(arm=arm, n_primaries=cell.n_primaries, stage="ft")
    model.table = b.CTX["table_ft"]
    b.load_pretrained(model=model, path=F.pretrained_path(arm=arm, month=cell.month, seed=seed))
    if arm.aux:
        torch.nn.init.zeros_(model.aux_head.weight)  # ty: ignore[invalid-argument-type, unresolved-attribute]
        torch.nn.init.zeros_(model.aux_head.bias)  # ty: ignore[invalid-argument-type, unresolved-attribute]
    model.to(cell.device)
    aux = (t["aux_flag"], t["aux_frac"]) if arm.aux else None
    F.train(model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=seed, lr=c.FINE_TUNE_LR,
            fresh_lr=b.MIXER_LR, device=cell.device, aux=aux, aux_weight=arm.aux_weight)  # fmt: skip
    return model


def q50(*, cell: F.Cell, model, rows: np.ndarray, x_full: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x_full[rows]).to(cell.device)
    p = torch.from_numpy(cell.test.primary[rows]).to(cell.device)
    out = []
    for a in range(0, len(x), 262_144):
        q = c.serving_quantiles(
            model=model, x=x[a : a + 262_144], p=p[a : a + 262_144], outlier=False
        )
        q = torch.sort(q, dim=1).values[:, 5]
        pp = p[a : a + 262_144]
        out.append((q * cell.sd_t[pp] + cell.mu_t[pp]).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    months = ARGS.months or list(F.MONTHS)
    events = c.labelled_events()
    grid, step = c.prep422.GRID_START_US, c.prep422.STEP_US
    shares_all = np.load(F.NRA_SHARES)["f1.0"]
    out = F.W / "scores" / "real_events.parquet"
    rows_out: list[dict] = pl.read_parquet(out).to_dicts() if out.exists() else []
    seen = {
        (r["arm"], r["month"], r["primary"], r["onset"], r["seed"], r["min_age"]) for r in rows_out
    }
    count = 0
    for month in months:
        cell = F.Cell(month=month, device="cuda")
        cuts = k.cutoffs(month=month, cadence="d")
        last_day = (cuts[-1] - c.month_start(month=month)).days
        first_cut = c.month_start(month=month)
        wanted: dict[datetime, list[tuple[int, int, int]]] = {}
        for p, spans in events.items():
            for onset, end in spans:
                end_us = grid + end * step
                if float(np.abs(shares_all[p, onset : end + 1]).max()) <= 0:
                    continue
                ok = [x for x in cuts if c.us(t=x) >= end_us + ARGS.min_age * DAY_US]
                if (
                    ok
                    and (ok[0] - first_cut).days <= last_day
                    and c.us(t=ok[0]) - end_us <= (14 + last_day) * DAY_US
                ):
                    wanted.setdefault(ok[0], []).append((p, onset, end))
        print(
            f"{month}: {sum(len(v) for v in wanted.values())} events at {len(wanted)} cutoffs",
            flush=True,
        )
        for tag in ARGS.arms:
            arm = F.make_arm(tag=tag)
            nbr_x = cell.variants(arm=arm, held={})["base"]
            for cut, evs in sorted(wanted.items()):
                lo, hi = c.us(t=cut), c.us(t=cut) + WINDOW_DAYS * DAY_US
                rows = np.nonzero((cell.test.origin >= lo) & (cell.test.origin < hi))[0]
                todo = [
                    e for e in evs if (tag, month, e[0], e[1], ARGS.seed, ARGS.min_age) not in seen
                ]
                if len(rows) == 0 or not todo:
                    continue
                model_m = fit_at(cell=cell, arm=arm, cut=cut, seed=ARGS.seed, shift=None)
                q_m = q50(cell=cell, model=model_m, rows=rows, x_full=nbr_x).astype(np.float64)
                del model_m
                prim_rows = cell.test.primary[rows]
                refit = cell.select(cut=cut)
                prim, tgt = cell.mr.primary[refit.index], cell.mr.target_idx[refit.index]
                for p, onset, end in todo:
                    end_us = grid + end * step
                    assert end_us <= c.us(t=cut) - c.CUTOFF_AGE_US, (
                        "an event ends after the refit cut minus 3.5 h"
                    )
                    corrected = -shares_all[prim, np.clip(tgt, 0, shares_all.shape[1] - 1)]
                    corrected = np.where(
                        (prim == p) & (tgt >= onset) & (tgt <= end), corrected, 0.0
                    ).astype(np.float32)
                    model_c = fit_at(cell=cell, arm=arm, cut=cut, seed=ARGS.seed, shift=corrected)
                    diff = q_m - q50(cell=cell, model=model_c, rows=rows, x_full=nbr_x).astype(
                        np.float64
                    )
                    sign = 1 if float(shares_all[p, onset : end + 1].sum()) > 0 else -1
                    depth = float(np.abs(shares_all[p, onset : end + 1]).max())
                    sel = prim_rows == p
                    rows_out.append({
                        "arm": tag, "month": month, "seed": ARGS.seed, "min_age": ARGS.min_age, "cut": str(cut.date()),
                        "primary": p, "onset": onset, "end": end, "depth_points": 100 * depth,
                        "age_days": (c.us(t=cut) - end_us) / DAY_US, "n_events_in_fit": len(evs),
                        "n_rows_corrected": int((corrected != 0).sum()),
                        "in_step2": False,
                        "absorbed_share": float(sign * diff[sel].mean() / depth) if sel.any() else None,
                    })  # fmt: skip
                    del model_c
                    count += 1
                    if count % 5 == 0:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        pl.DataFrame(
                            rows_out, schema_overrides={"absorbed_share": pl.Float64}
                        ).write_parquet(out)
                        print("events done", count, flush=True)
                    if ARGS.limit and count >= ARGS.limit:
                        break
                if ARGS.limit and count >= ARGS.limit:
                    break
        del cell
        torch.cuda.empty_cache()
        if ARGS.limit and count >= ARGS.limit:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows_out, schema_overrides={"absorbed_share": pl.Float64}).write_parquet(out)
    print("real events", len(rows_out), flush=True)


if __name__ == "__main__":
    main()
