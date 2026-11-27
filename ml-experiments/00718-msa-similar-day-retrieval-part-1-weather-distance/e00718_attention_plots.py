# ruff: noqa: E501
"""Experiments 718 and 719: the read-out weights and the across-day attention of fitted networks, from stored states (no refit).

Usage: e00718_attention_plots.py OUT_DIR PART [--month 2026-01] [--seed 0] [--device cuda]

Loads each arm's stage 3 state of the fold and seed, builds the arm's batch for one test origin at two Primaries (D5 of the six,
D8 of the eight) and lead day 1 (the day after the origin day), and uses `e00718_readouts.internals_for_case`. Writes two kinds of
chart: `attention_part<P>_<arm>_<Primary>.svg` (left, the read-out weight alpha of each history row at each half-hour of the target
day; right, the across-day attention of the last block's target-row query, averaged over heads and half-hours, per history row) and
`alpha_by_age_part<P>.svg` (the mean total read-out weight by row age over 120 random (Primary, origin) cases, the same for every arm, counting only the cases that hold a row of that age). Rows are
ordered by age, so a sorted order reads as recency. Plots only; the scores come from the contrast scripts.
"""

import argparse
import sys
from pathlib import Path

import altair as alt
import e00713_figures as f713  # ty: ignore[unresolved-import]
import e00713_layout as lay713  # ty: ignore[unresolved-import]
import e00718_arms as arms
import e00718_batch as fb
import e00718_figures as fig
import e00718_fit as fit_mod
import e00718_model as f9
import e00718_readouts as ro
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARMS = {1: ("Z", "R"), 2: ("Z", "U", "U27")}
N_CASES = 120


def load(*, arm: str, month: str, seed: int, device: str) -> tuple[f9.F9Net, fb.F9Source, object]:
    """Return the network and the batch source of `arm`, as `e00718_predict` builds them."""
    import e00713_inputs as inp  # ty: ignore[unresolved-import]

    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    expect = fit_mod.table_meta(arm=arm, month=month, fingerprint=data.fingerprint, phase="predict",
        digest=fit_mod.inputs_digest(source=data.source, cutoff_idx=data.cutoff_idx))  # fmt: skip
    tables = {
        fb.TABLE_ERA5: fit_mod.load_table(
            path=arms.table_file(
                data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, version="era5"
            ),
            expect=expect,
        ),
        fb.TABLE_ENS: fit_mod.load_table(
            path=arms.table_file(
                data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, version="ens"
            ),
            expect=expect,
        ),
    }
    source = fb.F9Source(base=data.source, tables=tables)
    model = f9.F9Net(seed=seed).to(device)
    state = arms.model_file(
        data_dir=ML_EXPERIMENTS_DATA_DIR, arm=arm, month=month, seed=seed, stage=3
    )
    model.load_state_dict(torch.load(state, map_location=device))
    return model, source, data


def case(
    *, model: f9.F9Net, source: fb.F9Source, primary: int, c_idx: int, lead_day: int
) -> dict[str, np.ndarray]:
    """Return the read-out weights, across-day attention, and present rows of one case, built as `e00718_predict.py` builds a scored sample (ensemble inputs, no target cut)."""
    dev = source.device
    one = lambda v: torch.tensor([v], dtype=torch.long, device=dev)  # noqa: E731
    batch = source.assemble(
        primary=one(primary),
        c_idx=one(c_idx),
        lead_day=one(lead_day),
        ensemble=True,
        target_end=10**9,
    )
    got = ro.extract(model=model, inputs=batch.inputs)
    last = max(int(k.split("_")[1]) for k in got if k.startswith("across_"))
    present = batch.inputs.row_exists[0].cpu().numpy().astype(bool)
    return {
        "alpha": got["alpha"][0].cpu().numpy()[present],
        "across": got[f"across_{last}"][0].cpu().numpy(),
        "present": present,
        "ages": batch.ages[0].cpu().numpy()[present],
    }


def main() -> int:
    """Write the charts."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("part", type=int, choices=[1, 2])
    parser.add_argument("--month", default="2026-01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sites = k578.sites().sort("primary")
    prim_of = dict(zip(sites["label"].to_list(), sites["primary"].to_list(), strict=True))
    rng = np.random.default_rng(0)
    c_pick = p_pick = None
    pooled = []
    for arm in ARMS[args.part]:
        model, source, data = load(arm=arm, month=args.month, seed=args.seed, device=args.device)
        c_all = data.test_origin_idx + lay713.NATIVE_OFFSET_STEPS
        if c_pick is None:
            c_pick = rng.choice(c_all, N_CASES)
            p_pick = rng.integers(0, 16, N_CASES)
        c_case = int(c_all[len(c_all) // 2])
        for label in ("D5", "D8"):
            got = case(
                model=model, source=source, primary=int(prim_of[label]), c_idx=c_case, lead_day=1
            )
            order = np.argsort(got["ages"])
            alpha, ages = got["alpha"][order], got["ages"][order]
            across = got["across"].mean(axis=(0, 1))
            across_present = np.concatenate([[True], got["present"]])
            across_kept = across[across_present]
            across_rows = np.concatenate([[across_kept[0]], across_kept[1:][order]])
            heat = pl.DataFrame(
                {
                    "age_days": np.repeat(ages, 48),
                    "half_hour": np.tile(np.arange(48), len(ages)),
                    "alpha": alpha.reshape(-1),
                }
            )
            hm = (
                alt.Chart(heat)
                .mark_rect()
                .encode(
                    x=alt.X(
                        "half_hour:O",
                        title="Half-hour of the target day (0 = 12:00 UTC)",
                        axis=alt.Axis(values=list(range(0, 48, 8))),
                    ),
                    y=alt.Y("age_days:O", title="Age of history row (days)", sort="ascending"),
                    color=alt.Color("alpha:Q", title="Read-out weight"),
                )
                .properties(
                    width=240, height=260, title="Read-out weight alpha (each column sums to 1)"
                )
            )
            bars = pl.DataFrame(
                {"row": ["target", *[str(int(a)) for a in ages]], "attention": across_rows}
            )
            bar = (
                alt.Chart(bars)
                .mark_bar(color=fig.COLOUR.get(arm, "#555"))
                .encode(
                    x=alt.X(
                        "attention:Q", title="Across-day attention (mean over heads and half-hours)"
                    ),
                    y=alt.Y(
                        "row:O",
                        title="Row (target, then history row age in days)",
                        sort=["target", *[str(int(a)) for a in ages]],
                    ),
                )
                .properties(width=240, height=260, title="Attention of the target row's query")
            )
            chart = (hm | bar).properties(
                title=f"Arm {arm}, fold {args.month}, seed {args.seed}, Primary {label}, one test origin, lead day 1"
            )
            f713.save(chart, path=args.out_dir / f"attention_part{args.part}_{arm}_{label}.svg")
        # pooled mean read-out weight by age, over the same cases for every arm
        for c, p in zip(c_pick, p_pick, strict=True):
            got = case(model=model, source=source, primary=int(p), c_idx=int(c), lead_day=1)
            mass = got["alpha"].sum(axis=1) / 48.0
            pooled += [
                {"arm": arm, "age_days": int(a), "mass": float(m)}
                for a, m in zip(got["ages"], mass, strict=True)
            ]
    df = (
        pl.DataFrame(pooled)
        .group_by("arm", "age_days")
        .agg(mass=pl.col("mass").mean(), n=pl.len())
        .sort("arm", "age_days")
    )
    chart = (
        alt.Chart(df)
        .mark_point(filled=True, size=22)
        .encode(
            x=alt.X(
                "age_days:Q", title="Age of history row (days)", scale=alt.Scale(type="symlog")
            ),
            y=alt.Y("mass:Q", title="Mean read-out weight of a row of that age"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(ARMS[args.part]), range=[fig.COLOUR[a] for a in ARMS[args.part]]
                ),
                title="Arm",
            ),
        )
        .properties(
            width=420,
            height=240,
            title=f"Read-out weight by row age, {N_CASES} random (Primary, origin) cases at lead day 1, fold {args.month}, seed {args.seed}",
        )
    )
    f713.save(chart, path=args.out_dir / f"alpha_by_age_part{args.part}.svg")
    print("wrote charts to", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
