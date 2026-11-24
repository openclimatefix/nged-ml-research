# ruff: noqa: E501, D103, PT018, PERF401
"""Experiment 713: attention maps of the trained `M*`, `E`, and `P` networks, and their figures.

Usage:
    e00713_attention.py extract ARM MONTH SEED [--device cuda] [--random-init]
    e00713_attention.py figures OUT_DIR MONTH SEED [--arms M* E P]

`extract` loads the saved stage 3 weights of ARM (`--random-init` instead builds an untrained network
with non-zero attention, for a smoke test of the code path; its output is labelled and never
published), runs the network with `return_attention=True` on chosen cases and on a sample of all test
origins, and writes `work/attention/<arm>_<month>_s<seed>.npz`. Nothing is refitted and the
forecast is the same as without the flag.

**Cases.** A D8 case is the test origin whose lead-day-1 actual has the largest half-hour step. A
clear day and a dull day are the test origins with the highest and the lowest mean ERA5 shortwave
over the target day at D5. Cases are chosen from the weather and the actuals, so every arm is read on
the same samples.

**Stored per case.** `across` (the last block's weights from each target-row token to the 14 history
rows, per head: 48 x 4 x 14), `within_top` (the last block's within-day weights of the target row,
4 x 48 x 48), `alpha` (the read-out weights, 14 x 48), the history grid indices, and the target-row
vector. **Stored in the summary** (every Primary, every test origin, lead days 1 and 7): the mean
across-day weight per history slot, split by whether the row has the target's weekday.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00713_arms as arms
import e00713_bank as bank
import e00713_figures as figs
import e00713_inputs as inp
import e00713_layout as lay
import e00713_model as mdl
import e00713_train as tr
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SUMMARY_LEADS: Final = (0, 6)
"""Lead days 1 and 7 as zero-based indices."""
BATCH: Final = 256


def attention_file(*, arm: str, month: str, seed: int) -> Path:
    return (
        arms.root(data_dir=ML_EXPERIMENTS_DATA_DIR)
        / "work"
        / "attention"
        / f"{arm}_{month}_s{seed}.npz"
    )


def pick_cases(*, data: inp.FoldData) -> dict[str, tuple[int, int, int]]:
    """Return the named cases as (Primary, contract-origin index, zero-based lead day)."""
    import e00578_common as k578  # ty: ignore[unresolved-import]

    assert data.test_origin_idx is not None
    sites = k578.sites()
    d8 = int(sites.filter(pl.col("label") == "D8")["primary"][0])
    d5 = int(sites.filter(pl.col("label") == "D5")["primary"][0])
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    steps = []
    for c in c_all:
        day = data.clean[d8, c : c + lay.STEPS_PER_DAY]
        steps.append(np.nanmax(np.abs(np.diff(day))) if np.isfinite(day).sum() > 40 else -1.0)
    short = np.array(
        [
            float(data.source.dense_weather[d5, c : c + lay.STEPS_PER_DAY, 0].float().mean())
            for c in c_all
        ]
    )
    return {
        "d8_steps": (d8, int(c_all[int(np.argmax(steps))]), 0),
        "clear_day": (d5, int(c_all[int(np.argmax(short))]), 0),
        "dull_day": (d5, int(c_all[int(np.argmin(short))]), 0),
    }


def run_model(
    *,
    model: mdl.MsaNet,
    data: inp.FoldData,
    spec: arms.ArmSpec,
    primary: np.ndarray,
    c: np.ndarray,
    lead: int,
    pool: np.ndarray | None,
) -> tuple[mdl.MsaOutput, object]:
    samples = tr.SampleSet(primary, c, np.zeros(len(c), bool))
    if spec.placebo:
        samples.ages = tr.placebo_ages_for(primary=primary, c_idx=c, pool=pool)  # ty: ignore[invalid-argument-type]
    sel = np.arange(len(c))
    b = tr.batch_for(
        source=data.source,
        samples=samples,
        index=sel,
        lead=np.full(len(c), lead),
        ensemble=True,
        cut=10**9,
        dropout=None,
    )
    with torch.no_grad():
        return model(b.inputs, return_attention=True), b


def cmd_extract(*, arm: str, month: str, seed: int, device: str, random_init: bool) -> None:
    path = attention_file(arm=arm, month=month, seed=seed)
    if random_init:
        path = path.with_name(path.stem + "_RANDOM_INIT_SMOKE_TEST.npz")
    if path.exists():
        return
    spec = arms.resolve(arm=arm, data_dir=ML_EXPERIMENTS_DATA_DIR)
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device=device)
    if random_init:
        torch.manual_seed(0)
        model = mdl.MsaNet(switches=spec.switches).to(device).eval()
        for p in model.parameters():
            if float(p.detach().abs().sum()) == 0.0:
                torch.nn.init.normal_(p, std=0.05)
    else:
        model = bank.load_model(
            spec=spec,
            arm=arm,
            month=month,
            seed=seed,
            device=device,
            data_dir=ML_EXPERIMENTS_DATA_DIR,
        )
    pool = tr.block_pool(clean=data.clean)
    out: dict[str, np.ndarray] = {}
    for name, (p, c, lead) in pick_cases(data=data).items():
        res, b = run_model(
            model=model,
            data=data,
            spec=spec,
            primary=np.array([p]),
            c=np.array([c]),
            lead=lead,
            pool=pool,
        )
        assert res.attention is not None and res.target_vector is not None
        last = max(int(k.split("_")[1]) for k in res.attention)
        out[f"{name}/across"] = res.attention[f"across_{last}"][0].cpu().numpy()
        if f"within_{last}" in res.attention:
            out[f"{name}/within_top"] = res.attention[f"within_{last}"][0, 0].cpu().numpy()
        out[f"{name}/alpha"] = res.alpha[0].cpu().numpy()
        out[f"{name}/hist_idx"] = b.hist_idx[0, :, 0].cpu().numpy()  # ty: ignore[unresolved-attribute]
        out[f"{name}/target_idx"] = b.target_idx[0, 0].cpu().numpy()  # ty: ignore[unresolved-attribute]
        out[f"{name}/meta"] = np.array([p, c, lead])
    assert data.test_origin_idx is not None
    c_all = data.test_origin_idx + lay.NATIVE_OFFSET_STEPS
    for lead in SUMMARY_LEADS:
        sums = np.zeros(15)
        same = np.zeros(15)
        count = 0
        for p in range(16):
            for start in range(0, len(c_all), BATCH):
                c = c_all[start : start + BATCH]
                res, b = run_model(
                    model=model,
                    data=data,
                    spec=spec,
                    primary=np.full(len(c), p),
                    c=c,
                    lead=lead,
                    pool=pool,
                )
                assert res.attention is not None
                w = (
                    res.attention[f"across_{max(int(k.split('_')[1]) for k in res.attention)}"]
                    .mean(dim=(1, 2))
                    .cpu()
                    .numpy()
                )  # (B, 15)
                weekday = (
                    ((b.target_idx[:, 0:1] - b.hist_idx[:, :, 0]) // lay.STEPS_PER_DAY % 7 == 0)  # ty: ignore[unresolved-attribute]
                    .cpu()
                    .numpy()
                )  # (B, 14)
                sums += w.sum(axis=0)
                same[1:] += (w[:, 1:] * weekday).sum(axis=0)
                count += len(c)
        out[f"summary_lead{lead + 1}/mean_weight"] = sums / count
        out[f"summary_lead{lead + 1}/same_weekday_weight"] = same / count
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **out)  # ty: ignore[invalid-argument-type]
    tmp.rename(path)
    print("stored", path, flush=True)


def heat_chart(
    *, frames: dict[str, np.ndarray], title: list[str], rows: str, cols: str
) -> alt.Chart:
    parts = []
    for name, m in frames.items():
        r, c = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
        parts.append(
            pl.DataFrame(
                {"arm": name, "row": r.ravel() + 1, "col": c.ravel() / 2.0, "weight": m.ravel()}
            )
        )
    data = pl.concat(parts)
    return (
        alt.Chart(data, width=WIDTH_HEAT, height=140)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X(
                "col:O",
                title=cols,
                axis=alt.Axis(values=[0, 6, 12, 18], labelAngle=0),
            ),
            y=alt.Y("row:O", title=rows, sort="ascending"),
            color=alt.Color(
                "weight:Q", title="Weight", scale=alt.Scale(range=["#FFFFFF", theme.BRAND_ORANGE])
            ),
            row=alt.Row("arm:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(title=alt.TitleParams(title, anchor="start", fontSize=12, subtitleFontSize=11))
    )


WIDTH_HEAT: Final = 520


def cmd_figures(*, out_dir: Path, shown: list[str], month: str, seed: int) -> None:
    star = arms.m_star(data_dir=ML_EXPERIMENTS_DATA_DIR)
    loaded = {
        s: np.load(attention_file(arm=figs.arm_name(shown=s, star=star), month=month, seed=seed))
        for s in shown
    }
    for case in ("d8_steps", "clear_day", "dull_day"):
        across = {
            s: z[f"{case}/across"].mean(axis=1).T for s, z in loaded.items()
        }  # (15, 48): rows 0..14 -> keep history 1..14
        frames = {s: m[1:] for s, m in across.items()}
        title = [
            f"Across-day attention of the target row, case {case.replace('_', ' ')}",
            "Each cell: the weight from the target row's token (x: its half-hour) to the history row (y: slot 1 is the nearest history row; for M* and E slot k is the row k+1 days old, for P the slots hold placebo rows 16 to 379 days old). Mean over 4 heads, last block.",
        ]
        figs.save(
            heat_chart(
                frames=frames,
                title=title,
                rows="History row (slot)",
                cols="Target half-hour (hours after 12:00 UTC on the origin day)",
            ),
            path=out_dir / f"attention_across_{case}.svg",
        )
    for lead in (1, 7):
        records = []
        for s, z in loaded.items():
            mean, same = (
                z[f"summary_lead{lead}/mean_weight"],
                z[f"summary_lead{lead}/same_weekday_weight"],
            )
            for slot in range(1, 15):
                records.append(
                    {
                        "arm": s,
                        "slot": slot,
                        "weight": float(mean[slot]),
                        "same_weekday": float(same[slot]),
                    }
                )
        data = pl.DataFrame(records)
        dom = list(loaded)
        colour = alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=dom, range=[figs.COLOUR.get(a, theme.ENSEMBLE_LINE) for a in dom]
            ),
            legend=alt.Legend(title=None, orient="top"),
        )
        x = alt.X(
            "slot:O",
            title="History slot (M* and E: slot k is the row k+1 days old; P: the k-th placebo row, 16 to 379 days old)",
        )
        base = alt.Chart(data, width=figs.WIDTH, height=150)
        chart = alt.vconcat(
            base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
                x=x,
                xOffset="arm:N",
                y=alt.Y("weight:Q", title="Mean attention weight"),
                color=colour,
            ),
            base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
                x=x,
                xOffset="arm:N",
                y=alt.Y("same_weekday:Q", title="Weight on a row with the target's weekday"),
                color=colour,
            ),
        ).properties(
            title=alt.TitleParams(
                [
                    f"Mean across-day attention by history age, lead day {lead}",
                    "Top: all rows. Bottom: only the weight that falls on a row with the same weekday as the target. Mean over Primaries, test origins, half-hours, and heads.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
        figs.save(chart, path=out_dir / f"attention_by_age_lead{lead}.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("arm")
    e.add_argument("month")
    e.add_argument("seed", type=int)
    e.add_argument("--device", default="cuda")
    e.add_argument("--random-init", action="store_true")
    f = sub.add_parser("figures")
    f.add_argument("out_dir", type=Path)
    f.add_argument("--arms", nargs="+", default=["M*", "E", "P"])
    f.add_argument("month")
    f.add_argument("seed", type=int)
    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(
            arm=args.arm,
            month=args.month,
            seed=args.seed,
            device=args.device,
            random_init=args.random_init,
        )
    else:
        cmd_figures(out_dir=args.out_dir, shown=args.arms, month=args.month, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
