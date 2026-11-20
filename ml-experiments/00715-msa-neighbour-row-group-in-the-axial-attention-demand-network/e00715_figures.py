# ruff: noqa: E501, D103, PT018
"""Experiment 715: the figures of design section 10, drawn from stored forecasts and stored network states.

Usage:
    e00715_figures.py demo OUT_DIR                 # render every chart type from random numbers, to check the code (not a result)
    e00715_figures.py attention MONTH SEED OUT_DIR # target row to neighbour rows (F4 beside F4-R) and the null-key share
    e00715_figures.py contrasts OUT_DIR            # N1 and N2 by Primary and by fold
    e00715_figures.py events OUT_DIR TAG           # at least 12 logged events: host power, forecasts, neighbours' power
    e00715_figures.py follow OUT_DIR               # follow share after permanent changes against after temporary moves

Every chart is written as SVG and optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.
**Anonymisation.** The charts show Primary substations by their D labels, which the rule on generators
does not cover, and count days from the event's start rather than showing a calendar date. No chart holds
the name, the identifier, or the time series of a metered generator. Colour is by arm and follows
`plotting.ocf_theme`: Candidate `C` Data Blue, `M*` Brand Orange, `F4` Data Green, `F4-R` Data Purple,
`F4-W` Data Sky.

**The attention maps show a product of attention weights, not influence.** The target row does not attend
to a neighbour directly. The map multiplies, for each age and column, the weight of the target token on
that host history row (the across-day attention of block 2, heads averaged) by the weight of that history
token on each neighbour slot (the across-substation attention of block 2, heads averaged), then averages
over the 48 columns and over the samples.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import altair as alt
import e00715_neighbours as nb
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from plotting import ocf_theme as theme

WIDTH: Final = 640
COLOUR: Final = {
    "C": theme.DATA_BLUE,
    "M*": theme.BRAND_ORANGE,
    "F4": theme.DATA_GREEN,
    "F4-R": theme.DATA_PURPLE,
    "F4-W": theme.DATA_SKY,
}
ARMS_SHOWN: Final = ("C", "M*", "F4", "F4-R")
AGES: Final = tuple(range(2, 16))
SLOTS: Final = ("slot 1", "slot 2", "slot 3", "slot 4", "slot 5", "slot 6", "null key")
MIN_EVENTS: Final = 12


def save(chart: alt.TopLevelMixin, *, path: Path) -> None:
    """Write `chart` as SVG and optimise it with svgo, as `CLAUDE.md` requires."""
    path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(path)
    if shutil.which("npx") is None:
        print("npx is missing: run svgo by hand on", path)
        return
    subprocess.run(
        [
            "npx",
            "svgo@4",
            "--multipass",
            "--precision=1",
            "--final-newline",
            str(path),
            "-o",
            str(path),
        ],
        check=True,
    )
    print("wrote", path, f"({path.stat().st_size // 1024} KB)")


def target_to_slot_map(*, across: torch.Tensor, nbr: torch.Tensor) -> np.ndarray:
    """Return the (samples, 14 ages, 7) product of the target row's weights on the history rows and the history tokens' weights on the slots and the null key.

    Args:
        across: (B, 48, heads, 15) the target token's across-day weights over the target row (index 0) and the 14 history rows.
        nbr: (B, 14, 48, heads, 7) the across-substation weights of the same block.
    """
    a = across.mean(dim=2)[:, :, 1:]  # (B, 48, 14)
    w = nbr.mean(dim=3).permute(0, 2, 1, 3)  # (B, 48, 14, 7)
    return (a[..., None] * w).mean(dim=1).cpu().numpy()  # (B, 14, 7)


def map_frame(*, maps: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the long table of mean slot-by-age weights, one block per arm."""
    rows = []
    for arm, m in maps.items():
        mean = m.mean(axis=0)
        rows += [
            {"arm": arm, "age": AGES[a], "slot": SLOTS[s], "weight": float(mean[a, s])}
            for a in range(14)
            for s in range(7)
        ]
    return pl.DataFrame(rows)


def attention_chart(*, maps: dict[str, np.ndarray], title: str) -> alt.Chart:
    data = map_frame(maps=maps)
    return (
        alt.Chart(data, width=260, height=260)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("slot:N", sort=list(SLOTS), title="Neighbour slot"),
            y=alt.Y("age:O", sort="descending", title="Age of the host history row (days)"),
            color=alt.Color(
                "weight:Q", scale=alt.Scale(scheme="oranges"), title="Product of weights"
            ),
            column=alt.Column("arm:N", title=None, sort=list(maps)),
        )
        .properties(
            title=alt.TitleParams(
                [
                    title,
                    "Product of attention weights from the host's target row, not a measure of influence.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
    )


def null_share_chart(*, table: pl.DataFrame) -> alt.Chart:
    """A bar of the null key's share of across-substation weight by Primary, with a panel per arm."""
    dom = sorted(table["arm"].unique().to_list())
    return (
        alt.Chart(table, width=300, height=220)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("label:N", sort=list(nb.LABELS), title="Primary"),
            y=alt.Y(
                "null_share:Q",
                title="Share of weight on the null key",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=dom, range=[COLOUR.get(a, theme.GREY_2) for a in dom]),
                legend=None,
            ),
            column=alt.Column("arm:N", title=None),
        )
        .properties(
            title=alt.TitleParams(
                [
                    "Share of the across-substation weight on the null key",
                    "A share near 1 means the neural network ignores its neighbours.",
                ],
                anchor="start",
                fontSize=12,
                subtitleFontSize=11,
            )
        )
    )


def contrast_chart(
    *, table: pl.DataFrame, by: str, title: str, neighbours: pl.DataFrame | None = None
) -> alt.Chart:
    """Mean difference and interval per `by` for the arms in `table` (columns `by`, `contrast`, `mean`, `lo`, `hi`)."""
    order = sorted(table[by].unique().to_list())
    dom = sorted(table["contrast"].unique().to_list())
    colour = alt.Color(
        "contrast:N",
        scale=alt.Scale(domain=dom, range=[theme.DATA_GREEN, theme.DATA_PURPLE][: len(dom)]),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y(f"{by}:N", sort=order, title=title)
    base = alt.Chart(table, width=WIDTH, height=max(160, 24 * len(order)))
    layered = alt.layer(
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X("lo:Q", title="CRPS difference (points of capacity; negative favours F4)"),
            x2="hi:Q",
            color=colour,
            yOffset="contrast:N",
        ),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=45).encode(  # ty: ignore[unresolved-attribute]
            y=y, x="mean:Q", color=colour, yOffset="contrast:N"
        ),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color=theme.GREY_1).encode(x="z:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        title=alt.TitleParams(
            [
                f"CRPS difference by {title.lower()}",
                "Dot: mean. Line: 95% interval from resampling whole months. N1 is F4 minus M*, N2 is F4 minus F4-R.",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        )
    )
    if neighbours is None:
        return layered  # ty: ignore[invalid-return-type]
    count = (
        alt.Chart(neighbours, width=80, height=max(160, 24 * len(order)))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=theme.GREY_2)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=None),
            x=alt.X("n_neighbours:Q", title="Neighbours"),
        )
    )  # ty: ignore[unresolved-attribute]
    return alt.hconcat(layered, count)  # ty: ignore[invalid-return-type]


def event_chart(
    *, series: pl.DataFrame, neighbours: pl.DataFrame, start_day: float, end_day: float, title: str
) -> alt.VConcatChart:
    """Two panels over days from the event's start: the host's power and forecasts, and the neighbours' power on the host's scale."""
    x = alt.X("day:Q", title="Days from the start of the event")
    shade = (
        alt.Chart(pl.DataFrame({"a": [start_day], "b": [end_day]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=theme.GREY_1, opacity=0.2)
        .encode(x="a:Q", x2="b:Q")
    )  # ty: ignore[unresolved-attribute]
    dom = [a for a in ("C", "M*", "F4", "F4-R") if a in series["arm"].unique().to_list()]
    lines = (
        alt.Chart(series.filter(pl.col("arm") != "actual"), width=WIDTH, height=190)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.5, clip=True)
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Host power (% of capacity)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=dom, range=[COLOUR[a] for a in dom]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    actual = (
        alt.Chart(series.filter(pl.col("arm") == "actual"), width=WIDTH, height=190)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#222222", strokeDash=[4, 3], clip=True)
        .encode(x=x, y="value:Q")
    )  # ty: ignore[unresolved-attribute]
    top = alt.layer(shade, lines, actual).properties(
        title=alt.TitleParams(
            [
                title,
                "Dashed black: the host's actual power. Coloured: the forecast of each arm. Shaded: the logged event.",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        )
    )
    bottom = (
        alt.Chart(neighbours, width=WIDTH, height=110)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2, clip=True)
        .encode(
            x=x,
            y=alt.Y(
                "value:Q", title="Neighbours' power (host's scale)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color("neighbour:N", legend=alt.Legend(title=None, orient="bottom")),
        )
    )
    return alt.vconcat(top, alt.layer(shade, bottom))


def follow_chart(*, table: pl.DataFrame, ceiling: pl.DataFrame) -> alt.Chart:
    """Follow share after temporary moves (x) against after permanent changes (y): one point per arm, the ideal point, and the ceiling classifiers."""
    pts = (
        alt.Chart(table, width=420, height=360)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "temporary:Q",
                title="Follow share after temporary moves",
                scale=alt.Scale(domain=[-0.2, 1.2]),
            ),
            y=alt.Y(
                "permanent:Q",
                title="Follow share after permanent changes",
                scale=alt.Scale(domain=[-0.2, 1.2]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(COLOUR), range=list(COLOUR.values())),
                legend=alt.Legend(title=None),
            ),
        )
    )
    ideal = (
        alt.Chart(pl.DataFrame({"temporary": [0.0], "permanent": [1.0], "arm": ["ideal"]}))  # ty: ignore[unresolved-attribute]
        .mark_point(shape="diamond", size=140, color="#222222", filled=False)
        .encode(x="temporary:Q", y="permanent:Q")
    )  # ty: ignore[unresolved-attribute]
    ref = (
        alt.Chart(ceiling)  # ty: ignore[unresolved-attribute]
        .mark_point(shape="cross", size=80, color=theme.GREY_2)
        .encode(x="temporary:Q", y="permanent:Q", tooltip="classifier:N")
    )  # ty: ignore[unresolved-attribute]
    return alt.layer(pts, ideal, ref).properties(  # ty: ignore[invalid-return-type]
        title=alt.TitleParams(
            [
                "Follow share: permanent changes against temporary moves",
                "Diamond: the ideal point (1 after a permanent change, 0 after a temporary move). Crosses: classifiers of experiment 714 (gap read as follow rates).",
            ],
            anchor="start",
            fontSize=12,
            subtitleFontSize=11,
        )
    )


def cmd_demo(*, out_dir: Path) -> None:
    """Render every chart type from random numbers; the output checks the plotting code, not a result."""
    import e00715_check_model as cm
    import e00715_model as f4

    torch.manual_seed(0)
    model = f4.F4Net(switches=cm.mdl.ARM_SWITCHES["M"])
    cm.randomise_output_projection(model=model)
    model.eval()
    inp, nbr = cm.random_inputs(b=6), cm.random_neighbours(b=6)
    out = model(inp, nbr, return_attention=True)
    assert out.attention is not None and out.nbr_attention is not None
    maps = {
        "F4": target_to_slot_map(across=out.attention["across_1"], nbr=out.nbr_attention[1]),
        "F4-R": target_to_slot_map(across=out.attention["across_1"], nbr=out.nbr_attention[1]),
    }
    assert (
        np.allclose(maps["F4"].sum(axis=(1, 2)), 1.0, atol=1e-3)
        or maps["F4"].sum(axis=(1, 2)).max() <= 1.0 + 1e-3
    )
    save(
        attention_chart(maps=maps, title="Demonstration with random weights"),
        path=out_dir / "demo_attention.svg",
    )
    rng = np.random.default_rng(0)
    null = pl.DataFrame(
        [
            {"arm": a, "label": lab, "null_share": float(rng.uniform(0.6, 1.0))}
            for a in ("F4", "F4-R")
            for lab in nb.LABELS
        ]
    )
    save(null_share_chart(table=null), path=out_dir / "demo_null_share.svg")
    rows = [
        {"label": lab, "contrast": c, "mean": float(rng.normal(0, 0.05)), "lo": -0.1, "hi": 0.1}
        for lab in nb.LABELS
        for c in ("N1", "N2")
    ]
    count = pl.DataFrame(
        {
            "label": list(nb.LABELS),
            "n_neighbours": [len(nb.load_table().neighbours[i]) for i in range(nb.N_PRIMARIES)],
        }
    )
    save(
        contrast_chart(table=pl.DataFrame(rows), by="label", title="Primary", neighbours=count),
        path=out_dir / "demo_contrast_primary.svg",
    )
    days = np.linspace(-14, 14, 200)
    series = pl.concat(
        [
            pl.DataFrame(
                {"arm": a, "day": days, "value": 50 + 5 * np.sin(days / 3) + rng.normal(0, 1, 200)}
            )
            for a in ("actual", "C", "M*", "F4", "F4-R")
        ]
    )
    nbrs = pl.concat(
        [
            pl.DataFrame({"neighbour": f"N{k}", "day": days, "value": 50 + rng.normal(0, 2, 200)})
            for k in range(3)
        ]
    )
    save(
        event_chart(
            series=series, neighbours=nbrs, start_day=0, end_day=7, title="Demonstration, host D1"
        ),
        path=out_dir / "demo_event.svg",
    )
    table = pl.DataFrame(
        {"arm": ["M*", "F4", "F4-R"], "temporary": [0.4, 0.35, 0.4], "permanent": [0.6, 0.65, 0.6]}
    )
    ceiling = pl.DataFrame(
        {"classifier": ["host", "host_nbr"], "temporary": [0.3, 0.2], "permanent": [0.5, 0.52]}
    )
    save(follow_chart(table=table, ceiling=ceiling), path=out_dir / "demo_follow.svg")


def cmd_attention(*, month: str, seed: int, out_dir: Path) -> None:
    """Attention maps of F4 beside F4-R on the real fold's origins, per Primary, from the stored stage 3 states."""
    import e00713_inputs as inp  # ty: ignore[unresolved-import]
    import e00715_arms as arms
    import e00715_predict as pr
    import e00715_train as tr
    from studies.sources import ML_EXPERIMENTS_DATA_DIR

    arms.install_own_cache()
    pipe = inp.load_pipeline(
        month=month, phase="predict", chain_root=inp.chain_root_of_seed(seed=seed)
    )
    data = inp.build_fold_data(pipe=pipe, device="cpu")
    assert data.test_origin_idx is not None
    c = data.test_origin_idx + pr.lay.NATIVE_OFFSET_STEPS
    maps: dict[str, np.ndarray] = {}
    null_rows = []
    table = nb.load_table()
    for arm in ("F4", "F4-R"):
        model, _, spec = pr.load_network(
            arm=arm,
            month=month,
            seed=seed,
            device="cpu",
            data_dir=ML_EXPERIMENTS_DATA_DIR,
            accept=[],
            record={},
        )
        assert spec is not None
        src = arms.f4_source(data=data, table=table, mode=spec.slots, rewire_seed=seed)
        per_primary = []
        for p in range(nb.N_PRIMARIES):
            samples = tr.F4Samples(
                primary=np.full(len(c), p),  # ty: ignore[unknown-argument]
                c_idx=c,  # ty: ignore[unknown-argument]
                is_val=np.zeros(len(c), bool),  # ty: ignore[unknown-argument]
            )
            idx = np.arange(min(len(c), 24))
            b = tr.batch_for(
                source=src,
                samples=samples,
                index=idx,
                lead=np.full(len(idx), 2),
                ensemble=True,
                cut=10**9,
                dropout=None,
            )
            with torch.no_grad():
                out = model(b.host.inputs, b.nbr, return_attention=True)
            assert out.attention is not None and out.nbr_attention is not None
            m = target_to_slot_map(across=out.attention["across_1"], nbr=out.nbr_attention[1])
            per_primary.append(m)
            null_rows.append(
                {
                    "arm": arm,
                    "label": nb.LABELS[p],
                    "null_share": float(m[:, :, 6].sum() / max(m.sum(), 1e-12)),
                }
            )
        maps[arm] = np.concatenate(per_primary)
    save(
        attention_chart(maps=maps, title=f"Fold {month}, seed {seed}"),
        path=out_dir / f"attention_{month}_s{seed}.svg",
    )
    save(
        null_share_chart(table=pl.DataFrame(null_rows)),
        path=out_dir / f"null_share_{month}_s{seed}.svg",
    )


def cmd_contrasts(*, out_dir: Path) -> None:
    """N1 and N2 by Primary and by fold, from the stored forecasts, with a month-block 95% interval."""
    import e00713_score as sc  # ty: ignore[unresolved-import]
    import e00715_score as s715

    groups = s715.group_ids()
    star = s715.star()
    truth = sc.truth_stack()
    frames = {}
    for name in ("F4", star, "F4-R"):
        raw = s715.load_arm(arm=name, seeds=s715.SEEDS, months=s715.arms.ALL_FOLDS)
        frames[name] = sc.add_losses(frame=sc.seed_average(frame=raw), truth=truth)
    sc.assert_same_keys(frames=frames)
    label_of = dict(
        zip(
            sc.k578.sites()["time_series_id"].to_list(),
            sc.k578.sites()["label"].to_list(),
            strict=True,
        )
    )
    out_rows: dict[str, list] = {"label": [], "month": []}
    for contrast, ref in (("N1", star), ("N2", "F4-R")):
        a = frames["F4"].filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 240))
        b = frames[ref].filter((pl.col("lead_hours") >= 24) & (pl.col("lead_hours") < 240))
        j = sc.paired(treatment=a, reference=b, metric="crps").with_columns(
            label=pl.col("time_series_id").replace_strict(label_of),
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        )
        for by in ("label", "month"):
            for key in sorted(j[by].unique().to_list()):
                g = j.filter(pl.col(by) == key)
                mean, lo, hi = month_interval(
                    month=g["month"].to_numpy(), values=g["d"].to_numpy().astype(np.float64)
                )
                out_rows[by].append(
                    {by: key, "contrast": contrast, "mean": mean, "lo": lo, "hi": hi}
                )
    del groups
    table = nb.load_table()
    count = pl.DataFrame(
        {
            "label": list(nb.LABELS),
            "n_neighbours": [len(table.neighbours[i]) for i in range(nb.N_PRIMARIES)],
        }
    )
    save(
        contrast_chart(
            table=pl.DataFrame(out_rows["label"]), by="label", title="Primary", neighbours=count
        ),
        path=out_dir / "contrast_by_primary.svg",
    )
    save(
        contrast_chart(
            table=pl.DataFrame(out_rows["month"]),
            by="month",
            title="Fold (month of the forecast origin)",
        ),
        path=out_dir / "contrast_by_fold.svg",
    )


def month_interval(
    *, month: np.ndarray, values: np.ndarray, seed: int = 715
) -> tuple[float, float, float]:
    """Return the mean and a 95% interval that resamples whole months."""
    labels, inv = np.unique(month, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(labels), size=(10_000, len(labels)))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def cmd_events(*, out_dir: Path, tag: str) -> None:
    """Draw at least 12 logged events (6 of the six, 6 of the eight): host power, four forecasts, and the neighbours' power, in days from the event's start."""
    from datetime import UTC, datetime, timedelta

    import e00713_figures as f713  # ty: ignore[unresolved-import]
    import e00713_score as sc  # ty: ignore[unresolved-import]
    import e00715_arms as arms
    import e00715_score as s715

    star = s715.star()
    truth = sc.truth_stack()
    rows = pl.read_parquet(sc.SCORE_DIR / f"protocol_{tag}" / "rows.parquet").filter(
        pl.col("arm") == "C"
    )
    picked = f713.event_runs(rows=rows, groups=sc.group_ids(), per_group=6)
    assert len(picked) >= MIN_EVENTS, (
        f"only {len(picked)} events with a run of origin days: the figure needs at least {MIN_EVENTS}"
    )
    table = nb.load_table()
    import e00422_prep as prep422  # ty: ignore[unresolved-import]

    start_date = datetime.fromtimestamp(prep422.GRID_START_US / 1e6, tz=UTC).date()
    for k, run in enumerate(picked):
        pad = timedelta(days=14)
        lo, hi = run["first"] - pad, run["last"] + timedelta(days=7)
        months = [m for m in arms.ALL_FOLDS if f"{lo:%Y-%m}" <= m <= f"{hi:%Y-%m}"]
        parts = []
        for shown, arm in (("C", "C"), ("M*", star), ("F4", "F4"), ("F4-R", "F4-R")):
            f = pl.concat(
                [
                    sc.seed_average(frame=s715.load_arm(arm=arm, seeds=s715.SEEDS, months=(m,)))
                    for m in months
                ]
            )
            f = f.filter(
                (pl.col("time_series_id") == run["time_series_id"])
                & ((pl.col("valid_time") - pl.col("power_fcst_init_time")) <= timedelta(hours=24))
            )
            daily = (
                f.with_columns(day=pl.col("power_fcst_init_time").dt.date())
                .group_by("day")
                .agg(value=pl.col("q50").mean() * 100.0)
                .sort("day")
            )
            parts.append(daily.with_columns(arm=pl.lit(shown)))
        n_days = (hi - lo).days + 1
        first_idx = (lo - start_date).days * 48
        day_axis = np.arange(n_days) + (lo - run["first"]).days

        def daily_power(
            *, primary: int, first_idx: int = first_idx, n_days: int = n_days
        ) -> np.ndarray:
            block = truth[primary, first_idx : first_idx + 48 * n_days].reshape(n_days, 48)
            return np.nanmean(block, axis=1) * 100.0

        actual = pl.DataFrame(
            {
                "arm": "actual",
                "day": day_axis.astype(float),
                "value": daily_power(primary=run["primary"]),
            }
        )
        series = pl.concat(
            [
                pl.concat(parts)
                .filter((pl.col("day") >= lo) & (pl.col("day") <= hi))
                .with_columns(day=(pl.col("day") - run["first"]).dt.total_days().cast(pl.Float64))
                .select("arm", "day", "value"),
                actual,
            ]
        )
        neighbours = pl.concat(
            [
                pl.DataFrame(
                    {
                        "neighbour": nb.LABELS[n],
                        "day": day_axis.astype(float),
                        "value": daily_power(primary=n),
                    }
                )
                for n in table.neighbours[run["primary"]]
            ]
        )
        title = f"Primary {run['label']} ({run['group']}), event {k + 1}"
        save(
            event_chart(
                series=series,
                neighbours=neighbours,
                start_day=0.0,
                end_day=float(run["length"]),
                title=title,
            ),
            path=out_dir / f"event_{run['group']}_{run['label']}_{k}.svg",
        )


def main() -> int:
    """Run one command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "attention", "contrasts", "events", "follow"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "demo":
        out = (
            Path(args.args[0]) if args.args else Path(tempfile.mkdtemp(dir=Path.home() / ".cache"))
        )
        cmd_demo(out_dir=out)
        return 0
    if args.command == "attention":
        cmd_attention(month=args.args[0], seed=int(args.args[1]), out_dir=Path(args.args[2]))
    elif args.command == "contrasts":
        cmd_contrasts(out_dir=Path(args.args[0]))
    elif args.command == "events":
        cmd_events(out_dir=Path(args.args[0]), tag=args.args[1])
    else:
        msg = (
            "the follow chart needs the follow table of `e00715_era.py`: run `era.py follow` first"
        )
        raise SystemExit(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
