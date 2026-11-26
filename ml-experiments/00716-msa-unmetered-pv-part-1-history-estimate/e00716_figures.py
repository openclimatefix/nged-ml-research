# ruff: noqa: E501
"""Experiment 716: the figures of part 1's page, drawn from the stored score files (WP7).

Usage: run.sh e00716_figures.py contrasts OUT_DIR
       # A1 and A2 with their 98.75% intervals, per fold, per group
       run.sh e00716_figures.py share OUT_DIR  # the first contrast per Primary against the PV share
       run.sh e00716_figures.py gains OUT_DIR        # the learned a and c per Primary
       run.sh e00716_figures.py --selftest

Every chart is written as SVG and optimised with `npx svgo@4 --multipass --precision=1 --final-newline`
(`CLAUDE.md`). Colour is by arm: `M*` Brand Orange, `H` Data Blue, the placebo `H-S` Data Purple,
`HT` Data Green, `HT-S` Data Sky. **Anonymisation:** every chart is of a Primary, never of a metered
generator, and counts days from the origin.

The builders take plain tables, so part 2's `e00717_figures.py` imports them. A builder that needs
arrays from the real data (`pv_and_power_chart`, which draws `p` beside the net power of 14 history
days) takes those arrays as arguments.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import altair as alt
import e00716_arms as arms
import e00716_score as score
import numpy as np
import polars as pl
from plotting import ocf_theme as theme

WIDTH: Final = 640
COLOUR: Final = {
    "M*": theme.BRAND_ORANGE, "H": theme.DATA_BLUE, "H-S": theme.DATA_PURPLE, "HT": theme.DATA_GREEN, "HT-S": theme.DATA_SKY,
    "first": theme.DATA_BLUE, "placebo": theme.DATA_PURPLE,
}  # fmt: skip


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


def interval_table(*, result: dict, prefix: str, labels: dict[str, str]) -> pl.DataFrame:
    """Return the counted contrasts of `result` as rows (contrast label, group, point, lower, upper)."""
    rows = []
    for key, label in labels.items():
        for g in score.GROUPS_COUNTED:
            c = result[f"{prefix}:{key}:{g}"]
            rows.append(
                {
                    "contrast": label,
                    "group": g,
                    "point": c["point"],
                    "lower": c["lower"],
                    "upper": c["upper"],
                    "delta": c["delta"],
                }
            )
    return pl.DataFrame(rows)


def contrast_chart(*, table: pl.DataFrame, title: str) -> alt.TopLevelMixin:
    """Return the counted contrasts as points with intervals, one panel per group.

    Each panel has the zero line and the detectable difference `delta`.
    """
    base = alt.Chart(table).encode(y=alt.Y("contrast:N", title=None))
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="CRPS difference, points of capacity (negative: the treatment is better)",
        ),
        x2="upper:Q",
        color=alt.Color("contrast:N", legend=None),
    )
    dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", color=alt.Color("contrast:N", legend=None)
    )
    zero = base.mark_rule(color=theme.GREY_2).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, dots, zero)
        .properties(width=WIDTH // 2, height=90, title=title)
        .facet(column=alt.Column("group:N", title=None))
        .resolve_scale(x="independent")
    )


def fold_chart(
    *, monthly: dict[str, list[float]], folds: tuple[str, ...], title: str
) -> alt.TopLevelMixin:
    """Return the first contrast's monthly difference per fold for each group.

    `monthly[group]` lists one value per fold in order.
    """
    rows = [
        {"fold": f, "group": g, "difference": v}
        for g, vals in monthly.items()
        for f, v in zip(folds, vals, strict=True)
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X("fold:N", title="Fold (origin month)"),
        y=alt.Y("difference:Q", title="CRPS difference, points of capacity"),
        color=alt.Color("group:N", title=None),
    )
    return alt.layer(
        base.mark_line(point=True),  # ty: ignore[invalid-argument-type]
        alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=theme.GREY_2).encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=220, title=title)


def share_chart(
    *, by_primary: dict[int, float], pv_share: dict[int, float], title: str
) -> alt.TopLevelMixin:
    """Return the first contrast per Primary against the PV share (logarithmic axis, as the stored analysis)."""
    ids = sorted(set(by_primary) & set(pv_share))
    table = pl.DataFrame(
        {
            "primary": [f"D{i}" for i in ids],
            "share": [pv_share[i] for i in ids],
            "difference": [by_primary[i] for i in ids],
        }
    )
    dots = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80, color=theme.DATA_BLUE)
        .encode(
            x=alt.X(
                "share:Q",
                scale=alt.Scale(type="log"),
                title="Unmetered PV capacity over the Primary's capacity",
            ),
            y=alt.Y("difference:Q", title="CRPS difference, points of capacity"),
        )
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(dx=8, align="left")
        .encode(x="share:Q", y="difference:Q", text="primary:N")
    )
    return alt.layer(dots, text).properties(width=WIDTH, height=260, title=title)


def gains_chart(*, a: dict[int, float], c: dict[int, float], title: str) -> alt.TopLevelMixin:
    """Return the learned gains `a` (history estimate) and `c` (target estimate) per Primary."""
    rows = [
        {"primary": f"D{i}", "gain": g, "kind": "a" if which == "a" else "c"}
        for which, d in (("a", a), ("c", c))
        for i, g in d.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("primary:N", title=None),
            y=alt.Y("gain:Q", title="Learned gain"),
            color=alt.Color("kind:N", title=None),
            xOffset="kind:N",
        )
        .properties(width=WIDTH, height=220, title=title)
    )


def pv_and_power_chart(
    *, hours: np.ndarray, p: np.ndarray, power: np.ndarray, title: str
) -> alt.TopLevelMixin:
    """Return `p` beside the net power for one Primary over several days.

    `p` and `power` are (days, 48), and the days count back from the origin.
    """
    n_days = p.shape[0]
    rows = [
        {
            "day": f"age {i + 2} d",
            "hour": float(h),
            "p": float(p[i, k]),
            "power": float(power[i, k]),
        }
        for i in range(n_days)
        for k, h in enumerate(hours)
    ]
    table = pl.DataFrame(rows).unpivot(
        index=["day", "hour"], on=["p", "power"], variable_name="series", value_name="value"
    )
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("hour:Q", title="Hour of day (UTC)"),
            y=alt.Y("value:Q", title="Standardised value"),
            color=alt.Color("series:N", title=None),
        )
        .properties(width=WIDTH // 3, height=100, title=title)
        .facet(facet=alt.Facet("day:N", title=None), columns=5)
    )


def cmd_contrasts(*, out_dir: Path) -> int:
    """Draw the counted contrasts and the monthly differences of part 1."""
    part = json.loads((score.score_dir_of(part=1) / "part1.json").read_text())
    table = interval_table(
        result=part,
        prefix="counted",
        labels={"first": "A1: H minus M*", "placebo": "A2: H minus H-S"},
    )
    save(
        contrast_chart(table=table, title="Part 1: the two counted contrasts at 98.75%"),
        path=out_dir / "contrasts.svg",
    )
    monthly = {g: part[f"counted:first:{g}"]["monthly"] for g in score.GROUPS_COUNTED}
    save(
        fold_chart(monthly=monthly, folds=arms.ALL_FOLDS, title="A1 by fold"),
        path=out_dir / "a1_by_fold.svg",
    )
    return 0


def selftest() -> int:
    """Draw every builder on synthetic tables and check that the SVG files are small."""
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="f7fig_", dir=Path.home()))
    try:
        fake = {
            f"counted:{k}:{g}": {"point": p, "lower": p - 0.05, "upper": p + 0.05, "delta": 0.06}
            for k, p in (("first", -0.03), ("placebo", 0.01))
            for g in score.GROUPS_COUNTED
        }
        save(
            contrast_chart(
                table=interval_table(
                    result=fake, prefix="counted", labels={"first": "A1", "placebo": "A2"}
                ),
                title="toy",
            ),
            path=out / "c.svg",
        )
        save(
            fold_chart(
                monthly={"six": [0.1] * 8, "eight": [-0.1] * 8}, folds=arms.ALL_FOLDS, title="toy"
            ),
            path=out / "f.svg",
        )
        ids = [5, 9, 10, 11, 14, 16, 2, 3, 4]
        save(
            share_chart(
                by_primary={i: 0.01 * i for i in ids},
                pv_share={i: 0.01 * i for i in ids},
                title="toy",
            ),
            path=out / "s.svg",
        )
        save(
            gains_chart(a=dict.fromkeys(ids, 0.1), c=dict.fromkeys(ids, -0.9), title="toy"),
            path=out / "g.svg",
        )
        hours = np.arange(48) / 2.0
        save(
            pv_and_power_chart(
                hours=hours,
                p=np.abs(np.sin(np.linspace(0, 3, 48)))[None].repeat(3, axis=0),
                power=np.cos(np.linspace(0, 3, 48))[None].repeat(3, axis=0),
                title="toy",
            ),
            path=out / "p.svg",
        )
        assert all(f.stat().st_size < 500_000 for f in out.glob("*.svg"))
        print("every builder drew an SVG under 500 KB")
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print("selftest passed")
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=("contrasts",))
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    return cmd_contrasts(out_dir=args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
