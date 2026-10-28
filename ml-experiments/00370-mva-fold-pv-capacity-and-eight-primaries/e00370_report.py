r"""Experiment 370: print every number the write-up quotes, and draw its two figures.

    PYTHONPATH=ml-experiments/00110-differentiable-physics-5-embedded-generation-and-capacity:ml-experiments/00358-mva-fold-known-answer-at-folded-mw-primaries:ml-experiments/00366-mva-fold-irradiance-correlation-reverse-flow-rule \
        uv run --no-sync python ml-experiments/00370-*/e00370_report.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00110_common as common  # ty: ignore[unresolved-import]
import e00358_known_answer as known  # ty: ignore[unresolved-import]
import e00366_rule as rule  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-pv-capacity-and-eight-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00370-{SLUG}" / "work"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00370-{SLUG}"
)
SIGNED_SITES: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
CAPACITY_TOLERANCE: Final[float] = 0.15
PIN_TOLERANCE: Final[float] = 0.25
NGED_MARKS: Final[dict[str, str]] = {
    "D8": "Yes",
    "D12": "Maybe",
    "D22": "Maybe",
    "D2": "No",
    "D3": "No",
    "D4": "No",
    "D19": "No",
    "D21": "No",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def signed_cells() -> pl.DataFrame:
    """Return every signed-MW cell with the rule's choice, from 358's folder, this one, and 366."""
    rows = []
    for folder in (known.OUT_DIR, WORK / "signed"):
        for path in sorted(folder.glob("*.json")):
            rows.append(json.loads(path.read_text()))
    cells = pl.DataFrame(rows, infer_schema_length=None).filter(pl.col("site").is_in(SIGNED_SITES))
    chosen = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR
        / "00366-mva-fold-irradiance-correlation-reverse-flow-rule"
        / "work"
        / "rule.parquet"
    ).select("site", "cut", "level", "reactive", "chosen")
    extra = []
    for cut in known.CUTS:
        for level in known.LEVELS:
            for reactive in known.REACTIVE:
                if reactive != "Q0":
                    continue
                job = ("D9", cut, level, reactive)
                result = rule.cell(job)
                extra.append(
                    {
                        "site": "D9",
                        "cut": cut,
                        "level": level,
                        "reactive": reactive,
                        "chosen": result["chosen"],
                    }
                )
    chosen = pl.concat([chosen, pl.DataFrame(extra)], how="diagonal_relaxed")
    return cells.join(chosen, on=["site", "cut", "level", "reactive"], how="left").with_columns(
        capacity_chosen=pl.when(pl.col("chosen") == "l0")
        .then(pl.col("capacity_l0"))
        .otherwise(pl.col("capacity_n"))
    )


def main() -> None:
    """Print the numbers and draw the figures."""
    cells = signed_cells().with_columns(
        ratio_chosen=pl.col("capacity_chosen") / pl.col("capacity_s"),
        ratio_l0=pl.col("capacity_l0") / pl.col("capacity_s"),
    )
    cache = common.load_cache()
    pins = {
        site: pinned_capacity(
            vintages=read_vintages(), table=cache.sites[site].table, cut=known.CUTS["scored"]
        ).held_points
        for site in SIGNED_SITES
    }
    scored = cells.filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0").sort("site")
    print("P1: chosen-fit capacity over signed-fit capacity, scored cut, Q0")
    for level in (0.0, 1.0):
        part = scored.filter(pl.col("level") == level)
        within = (part["ratio_chosen"] - 1).abs() <= CAPACITY_TOLERANCE
        print(f" level {level}: within 15% at {int(within.sum())} of {part.height}")
        print(part.select("site", "chosen", "ratio_chosen", "ratio_l0"))
    print("P2: pin over signed-fit capacity, level 0.0, scored cut")
    pin_rows = []
    for site in SIGNED_SITES:
        s = scored.filter(pl.col("site") == site, pl.col("level") == 0.0)["capacity_s"][0]
        pin_rows.append({"site": site, "pin": pins[site], "capacity_s": s, "ratio": pins[site] / s})
    pin_table = pl.DataFrame(pin_rows)
    print(pin_table)
    print(f" within 25%: {int(((pin_table['ratio'] - 1).abs() <= PIN_TOLERANCE).sum())} of 6")
    census = pl.DataFrame(
        [json.loads(p.read_text()) for p in sorted((WORK / "census").glob("*.json"))]
    ).with_columns(nged_mark=pl.col("site").replace_strict(NGED_MARKS, default="?"))
    with pl.Config(tbl_rows=40, tbl_cols=20):
        print("P3 and P4: census")
        print(
            census.select(
                "site",
                "nged_mark",
                "cut",
                "chosen",
                "correlation",
                "p1_training_sunny",
                "reverse_share_sunny_train_l0",
                "capacity_n",
                "capacity_l0",
                "mae_n_sunny",
                "mae_l0_sunny",
                "mae_n_all",
                "mae_l0_all",
            ).sort("site", "cut")
        )
    for cut in known.CUTS:
        part = census.filter(pl.col("cut") == cut)
        above = int((part["reverse_share_sunny_train_l0"] > 0.5).sum())
        print(f"P3 {cut}: L0 sunny training reverse-flow share above 50% at {above} of {part.height}")
    d12 = census.filter(pl.col("site") == "D12").sort("cut")
    print("P4: D12 mae_n - mae_l0 (points)")
    print(d12.select("cut", "chosen", gap_sunny=pl.col("mae_n_sunny") - pl.col("mae_l0_sunny")))
    print(" reactive power zero only; the rule fails at power factor 0.97 (experiments 358, 366)")
    ratios = pl.concat(
        [
            scored.filter(pl.col("level") == 0.0).select(
                "site", kind=pl.lit("Chosen fit, no injection"), value="ratio_chosen"
            ),
            scored.filter(pl.col("level") == 1.0).select(
                "site", kind=pl.lit("Chosen fit, level 1.0"), value="ratio_chosen"
            ),
            pin_table.select("site", kind=pl.lit("Pin, no injection"), value="ratio"),
        ]
    )
    points = (
        _chart(ratios.to_pandas())
        .mark_circle(size=90)
        .encode(
            x=alt.X("site:N", sort=SIGNED_SITES, title="Primary substation"),
            y=alt.Y("value:Q", title="Capacity divided by the signed fit's capacity"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Chosen fit, no injection", "Chosen fit, level 1.0", "Pin, no injection"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_AMBER],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
    )
    band = (
        _chart(pl.DataFrame({"y": [1 - CAPACITY_TOLERANCE, 1.0, 1 + CAPACITY_TOLERANCE]}).to_pandas())
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (points + band).properties(width=560, height=260).save(
        DOCS_DIR / "figure_1_capacity_ratio_signed_primaries.svg"
    )
    census_plot = census.filter(pl.col("cut") == "scored").select(
        "site", value="reverse_share_sunny_train_l0", mark="nged_mark"
    )
    bars = (
        _chart(census_plot.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("site:N", sort=list(NGED_MARKS), title="MVA-only Primary"),
            y=alt.Y("value:Q", title="Share of sunny training half-hours with fitted reverse flow"),
            color=alt.Color(
                "mark:N",
                title="NGED monitor file",
                scale=alt.Scale(
                    domain=["Yes", "Maybe", "No"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_AMBER, ocf.DATA_BLUE],
                ),
            ),
        )
    )
    (bars).properties(width=560, height=260).save(DOCS_DIR / "figure_2_census_fitted_reverse_flow.svg")


if __name__ == "__main__":
    main()
