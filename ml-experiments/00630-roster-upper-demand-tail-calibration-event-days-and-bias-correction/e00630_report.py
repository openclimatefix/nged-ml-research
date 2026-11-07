# ruff: noqa: E501
"""Experiment 630, part 3: print every number the write-up quotes, and the decision-rule verdicts."""

import polars as pl
from e00630_common import NETWORKS, TABLES, load

pl.Config.set_tbl_rows(60).set_tbl_cols(30).set_tbl_width_chars(250).set_fmt_str_lengths(30)


def pct(col: str) -> pl.Expr:
    """Format a share and its interval as a percentage string."""
    return (
        pl.format(
            "{}% [{}, {}]",
            (pl.col(col) * 100).round(1),
            (pl.col(f"{col}_lo") * 100).round(1),
            (pl.col(f"{col}_hi") * 100).round(1),
        )
    ).alias(col)


def read(name: str, network: str) -> pl.DataFrame:
    """Read one exceedance table."""
    return pl.read_csv(TABLES / f"exceedance_by_{name}_{network}.csv")


def show(title: str, df: pl.DataFrame) -> None:
    """Print a titled table."""
    print(f"\n### {title}")
    print(df)


def main() -> None:
    """Print every number the write-up quotes."""
    for network in NETWORKS:
        o = read("overall", network)
        show(
            f"{network} overall (exceedance share [95% day interval])",
            o.select(
                "subset",
                "n",
                "n_days",
                pct("a90"),
                pct("a95"),
                pct("b10"),
                pct("b05"),
                pl.col("gap").round(3).alias("gap_mw"),
            ),
        )
    cand = read("month", "Candidate")
    show(
        "Candidate by month",
        cand.select(
            "subset",
            "month",
            "n",
            pct("a90"),
            pct("a95"),
            pct("b10"),
            pct("b05"),
            pl.col("gap").round(3),
        ).sort("subset", "month"),
    )
    for network in ("K1", "K3"):
        m = read("month", network).filter(pl.col("subset") == "event-free")
        show(
            f"{network} event-free by month",
            m.select("month", "n", pct("a90"), pct("a95")).sort("month"),
        )
    # decision rule
    ov = read("overall", "Candidate")
    ef = ov.filter(pl.col("subset") == "event-free").row(0, named=True)
    ev = ov.filter(pl.col("subset") == "event").row(0, named=True)
    months_above = cand.filter((pl.col("subset") == "event-free") & (pl.col("a90") > 0.12)).height
    c1 = 0.08 <= ef["a90"] <= 0.12 and ev["a90"] - ef["a90"] >= 0.05
    c2 = months_above >= 3
    print(
        f"\n### Decision rule: event-free pooled q90 {ef['a90']:.4f}, event pooled q90 {ev['a90']:.4f}, difference {ev['a90'] - ef['a90']:.4f}; months above 12%: {months_above}/5"
    )
    print(
        f"event-driven: {c1}; bias on ordinary days: {c2}; answer: {'mixed' if c1 == c2 else ('event-driven' if c1 else 'bias on ordinary days')}"
    )
    for name in ("band", "tercile", "cold_night", "group"):
        for network in ("K1", "Candidate", "K3"):
            d = read(name, network).filter(pl.col("subset") == "event-free")
            key = {
                "band": "band",
                "tercile": "tercile",
                "cold_night": "cold_night",
                "group": "group",
            }[name]
            show(
                f"{network} event-free by {name}",
                d.select(
                    key, "n", pct("a90"), pct("a95"), pct("b10"), pct("b05"), pl.col("gap").round(3)
                ),
            )
    # hour blocks
    for network in ("K1", "Candidate"):
        h = read("hour", network).filter(pl.col("subset") == "event-free")
        h = h.with_columns((pl.col("hour") // 4 * 4).alias("block"))
        show(
            f"{network} event-free by hour (4-hour blocks, point estimates)",
            h.group_by("block")
            .agg(
                (pl.col("a90") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("a95") * pl.col("n")).sum() / pl.col("n").sum(),
                (pl.col("gap") * pl.col("n")).sum() / pl.col("n").sum(),
            )
            .sort("block")
            .with_columns(pl.all().round(3)),
        )
    for network in ("K1", "Candidate", "K3"):
        p = read("primary", network).filter(pl.col("subset") == "event-free")
        show(
            f"{network} event-free by Primary",
            p.select("primary_name", "n", pct("a90"), pct("a95"), pl.col("gap").round(3)).sort(
                "primary_name"
            ),
        )
    # gap relative to median demand, by Primary
    for network in ("Candidate",):
        df = load(network=network).filter(pl.col("event_free"))
        g = (
            df.group_by("primary_name", "group", "unit")
            .agg(
                pl.col("gap").mean().alias("mean_gap_mw"),
                pl.col("observed_power").median().alias("median_demand"),
                pl.col("capacity").first(),
            )
            .with_columns((pl.col("mean_gap_mw") / pl.col("capacity")).alias("gap_over_capacity"))
            .sort("primary_name")
        )
        show(
            f"{network} event-free mean(actual - q50) by Primary",
            g.with_columns(pl.col("mean_gap_mw", "median_demand", "gap_over_capacity").round(3)),
        )
    # correction
    for network in NETWORKS:
        c = pl.read_csv(TABLES / f"correction_{network}.csv")
        for sl in ("overall", "group", "band", "month"):
            d = c.filter(pl.col("slice") == sl)
            keys = {"overall": [], "group": ["group"], "band": ["band"], "month": ["month"]}[sl]
            rows = []
            for arm in ("base", "S", "R"):
                rows.append(  # noqa: PERF401
                    d.select(
                        "subset",
                        *keys,
                        "n",
                        pl.lit(arm).alias("arm"),
                        pct(f"a90_{arm}"),
                        pct(f"a95_{arm}"),
                        pl.col(f"crps_{arm}").round(4).alias("crps"),
                        (
                            pl.col(f"crps_ratio_{arm}").round(4) if arm != "base" else pl.lit(1.0)
                        ).alias("crps_ratio_vs_base"),
                        (
                            pl.format(
                                "[{}, {}]",
                                pl.col(f"crps_ratio_{arm}_lo").round(4),
                                pl.col(f"crps_ratio_{arm}_hi").round(4),
                            )
                            if arm != "base"
                            else pl.lit("")
                        ).alias("crps_ratio_interval"),
                    ).rename({f"a90_{arm}": "a90", f"a95_{arm}": "a95"})
                )
            show(
                f"{network} correction test, last two months, by {sl}",
                pl.concat(rows).sort("subset", *keys, "arm"),
            )
    pr = pl.read_csv(TABLES / "correction_parameters.csv")
    show(
        "fitted parameters by network and band (median over Primaries)",
        pr.group_by("network", "band")
        .agg(
            pl.col("shift_mw").median().round(3),
            pl.col("shift_mw").min().round(3).alias("shift_min"),
            pl.col("shift_mw").max().round(3).alias("shift_max"),
            pl.col("factor").median(),
            pl.col("factor").min().alias("f_min"),
            pl.col("factor").max().alias("f_max"),
            pl.col("train_q90_exceedance").mean().round(3),
        )
        .sort("network", "band"),
    )
    # usable-correction rule for the candidate
    c = (
        pl.read_csv(TABLES / "correction_Candidate.csv")
        .filter((pl.col("slice") == "overall") & (pl.col("subset") == "event-free"))
        .row(0, named=True)
    )
    for arm in ("S", "R"):
        ok = (
            all(
                c[f"{k}_{arm}_lo"] <= nom <= c[f"{k}_{arm}_hi"]
                for k, nom in (("a90", 0.10), ("a95", 0.05))
            )
            and c[f"crps_ratio_{arm}_hi"] <= 1.01
            and c[f"crps_ratio_{arm}"] <= 1.01
        )
        print(
            f"\ncorrection {arm}: q90 {c[f'a90_{arm}']:.4f} [{c[f'a90_{arm}_lo']:.4f}, {c[f'a90_{arm}_hi']:.4f}], q95 {c[f'a95_{arm}']:.4f} [{c[f'a95_{arm}_lo']:.4f}, {c[f'a95_{arm}_hi']:.4f}], crps ratio {c[f'crps_ratio_{arm}']:.4f}; correction usable by the rule (point CRPS ratio <= 1.01): {ok}"
        )


if __name__ == "__main__":
    main()
