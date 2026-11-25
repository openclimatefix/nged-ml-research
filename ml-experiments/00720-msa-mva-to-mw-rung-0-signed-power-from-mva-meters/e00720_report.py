# ruff: noqa: E501, PT018
"""Experiment 720: the report script.

Reads the private tables, writes tables and figures, and prints nothing dated.

Inputs (all under the private folder, written by `e00720_rung0b.py --run`, `e00720_v2.py --run`, and
`e00720_analyses.py`): `rung0b_readouts.csv`, `rung0b_references.csv`, `v2_bar.csv`, `k1_v3.csv`,
`v2_fits.csv`, `d8_running_vs_idle_profiles.csv`. Outputs: a markdown report and the figures under
the private folder, and with `--publish-dir` a pooled table of the sign read-outs, which passes
`e00720_guard.public_aggregate` or is not written. Any table that holds a generator profile, a
state, or a date is never published.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import e00720_figures as figs
import e00720_guard as guard
import polars as pl


def pooled_sign_table(*, readouts: pl.DataFrame) -> pl.DataFrame:
    """Pool the rung 0b rows over sites and folds: mean area and recall per read-out, sky, and level."""
    return (
        readouts.group_by("readout", "sky", "level")
        .agg(
            auc=pl.col("auc").mean(),
            recall_at_5pct_false=pl.col("recall_at_5pct_false").mean(),
            n_rows=pl.col("n_rows").sum(),
        )
        .sort("level", "readout", "sky")
    )


def bars_markdown(*, tables: dict[str, pl.DataFrame]) -> str:
    """Return a markdown report of the bar tables that exist."""
    out = ["# Experiment 720 rung 0 report (private)\n"]
    for name, frame in tables.items():
        out.append(f"## {name}\n")
        out.append(frame.write_csv(separator="|"))
    return "\n".join(out)


def build(*, private: Path, publish_dir: Path | None) -> list[Path]:
    """Read whatever tables exist, draw the figures, write the report; return the written paths."""
    written: list[Path] = []
    tables: dict[str, pl.DataFrame] = {}
    path = private / "rung0b_readouts.csv"
    if path.exists():
        readouts = pl.read_csv(path)
        pooled = pooled_sign_table(readouts=readouts)
        tables["sign read-outs, pooled"] = pooled
        written.append(
            figs.write_chart(
                chart=figs.sign_auc_chart(rows=pooled.select("readout", "sky", "level", "auc")),
                table=pooled,
                name="sign_auc",
                private_dir=private / "figures",
            )
        )
        if publish_dir is not None:
            guard.public_aggregate(frame=pooled)
            written.append(publish_dir / "sign_pooled.csv")
            publish_dir.mkdir(parents=True, exist_ok=True)
            pooled.write_csv(written[-1])
    for name in ("rung0b_references", "v2_bar", "k1_v3"):
        if (private / f"{name}.csv").exists():
            tables[name] = pl.read_csv(private / f"{name}.csv")
    fits = private / "v2_fits.csv"
    if fits.exists():
        f = pl.read_csv(fits).filter(pl.col("primary"))
        written.append(
            figs.write_chart(
                chart=figs.paired_f1_chart(per_cluster=f.select("f1_m2", "f1_hindsight")),
                table=f,
                name="paired_f1",
                private_dir=private / "figures",
            )
        )
    prof = private / "d8_running_vs_idle_profiles.csv"
    if prof.exists():
        p = pl.read_csv(prof)
        written.append(
            figs.write_chart(
                chart=figs.profile_chart(profile=p),
                table=p,
                name="profile",
                private_dir=private / "figures",
            )
        )
    report = private / "report.md"
    report.write_text(bars_markdown(tables=tables))
    written.append(report)
    return written


def selftest() -> int:
    """Run `build` on synthetic private tables in a temporary folder."""
    readouts = pl.DataFrame(
        {"readout": ["anchored", "hindsight"] * 4, "sky": ["real", "real", "shifted", "shifted"] * 2, "level": [1.0] * 4 + [2.0] * 4,
         "auc": [0.9, 0.97, 0.6, 0.7, 0.95, 0.99, 0.55, 0.7], "recall_at_5pct_false": [0.5] * 8, "n_rows": [100.0] * 8}
    )  # fmt: skip
    with tempfile.TemporaryDirectory(dir=str(Path.home() / ".cache")) as d:
        tmp = Path(d)
        readouts.write_csv(tmp / "rung0b_readouts.csv")
        pl.DataFrame(
            {"primary": [True] * 3, "f1_m2": [0.3, 0.6, 0.8], "f1_hindsight": [0.5, 0.8, 0.9]}
        ).write_csv(tmp / "v2_fits.csv")
        pl.DataFrame({"point": [0.1], "lower": [0.05]}).write_csv(tmp / "v2_bar.csv")
        out = build(private=tmp, publish_dir=tmp / "pub")
        assert (tmp / "pub" / "sign_pooled.csv").exists() and (tmp / "report.md").exists()
        assert len(out) >= 4
        assert "v2_bar" in (tmp / "report.md").read_text()
    print(
        "PASS report: pooled sign table published through the guard, figures and report written privately"
    )
    return 0


def main() -> int:
    """Parse arguments and run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--publish-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.build:
        paths = build(private=guard.PRIVATE_DIR, publish_dir=args.publish_dir)
        print(
            f"wrote {len(paths)} files under the private folder"
            + (" and the pooled table to the publish folder" if args.publish_dir else "")
        )
        return 0
    ap.error("give --selftest or --build")
    return 2


if __name__ == "__main__":
    sys.exit(main())
