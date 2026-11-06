# ruff: noqa: E501, D103
"""Experiment 589: check the exported files against experiment 584's headline numbers.

Reads only the exported Parquet files and `substations.csv` (plus the series table, to prove that no
generator is present). Prints the checks and writes them to `checks.txt` in the export folder.

Usage: python e00589_check.py
"""

import io
import itertools
from contextlib import redirect_stdout

import e00048_fit as fd  # ty: ignore[unresolved-import]
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "backtest_export_2026-11-06"
ARMS = ("C0", "C1", "C2", "C3", "E2", "P5plain")
Q = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
TAU = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
KEYS = ["time_series_id", "power_fcst_init_time", "valid_time"]
TARGET_CRPS = {  # experiment 584, event-free rows, 24-120 h, points of capacity
    ("C0", "eight MVA Primaries"): 3.146, ("C0", "six MW Primaries"): 2.611,
    ("C1", "eight MVA Primaries"): 3.081, ("C1", "six MW Primaries"): 2.465,
}  # fmt: skip
TARGET_PAIR = {  # experiment 584, verified December pair, 0-7 days after the end, 24-120 h, q05-q95
    ("D16", "C0"): 0.479, ("D16", "C1"): 0.824, ("D16", "C2"): 0.840, ("D16", "C3"): 0.840,
    ("D5", "C0"): 0.313, ("D5", "C1"): 0.369, ("D5", "C2"): 0.583, ("D5", "C3"): 0.583,
}  # fmt: skip


def load(arm: str) -> pl.DataFrame:
    sub = pl.read_csv(OUT / "substations.csv").select(
        "time_series_id", "primary_name", "group", "capacity"
    )
    frames = [pl.read_parquet(p) for p in sorted(OUT.glob(f"{arm}_20*.parquet"))]
    return pl.concat(frames).join(sub, on="time_series_id", how="left")


def anonymisation() -> None:
    series, _ = fd._series_and_arrays()
    primaries = set(
        series.filter(pl.col("substation_type") == "Primary")["time_series_id"].to_list()
    )
    sub_ids = set(pl.read_csv(OUT / "substations.csv")["time_series_id"].to_list())
    print("== Anonymisation")
    assert sub_ids <= primaries, sub_ids - primaries
    assert len(sub_ids) == 16
    for p in sorted(OUT.glob("*.parquet")):
        ids = set(
            pl.read_parquet(p, columns=["time_series_id"])["time_series_id"].unique().to_list()
        )
        assert ids <= primaries, (p.name, ids - primaries)
    print(
        f"all {len(list(OUT.glob('*.parquet')))} files hold only the 16 Primaries of substations.csv; no generator series"
    )


def crps_check() -> None:
    print(
        "\n== CRPS, event-free rows, 24-120 h, against the NRA, points of capacity (C0 and C1 on their shared rows)"
    )
    a0, a1 = (
        load(a).filter(
            pl.col("event_free") & (pl.col("lead_hours") >= 24) & pl.col("nra_power").is_not_null()
        )
        for a in ("C0", "C1")
    )
    both = a0.join(a1.select(KEYS), on=KEYS, how="inner")
    both1 = a1.join(a0.select(KEYS), on=KEYS, how="inner")
    for name, df in (("C0", both), ("C1", both1)):
        pin = [
            (
                pl.max_horizontal(
                    t * (pl.col("nra_power") - pl.col(q)),
                    (t - 1) * (pl.col("nra_power") - pl.col(q)),
                )
            )
            for q, t in zip(Q, TAU, strict=True)
        ]
        s = df.with_columns(crps=2.0 * pl.mean_horizontal(*pin) / pl.col("capacity") * 100)
        for g, v, _n in s.group_by("group").agg(pl.col("crps").mean(), pl.len()).iter_rows():
            if g in ("eight MVA Primaries", "six MW Primaries"):
                want = TARGET_CRPS[(name, g)]
                flag = "ok" if abs(v - want) < 0.0015 else "MISMATCH"
                print(f"{name} {g}: {v:.3f} (experiment 584: {want:.3f}) {flag}")


def pair_check() -> None:
    print(
        "\n== Coverage of the NRA by q05-q95, verified December pair, 0-7 days after the end, 24-120 h"
    )
    for arm in ("C0", "C1", "C2", "C3"):
        df = load(arm)
        for name in ("D16", "D5"):
            ev = df.filter((pl.col("primary_name") == name) & (pl.col("event_id") == 23))
            end = ev["valid_time"].max() + pl.duration(minutes=30)
            x = df.filter(
                (pl.col("primary_name") == name)
                & (pl.col("valid_time") >= end)
                & (pl.col("valid_time") < end + pl.duration(days=7))
                & (pl.col("lead_hours") >= 24)
                & pl.col("nra_power").is_not_null()
            )
            cov = x.select(
                (
                    (pl.col("q05") <= pl.col("nra_power")) & (pl.col("nra_power") <= pl.col("q95"))
                ).mean()
            ).item()
            want = TARGET_PAIR[(name, arm)]
            print(
                f"{arm} {name}: {cov:.3f} (experiment 584: {want:.3f}, n={x.height}) {'ok' if abs(cov - want) < 0.0015 else 'MISMATCH'}"
            )


def structure_check() -> None:
    print("\n== Structure")
    c1, c2, c3 = load("C1"), load("C2"), load("C3")
    assert c1.height == c2.height == c3.height
    j = c1.join(c2, on=KEYS, suffix="_2").join(
        c3.select(*KEYS, *[pl.col(q).alias(f"{q}_3") for q in Q]), on=KEYS
    )
    same = j.filter(~pl.col("gate_on_2"))
    print("C2 equals C1 where the gate is off:", all(same[q].equals(same[f"{q}_2"]) for q in Q))
    on = j.filter(pl.col("gate_on_2"))
    print(
        f"gate on for {on.height / j.height:.3%} of C2 rows; mean q05-q95 width there: C1 {(on['q95'] - on['q05']).mean():.3f}, C2 {(on['q95_2'] - on['q05_2']).mean():.3f}, C3 {(on['q95_3'] - on['q05_3']).mean():.3f}"
    )
    for arm in ARMS:
        d = load(arm)
        mono = d.select(
            pl.all_horizontal(*[pl.col(a) <= pl.col(b) for a, b in itertools.pairwise(Q)]).all()
        ).item()
        print(
            f"{arm}: quantiles non-decreasing on every row: {mono}; rows {d.height}; observed_power null {d['observed_power'].null_count()}; drop_reason set {d['drop_reason'].drop_nulls().len()}; in verified event {int(d['in_verified_event'].sum())}"
        )


def file_list() -> None:
    print("\n== Files")
    print("| File | Rows | Size, MB |\n|---|---|---|")
    total = 0
    for p in sorted(OUT.iterdir()):
        n = pl.scan_parquet(p).select(pl.len()).collect().item() if p.suffix == ".parquet" else ""
        total += p.stat().st_size
        print(f"| `{p.name}` | {n} | {p.stat().st_size / 1e6:.1f} |")
    print(f"\nFolder total: {total / 1e6:.0f} MB")


def main() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        anonymisation()
        crps_check()
        pair_check()
        structure_check()
        file_list()
    text = buf.getvalue()
    print(text)
    (OUT / "checks.txt").write_text(text)


if __name__ == "__main__":
    main()
