# ruff: noqa: D103
"""Experiment 648: the screen table. Each arm minus its network's reference, with guards (CPU).

For every scored arm folder `score/<net>_<arm>` (net `k1` or `c2`), reads the network's own
quantiles (system `K2`; the G4 gate belongs to K1 only and is not part of the question) and writes
`score/screen.csv`: the pooled event-free CRPS gain (reference minus arm, points of capacity, 24-120
h, all Primaries, cells weighted by rows) with a 95% interval from resampling Primary-month cells,
the gain per fold, the gain without D15 and D18, and the guards on the injected bank (both groups,
all event lengths, day 4 to the end): the rise in the follow ratio (`follow_q50`), the fall in the
safe-side share (`nra_side95`), and the rise in q95 and q99 pinball loss on the bank and q05 and q95
pinball loss on event-free rows (points of capacity). Pre-registered pass rule: gain at least the
margin, at least 2 of 3 folds improved, the gain at least 0 without D15 and D18, follow rise at most
0.03, safe-side fall at most 0.03, and no pinball rise above 0.02 points.
"""

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

NAME = "00648-demand-encoder-size-and-hyperparameter-search-on-k1-and-the-class-2-candidate"
SCORE = ML_EXPERIMENTS_DATA_DIR / NAME / "score"
MARGIN = {"k1": 0.113, "c2": 0.109}
BAND = "24-120 h"
N_BOOT = 4000
RNG = np.random.default_rng(648)


def cells(*, name: str) -> pl.DataFrame:
    return pl.read_csv(SCORE / name / "event_free_cells.csv").filter(
        (pl.col("band") == BAND) & (pl.col("system") == "K2")
    )


def bank(*, name: str) -> dict[str, float]:
    row = (
        pl.read_csv(SCORE / name / "bank_summary.csv")
        .filter(
            (pl.col("system") == "K2")
            & (pl.col("month_set") == "all")
            & (pl.col("band") == BAND)
            & (pl.col("group") == "both groups")
            & (pl.col("length_class") == "all lengths")
            & (pl.col("phase") == "days4-end")
        )
        .row(0, named=True)
    )
    return {k: row[k] for k in ("follow_q50", "nra_side95", "pin95", "pin99")}


def gain(*, ref: pl.DataFrame, arm: pl.DataFrame, col: str = "crps") -> pl.DataFrame:
    keys = ["prim", "label", "month"]
    return ref.join(arm, on=keys, suffix="_a").select(
        *keys, pl.col("n_rows"), (100 * (pl.col(col) - pl.col(f"{col}_a"))).alias("gain")
    )


def pooled(*, g: pl.DataFrame) -> float:
    return float((g["gain"] * g["n_rows"]).sum() / g["n_rows"].sum())  # ty: ignore[unsupported-operator]


def interval(*, g: pl.DataFrame) -> tuple[float, float]:
    gain_, w = g["gain"].to_numpy(), g["n_rows"].to_numpy().astype(float)
    idx = RNG.integers(0, len(g), size=(N_BOOT, len(g)))
    boots = (gain_[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


out = []
for net in ("k1", "c2"):
    ref_name = f"ref_{net}"
    if not (SCORE / ref_name / "event_free_cells.csv").exists():
        continue
    ref, ref_bank = cells(name=ref_name), bank(name=ref_name)
    for d in sorted(SCORE.iterdir()):
        if not d.name.startswith(f"{net}_") or not (d / "bank_summary.csv").exists():
            continue
        a, a_bank = cells(name=d.name), bank(name=d.name)
        g = gain(ref=ref, arm=a)
        lo, hi = interval(g=g)
        folds = {m: pooled(g=g.filter(pl.col("month") == m)) for m in sorted(g["month"].unique())}
        no_dd = pooled(g=g.filter(~pl.col("label").is_in(["D15", "D18"])))
        j = ref.join(a, on=["prim", "label", "month"], suffix="_a")
        pin = {
            c: float((100 * (j[f"{c}_a"] - j[c]) * j["n_rows"]).sum() / j["n_rows"].sum())  # ty: ignore[unsupported-operator]
            for c in ("pin05", "pin95")
        }
        guards = {
            "d_follow": a_bank["follow_q50"] - ref_bank["follow_q50"],
            "d_safe95": a_bank["nra_side95"] - ref_bank["nra_side95"],
            "d_bank_pin95": 100 * (a_bank["pin95"] - ref_bank["pin95"]),
            "d_bank_pin99": 100 * (a_bank["pin99"] - ref_bank["pin99"]),
            "d_ef_pin05": pin["pin05"],
            "d_ef_pin95": pin["pin95"],
        }
        guard_ok = (
            guards["d_follow"] <= 0.03
            and guards["d_safe95"] >= -0.03
            and max(guards[k] for k in guards if "pin" in k) <= 0.02
        )
        gain_ = pooled(g=g)
        passed = (
            gain_ >= MARGIN[net]
            and sum(v > 0 for v in folds.values()) >= 2
            and no_dd >= 0
            and guard_ok
        )
        out.append(
            {
                "net": net,
                "arm": d.name.removeprefix(f"{net}_"),
                "gain": gain_,
                "lo": lo,
                "hi": hi,
                "margin": MARGIN[net],
                **{f"fold_{m}": v for m, v in folds.items()},
                "gain_without_d15_d18": no_dd,
                **guards,
                "guards_ok": guard_ok,
                "passes_screen": passed,
            }
        )
df = pl.DataFrame(out)
df.write_csv(SCORE / "screen.csv")
with pl.Config(tbl_rows=60, tbl_cols=30, float_precision=4):
    print(df)
