# ruff: noqa: E501, D103, ANN201
"""Step 1 of experiment 569: why did school-term columns (B3a) have a wide range of effects in 562?

CPU only, reads 562's stored per-row CRPS and 437's stored school-term columns. Writes a per-Primary
table of B3a minus its shuffled-day placebo (1 seed), the same per-Primary difference between two
seeds of plain M0 as a noise reference, and the split by school-term state.
"""

import e00521_stats as s  # ty: ignore[unresolved-import]
import e00562_columns as cols  # ty: ignore[unresolved-import]
import e00562_score as sc  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT = ML_EXPERIMENTS_DATA_DIR / "00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1"


def main():
    t = sc.table(arms=["B3a", "B3ap", "M0s0", "M0s1"])
    need = [f"{a}__crps" for a in ("B3a", "B3ap", "M0s0", "M0s1")]
    t = t.filter(pl.col("main16") & pl.all_horizontal([pl.col(c).is_finite() for c in need]))
    # school-term state per row (437's calendar columns, identical for every Primary)
    look = cols.TableLookup(columns=("school_term_flag", "school_window_flag"))
    vals, found = look.get(
        primary=t["primary"].to_numpy(), origin_idx=t["origin"].to_numpy(), valid_idx=t["target"].to_numpy()
    )
    t = t.with_columns(term=pl.Series(vals[:, 0]), window=pl.Series(vals[:, 1]), found=pl.Series(found))
    print("rows", t.height, "school flag found", float(t["found"].mean()))
    labels = dict(zip(*pl.read_csv(s.DATA / "site_map.csv").select("primary", "label"), strict=True))
    groups = dict(zip(*pl.read_csv(s.DATA / "site_map.csv").select("primary", "group"), strict=True))
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted(t["primary"].unique().to_list()):
        sub = t.filter(pl.col("primary") == p)
        rec = {"primary": p, "label": labels[p], "group": groups[p], "n": sub.height}
        for name, a, b in (("real_minus_placebo", "B3a", "B3ap"), ("seed_noise", "M0s0", "M0s1"), ("placebo_minus_m0", "B3ap", "M0s0"), ("real_minus_m0", "B3a", "M0s0")):
            d = sub.select("month", "day", d=pl.col(f"{a}__crps") - pl.col(f"{b}__crps"))
            st = s.block_stats(diff=d, rng=rng, tests=16)
            rec |= {name: st["difference"], f"{name}_low": st["low_month"], f"{name}_high": st["high_month"], f"{name}_folds_neg": st["folds"] - st["folds_positive"]}
        rec["m0_crps"] = float(sub["M0s0__crps"].mean())
        for state, flt in (("term", pl.col("term") > 0.5), ("holiday", pl.col("term") <= 0.5)):
            part = sub.filter(flt)
            rec[f"real_minus_placebo_{state}"] = float((part["B3a__crps"] - part["B3ap__crps"]).mean()) if part.height else None
            rec[f"n_{state}"] = part.height
        rows.append(rec)
    out = pl.DataFrame(rows).sort("real_minus_placebo")
    OUT.mkdir(exist_ok=True, parents=True)
    out.write_csv(OUT / "school_per_primary.csv")
    cap = pl.read_csv(s.DATA / "site_map.csv").select("primary", "capacity_mw")
    out = out.join(cap, on="primary")
    with pl.Config(tbl_rows=30, tbl_width_chars=250):
        print(out.select("label", "group", "n", "real_minus_placebo", "real_minus_placebo_low", "real_minus_placebo_high", "real_minus_placebo_folds_neg", "seed_noise", "placebo_minus_m0", "real_minus_m0", "real_minus_placebo_term", "real_minus_placebo_holiday", "m0_crps", "capacity_mw"))
    eff, noise = out["real_minus_placebo"].to_numpy(), out["seed_noise"].to_numpy()
    print("range real-minus-placebo", eff.min(), eff.max(), "sd", eff.std(ddof=1))
    print("range seed noise (M0 s0 - s1)", noise.min(), noise.max(), "sd", noise.std(ddof=1))
    print("sd ratio", eff.std(ddof=1) / noise.std(ddof=1))
    print("corr(effect, seed noise) Spearman", pl.DataFrame({"a": eff, "b": noise}).select(pl.corr("a", "b", method="spearman")).item())
    for c in ("capacity_mw", "m0_crps"):
        print("Spearman effect vs", c, pl.DataFrame({"a": eff, "b": out[c].to_numpy()}).select(pl.corr("a", "b", method="spearman")).item())
    print("share of CRPS-difference from term rows", out["real_minus_placebo_term"].mean(), "holiday rows", out["real_minus_placebo_holiday"].mean())
    # share of rows in school-term and holiday windows
    print("term share", float((t["term"] > 0.5).mean()), "window share", float((t["window"] > 0.5).mean()))


main()
