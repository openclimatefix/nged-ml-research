# ruff: noqa: ANN001, ANN201, D100, D103, E501, PTH123, SIM115
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

D = str(ML_EXPERIMENTS_DATA_DIR / "00533-l1d-vs-a-pool-same-rows") + "/"
c = pl.read_csv(D + "scores/contrasts.csv")
ORDER = ["0-6h", "6-12h", "12-24h", "24-48h", "48-72h", "72-120h", "120-240h", "all"]


def tab(sl, st, names):
    s = "| Band | Contrast | Difference | Interval | Folds positive | MDE |\n|---|---|---|---|---|---|\n"
    for n in names:
        for b in ORDER:
            r = c.filter(
                (pl.col("slice") == sl)
                & (pl.col("set") == st)
                & (pl.col("contrast") == n)
                & (pl.col("band") == b)
            )
            if r.height == 0:
                continue
            r = r.row(0, named=True)
            s += f"| {b} | {n} | {r['difference']:+.3f} | [{r['low_month']:+.3f}, {r['high_month']:+.3f}] | {r['folds_positive']}/{r['folds']} | {r['mde']:.2f} |\n"
    return s


N = ["L-1d - A-pool", "L-1d-4q - A-pool", "A-pool - K2-S1"]
md = f"""# Experiment 533 interim numbers (INTERIM, UNAUDITED: no code review, no write-up)

## Overlap (stated before any scoring; no fit was run)

- **A-pool's stored rows cover 511's rows completely.** The five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) hold 1,145,256 rows in 511's row tables. Of these, 1,142,520 are in the all-arms-present set `main16`, and every one has finite A-pool, L-1d (2 seeds), L-1d-4q (527), L-Ww, S2-J-0.4x (524), and K2-S1 (cell c0) scores. The 2,736 rows left out are 2026-06 rows that 511 and 524 did not score.
- A-pool and manual_heuristic scores in 511's and 524's row tables are identical (maximum absolute difference 0.0), and the keys `(primary, origin, target)` match row for row.
- manual_heuristic has finite scores on about 80% of the rows (none beyond its horizon), so it is not in the intersection. Its gaps are paired over the rows where it exists, and its absolute column is the mean over those rows only.
- Row sets: eight = MVA-only, six = signed-MW (rows with no flagged class), D15/D18 report-only. Slices: all rows, no labelled event (`tgt_real` false), cold nights, Christmas week.

Code: `code/e00533_score.py` (imports 521's row sets, lead bands, bootstrap, and MDE formula; reads the stored per-row scores of 511, 521, 524, and 527). Tables: `scores/absolute.csv`, `scores/contrasts.csv`, `scores/checks.json`, and `tables_full.md` (every slice, set, and band, with per-fold values).

## Take-home

**The lag-free L-1d is worse than A-pool at the eight up to 120 h (every interval excludes zero, 5 of 5 folds), and about level with it at the six except at 12-24 h (+0.196, interval excludes zero).** At 120-240 h the L-1d minus A-pool gap is +0.082 (interval -0.034 to +0.213; MDE 0.157) at the eight and +0.002 (-0.070 to +0.083; MDE 0.134) at the six, so a long-lead difference is not shown. At 24-48 h at the eight, L-1d is worse by 0.288 (0.189 to 0.374; 5 of 5 folds). Four seeds (L-1d-4q) move the 120-240 h gap by under 0.02 at the eight and to -0.040 at the six.

- A-pool minus K2-S1 at 120-240 h is +0.091 (-0.115 to +0.289) at the eight and +0.194 (-0.210 to +0.528) at the six: intervals include zero at both sets, so A-pool is not shown to differ from K2-S1 at long lead. A-pool is better than K2-S1 at 24-48 h (-0.309 eight, -0.186 six; intervals exclude zero).
- Positive "L-1d - A-pool" means A-pool is better. Gaps are in points of capacity.
- Cold nights: L-1d is worse than A-pool by 0.318 pooled at the eight (0.064 to 0.428; positive in 3 of the 3 folds that have cold nights), driven by 24-48 h (+0.535). At 120-240 h the cold-night gaps are +0.111 (eight) and -0.116 (six), both with intervals that include zero.
- Christmas week lies in one fold (2025-12), so its interval is a single point and its folds-positive count is 0 or 1 of 1. Treat those rows as descriptive.

## Caveats

- Bonferroni count: 28 (2 counted contrasts, L-1d and L-1d-4q each minus A-pool, x 7 bands x 2 sets). All other contrasts are report-only. Intervals shown are 95% month-block bootstraps with 5 blocks (near min-to-max); Bonferroni-level intervals are in `contrasts.csv`.
- The MDE beside each contrast is 1.96 x sqrt(2) x sd / sqrt(k) of L-1d's seed spread over the five folds (k = 2 for L-1d, 4 for L-1d-4q), as in 521 and 527. It ignores A-pool's own seed spread. A one-fold slice (Christmas week) gives a meaningless MDE.
- Weather for 511's rows is origin-time (532 PASS on the 397 summary tables).
- Everything is interim and unaudited. Nothing was fitted.

## Paired gaps by lead band

### Eight (MVA-only), all rows

{tab("all rows", "eight", N)}
### Six (signed-MW), all rows

{tab("all rows", "six", N)}
### Eight, rows with no labelled event

{tab("no labelled event", "eight", N[:2])}
### Eight, cold nights

{tab("cold nights", "eight", N[:2])}
## Full tables

CRPS tables and gaps against K2-S1, S2-J-0.4x, L-Ww, and manual_heuristic, for every slice and set, are in `tables_full.md`.
"""
open(D + "interim_numbers.md", "w").write(md)
