# Scoreboard for Primary demand

**On the current rows, Primary demand research default 3 scores {signed_d3} against BG2's
{signed_bg2} at the six Primaries that report signed megawatts, and {mva_d3} against {mva_bg2} at
the eight that report apparent power only.** Both scores are the mean continuous ranked probability
score (CRPS) for forecasts 1 to 10 days ahead, in points of effective capacity, and smaller is
better. The difference from BG2 is {signed_gap_text} at the signed-MW six and {mva_gap_text} at the
MVA-only eight (D8 included, with donor spread; 99% intervals). Without D8, the MVA-only gap is
{mva7_gap_text}. This page is descriptive: it adopts nothing, and an adoption is decided by the
[evaluation protocol](evaluation-protocol-demand.md) in an experiment.

BG2's score here differs from experiment 379's by up to 0.006 points (3.78 against 3.788 at the MVA-only eight), because the shared half-hours differ slightly.

The page scores every stored forecast on one shared set of half-hours, so progress is read from one
table and never from a chain of contrasts on different rows. The script
`scripts/scoreboard_demand_rows.py` scores the stored forecasts and `scripts/scoreboard_demand_page.py`
rebuilds this page from them. Run both again whenever the protocol version changes. No forecast is
refitted.

## What is scored

**Rows.** Two shared sets of half-hours are scored, because the stored forecasts do not all reach
the same lead. Each row set holds the half-hours that every forecast in it covers, scored under the
protocol's fold regime `fold_a_daily_gap_v1` (292 daily runs, July 2025 to June 2026), at all 16
Primaries.

| Row set | Half-hours | Lead range | Forecasts |
|---|---|---|---|
| Leads to 240 hours | {n_240} | 0.5 to 240 hours | All eight forecasts in the first table, for days 1 to 10 and each lead band to day 10 |
| Leads to 336 hours | {n_336} | 0.5 to 336 hours | Stack without S, BG2, `manual_heuristic`, `climatology` (scored to 336 hours; the tables read only days 11 to 14 from this set) |

**Forecasts and where they are stored.** Every score is computed from the private forecast folders
below. Every forecast was stored under `demand-v1` and is rescored here with the metrics of `studies.eval_protocol`. The floor check below is the per-Primary floor that `demand-v2` added.

| Forecast | Stored in | Lead range stored | Notes |
|---|---|---|---|
| Primary demand research default 1 | Experiment 378, `work/arms` (`default_1`) | To 240 hours | Two-number solar summary S below 168 hours, stack without S above |
| Primary demand research default 2 | Experiment 378, `work/arms` (`x1` columns) | To 240 hours | Default 1 with S extended to 240 hours. Rebuilt here from the stored columns, never scored before on these rows |
| Primary demand research default 3 | Experiment 378, `work/arms` (`d3` columns) | To 240 hours | Default 2 plus the heuristic blend on daylight half-hours above 168 hours |
| Stack without S | Experiment 377 (to 240 hours: through experiment 378's copy) | To 336 hours | No S at any lead |
| BG2 | Experiment 87, `predictions/p_s6_bg2` | To 351 hours | Production does not run BG2 |
| `manual_heuristic`, `climatology` | Experiment 87, `ref_*` | To 351 hours | The two fixed reference forecasts |
| `xgboost_baseline` | Experiment 87, `ref_xgboost_baseline` | To 240 hours scored here | The production-style XGBoost model |

**D8 appears in two ways.** In the leads-to-240-hours rows every forecast uses the donor spread at D8,
as experiment 378 did. In the leads-to-336-hours rows the stack without S uses D8's own-residual
spread, which is the protocol's D8 row. The two sets of rows therefore disagree about D8 for that
one forecast, so the days 11 to 14 cell of the stack without S in the MVA-only band table is not like-for-like with the other cells of its row. Separately, every number for the MVA-only group assumes reactive power is zero.

**Forecasts that cannot be put on the current rows.** No such forecast is mixed in. Each needs a rerun before it can be added.

- Primary demand research default 1, 2, and 3 at days 11 to 14. The stored arms stop at 240 hours.
- Primary demand research default 3 at D8 with its own-residual spread. Experiment 378 scored D8
  with the donor spread only.
- Release candidates. The [model release log](https://openclimatefix.github.io/nged-substation-forecast/live_service/model-releases/)
  lists no numbered release yet, so there is no release-candidate row.
- Every score from before `demand-v1`. Those scores used six Primaries and a different scored year.

## Progress

![Absolute CRPS of each Primary demand forecast, days 1 to 10, with the benchmarks as dashed lines](scoreboard-demand-figure.svg)

## Days 1 to 10

**All forecasts, one table per group.** CRPS is the mean over the half-hours of the group. The
"Minus BG2" column is the paired difference from BG2 on the same half-hours, as a 99% interval from
20,000 resamples of whole months; the CRPS interval is 95% from 2,000 resamples. A negative
difference means the forecast beats BG2. The contrast is not shown for the benchmarks, which are
not candidates.

### Signed-MW six (D5, D9, D10, D11, D14, D16)

{headline_signed}

### MVA-only eight (D2, D3, D4, D8, D12, D19, D21, D22), reactive power assumed zero

{headline_mva}

### MVA-only seven without D8, reactive power assumed zero

{headline_mva_no_d8}

### Short history (D15, D18), report only

{headline_short}

## CRPS by lead band

**Each cell is the CRPS in points of effective capacity.** Days 1 to 10 are from the leads-to-240-hours
rows. Days 11 to 14 are from the leads-to-336-hours rows and cover four forecasts only.

### Signed-MW six

{bands_signed}

### MVA-only eight, reactive power assumed zero

{bands_mva}

### MVA-only seven without D8, reactive power assumed zero

{bands_mva_no_d8}

### Short history (D15, D18), report only

{bands_short}

## Floor check of the evaluation protocol

**The floor asks for two things at each of the 14 signed-MW and MVA-only Primaries.** The CRPS over
days 1 to 10 must not exceed the CRPS of `manual_heuristic`, and must stay within 1.10 times the CRPS
of BG2. The check below uses point estimates, as the protocol does. D8 is judged with the donor
spread, which is the weaker arm, so its row is stricter than the protocol's.

{floor_summary}

The same check for Primary demand research default 3, at each Primary:

{floor_detail}

## CRPS at every Primary, days 1 to 10

**One row per Primary, from the leads-to-240-hours rows.**

{per_primary}
