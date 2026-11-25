---
id: "00725"
slug: msa-live-buildable-history-weather
title: >-
  Does the axial-attention demand network still forecast better than the Candidate when every
  history token reads only weather that a live service holds at the forecast origin?
theme: lagged-power-and-error-correction
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts and network weights for about 61 fits; (2) more than one
  rule for the history weather defensibly tests the hypothesis, and the page fixes one; (3) it
  subclasses the data source of experiment 713, and the callers of that class cannot be listed
  without searching the repository; (4) it touches no serving path or degradation rule; (5) it
  changes no contract. Full set of reviews: plan, code (Opus, each new script, twice), results,
  write-up, prose.
hypothesis: >-
  The gain of the axial-attention demand network of experiment 713 over the Candidate at the six
  signed-MW Primaries does not depend on analysis weather in the newest history rows, because the
  across-day attention reads the power of the recent days and needs the weather only to explain
  that power.
prediction: >-
  Written before any fit. L1 (`L` minus `C`) lies between -0.22 and -0.08 points of capacity at the
  six signed-MW Primaries and between -0.14 and +0.19 at the eight MVA-only Primaries. L2 (`L` minus
  `M*`) lies inside +/-0.054 at the six and inside +/-0.142 at the eight. The predicted verdict is
  `small gain` or `inconclusive`. A prediction fails if a point estimate falls outside its range.
decision_rule: >-
  Two counted contrasts, L1 and L2, each at the six and at the eight, so four counted readings, each
  read at 98.75% from a month-block bootstrap, with experiment 713's three guards, noise floors,
  and verdict tables. The verdict is read from L1 alone. The smallest gain worth porting is 0.05
  points of capacity, experiment 713's value. Seeds 2 to 4 run only if L1 has a point estimate
  below zero in at least one group after the 8-fold, 2-seed reading. Every verdict, headline, and
  take-home message states that experiment 713's known-answer gate was waived and that the target
  row of every arm still reads a CAMS day summary that a live service may not hold.
benchmark: >-
  `C` (the Candidate of experiment 667, stored seeds 0 to 4) and `M*` (experiment 713's selected
  arm, `M1`, stored seeds 0 to 4); neither is refitted.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity on leads from 24 h up to but not including 240 h after the 12:00 UTC
  origin, with the mean absolute error of the median forecast normalised by capacity beside it; q01
  and q99 pinball loss; the stopping pass of every fit; the read-out weights by age.
---

**We ask whether the axial-attention demand network of experiment 713 (the "MSA") still forecasts
better than the Candidate when every history token reads only weather that a live service holds at
the native origin.** Experiment 713 found `M*` minus `C` of -0.147 points of CRPS [-0.262, -0.020]
at the six signed-MW Primaries and +0.022 [-0.092, +0.112] at the eight MVA-only Primaries. The
history rows of `M*` read ERA5 and CAMS analyses at every age, and a live service does not hold
those analyses for the newest rows. Experiment 713 therefore labelled both scores a ceiling. This
page is the follow-up named F8. It is the plan: no neural network has been fitted, no GPU has been
used, and every number below is a design value or a figure from an earlier experiment.

**The honest expectation is that the gain at the six signed-MW Primaries survives.** Experiment
713's swap (a) gave every history token ENS weather at prediction, with no retraining, and `M*`
minus `C` stayed at -0.152 [-0.258, -0.048] at the six (95%, seeds 0 and 1).

## Names used on this page

- **Experiment 713** is the axial-attention demand network. **`M*`** is its selected arm, `M1`: the
  weather and date-time encoders are loaded from the Candidate chain and frozen. **`C`** is the
  Candidate of experiment 667.
- **ENS** is the 51-member ensemble forecast of the European Centre for Medium-Range Weather
  Forecasts. **ERA5** is the same centre's reanalysis, and **ERA5T** is the preliminary release of
  ERA5. **CAMS** is the Copernicus Atmosphere Monitoring Service's satellite irradiance estimate.
  **CRPS** is the continuous ranked probability score, in points of capacity.
- **The contract origin** `c` is 12:00 UTC on day D. **The native origin** `o` is 09:00 UTC on
  day D, the time at which the forecast is made.
- **The `ft` table** holds, for each Primary and each half-hour, the ENS-mean weather of the stored
  run with the smallest lead for that half-hour, and that run's forecast origin (the issue time plus
  9 hours).
- **A history row of age `a`** holds the 48 half-hours stamped (period end) from 12:00 UTC on
  D - `a` to 11:30 UTC on D - `a` + 1.
- **The quick-look folds** are experiment 713's four screen folds: 2025-11, 2026-01, 2026-03, and
  2026-06.

## Why this runs

**This experiment runs because three prediction-only swaps in experiment 713 bracket the answer and
none of them is a like-for-like score.** Each swap changed an input of the stored `M*` at prediction
and scored `M*` minus `C` again (95%, seeds 0 and 1).

| Reading | Six signed-MW Primaries | Eight MVA-only Primaries |
|---|---|---|
| `M*` minus `C`, seeds 0 and 1 | -0.150 | +0.027 |
| Swap (a): ENS in the 7 weather columns of every history token; the 14 B1 columns kept as analyses | -0.152 [-0.258, -0.048] | +0.028 [-0.070, +0.125] |
| Swap (b): as swap (a), with the 14 B1 columns set to zero | -0.109 [-0.236, +0.010] | +0.234 [+0.079, +0.365] |

**Swap (a) leaves two gaps that F8 closes.** Swap (a) kept the 14 B1 columns (trailing temperature
and irradiance windows, and the CAMS day summary) as analyses, so the swapped inputs still held
values that a live service lacks. The neural network was also not trained on the swapped inputs.
Swap (b) shows that the B1 columns carry skill at the eight MVA-only Primaries, where zeroing the
columns moved the contrast by about +0.21. F8 therefore replaces the B1 columns with live values and
cannot drop them.

## What is compared

| Arm | History weather | Fitted | Seeds | Role |
|---|---|---|---|---|
| `C` | ENS at the two lag times (the Candidate's own inputs) | Stored | 0 to 4 | Reference of L1 |
| `M*` | ERA5 and CAMS analyses at every age | Stored | 0 to 4 | Reference of L2 |
| `L` | The live rule of the next section | Stages 2 and 3, from the stored stage 1 state of `M*` | 0 to 4 | Treatment |
| `S` | The live rule | No fit: the stored stage 3 state of `M*` predicts | 0, 1 | Report-only: the live rule with no retraining |
| `Z` | As `M*` | As `L`, through F8's code | 0, 1 on the quick-look folds | Report-only: the refit without the weather change. The fit of fold 2025-11, seed 0, runs in phase 0 as a reproduction check |
| `E` | ENS at every age | As `L` | 0, 1 on the quick-look folds | Report-only: no analysis at any age |

**The counted contrasts are L1 (`L` minus `C`) and L2 (`L` minus `M*`), each at the six signed-MW
Primaries and at the eight MVA-only Primaries.** L1 is experiment 713's H1 made like for like. L2 is
the change in CRPS from removing the lab shortcut. Every contrast uses experiment 713's rows, folds,
headline band, interval, guards, and noise floors, and pairs seed s of one arm with seed s of the
other. A two-seed reading uses seeds 0 and 1 of `L`, `C`, and `M*`. The 5-seed reading uses seeds 0
to 4 of all three.

**Each fitted arm starts from the stored stage 1 state of `M*` of the same fold and seed, and the
arms differ in the history weather alone.** `L`, `Z`, and `E` skip stage 1 and run experiment 713's
stages 2 and 3 from that state. F8 adds no parameter and no random draw, so the seed sets the same
pass order, lead-day draws, and dropout draws as in the `M*` fit of the same fold and seed. The
validation loss of each arm is computed on that arm's own inputs, so early stopping in `L` reads
live-rule inputs, as a live refit would. `L` minus `S` is what training on the live inputs adds, and
`S` minus `M*` is what the inputs alone change. `Z` minus `M*` is the run-to-run distance of a
stage 2 and stage 3 refit, and `L` minus `Z` is the weather change with the refit held equal. `E`
is the first specification of F8 in the design of experiment 713, and a live service running `E`
needs no ERA5T feed and no half-hourly CAMS feed for the history rows.

## What changes in the inputs

### Three availability lags

**The live rule rests on three lags, each measured against the native origin.**

| Source | Lag assumed | Latest usable value at native origin 09:00 UTC on day D | Basis |
|---|---|---|---|
| ENS run | 9 h from issue time | The 00Z run of day D | The Candidate's own rule; experiment 713 asserts `nwp_init_time <= o - 9 h` on every target row |
| CAMS irradiance | 36 h after the end of a UTC day | UTC day D - 3, whose last half-hour is stamped 00:00 UTC on D - 2 | Experiment 504's rule; the stored CAMS table has no availability stamp |
| ERA5T | 120 h after the valid time | The half-hour stamped 09:00 UTC on D - 5 | "About 5 days late" in the design of experiment 713 |

**The CAMS lag of 36 hours is fixed, and UTC day D - 3 is in hand at the native origin under both
latencies that CAMS documents.** Day D - 3 ends 57 hours before the native origin, so the rule needs
a latency of at most 57 hours after the end of a day, whatever the hour of the daily update. Until
February 2026 CAMS documents a latency of 2 days, which puts day D - 3 in hand at least 9 hours
before the native origin. Since February 2026 CAMS documents a latency of 1 day, which puts day D - 3
in hand about a day before the native origin. The lag is not loosened to D - 2, because holding
day D - 2 depends on the hour of the daily update, which no source gives.

**Two caveats on CAMS go on the page.** The stored CAMS series may hold reprocessed values, because
`cams_demand_points.parquet` has no availability stamp. The grid stamp of a CAMS value is also not
the CAMS source stamp, because the dense weather copies each hourly CAMS value to both half-hours of
its hour, so check KL4 and the code review test the CAMS source stamp. At the age-3 boundary (column
24, stamped 00:00 UTC on D - 2), both stamps give the same answer.

**The ERA5T lag is measured against the native origin, and experiment 713's look-ahead statement
measured the same lag against the contract origin.** Against the native origin, columns 43 to 47 of
the age-6 row are not buildable, and two B1 columns of an age-6 token read later half-hours of the
token's local day. F8 reads ERA5 from age 7, which is buildable in every column with a margin of one
day.

### The table of cells

**Each history token has five groups of weather columns, and the live rule fixes one source for each
group in each block of the history matrix.** The 21 weather columns of a token are
`hist_weather[..., 0:21]` of experiment 713. "Analysis" means the dense array that `M*` reads.
"ENS" means the value described under "Why each cell is buildable".

| Column group (index in the 21 columns) | Age 2 | Age 3, columns 25 to 47 | Age 3, columns 0 to 24 | Ages 4 to 6 | Ages 7 to 15 |
|---|---|---|---|---|---|
| Shortwave irradiance (0) | ENS | ENS | Analysis (CAMS) | Analysis (CAMS) | Analysis (CAMS) |
| Longwave, temperature, dew point, pressure, precipitation, and wind speed (1 to 6) | ENS | ENS | ENS | ENS | Analysis (ERA5) |
| B1 temperature columns: `t_mean_3h`, `t_mean_6h`, `t_mean_24h`, `t_change_3h`, `t_change_6h`, `t_min_day`, `t_max_day`, and `t_cold_snap_days` (7 to 14) | ENS | ENS | ENS | ENS | Analysis (ERA5) |
| B1 irradiance columns: `sw_mean_3h`, `sw_mean_6h`, `sw_mean_24h`, and `csi_change_3h` (15 to 18) | ENS | ENS | Analysis (CAMS) | Analysis (CAMS) | Analysis (CAMS) |
| CAMS day summary: `cams_prev_day_mean` and `cams_prev_day_csi` (19, 20) | Columns 0 to 23: unchanged. Columns 24 to 47: UTC day D - 3 | Unchanged | Unchanged | Unchanged | Unchanged |

**The power, the date-time columns, the day type, and the whole target row are unchanged.** The
power keeps experiment 713's 27-hour rule. **Of the 14,112 weather values of a history matrix (14
rows by 48 half-hours by 21 columns), the rule changes 3,763, or 27%.** The changed values are
columns 0 to 18 at 71 tokens (1,349 values), the CAMS day summary at 24 tokens (48 values), and
columns 1 to 14 at 169 tokens (2,366 values). Check KL2 recomputes the count.

### Why each cell is buildable

- **ENS cells, weather columns 0 to 6.** The value is the `ft` table's value at the token's own
  half-hour. The newest history token is stamped 11:30 UTC on D - 1, so the run behind any history
  token was issued at or before 00:00 UTC on D - 1, which is 33 hours before the native origin,
  against a requirement of 9 hours. No history token reads the run issued at 00:00 UTC on D.
- **ENS cells, the 12 B1 window columns.** The value is experiment 437's stored value, keyed by the
  forecast origin that the `ft` table stores for the token and by the token's own half-hour.
  Experiment 437 built each stored value from that run and from the first days of earlier runs
  alone. The Candidate reads the same stored values in its target rows. No download is needed.
- **Analysis cells, CAMS (columns 0 and 15 to 18).** The CAMS source stamp of the token is at or
  before 00:00 UTC on D - 2, so the token lies in UTC day D - 3 or earlier.
- **Analysis cells, ERA5 (columns 1 to 14 at ages 7 to 15).** The newest token of the age-7 row is
  stamped 11:30 UTC on D - 6, 21.5 hours older than the 120-hour limit. `t_min_day` and `t_max_day`
  read at the latest to 00:00 UTC on D - 5, 9 hours older than the limit.
- **CAMS day summary (columns 19 and 20).** The day read is the earlier of the day before the
  token's stamp day and D - 3. One formula covers every token.

**The training phase and the prediction phase of F8's history weather read different stored tables.**
Training (stages 2 and 3) reads the `ft` table of experiment 560 and the B1 table of experiment 437
(821 runs, 2024-04-01 to 2026-06-30). Prediction reads experiment 667's extended copy of both (944
runs, to 2026-10-31). The Candidate's training rows and experiment 713's ensemble target row read
experiment 667's extended copies in both phases, because `e00667_fold.install` redirects the inputs in
both phases. The history ENS cells of F8 in training therefore come from a different table than the
target row of the same sample. F8 reads every stored table without writing to it. The training-phase
`ft` table has gaps where the prediction-phase table has none, so checks KL1, KL5, and KL7 run once in
each phase.

### Missing cells and the target row

**A training cell with no `ft` row reads the analysis, and a scored cell with no `ft` row reads
zero.** Cells before the start of the `ft` table (2024-04-01) and cells in the gaps of the
training-phase table have no `ft` row. A fallback cell at ages 2 to 6 holds an analysis value that a
live service lacks at the origin, and the fallback is confined to training. A scored cell reads the
standardised value zero, as the Candidate's standardiser does, and check KL7 counts every case.

**F8 leaves the target row as experiment 713 and the Candidate read the row, so L1 stays like for
like.** The page carries this fixed sentence beside L1:

> `C`, `M*`, and `L` all read the CAMS mean and clear-sky ratio of UTC day D - 1 in the target row.
> Under the latency that CAMS documents, a live service could not have held those two values at
> 09:00 UTC on day D before February 2026, and may not hold them since. L1 is therefore like for
> like in the history weather, and neither score is a live score.

**A report-only read-out, `T3`, bounds how much of L1 rests on the two target-row columns.** In
`T3`, the stored states of `C`, `M*`, and `L` predict with the target row's CAMS day summary moved
from D - 1 to D - 3, on 8 folds with seeds 0 and 1. No arm is trained on the moved summary, and `T3`
changes no verdict.

**Stage 1 is not refitted, and the stage 1 state of `M*` read analyses at every age.** The
Candidate's pre-training reads ERA5 and CAMS pseudo-runs up to the same cutoff, so the two networks
share the property.

## Wiring checks

**Every check from KL1 to KL10 is a wiring check, which cannot fail on a correct build, and every
threshold is fixed now.** `e<id>_checks.py` writes `work/wiring_checks_passed.json`, and the fit
script refuses a real fold without the file. A failed check stops the experiment and goes to the
manager. F8 has no capability gate, because F8 adds no parameter and asks the neural network for no
new capability, so the wiring checks cannot show that the network uses the history weather.

| Check | What must hold | Threshold |
|---|---|---|
| KL1, rule off | With every cell set to "analysis", F8's batch equals the batch of experiment 713's `Source.assemble` for 64 samples per fold, in both phases; F8's `predict` on the stored stage 3 state of `M*`, fold 2025-11, seed 0, equals the stored forecast file | Bit-identical inputs; at most 1e-6 of capacity |
| KL2, the table | For 3 origins per fold, including one in each week that holds a clock change, the source of every history value equals the table above, computed a second time with `datetime` arithmetic and no grid index; the count of changed values is 3,763 per sample | Exact |
| KL3, poisoned sources | For 64 training samples per fold and every scored origin, each stored source array is copied and every forbidden value is set to 1e6 and then to NaN: ERA5 values stamped after 09:00 UTC on D - 5; half-hourly CAMS values with a CAMS source stamp after 00:00 UTC on D - 2; CAMS day summaries of days after D - 3; `ft` values and stored B1 values whose run has a forecast origin at or after `o` (including the run of day D). The assembled history inputs must not change, except in the counted fallback cells of the training samples | Bit-identical outside the exempt cells; the number of exempt cells equals KL7's count for the same samples; no exempt cell at a scored origin |
| KL3r, rebuilt analysis window columns | For 2 Primaries and the 3 origins per fold of KL2, the 12 analysis B1 window columns and the CAMS day summary are rebuilt from the poisoned ERA5 and CAMS weather of KL3, because the cached columns are built once from clean data and a poison of the raw arrays never reaches them | Bit-identical |
| KL3n, negative controls | Each poison of KL3 and KL3r moved one step earlier must change the inputs: ERA5 after 09:00 UTC on D - 6; half-hourly CAMS after 00:00 UTC on D - 3; day summaries after D - 4; `ft` runs and stored B1 runs with a forecast origin at or after 09:00 UTC on D - 1 | Every control changes at least one input |
| KL4, per-batch assertions | On every batch of every fit and prediction: each ENS cell's stored forecast origin is at or before the token's stamp and at or before `o`, or the cell is a counted fallback; each ERA5 cell is stamped at or before `o` - 120 h; each CAMS cell has a CAMS source stamp at or before 00:00 UTC on D - 2; each day summary is of D - 3 or earlier | No assertion raises |
| KL5, stored B1 against the `ft` series | For every ENS cell of 3 origins per fold, in both phases, the nine trailing columns recomputed from the `ft` table's temperature and shortwave equal experiment 437's stored values wherever the `ft` series has no gap inside the window | At most 1e-4 in raw units; the share of cells compared is printed per column |
| KL6, inherited checks | Experiment 713's poisoned-power check and next-day-run canary pass through F8's source | As experiment 713 |
| KL7, coverage | Per fold, in both phases: the share of ENS cells that fall back to an analysis in training, and the share that read zero at scored origins. Phase 0 prints the fallback share per fold, per arm (`L` and `E`), and per Primary, before KL7 is evaluated | Under 2% of training cells, pooled over the 16 Primaries per fold and arm, and never raised after the measurement; under 1% of scored cells; no scored cell reads an analysis fallback |
| KL8, look-ahead statement | Experiment 713's row-by-row statement is printed for `L`, `Z`, and `E`, with one line per block of the table, the source, the issue or valid time, and the margin to the lag; `--check-data` prints PASS | The statement prints PASS |
| KL9, the type of the source | In stage 2, in stage 3, and at prediction, the number of calls of `LiveSource.assemble` equals the number of batches; `type(world.source) is LiveSource` after the stage 3 source is built; in one stage 3 batch of `L`, the history inputs differ from the same batch of `Z` only in the changed cells, and in at least one value of each of the three changed blocks | Exact |
| KL10, substitution | With the rule on, the `ft` weather is replaced by the analysis array and the stored B1 lookup by the dense analysis B1, on the same grid keys. The assembled batch must equal the rule-off batch in every value except the 48 day-summary values, which must equal the summary of D - 3, and except the scored cells that read zero for a missing `ft` row, which KL7 counts and the check exempts. For 50 random ENS cells, the assembled raw value also equals an independent Polars lookup of the raw summary row at the token's valid time | Bit-identical; at most 1e-4 in raw units for the 50 cells |

**KL3, KL3r, and KL3n together are the known-answer test that no history input uses information
unavailable at the origin, and KL10 catches an ENS value placed at the wrong half-hour.** KL1 runs
with the rule off, KL2 checks the source labels, KL3 checks only what is available at the origin, so
a shift inside the first day of one run stays available, and KL5 compares two ENS sources read
through the same index, so a shift shared by both cancels out.

**Stage 3 would train on the analysis weather with every other check passing, unless `LiveSource`
overrides `with_power_in`.** In experiment 713, `Source.with_power_in` builds a plain `Source`, and
the stage 3 code calls it for every fit. KL9 counts the calls of `LiveSource.assemble` per batch and
asserts the type of the source.

**The `Z` fit of fold 2025-11, seed 0, is a reproduction check in phase 0, with tolerances fixed
now.** Before the `Z` fit, phase 0 measures on the CPU the seed-to-seed spread of the best
validation loss and of the CRPS: from the 40 stored `M1` logs, the across-seed standard deviation of
`best_val` per stage (fold 2025-11, and all folds pooled), and from the stored `M*` forecasts the
across-seed standard deviation of fold 2025-11's headline CRPS in each group. The reproduction check
requires the best validation loss of stage 2 and of stage 3 within 0.01 nats of the stored `M*`
value of the same fold and seed, and the headline-band CRPS of the fold within 0.02 points of the
stored `M*` forecast of the same seed, in each group. A miss inside one across-seed standard
deviation goes to the manager as GPU non-determinism. A miss beyond one standard deviation stops the
experiment as a suspected fault. Neither tolerance changes after the `Z` fit.

## Contrasts, thresholds, and decision rule

**F8 has four counted readings, so each interval is read at 98.75%, as in experiment 713.** Each
contrast is CRPS on experiment 713's row set, on the headline band, over the 8 folds, from
seed-averaged forecasts. The interval is the month-block bootstrap with 100,000 resamples. The t
guard is the t interval on the 8 monthly differences at 98.75% with 7 degrees of freedom. The seed
guard is the same sign for each paired seed taken alone. The noise-floor guard is a point estimate
of at least the noise floor.

| Seeds per arm | Noise floor, six signed-MW Primaries | Noise floor, eight MVA-only Primaries | `δ`, six | `δ`, eight |
|---|---|---|---|---|
| 2 | 0.054 | 0.142 | 0.094 | 0.25 |
| 5 | 0.034 | 0.090 | 0.06 | 0.16 |

**The noise floors and `δ` are the pre-registered values of experiment 713, taken from the
Candidate's stored seeds, and the decision rule keeps them whatever the measured noise is.** The
smallest gain worth porting is τ = 0.05 points, because L1 asks experiment 713's question of
replacing the Candidate's whole neural network.

**The verdict is read from L1 alone with experiment 713's tables.** With D the point estimate and
[L, U] the 98.75% interval, a group's reading is the first row that holds.

| Order | Group reading | Condition |
|---|---|---|
| 1 | clear loss | L > +τ, and all three guards hold |
| 2 | loss | L > 0, and all three guards hold |
| 3 | gain | U < 0, D <= -τ, and all three guards hold |
| 4 | small gain | U < 0, and the seed guard holds for at least 4 of the 5 seeds (both seeds in the two-seed case) |
| 5 | no detectable difference | L > -`δ` and U < +`δ` |
| 6 | cannot tell | anything else |

| Order | Verdict | Condition on L1 |
|---|---|---|
| 1 | `refuted` | "clear loss" in either group |
| 2 | `not supported` | "loss" in either group; or "no detectable difference" in both groups |
| 3 | `supported` | "gain" in both groups, and every guard below holds |
| 4 | `partly supported` | "gain" in one group and no "loss" in the other; or "gain" in both groups with a guard below failing |
| 5 | `small gain` | "small gain" in at least one group, with no "gain" and no "loss" in either group |
| 6 | `inconclusive` | anything else |

**Guards for `supported`:** the sign of L1 holds with each Primary left out in turn and with each
month left out in turn; the q01 and q99 pinball loss of `L` is within 0.05 of the Candidate's in
each group; and the normalised mean absolute error of the median moves the same way as CRPS. A
`supported` verdict does not require a capability test, because F8 has none by design.

**L2 gives one pre-registered sentence per group, changes no verdict word, and adds no seed.**

| Order | Reading of L2 | Sentence on the page |
|---|---|---|
| 1 | "loss" or "clear loss" | The lab shortcut accounted for a detectable part of the score of `M*` in this group |
| 2 | "no detectable difference" | Removing the lab shortcut changed the score of `M*` by less than `δ` in this group |
| 3 | "gain" or "small gain" | `L` scored better than `M*` in this group, which the design did not predict; audit trigger 5 of experiment 713 fires |
| 4 | "cannot tell" | The interval of L2 is too wide to say |

**One sentence about experiment 713 is fixed now.** The page says that experiment 713's small gain at
the six signed-MW Primaries survives live-buildable history weather only if L1 reads "gain" or
"small gain" in that group. F8's verdict does not close or reopen the last-14-days axial line. A
`not supported` or `refuted` verdict in F8 means that the line's only gain was a product of the lab
shortcut, and the page says so.

**A rule on arm `Z`, fixed now.** If `Z` minus `M*` on the quick-look folds, with seeds 0 and 1
paired, exceeds the two-seed noise floor in absolute value in either group, experiment 713's audit
trigger 1 fires before the counted L2 is read. Phase 3 therefore runs before the counted two-seed
reading.

### Report-only readings

**Every reading in this list is read at 95% and changes no verdict.**

- `S` minus `C`, `S` minus `M*`, and `L` minus `S` on 8 folds, seeds 0 and 1.
- `L` minus `Z`, `Z` minus `M*`, `L` minus `E`, and `E` minus `C` on the quick-look folds, seeds 0
  and 1.
- `T3`: `L` minus `C` and `L` minus `M*` with the target row's CAMS day summary moved to D - 3, on 8
  folds, seeds 0 and 1, printed beside L1 and beside the two-seed L1.
- L1 and L2 for all 16 Primaries pooled, by lead band, by Primary, by fold, by half-hour of valid
  time, with one Primary left out, and with one month left out, with the normalised mean absolute
  error of the median beside every CRPS contrast.
- L1 and L2 on lead days 2 to 5 against lead days 6 to 10, because the newest history rows are the
  rows that the rule changes.
- The tails, the cold winter nights, and the curtailment figure through experiment 713's protocol
  scorer.
- The across-day attention and the read-out weights of `L` and `M*` on the same samples, by age.

## Order of work and the staging rule

**The quick look comes first: `L` on the four quick-look folds with seeds 0 and 1, and `S` on the
same folds.** Every quick-look number is labelled interim, and the quick look decides no verdict.
If `L` minus `M*` on the quick-look folds is above +0.15 in both groups, or above +0.10 in both
groups and positive in at least 3 of 4 folds in both groups, the implementer stops and asks the
manager for a wiring audit before the other four folds run. Swap (a) predicts a change near zero, so
a large loss is more likely a fault than a finding. The other four folds run after the audit
whatever the audit finds.

**The staging rule for seeds 2 to 4 is fixed now.**

> After the 8-fold, 2-seed reading of the counted contrasts, seeds 2 to 4 run only if the point
> estimate of L1 (`L` minus `C`), the one counted treatment contrast of this rule, is below zero in
> at least one group. If L1 has a point estimate at or above zero in both groups, the experiment is
> written up at 2 seeds with the pre-registered two-seed values, and the page states that the
> futility rule fired. L2 (`L` minus `M*`) is not a counted treatment contrast of this rule: no
> reading of L2 adds or removes a seed. This rule was added before any treatment score was read.

**L1 alone is the counted treatment contrast of the rule, because L1 is F8's gain contrast.** L2 is
predicted to lie near zero, so a clause on L2 would send the experiment to 5 seeds about three times
in four and would spend 3.4 GPU-process-hours on a sentence that changes no verdict word. No seed is
added or dropped for any other reason after a score is read.

**Every fit writes a stopping-pass record, and the records go to the manager with the quick look and
again with the 8-fold reading.** The record holds, for stage 2 and for stage 3: the passes run, the
best pass, the optimiser steps at the best pass, the validation loss at pass 1 and at the best pass,
whether the 60-pass cap was hit, and whether the 120-pass rerun ran. The steps for `L` and `M*` are
derived by one formula, the best pass times ceil(`pairs` / 256), and printed beside the stored
record of the `M*` fit of the same fold and seed. The 40 stored `M1` fits ran 11 to 20 passes of
stage 2 (median 13, median best pass 5) and 10 to 23 passes of stage 3 (median 15, median best pass
7). None hit the cap.

## The training-budget check

**F8 is less exposed to slow learning than earlier follow-ups, because every parameter of `L` starts
at a trained value, F8 asks for no new capability, and the untrained swap (a) lost no skill at the
six signed-MW Primaries.** Two exposures remain, and both would bias L2 upwards (against `L`). The
best state of stage 2 lay inside the 300-step warm-up for 39 of the 40 stored fits of `M1`, so the
trunk moves little in stage 2. The validation loss also moves by up to 0.065 nats between passes, so
patience 8 can stop a fit whose slow adaptation is still under way. A positive L2 could therefore
mean that `L` stopped before the trunk adapted to the new inputs, and not that the live inputs hold
less information.

**Three checks separate the two readings, and none of the three changes the counted recipe.**

1. **The stopping-pass records.** An `L` fit whose best pass in stage 3 is under half the best pass
   of the `M*` fit of the same fold and seed is flagged and listed on the page.
2. **`L` minus `S`.** If `L` minus `S` is inside +/-0.02 in both groups, the page says that the
   retraining did not detectably change the forecasts.
3. **A budget probe, report-only, which runs only on a flag fixed now.** The probe runs if the
   two-seed point estimate of L2 on 8 folds is above the two-seed noise floor (+0.054 or +0.142) in
   either group, or if check 1 flags more than 4 of the 16 fits of `L` in phases 1 and 2. The probe
   runs before phase 6. Arms `L-P` and `Z-P` are `L` and `Z` with patience 30 and a cap of 120
   passes in stages 2 and 3, on folds 2025-11 and 2026-06, seed 0. The probe reads the validation
   loss alone: for each fit, the median over 5 passes ending at the global best pass minus the same
   median at the patience-8 pass. If the gain of `L-P` exceeds the gain of `Z-P` by more than 0.01
   nats on both folds, the page states that patience 8 may under-train `L`, and the manager is told
   before phase 6. The page prints the spread of the 5 passes beside each gain, and says whether the
   probe ran and gives the two values of the flag.

**A change of the recipe after a probe flag belongs to the maintainer.** A longer budget for `L`
alone would break the pairing with `M*`, so the change would need `M*` refitted on the same budget,
which is a new experiment.

## Compute and machine rules

**The plan needs about 11.7 GPU-process-hours at the slowest measured fit time of 0.14 per
warm-started fit with its prediction, and about 6 at 0.07, against a reporting ceiling of 12.**
F7's warm-started fits measured 0.07 and F4's measured 0.14. The median GPU time of stage 2 plus
stage 3 in the 40 stored `M1` logs is 124 s per fit, so data building takes most of the 0.14. The
counted two-seed reading on 8 folds needs about 4.0 GPU-process-hours, because the reading is taken
after phase 3.

| Phase | Fits | GPU-process-hours at 0.14 |
|---|---|---|
| 0. The cache of the ENS arrays, built on the CPU; KL1 to KL10; a smoke fit with timing and memory peak of `L` on fold 2025-11, seed 0, whose outputs are never scored; the reproduction fit of `Z` on fold 2025-11, seed 0 | 2 | 0.4 |
| 1. Quick look: `L` on the quick-look folds, seeds 0 and 1; `S` predictions on the same folds | 8 | 1.3 |
| 2. `L` on the other 4 folds, seeds 0 and 1; `S` predictions | 8 | 1.3 |
| 3. `Z` on the quick-look folds, seeds 0 and 1, less the fit of phase 0; then the counted two-seed reading and the futility rule | 7 | 1.0 |
| 4. `T3`: predictions of `C`, `M*`, and `L` with the target-row summary moved, 8 folds, seeds 0 and 1 | 0 | 0.3 |
| 5. Budget probe, only if the flag fires: `L-P` and `Z-P` on folds 2025-11 and 2026-06, seed 0 | 4 | 1.6 |
| 6. `L` on 8 folds, seeds 2 to 4, unless the futility rule fired | 24 | 3.4 |
| 7. `E` on the quick-look folds, seeds 0 and 1 | 8 | 1.1 |
| 8. Attention extraction and the look-ahead statements | 0 | 0.5 |
| 9. Reserve for reruns at the 120-pass cap | | 0.8 |
| **Total** | **61** | **about 11.7** |

**Phase 0 makes the first projection, and the order of cuts is fixed now.** Phase 0 projects the
total from the GPU time of the two phase-0 fits (the smoke `L` fit and the reproduction `Z` fit)
and from one timed `T3` prediction (fold 2025-11, seed 0, the stored states of `C` and `M*`, and the
smoke state in place of `L`). If that projection, with the probe, exceeds 12 GPU-process-hours, arm
`E` is cut before phase 1. The implementer reports a second projection to the manager from the
median of the first 3 fits of phase 1, and stops and asks before crossing the ceiling. The order of
cuts is `E` (phase 7), then `Z` cut to the reproduction fit of phase 0, then the probe cut to fold
2025-11 alone. Phases 1, 2, and 6 carry the counted readings and are cut only by the futility rule
for phase 6. If `Z` is cut, the rule on `Z` cannot be applied, and the page says so.

**Machine rules.**

- **The ENS arrays are cached first, in a CPU step, so that the fits stay inside the 9G scope.**
  Phase 0 builds the ENS weather and the ENS B1 values on the grid once, as an `.npy` cache under
  F8's own folder, with the forecast origin of each cell and the `found` mask. The fits load the
  cache alone and never the parquet tables.
- Every fit runs under `systemd-run --user --scope -p MemoryMax=9G -p MemorySwapMax=0 -p
  OOMPolicy=continue`, through a `run.sh` that calls the virtual environment's Python directly. The
  smoke fit prints the host peak, and the scope is confirmed or raised with the manager before
  phase 1.
- One thread for every numerical library, and each GPU process stays at or under 6 GB of GPU memory
  as `nvidia-smi` reports the process total. At most two GPU processes of F8 run at once, and only
  with the manager's slot.
- The implementer sends "fits starting" to the manager before phase 0 and before phase 1, and checks
  in before any workstation job. The rebuild of KL3r runs on the CPU with one thread and is
  estimated at 2 to 3 CPU-hours.
- No download: every input is a stored table.

## Code structure and review gates

**F8 imports experiment 713 from a frozen checkout, edits no file of experiment 713, and asserts
that the code hash logged with each stage 1 state equals the hash that the frozen checkout
computes.** `LiveSource` subclasses experiment 713's `Source`, replaces `hist_weather` by the rule
after the parent's `assemble`, carries the assertions of KL4, and overrides `with_power_in`. F8
imports experiment 713's `train_stage` unchanged, and copies a function only if its body must change.
F8 writes nothing into the folder of experiment 713, 667, 437, 560, or 632.

**The review gates run in this order.**

1. An Opus design review of the design note, resolved before this page. The review asks for no
   second design review after the fixes.
2. This plan page and its row on the theme's plan page, pushed to the experiment's branch.
3. An Opus code review of every new script by a fresh agent before any run. The checklist: the three
   lags and the CAMS stamp convention (period end), tested on the CAMS source stamp; the override of
   `with_power_in`, and every other place where experiment 713 constructs a `Source`; the key of the
   stored B1 lookup; the standardisation of the `ft` and stored B1 values with the fold's
   statistics; the fallback in training and the zero at scored origins; the poison applied before any
   lookup, the rebuild of KL3r, and the exemption limited to counted cells; the table that each phase
   reads; the seed pairing in the scorer; and the frozen checkout and the code hash.
4. A second Opus review of the fixed code, before phase 0.
5. The wiring checks, phase 0 with the reproduction check, and "fits starting" to the manager.
6. The fits, a result review, an audit if a trigger fires, the write-up, a numbers review, and a
   prose review.

## Disclosures

- **Experiment 713's known-answer gate failed KA3 and KA12 and was waived by the maintainer on
  2026-11-23.** F8 builds on `M*`, so F8 makes no statement about following level shifts or
  recognising an era.
- **F8 has no capability test of its own,** and the wiring checks cannot show that the network uses
  the history weather.
- **The three lags are assumptions,** with the source of each. The CAMS table has no availability
  stamp and may hold reprocessed values.
- **The target row's CAMS day summary of D - 1** carries the fixed sentence above, with the `T3`
  read-out beside L1.
- **The training fallback reads analyses** in the counted cells of KL7, and no scored cell does.
- **Stage 1 is experiment 713's,** trained on analyses at every age.
- **A positive L2 may come from a short training budget,** read beside the three checks above, and
  the page says whether the probe ran.
- **The tolerances of the reproduction check (0.01 nats and 0.02 points) have no measurement behind
  them,** so a miss inside one across-seed standard deviation goes to the manager and is not read
  alone as a fault.
- **Estimates without a measurement:** the probe's 0.4 GPU-process-hours per fit, the 0.3 of `T3`,
  the 85 MB of added device memory, the 77 MB of the cache per phase, and the 2 to 3 CPU-hours of
  KL3r.
- **Every Primary score, forecast, and plot is of a Primary substation.** No page names a generator,
  and no page plots or tabulates a metered generator's time series.

## What each outcome would mean

- **L1 reads "small gain" or "gain" at the six signed-MW Primaries, and L2 reads "no detectable
  difference".** Experiment 713's result did not rest on the shortcut, and the increments that
  follow-ups F4, F7, and F9 measure are real in live terms.
- **L1 loses its gain and L2 reads "loss".** The shortcut produced the gain, and each follow-up
  increment would need a rerun on live inputs before any port.
- **L1 loses its gain and `L` minus `S` is near zero with early stopping passes.** The training
  budget is the first suspect, and the probe's reading decides whether to ask the maintainer for a
  refit of `M*` and `L` on a longer budget.
- **`T3` differs from the two-seed L1 by more than the two-seed noise floor in a group.** A
  detectable part of L1 in that group rests on two target-row values that a live service may not
  hold.
- **`E` matches `L`.** A live service can drop the ERA5T feed and the half-hourly CAMS feed for the
  history rows.
- **The futility rule fires.** L1 is at or above zero in both groups at 2 seeds, and the page
  reports the two-seed reading with the two-seed values.

## Method

The experiment is planned and has no fitted result yet, so these bullets describe the design.

- **Architecture and training regime:** experiment 713's `MsaNet`, stages 2 and 3, warm-started from
  the stored stage 1 state, with frozen encoders and no new parameter.
- **Measurement:** CRPS in points of capacity on experiment 713's row set, month-block bootstrap
  intervals, no valid time on or after 2026-07-01 scored.
- **Look-ahead statement for `Z`:** the statement of `Z` shows negative margins of up to 111 h by
  design, because `Z` reads the analysis in every cell. KL8 requires that `Z` shows them, which proves
  that the statement detects a look-ahead. KL8 requires no negative margin for `L` or `E`.
- **Standing practices followed:** monthly refit; cleaned data; fitted states saved outside git;
  stored forecasts reused as controls; a screen on four folds first; forecast example plots.
- **Plots, many, as every MSA write-up carries:** the table of cells drawn as a 14-by-48 grid per
  column group; the ENS value against the analysis at the changed cells, for temperature and for
  shortwave, by age; L1 and L2 per fold and per Primary, with the two groups kept apart; L1 and L2
  by lead day; day-ahead and 14-day forecast time series of `C`, `M*`, and `L` at one Primary of each
  group, on a sunny window and a dull window; the across-day attention maps and read-out weights of
  `M*` and `L` on the same samples; the stopping passes of `L` against `M*`; and the validation
  curves of the probe if the probe ran.

## Reviews

An Opus agent reviewed the design twice before this page was written, and the design was revised
after both reviews. Still to come: an Opus code review of each new script before any fit, a second
review of the fixed code, the wiring checks, a result review, an audit if a trigger fires, a numbers
review, and a prose review.

## Files

**This page names experiment 713 and experiment 667 in plain text, because their pages are not yet on
`main`.**

- `data/ml-experiments/briefs/msa_architecture/f8_design.md` and `f8_design_review.md` (read,
  private): the design that this page condenses.
- `ml-experiments/00725-msa-live-buildable-history-weather/` (to be created): `run.sh`, the rule,
  the cache builder, `LiveSource`, the fit, the prediction, the checks, the scorer, the figures, and
  the report script, none of which exists yet.
- Read from experiment 713: the stored states, the stored forecasts of `M*`, the scorer, and the code
  at the frozen checkout.
- Written by this experiment (to be created): fitted states under
  `data/ml-experiments/00725-msa-live-buildable-history-weather/work/` and predictions under
  `predictions/`, never committed.
