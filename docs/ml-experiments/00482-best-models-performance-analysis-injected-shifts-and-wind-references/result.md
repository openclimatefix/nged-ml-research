---
id: "00482"
slug: best-models-performance-analysis-injected-shifts-and-wind-references
title: >-
  How do the lagged and lagless demand forecasts respond to injected load shifts and level changes,
  and how does the wind forecast compare with its references?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: none
size: >-
  medium. Stored: tables and figures only, nothing fitted (private folder
  data/ml-experiments/00482-best-models-performance-analysis-injected-shifts-and-wind-references/).
  Several designs would show the same behaviour, so the plan fixes the eight scenarios and the lag
  spread before any chart is read. Callers: the scripts import the folders of experiments 402, 404,
  428, 435, 446, 454, 456, 462, and 468. Production serving path and degradation rule: not touched.
  One Sonnet code review before the first scoring run, one Sonnet results review, one Sonnet prose
  review.
hypothesis: >-
  A forecast that reads lagged power follows an injected load shift that the neighbours mirror less
  than one that does not, the forecast that mixes lagged and lagless heads (K2-S1) limits the harm
  of an unmirrored shift, and the pooled plain network (A-pool), which never saw an injected shift
  in training, follows the shift most.
prediction: >-
  Written before any chart is read. P1. For a shift of 3 times the lag spread lasting 14 days, the
  median of A-pool moves by at least 50% of the shift on the first day the lags are inside the
  event, the median of K2-S1 moves by under 50% of it, and the lagless forecast moves by under 5%.
  P2. For a genuine level change (lags and target move together), K2-S1 and S1 follow the new
  level within 1 day, and A-pool follows it too. P3. The 90% band of K2-S1 is wider inside an
  unmirrored shift than outside it, and the share of actuals inside the band stays above 80%.
  P4. Wind: research default 2 has the lowest median absolute error of the stored wind references
  at each farm, in the rows experiment 461 scored.
decision_rule: >-
  This is a descriptive analysis, so no arm is adopted. A prediction counts as holding when the
  stated number holds on the rows shown. Contrasts are reported with month-block bootstrap
  intervals (wind) at a Bonferroni level set by the count of contrasts, and with the minimum
  detectable effect beside each. Demand scenario scores are blocked by Primary and by test month,
  and reported with and without the largest-scoring scenario. Interim numbers are labelled interim
  and unaudited.
benchmark: >-
  Wind: research defaults 1 and 2 (experiments 340 and 347), climatology (experiment 230), and the
  manual heuristic recomputed from experiment 303's rows. Demand: the lagless network, S1, R-inj,
  B-mix, K2-S1 (injected-trained, experiments 404, 428, and 454), and A-pool (experiment 425's arm
  A, never trained on injected shifts), each forecast on the same injected worlds.
row_set_id: none
metric: >-
  Wind: median absolute error in points of effective capacity, with the minimum detectable effect.
  Demand: CRPS and median error in points of effective capacity per scenario, coverage of q01, q05,
  q95, and q99, the 12 to 36 hour band, and the avoidable-curtailment cost at f = 0.9, 0.75, and
  0.5 at the signed-MW six.
code: ml-experiments/00482-best-models-performance-analysis-injected-shifts-and-wind-references/
---

**Plan.** Experiment 468 did Part A and the real-event half of Part B. This experiment does the
rest: wind reference contrasts, the 14-day wind chart, the normal-running-arrangement (NRA)
corrected actual and the surprise inputs beside the real-event charts, and about eight injected
scenarios on demand. Nothing is fitted, because every network already has stored weights.

## Plan details

- **Fits: none.** S1, R-inj, B-mix, and K2-S1 come from the stored networks of experiments 404, 428,
  and 454. A-pool is experiment 425's arm A, forecast from its stored weights
  (`00425.../work/models/A`) with the injection applied to its lag columns. One stored network
  forecasts a whole test month on the GPU in 0.5 s (238,080 rows, 0.33 GB), so the whole grid is
  well under 0.5 GPU-hours. Inference uses one GPU process under 8 GB and one feeder core; the
  scoring uses up to 4 CPU cores.
- **The eight scenarios** (all on the five test months of experiment 454, seeds 0 to 2):
    1. 1 times the lag spread, 1 day, positive, no mirror.
    2. 3 times, 1 day, negative, no mirror.
    3. 1 times, 14 days, positive, mirror at two neighbours.
    4. 3 times, 14 days, negative, no mirror.
    5. 3 times, 14 days, positive, mirror at two neighbours.
    6. 1 times, 14 days, negative, no mirror.
    7. 3 times, 1 day, positive, mirror at two neighbours.
    8. 2 times, 3 days, positive, genuine level change (lags and target shifted together, no mirror).
- **Lag spread** is, for each Primary, the standard deviation of the 24-hour change in power over the
  clean rows before the test month, in fractions of effective capacity. It is fixed here and is not
  tuned.
- **Where the shift starts** is the same 72 hours into the test month for every Primary in a world,
  so lags fall inside the shift for some origins and outside it for others. The charts show origins
  before, inside, and after the shift.
- **Reuse.** Lagless head: experiment 404's frozen cross-fitted table. Mirror construction:
  experiment 462's `add_mirrors` and neighbour list. Forecast hook: experiment 428's
  `forecast_rows(modify=...)`. Real events: experiment 454's stored forecasts and experiment 435's
  `nra_correction_share.npz`.
- **Wind references.** Research defaults 1 and 2, climatology, and the manual heuristic are read
  through experiments 303, 340, and 347 (PyTorch environment), converted to MW with the old
  capacity, and cut to the half-hours of experiment 461's scored rows. Contrasts are `member_plain`
  (p99.9) minus each reference, in points of the p99.9 capacity, by month-block bootstrap.
- **14-day wind chart.** Wind only (solar and demand end at 240 hours). The forecast is the stored
  `member_plain` table before table P, because table P is fitted to 24 to 240 hours.
- **NRA and surprise inputs.** The real-event charts add the NRA-corrected actual (experiment 435) and
  the surprise of each lag: the lag minus the lagless forecast at the same origin, in points of
  capacity.
- **Standing practices followed.** Cleaned data and the D15 and D18 cut come with the stored worlds
  and forecasts. Refits are those of the stored networks (monthly). Example forecasts and
  load-shift time series plots are included. The weekly retraining and the screen folds do not apply
  because nothing is fitted.
- **Anonymisation.** Wind charts that carry megawatts stay in the private store. Any chart on the
  page uses capacity fractions and no generator time series.

## Not planned

- Solar contrasts (done in experiment 468) and the curtailment week (no label exists).
- The experiment 470 assembly winner as a reference (not yet chosen).
