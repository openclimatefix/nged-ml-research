---
id: "00677"
slug: building-up-the-candidate-demand-network-one-component-at-a-time
title: >-
  How much does each component of the Candidate demand network contribute to its one-year backtest
  score, when the network is built up from the date-time encoder alone, one component per step?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores about 13 networks per fold and seed as private Parquet files and weights;
  (2) several designs are defensible (what "without injected events" removes, which network builds
  the lag-free table, quarterly or monthly refits); (3) it spans code from experiments 545, 559, 560,
  574, 591, 598, 606, 610, 612, 620, and 667, whose callers cannot be named without searching; (4) it
  touches no serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  Descriptive, not a hypothesis test. A reverse ablation measures the order-dependent contribution
  of each Candidate component. Predictions that could fail are listed under "Predictions" below.
prediction: >-
  P1: step 8 (the full Candidate) reproduces experiment 667's NRA output within 0.10 points of
  capacity CRPS at monthly refits. P2: the two lags (steps 4 and 5) give the largest single drop in
  median MAE at days 0 to 3. P3: steps 6 to 8 together lower the follow of a held-out injected shift
  by at least 0.03 against step 5 trained with the same injected events. P4: no step before step 4
  beats the manual heuristic on the normalised MAE of the median at days 0 to 14.
decision_rule: >-
  Descriptive; no ranking and no claim of benefit. A step contrast is called separable only if the
  block-bootstrap interval over weeks excludes zero and the contrast exceeds the spread between
  seeds of the same step. Stop rules: if step 8 misses P1 by more than 0.25 points, stop and report
  to the manager before steps 1 to 7 are interpreted.
benchmark: >-
  The manual heuristic, climatology, the stored Candidate of experiment 620, experiment 667's
  Primaries file (step 8 at monthly refits), the Primary demand research default 3 (the GAM,
  experiment 322 and 251), and BG2 (the XGBoost model, experiment 87 arm p_s6_bg2).
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  Normalised MAE of the median, CRPS and twCRPS in points of capacity, q01, q05, q95, and q99
  coverage and pinball loss, by lead band (days 0 to 1, 1 to 3, 3 to 7, 7 to 14), by regime slice,
  and by condition; follow and absorption on a held-out injected bank; avoidable-curtailment cost.
---

**Plan only.** This page holds the plan; the results sections are written after the fits.

## Plan

### Question and reading order

The Candidate is the class 2 network of experiment 620 (arm P7+D50 of experiment 610). The ladder
adds one component to the previous step and scores every step the same way. The order matters: a
reverse ablation gives each component's gain given the components already present, and says nothing
about what removing that component from the full Candidate would cost.

### Components of the Candidate and where each sits on the ladder

Held fixed in every step (present wherever the step has the part they belong to):

- Training loss and head: a three-component Gaussian mixture, negative log-likelihood, 11 quantiles
  from the mixture CDF, sorted (so no step has crossing quantiles).
- Widths (8-wide Primary embedding, encoders 64 to 32, mixer 64), dropout 0.1, AdamW with weight
  decay 1e-3, batch 8192, 20% of rows per epoch, patience 8, gradient-norm clip 1.0.
- The 730-day window, the 90-day recency half-life, the E3 refit rule (7-day embargo), the slow
  per-Primary output offset D5, and the cut of D15 and D18 before 2025-03-01.
- The 14 weather means, the 7 spread columns, the 2 lead columns, the 21 date-time columns, and the
  14 B1 columns (thermal lag and solar gain) read by the weather encoder.
- No band-width factor is applied at any step (667's default); the factor 1.0934 is a Candidate-only
  post-processing step and is reported as a separate row at step 8 if 667 applies it.

Added where:

| Step | Components added |
|---|---|
| 1 | Date-time encoder, the Primary embedding, the mixer. The weather input is removed, not zeroed, so the mixer sees only the date-time embedding. |
| 2 | Weather encoder alone (B1 columns included), no date-time encoder. |
| 3 | Both encoders into the mixer, with the spread and lead columns. Version a: trained on ENS only (2024-04 onward, no pre-training, nothing frozen). Version b: ERA5 and CAMS pre-training, then fine-tuning on ENS with both encoders frozen (the chain of experiment 620). |
| 4 | One lag encoder at 24 hours, with its expected-weather and expected-date-time embeddings at the lag's own time, the lag-ID embedding, and block dropout of lags (30% of days). |
| 5 | A second lag at one week through the shared lag encoder. Two versions: with and without injected events (below). |
| 6 | The gap columns: power at the lag time minus the lag-free table, plus the `exp_missing` flags (arm P3 of experiment 606). |
| 7 | The exponentially weighted averages of the gap, half-lives 1, 3, and 7 days (arm P5 of experiment 606). Signal dropout D50 applies from here on. |
| 8 | The 42 neighbour columns (arm P7 of experiment 606). Step 8 with injected events is the Candidate. |

Steps 6 to 8 need the lag-free table T. T is the median of the step-5 network of the same version and
seed with both lags masked, taken at the freshest origin with lead at most 24 hours. T is built once
per fold, version, and seed and shared by steps 6 to 8 (in-fold, as in the Candidate). Steps 6 to 8
therefore depend on the step-5 fits, and each is fine-tuned from the step-5 weights with the new
columns zero-initialised, as the Candidate does. The 606 reference month for the signal noise is the
latest month before each fold's cutoff, not the Candidate's fixed 2026-06 (that month is future data
for earlier folds).

### What "without injected events" removes

The Candidate trains on the lag-only synthetic shifts of experiment 581 (5% of each Primary's row
span, depths 0.15, 0.30, 0.50 of capacity, lengths 3, 7, 15, 30 days), mirrored into one
neighbour's summaries from step 8 (and from step 6 for the gap columns). The brief names the 653
injector; the Candidate does not use it, and the 660 page tells readers to import its own module. The
plan resolves the disagreement as follows, and the plan review is asked to confirm it:

- The **with** version is the Candidate recipe as it stands (581 shifts), so step 8 with injection
  can reproduce 667.
- The **without** version removes those shifts from the lag inputs of the fine-tuning rows, and from
  the neighbour mirror at step 8; nothing else changes.
- The **held-out scoring bank** is built with the experiment 653 injector (the canonical generator),
  with a seed family disjoint from every training seed. It injects lag-only shifts, as the Candidate
  trains on, so follow (the share of an injected shift that the forecast median copies) and event-free
  CRPS are read on a bank that neither version trained on. Whether to inject into targets as well is
  a design question for the Discuss session.

The real long-and-deep cell is nearly empty (experiment 665 found none of NGED's 77 logged shifts
lasting more than a day in the long-and-deep type), so the real-record slices are descriptive and
every slice reports its rows and events first.

### Refit cadence, window, and seeds

- Window: the 12 months of experiment 667 (daily 12:00 UTC origins, 2025-11-01 to 2026-10-31,
  14-day horizon). Experiment 667's folds use ENS history from 2024-04-01, so the early folds have
  fewer ENS months. The window overlaps the development months (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06) and the final-test window of experiment 620 (2026-07 to 2026-10); the Candidate's settings
  were tuned on 5 of the 12 months, which favours the final steps. The year is descriptive, and no
  model-selection choice is made from it.
- Default: **quarterly refits** at the starts of 2025-11, 2026-02, 2026-05, and 2026-08, each scoring
  three months. Step 8 with injection at the monthly cadence is experiment 667's own file, reused
  without refitting; the quarterly step 8 will differ from it by cadence alone, and the page says so.
- Seeds: 3 for steps 1 to 5, 5 for steps 6 to 8. Experiment 610's audit found a seed spread of 0.011
  against an effect of 0.153, so each contrast shows the per-seed values beside it.
- Intervals: block bootstrap over weeks (7-day blocks of origin days), with a minimum detectable
  effect printed beside each contrast and a Bonferroni count over the contrasts.
- Leakage check: a per-row assert that every lag and every T entry was observed before 06:00 UTC on
  the origin day (the Candidate's lags are at least 24 hours before its 09:00 origin). The ENS 00Z run
  counts as available at 09:00.

### Reference arms

- The manual heuristic and climatology, from `baseline_forecasters`, as in 620 and 668.
- The stored Candidate of 620 (one fit, 2026-07 fold only) and 667's Primaries file.
- **GAM**: the Primary demand research default 3 (experiments 251 below 168 h and 322 above; six MW
  Primaries, 193 of the 241 scored origin days up to 2026-06-29, origin 09:00 UTC, joined on valid
  time and series). Weekly refit on the whole history, as stored.
- **XGBoost**: BG2, arm `p_s6_bg2` of experiment 87 (16 Primaries, leads to 168 h, origins to
  2026-06-30). Weekly refit, as stored.
- Neither stored reference reaches 2026-11-01 to 2026-10-31 in full, so both are scored on their
  overlap with the ladder's rows; the ladder is also scored on those shared rows. A CPU re-run of
  either to cover the full year is not planned; the manager is asked first if it is wanted.

### Outputs

For each step and version: a horizon plot (0 to 14 days, normalised MAE of the median, y axis not from
zero) against the manual heuristic, the previous step, the GAM, and XGBoost; the regime table (all
rows; inside long and deep events; the 7 days after; none and at least 7 days after one ended; long
and deep); CRPS, twCRPS, tail calibration, and avoidable-curtailment cost beside MAE; results by
condition through `studies.eval_protocol`; per-Primary CRPS for the six MW and eight MVA groups;
step-to-step contrast plot with intervals; time-series figures (a cold week, a warm week, the
verified D5/D16 transfer of 2025-12-30 to 2026-01-14) at several lead times, at least 280 px per
panel and at most two panels per figure. The headline figure (all steps on one horizon plot) and the
take-home lead the page.

Each step's forecasts are written to a private Parquet file
`data/ml-experiments/00677-*/backtests/step<N>_<version>.parquet` with the 667 columns, and weights
are saved if the total is under about 2 GB.

### Prevailing output

The survival head (R-haz) is trained on the frozen Candidate's quantiles with a bank of injected
shifts and needs the signal columns, so it cannot exist at steps 1 to 7. Prevailing rows come from
667's file at step 8 or are left out; `NRA_or_prevailing` is `"NRA"` in every other file.

### Compute estimate

Per seed and fold, from 667's logs (pre-training about 80 s, step 2 about 20 s, fine-tuning about
65 s): pre-training for three lag configurations (lag-free, one lag, two lags) about 240 s; step 2
and ENS-only fits for about 13 networks about 350 s; fine-tuning for about 10 lag-bearing networks
about 650 s; total about 1,240 s, or 0.34 GPU-hours.

| Design | GPU-hours |
|---|---|
| Quarterly, 3 seeds (steps 1 to 5) + 5 seeds (steps 6 to 8) | about 5 |
| Monthly, same seeds | about 16 |

Both are under the 20 GPU-hour ask line before table builds (CPU), scoring, and the injected bank.
The plan uses the quarterly design; monthly refits with extra seeds are asked before they run. The
GPU is shared and other backtests have priority.

### Predictions that could fail and stop rules

- P1 to P4 (front matter). A miss on P1 is reported before the other steps are interpreted.
- Stop and tell the manager if a fit would take the total above 20 GPU-hours.

## Method

To be written after the fits.

## Files

To be written after the fits.
