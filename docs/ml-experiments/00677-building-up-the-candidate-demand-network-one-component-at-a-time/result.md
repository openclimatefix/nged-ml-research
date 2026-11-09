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
  Complex. (1) Changes what is stored: yes, about 13 networks per fold and seed, as private Parquet
  files and weights. (2) Touches the production serving path: no. (3) Touches a degradation rule:
  no. (4) Admits more than one defensible design: yes (what "without injected events" removes, which
  checkpoint builds the lag-free table, quarterly or monthly refits). (5) Spans code whose callers
  cannot be named without searching: yes (experiments 545, 559, 560, 574, 591, 598, 606, 610, 612,
  620, and 667).
hypothesis: >-
  Descriptive, not a hypothesis test. A reverse ablation measures the order-dependent contribution
  of each Candidate component. Predictions that could fail are listed in the front matter field
  `prediction` and repeated under "Predictions that could fail and stop rules" below.
prediction: >-
  P1: on the pilot fold 2026-05 (monthly refit, 2 seeds, all 14 Primaries with data, days 0 to 14),
  step 8 with injected events matches experiment 667's NRA output within 0.10 points of capacity
  CRPS. P2: the drop in median MAE from step 3b to step 5, pooled over days 0 to 3, exceeds every
  other adjacent-step drop over the same days. P3: step 8 lowers the follow of a held-out injected
  shift by at least 0.03 against step 5, both with injected events (step 5 is expected near the 0.035
  to 0.067 that experiment 610 measured, so 0.03 is a large share of it). P4: no step before step 4
  beats the manual heuristic on the normalised MAE of the median pooled over days 0 to 14.
decision_rule: >-
  Descriptive; no ranking and no claim of benefit. A step contrast is called separable only if the
  block-bootstrap interval over weeks excludes zero and the contrast exceeds the standard deviation
  of the pooled CRPS across seeds 0 to 2 of the same step. The Bonferroni family is the adjacent-step
  contrasts pooled over days 0 to 14. Stop rule: if the pilot fold misses P1 by more than 0.25 points
  (a judgement call: about twice the 0.10 tolerance and well above experiment 667's seed spread), stop
  and report to the manager before the other folds run.
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

**Start conditions.** The ladder starts its fits only after the maintainer's backtest queue is done
(the manager says when) and after [experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md)
has merged, because step 8 at monthly refits is 667's own Primaries file. Until 667 merges, the
reproduction target is experiment 620's stored Candidate. The manager approved quarterly refits
(about 5 GPU-hours) on 2026-11-09; monthly refits are not approved.

### Question and reading order

The Candidate is the class 2 neural network of experiment 620 (arm P7+D50 of experiment 610). The
ladder adds one component to the previous step and scores every step the same way. The order
matters: a reverse ablation gives each component's gain given the components already present, and
says nothing about what removing that component from the full Candidate would cost.

### Components of the Candidate and where each sits on the ladder

Held fixed in every step that has the part they belong to:

- Loss and output: a three-component Gaussian mixture, negative log-likelihood, 11 quantiles from
  the mixture CDF, sorted (so no step has crossing quantiles).
- Widths (8-wide Primary embedding, encoders 64 to 32, mixer 64), dropout 0.1, AdamW with weight
  decay 1e-3, batch 8192, 20% of rows per epoch, patience 8, gradient-norm clip 1.0.
- The 730-day window, the 90-day recency half-life, the E3 refit rule (7-day embargo), and the slow
  per-Primary output offset D5. ENS-only fits (steps 1, 2, and 3a) have no pre-trained D5 knots: the offset keeps only
  the newest knot, which departs from the Candidate and is named as a caveat.
- The 7 weather means, the 7 spread columns, the 2 lead columns, the 21 date-time columns, and the
  14 B1 columns (thermal lag and solar gain) read by the weather encoder.
- The cut of D15 and D18 before 2025-03-01.
- No band-width factor at any step. The Candidate's factor 1.0934 is post-processing; step 8 is
  reported without it, and with it as one extra row if 667 applies it.

Added where:

| Step | Components added |
|---|---|
| 1 | Date-time encoder, Primary embedding, and mixer. The weather, spread, and lead inputs are removed, so the mixer sees only the date-time embedding. |
| 2 | Weather encoder alone (B1 columns included); the date-time and lead inputs are removed. The mixer sees the weather embedding and the Primary embedding. |
| 3 | Both encoders into the mixer; the lead columns enter here (the spread columns enter with the weather encoder at step 2). Version a: trained on ENS only, no pre-training, nothing frozen. Version b: ERA5 and CAMS pre-training, then fine-tuning on ENS with both encoders frozen (the chain of experiment 620). |
| 4 | One lag encoder at 24 hours, with its expected-weather and expected-date-time embeddings at the lag's own time, the lag-ID embedding, and block dropout of lags (30% of days). |
| 5 | A second lag at one week through the shared lag encoder. Two versions: with and without injected events. |
| 6 | The gap columns: power at the lag time minus the lag-free table, plus the `exp_missing` flags (arm P3 of experiment 606). Signal dropout D50 applies from here on, because the Candidate applies it whenever signal columns exist. |
| 7 | The exponentially weighted averages of the gap, half-lives 1, 3, and 7 days (arm P5 of experiment 606). |
| 8 | The 42 neighbour columns (arm P7 of experiment 606). Step 8 with injected events is the Candidate. |

### Schedule and frozen parts of each step

All steps use the Candidate's three-stage schedule where it applies: stage 1 pre-training, stage 2
(encoders frozen, fresh mixer), stage 3 fine-tuning (recency weights, E3). "Stage" below never
means a ladder step.

| Step | Stage 1 (pre-training) | Stage 2 | Stage 3 | Frozen after stage 1 |
|---|---|---|---|---|
| 1, 2, 3a | None; random initialisation | Same as other steps, learning rate 5e-4 throughout | Same as other steps | Nothing |
| 3b | ERA5 and CAMS, lag-free | Yes | Yes | Both encoders |
| 4 | ERA5 and CAMS, one lag | Yes | Yes | Both encoders |
| 5 (with, without) | ERA5 and CAMS, two lags (shared) | Yes, shared by the two versions | Yes; "with" adds the 581 shifts to the lag inputs | Both encoders |
| 6, 7, 8 | The step-5 chain | The step-5 stage-2 checkpoint | Starts from it with the signal columns zero-initialised | Both encoders |

Steps 3b, 4, and 5 each have their own pre-training, because the lag encoder is trained in stage 1,
so the contrast between adjacent steps includes a re-pre-trained encoder. Steps 4 and onward do not
start from the step-3b checkpoint. The verification item for the code review is which stages of the
Candidate see the 581 shifts (experiment 620, lines 229 to 252); the plan assumes only stage 3.

**The lag-free table T** is the median of the step-5 stage-2 checkpoint of the same version and
seed with both lags masked, taken at the freshest origin with lead at most 24 hours. This is the
Candidate's own construction (its stage-2 network), not a fine-tuned network. Steps 6 to 8 build T
once per fold, version, and seed and share it (in-fold, as the Candidate does). The earlier review
suggested the step-3 network instead; the plan departs from that because the Candidate uses its
stage-2 lag network, and step 8 must reproduce the Candidate. The 606 reference month for the signal
noise is the latest month before each fold's cutoff, not the Candidate's fixed 2026-06, which is
future data for earlier folds.

### What "without injected events" removes

The Candidate trains on the lag-only synthetic shifts of experiment 581 (5% of each Primary's row
span, depths 0.15, 0.30, 0.50 of capacity, lengths 3, 7, 15, 30 days). From step 8, each shift is
also mirrored into one neighbour's summaries; at steps 4 to 7 the shifts reach only the own lag
inputs. Steps 4 and below have no injected-event version (step 4 trains with the 581 shifts, as in
the Candidate, in both ladders). Decisions:

- The **with** version is the Candidate recipe as it stands (581 shifts), so step 8 with injection
  can reproduce 667.
- The **without** version removes the 581 shifts from the lag inputs of stage 3 and the neighbour
  mirror at step 8; nothing else changes. The contrast therefore means different things at different
  steps (own lags only before step 8, own lags plus neighbours at step 8).
- The **held-out scoring bank** is built by experiment 581's own generator with seeds disjoint from
  every training seed, so the bank matches the "with" arm's training distribution, which favours the
  "with" version by design. A second bank from experiment 660's generator
  (`e00660_generator.py`, the shared generator its page tells readers to import) is scored as a
  check on that bias. Both inject into lag inputs only. Each bank also mirrors the shift into one
  neighbour's summaries, so step 8's neighbour columns can show their effect. Whether to inject into
  targets too is a question for the Discuss session and is settled before the code review.
- **Follow** is the share of an injected shift's depth that the median forecast copies (0 means the
  forecast ignores the shift, 1 means it copies all of it); lower is better for an NRA forecast.
  **Absorption** is the share of the shift that the network's fitted values take up in its training
  power (experiment 595's definition). The code review checks both formulas against experiments 595
  and 610.

The real long-and-deep cell is nearly empty (experiment 665 found none of NGED's 77 logged shifts
lasting more than a day in the long-and-deep type), so the real-record slices are descriptive and
every slice reports its rows and events first.

### Refit cadence, window, and seeds

- Window: the 12 months of experiment 667 (daily 12:00 UTC origins, 2025-11-01 to 2026-10-31,
  14-day horizon). Origins after about 2026-10-26 have valid times beyond the stored data, so each
  lead band reports its row count. The native origin of the networks is 09:00 UTC; the 12:00 contract
  origin follows 667's convention (`power_fcst_init_time` carries the 12:00 origin). The window
  overlaps the development months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06) and the final-test
  window of experiment 620 (2026-07 to 2026-10); the Candidate's settings were tuned on 5 of the 12
  months, which favours the later steps. The year is descriptive, and no model-selection choice is
  made from it.
- Default: **quarterly refits** at the starts of 2025-11, 2026-02, 2026-05, and 2026-08, each scoring
  three months. Step 8 with injection at the monthly cadence is 667's own file, reused without
  refitting; the quarterly step 8 differs from it by cadence, seed count (5 against 2), and the
  per-fold reference month.
- **Pilot fold.** Before the quarterly folds run, the 2026-05 fold runs step 8 with injection at the
  monthly cadence and 2 seeds, and is compared with 667's file on the same rows (all Primaries with
  data, days 0 to 14). P1 is read there, and the stop rule fires there.
- Seeds: seeds 0 to 2 for every step; seeds 3 and 4 are added for steps 6 to 8 (and for step 5, whose
  chain they share). Every adjacent-step contrast is scored on seeds 0 to 2 only, so the number of
  averaged seeds is the same on both sides. The five-seed rows are extra rows. Each contrast shows
  the per-seed values beside it. Experiment 610's audit found the extra error of arm P7 to be +0.011
  for seed 0 and +0.153 for seed 1, so seed noise is about 0.14 there; the minimum detectable effect
  is computed from the observed per-seed standard deviation and the page says whether 3 seeds resolve
  it.
- Intervals: block bootstrap over weeks (7-day blocks of origin days), with a minimum detectable
  effect printed beside each contrast.
- Leakage checks: a per-row assert that every lag time is at most the native origin minus 24 hours
  (the Candidate's rule), the same for every T entry, and a second check that every lag was observed
  before 06:00 UTC on the origin day. The ENS 00Z run counts as available at 09:00.

### Reference arms

- The manual heuristic and climatology, from `baseline_forecasters`, as in 620 and 668.
- The stored Candidate of 620 (one fit, 2026-07 fold only) and 667's Primaries file.
- **GAM**: the Primary demand research default 3 (experiment 251 below 168 h, experiment 322 above;
  six MW Primaries, origin days 2025-11-01 to 2026-06-29 that were scored, 09:00 UTC origins).
  Weekly refit on the whole history, as stored.
- **XGBoost**: BG2, arm `p_s6_bg2` of experiment 87 (16 Primaries, leads to 168 h, origins to
  2026-06-30). Weekly refit, as stored.
- Both stored arms end by 2026-06-30. XGBoost's line therefore stops at day 7 on the horizon plots,
  and both lines are scored only on the valid times the ladder shares with them. The ladder is also
  scored on those shared rows, and the page states the row counts and Primaries. Because the stored
  origins are at 09:00 and the ladder's at 12:00, the two are joined on valid time and series, each
  with its own lead. The cost of a CPU re-run to cover the full year is not estimated here; the
  manager is asked before one is planned.

### Outputs

For each step and version: a horizon plot (0 to 14 days, normalised MAE of the median, y axis not from
zero) against the manual heuristic, the previous step, the GAM, and XGBoost; the regime table (all
rows; inside long and deep events; the 7 days after; none and at least 7 days after one ended; the
long-and-deep type is the "inside" row, so there is no separate row); CRPS, twCRPS, tail calibration,
and avoidable-curtailment cost beside MAE; results by condition through `studies.eval_protocol`;
per-Primary CRPS for the six MW and eight MVA groups; a step-to-step contrast plot with intervals;
time-series figures (a cold week, a warm week, the verified D5/D16 transfer of 2025-12-30 to
2026-01-14) at several lead times, at least 280 px per panel and at most two panels per figure.
The headline figure (all steps on one horizon plot), the take-home, the first step that beats the
manual heuristic, and whether step 8 reproduces 667 (with the reason for any gap) come first.

Each step's forecasts are written to a private Parquet file
`data/ml-experiments/00677-*/backtests/step<N>_<version>.parquet` with the 667 columns plus a `seed`
column and the `step` and `version` labels, and weights are saved if the total is under about
2 GB. The injected-event tables per fold, the T tables (or the settings and seeds to rebuild them),
and the list of Primaries and rows per fold are saved beside them.

### Prevailing output

The survival head (R-haz) is trained on the frozen Candidate's quantiles with a bank of injected
shifts and needs the signal columns, so the survival head cannot exist at steps 1 to 7. Prevailing
rows come from 667's file at step 8 or are left out; `NRA_or_prevailing` is `"NRA"` in every other
file.

### Compute estimate

Per seed and fold, from 667's logs (stage 1 about 80 s, stage 2 about 20 s, stage 3 about 65 s; the
logs are for 10-day horizons and the 14-day horizon adds some):

| Item | Networks | Seconds |
|---|---|---|
| Stage 1 for three lag configurations (lag-free, one lag, two lags) | 3 | 240 |
| ENS-only fits (steps 1, 2, 3a) | 3 | 150 |
| Stage 2 (steps 3b, 4, and 5 twice) | 4 | 80 |
| Stage 3 (steps 3b and 4; steps 5 to 8, each twice) | 10 | 650 |
| Total for seeds 0 to 2 | | 1,120 per seed |
| Extra seeds 3 and 4 for the two-lag chain (stage 1 80, stage 2 40, stage 3 for steps 5 to 8 twice 390) | | 510 per seed |

Per fold: 3 x 1,120 + 2 x 510 = 4,380 s, or 1.22 GPU-hours. Quarterly (4 folds): 4.9 GPU-hours;
monthly (12 folds): 14.6 GPU-hours. Adding about 10% for scoring the 13 networks on the injected
banks gives about 5.4 and 16. The lag-free table builds run on the CPU and are not timed. The
estimate is checked against the running total before each stage, and the manager is told if the
total would pass 20 GPU-hours. The GPU is shared and other backtests have priority; `nvidia-smi` is
checked before each fit.

### Predictions that could fail and stop rules

- P1 to P4 are in the front matter. P1 is read on the pilot fold, before the other folds run.
- P2 and P3 are read on the quarterly ladder, pooled over the Primaries with data.

### Caveats to carry onto the page

- NGED event labels are an incomplete dump, so event counts are lower bounds and the "none and at
  least 7 days after" slice can hold unlabelled events.
- Days 10 to 14 have not been tested for the demand networks before experiment 667.
- D15 and D18 are cut before 2025-03-01.
- The ladder is one order of adding components.
- ENS history starts 2024-04-01, so early folds have less ENS data.
- ENS-only networks (steps 1, 2, and 3a) differ from step 3b in three ways, not one: no ERA5 and CAMS
  pre-training, no frozen encoders, and only the newest D5 knot.

## Method

To be written after the fits.

## Files

To be written after the fits.
