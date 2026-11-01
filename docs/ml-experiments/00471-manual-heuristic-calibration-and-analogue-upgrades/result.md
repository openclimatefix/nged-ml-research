---
id: "00471"
slug: manual-heuristic-calibration-and-analogue-upgrades
title: >-
  Which simple changes to NGED's manual heuristic lower its quantile score and its
  avoidable-curtailment cost?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: other
size: >-
  medium. Stored: one table of per-variant quantile forecasts in this experiment's private folder.
  More than one design is defensible (window length, quantile level, what the fit minimises), so the
  plan fixes one primary variant and counts the rest in the Bonferroni number. Production serving
  path: not touched (this repository holds none). Degradation rule: not touched. Callers: scripts
  import the row builder of experiment 448 and the scorers of experiments 457 and 449, whose callers
  sit in those folders. No fit loop, no neural network, no GPU.
hypothesis: >-
  If the 13 (and later 12) equally weighted historical analogues of the manual heuristic are
  recalibrated with a correction fitted on a trailing window of past errors, the quantile score
  falls against the faithful heuristic, because the heuristic's empirical 95th percentile of 13
  members is exceeded about 11.4% of the time even when the analogues are perfect and the heuristic
  carries a level bias.
prediction: >-
  Written before any score is read. P1. Perfectly calibrated analogues give a p95 exceedance floor of
  11.4% (13 members) and 11.9% (12 members); only 13 and 12 members occur at 0 to 240 hours, so
  12.5% cannot appear. The real heuristic is above that floor (experiment 13 found a level bias,
  and q01 coverage is 5 times nominal at the signed-MW six). P2. After calibration the p95 exceedance
  moves from the 11% to 14% range towards 5%. P3. The quantile score improves most at long lead
  (days 5 to 10). P4. The variance half of EMOS has something to fit only if the rank histogram is
  not flat; analogue sets are likely over-dispersed at short lead.
decision_rule: >-
  Score is the mean quantile score (pinball over the 11 levels) and CRPS in points of capacity, at
  leads 0 to 240 hours, on one row intersection stated in the write-up. A variant is an improvement
  only if it lowers the quantile score against the faithful `manual_heuristic` by more than the
  minimum detectable effect at the eight MVA-only Primaries AND at the signed-MW six, and does not
  worsen cold winter nights or the q99 pinball loss by more than 0.02. The minimum detectable effect
  is the half-width of the 95% month-block bootstrap interval of the paired difference of a placebo
  variant (the faithful heuristic against itself with the quantile levels permuted by half a step),
  printed before any contrast is read. Stop rule: if stage 0 shows calibrated analogues and no
  stage-1 variant beats the faithful heuristic by the minimum detectable effect, the line closes.
  Choosing windows and quantile levels on the five scored months is selection-biased; the primary
  variant is fixed in the plan and every other variant is counted in the stated Bonferroni number.
benchmark: >-
  The faithful `manual_heuristic` (empirical q01 to q99 of the stored analogue members), with
  climatology, BG2 (to 168 hours), the stored arm A-avg5, and the monthly XGBoost model as
  references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score and CRPS in points of capacity; q01, q05, q95, q99 coverage and pinball loss; the
  12 to 36 hour band; cold winter nights; avoidable-curtailment cost at f = 0.9, 0.75, 0.5 at the
  signed-MW six; per group and per Primary.
---

Plan, written 2026-11-01 before any score. Rewritten as the write-up when finished.

NGED's manual heuristic reads one number from 13 equally weighted historical analogues: their 95th
percentile. This experiment asks whether cheap changes to that rule, with no trained neural network,
lower its quantile score and its avoidable-curtailment cost. A stronger heuristic is a tougher bar for
every network we propose. The faithful `manual_heuristic` stays the headline bar. Calibration follows
[Gneiting et al. (2005)](https://doi.org/10.1175/MWR2904.1), who introduced ensemble model output
statistics (EMOS) for weather ensembles; nothing in EMOS requires the ensemble to come from a weather
model.

## Plan

- **Inputs.** The stored analogue members, `data/power_forecasts/experiment_name=manual_heuristic`
  (31 series, 380 daily runs from 2025-06-16 to 2026-06-30, 13 members to about day 7, 12 to about
  day 14). They are in MW. Step 0 reproduces experiment 87's stored `ref_manual_heuristic` 11
  quantiles from the members (linear interpolation) before any variant is built.
- **Scorer.** Experiment 448's row builder (`e00448_refs.py`), experiment 457's scoring code
  (`e00457_score.py`), and experiment 449's curtailment code (`e00449_curtailment.py`), imported
  from their folders. No fit loop and no GPU. Compute: about 1 to 3 CPU-hours, about 3 GB, 4 cores.
- **Rows.** Experiment 457's row set (`fold_a_daily_gap_v1`) over its five months (2025-10,
  2025-12, 2026-01, 2026-03, 2026-06), leads 0 to 240 hours, the eight MVA-only and the signed-MW six
  separately, D15 and D18 report-only. Every arm is scored on one intersection with BG2 (to 168
  hours); 168 to 240 hours is also reported without BG2. The intersection is stated in the write-up.
- **Standing practices followed:** cleaned data, the D15 and D18 cut before 2025-03-01, the tails
  and the cost reported, example forecasts plotted. Not applicable: refits (no neural or XGBoost
  fit), saved models, the few-fold screen.
- **Stage 0 (measure first).** Rank histogram of the actual among the analogues per lead band;
  correlation between analogue spread and the absolute error of the analogue mean; exceedance of the
  empirical p95 and of each other level. Says whether the variance half of EMOS has anything to fit.
- **Stage 1 (calibration variants).** (a) EMOS: mean = a + b x analogue mean, spread = c + d x
  analogue spread (log link), fitted by minimum CRPS on a trailing window of 28 or 56 days (112 days
  for 2025-12 and later only, because the store starts in mid-June), refit weekly, per Primary and
  lead band; (b) quantile mapping of the analogue q95 only; (c) the faithful method at a fitted
  quantile level; (d) the fitted distribution emitted as m equiprobable members `(i - 0.5)/m` with
  m = 99, then the 11 quantiles rebuilt from them. A trailing window may use only pairs whose valid
  time is observed at the forecast origin minus the data delay of experiments 454 and 457.
- **Stage 2 (analogue upgrades).** A recent-weeks level adjustment from the last observed days (no
  lookahead past the origin); holiday-aligned analogues (issue 1088 has no code; new code reusing
  experiment 6's special-day calendar, dropped if it takes more than half a day); a clock-change arm
  that shifts the fixed-UTC analogue lags to local time. The 26-analogue and load-growth arms are
  report-only pending the neural Discuss session's agreement.
- **Primary variant (fixed now):** EMOS (a) with a 56-day window and weekly refit. All other variants
  count in the Bonferroni number, stated in the scorer's printout before any contrast is read.
- **Reviews:** Sonnet code review before the first scored run, Sonnet results review, Sonnet
  write-up and prose reviews. Opus only if a trigger in the audit section fires and the result is
  very surprising.
- **Not edited:** GitHub issue 715 of the production repository, and any file in `src/` or
  `packages/`.
