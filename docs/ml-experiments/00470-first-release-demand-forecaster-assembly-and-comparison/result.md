---
id: "00470"
slug: first-release-demand-forecaster-assembly-and-comparison
title: >-
  Which assembled demand forecaster should be the first release: one pretrained network, a mixture
  of lagged and lagless networks, or either of those with the GAM forecast as an input?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: one 11-quantile table per candidate, month, and cell in this experiment's private
  folder, plus the mixture parameters of experiment 425's arm A networks on the scored rows. No fit
  is planned (0 fits, about 4 to 8 CPU-hours on 4 cores, no GPU). More than one design would
  defensibly rank the candidates (the choice of common rows, the number of tests, which tolerance
  decides a tie), so the plan fixes each before any score is read. Production serving path: not
  touched (this repository holds none). Degradation rule: not touched. Callers: the scripts import
  experiments 404, 425, 435, 448, 454, and 457, whose callers sit in those folders.
hypothesis: >-
  Two pieces of evidence point to different first-release forecasters. Experiment 425's arm A (one
  network pretrained on ERA5 and CAMS weather, five seeds averaged) has the lowest CRPS at the eight
  MVA-only Primaries. Experiment 454's K2-S1 (two lagged networks mixed with two lagless networks)
  has the lowest CRPS on rows with no labelled event at the signed-MW six. Scored on one set of
  rows, the simplest candidate within the stated tolerance of the best is a pretrained network
  without a lagless head at the eight, and the lagged-plus-lagless mixture at the six.
prediction: >-
  Written before any score is read. P1. A-pool and K2-425A both score within 0.03 points of
  effective capacity of the best candidate at the eight. P2. K2-S1 beats A-pool by 0.05 to 0.15
  points on rows with no labelled event at the six (experiment 454 found 0.121 against B-mix, and
  arm A beats B-mix by an unknown amount there). P3. K2-425A and K2-425A+S1 are no better than
  K2-S1 on rows inside real events scored against normal-running-arrangement actuals, because arm A
  has seen real lags with their events and no injected worlds, and S1 was trained to distrust them.
  P4. A-pool plus the GAM input lowers CRPS at the eight by about the 0.036 that experiment 457
  found, and no candidate beats the others by more than the minimum detectable effect at the six.
  The prediction could fail in either direction: A-pool may win at the six too, which would mean
  the lag design adds nothing over pretraining.
decision_rule: >-
  Score = CRPS in points of effective capacity (mass-weighted pinball loss over the 11 quantiles),
  1.15 widening unless stated, seeds averaged within a candidate as its cells define, a month-block
  bootstrap (month by origin day blocks, 20,000 resamples) with Bonferroni intervals. The number of
  tested contrasts is 54, fixed before any score is read: 6 contrasts (A-pool against each other
  candidate, plus K2-425A and K2-425A+S1 each against K2-S1) times 7 CRPS row sets (the eight; the
  six on rows with no labelled event; the six on real-event rows against normal-running-arrangement
  actuals; cold winter nights at the eight and at the six; the 12 to 36 hour band at the eight and
  at the six), plus the q99 pinball loss at the eight and at the six on rows with no labelled event
  (12). Christmas week, D15 and D18, tails other than q99, the real-event rows with and without the
  largest event, contrasts involving B-mix, `S1-ens3`, and A-ens2, and curtailment are descriptive
  and enter no count. The curtailment clause uses the Bonferroni block-bootstrap interval of the
  candidate's contrast with the best candidate. At the six the minimum detectable effect for a
  single-seed pool is about 0.07, above 0.03, so P2 and P4 can only be conclusive if the gap
  exceeds the minimum detectable effect; otherwise they are stated as inconclusive. The minimum detectable effect (experiment 428's formula on the seed-to-seed
  spread of arm A's five single seeds at the eight and the six; the three cells of each mixture
  construct for the mixtures) is printed before any contrast is read. A contrast inside the
  minimum detectable effect is inconclusive. Rule for the first release: among candidates whose
  CRPS is within 0.03 of the best on the eight and on the six (rows with no labelled event), no
  worse than K2-S1 by more than 0.03 on real-event rows against normal-running-arrangement actuals
  and on cold winter nights, within +0.02 on q99 pinball loss, and not worse than the best
  candidate on avoidable-curtailment cost by more than its interval at f = 0.9, choose the one with
  the fewest serving networks and no GAM in the serving path; a candidate is admissible when its
  contrast with the best candidate is inconclusive (inside the minimum detectable effect) or
  conclusive but under 0.03; if every candidate is admissible at the six, the six decides nothing
  and the choice rests on the eight and the simplicity rule. Serving cost counts the networks per
  forecast at the stated serving seed count (A-pool 5, K2-S1 4, K2-425A 4, K2-425A+S1 4, A-gam-in 5),
  and counts the GAM, the lagless table, and its store of past lagless forecasts as extra serving
  cost; A-pool at 1 network is reported as a sensitivity, not a candidate. If A-pool and K2-S1 tie at the eight and the six, the lag design adds
  nothing over pretraining and A-pool is the recommendation. The smallest gain worth porting is
  0.03 points, because a mixture doubles the networks run per forecast and adds the lagless table
  and its store of past lagless forecasts. Every score is selection-biased, because the five folds
  that choose the candidate also score it; the page names the first later month that lies in no
  selection fold and the scoring recipe for it, and that check is not a gate here.
benchmark: >-
  A-pool (experiment 425's arm A, five-seed quantile average), experiment 454's B-mix, K2-S1, and
  the S1 three-seed ensemble, experiment 457's A-gam-in, BG2, `manual_heuristic`, climatology, and
  experiment 449's monthly XGBoost forecast, all stored.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS from the 11 quantiles in points of effective capacity; coverage and pinball loss at q01,
  q05, q95, and q99 raw, with the 1.15 widening, with a sweep from 1.0 to 1.15 for the mixtures,
  and after table P of experiment 435; avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the
  signed-MW six with block intervals. Per group (the eight, the six, D15 and D18) and per
  Primary.
---

## Which assembled demand forecaster should be the first release?

*Planned: no score has been read. The plan below is final unless a review changes it; the finished
write-up replaces this section.*

### Plan

- **Question.** The parts of the first-release demand forecaster were tested one at a time, in two
  harnesses that score slightly different rows. This experiment scores the assembled candidates
  on one set of rows and applies the pre-registered rule above.
- **Candidates (5, plus references).** A-pool; K2-S1 (2 S1 networks plus 2 lagless networks,
  experiment 454); K2-425A (2 arm A seeds plus 2 lagless networks); K2-425A+S1 (1 arm A seed,
  1 S1 network, and 2 lagless networks); A-pool plus the GAM input (A-gam-in of experiment 457).
  Seed-matched control: A-ens2, a density mixture of 2 arm A seeds, so a mixture is compared with a
  pool of equal seed count. References: B-mix, the S1 three-seed ensemble (`S1-ens3`), BG2,
  `manual_heuristic`, climatology, and the monthly XGBoost forecast.
- **Reuse (0 fits).** B-mix, K2-S1, and `S1-ens3` are read from experiment 454's predictions
  (`work/predictions/<arm>/<month>_s<cell>_w-1.parquet`). A-pool and A-gam-in are read from
  experiments 425 (`work/seed_preds/A`) and 457 (`work/seed_preds/Gin`). BG2, `manual_heuristic`,
  and climatology are read from experiment 457's reference builder, and the XGBoost forecast from
  experiment 449 through `e00448_refs`. K2-425A, K2-425A+S1, and A-ens2 need arm A's mixture
  parameters, which no earlier experiment stored (it wrote quantiles only). They come from a
  forward pass of the saved arm A networks (`work/models/A/<month>/s<seed>`, seeds 0 to 4) on the
  scored rows, with experiment 454's frozen lagless tables (`00404.../work/lagless`) and S1 heads
  (experiment 454's `work/heads/S`). A forward pass is not a fit. It runs only on the intersection rows (the lagless head requires every
  row present), uses the saved standardiser and target statistics, and is checked by recomputing
  A-pool's quantiles and requiring a maximum difference under 5e-4 against `seed_preds/A` as
  `check_against_stored` does in experiment 454. The scorer also checks that a forecast origin means
  the same lag cutoff in both harnesses. Arm A seed s pairs with lagless seed s and S1 seed s (seeds
  0 to 2; the mixtures use the pairs (0,1), (0,2), (1,2) as experiment 454 does). The lagless tables
  are cross-fitted, so no leakage arises.
- **Rows.** All candidates are scored on the intersection of (Primary, forecast origin, target
  half-hour) over the five folds (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), after the cleaned-data
  drops and the D15 and D18 start cut. Experiments 425 and 454 key the same rows differently
  (`time_series_id`, `power_fcst_init_time`, `valid_time` against `primary`, `origin`, `target`), so
  the scorer first maps one key to the other and checks the mapping on the shared rows. The scorer
  prints the intersection size, the sizes of the two source row sets, and the rows lost to each.
  Three intersections are reported: the 14 Primaries that have an A-gam-in forecast (the gating
  set, the eight plus the six), all 16 Primaries without A-gam-in (D15 and D18 report-only), and
  days 1 to 7 only when BG2 is a party to the contrast (BG2 stops at 168 hours). Quoted scores of
  experiments 454 and 457 appear as absolute references only, labelled as scored on different rows.
- **Scoring.** The scorer copies experiment 454's `e00454_score.py` for the row classes (event,
  harm, rows with no labelled event), the NRA basis, and the variants (raw, widening, table P), and
  experiment 457's `e00457_score.py` for cold nights, Christmas week, and the block bootstrap. Equal
  rows per slice are printed. Absolute CRPS in points of effective capacity is printed beside every
  contrast.
- **Tails.** Coverage and pinball loss at q01, q05, q95, and q99 for each candidate; the lower tails
  need a ceiling because S1-family mixtures are over-wide at q05, so q05 and q01 coverage above
  0.075 and 0.02 respectively is flagged and reported, and does not decide the rule.
- **Curtailment.** Avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six,
  through `studies.node_curtailment`, as experiment 454's `e00454_curtailment.py` does. Only the
  cross-construct contrasts (A-pool against K2-S1 and the two new mixtures, on the intersection)
  are new; experiments 454 and 457 stored the rest.
- **Production cost per candidate.** Networks per serving pass, fits per monthly refit, the lagless
  table and the store of past lagless forecasts, whether the GAM runs in serving, and the extra
  storage.
- **Standing practices followed.** Monthly refits (nothing is refitted); cleaned data (the flagged
  `substation_zero` half-hours are dropped, counted per fold); the D15 and D18 cut before
  2025-03-01; stored forecasts reused; tails and cost reported; examples plotted (a day-ahead and a
  14-day forecast for the winning candidate and its main rival at one Primary, plus at least six
  real-event plots across both metering groups with generators anonymised); the unlogged-event
  caveat stated. No fit means no new saved models.
- **Earlier advice followed or departed from.** Follows 454 (do not pool a constant across the six
  and the eight; "rows with no labelled event"), 457 (state the minimum detectable effect first),
  and 435 (table P as a final stage). Departs from the brief's "final test once on the winner":
  no held-out month exists, so the page labels every score selection-biased instead.
- **Process.** Sonnet writes the code; a fresh Sonnet agent reviews the plan, then the code before
  the first scored run; a fresh Sonnet agent re-derives the headline numbers; a fresh Sonnet agent
  reviews the write-up and its prose. An audit follows only if a trigger in the `ml-experiment`
  skill fires.
- **Further standing rules.** Run scripts under `systemd-run --user --scope -p MemoryMax=16G -p
  OOMPolicy=continue`. Report threshold-weighted CRPS beside the tails, per-Primary tables, the
  avoidable-curtailment cost during events, in the 7 days after, and across all rows, with the saving
  against `manual_heuristic` in GBP per MW per day, interval width and coverage on rows where a lag
  is nulled, and the "Checks" bullets (median minus mean forecast per site, mean-forecast error
  beside each contrast, which audit triggers fired). `take_home_message` is written in the finished
  commit.
- **First later month.** The first month of data in no selection fold is named when the page is
  written, with the scoring recipe of this scorer.
- **Compute.** CPU only, at most 4 cores, under 12 GB of memory, no GPU. If a fit proves necessary,
  the worker reports its count and GPU-hours to the manager before starting it.
