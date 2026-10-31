---
id: "00434"
slug: demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts
title: >-
  Which re-ordering of the stored demand forecast's 51 sample paths, by ensemble member rank or by
  past observed paths, makes the paths coherent across the 16 Primaries and across leads?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - hierarchy-and-pooling
status: finished
kind: other
protocol: demand-v2
date: 2026-10-31
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted or written for reuse. Serving path: not touched. Degradation rule: not touched. Designs:
  more than one defensible (which variable ranks the members, which past dates feed the Schaake
  shuffle), so the plan registers each. Callers: this folder's scripts only, which read experiment
  425's stored arm A, the stored 51-member weather frames, and the cleaned actuals, and call no
  reviewed module beyond `studies.*`. One Sonnet plan review, one Sonnet code review before the
  first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The stored forecast of experiment 425's arm A holds 11 quantiles per Primary and half-hour.
  Drawing 51 values from those quantiles and assigning them to the 51 ensemble members by the rank
  of each member's 2-metre temperature (ensemble copula coupling, ECC) gives sample paths that match
  the observed dependence across Primaries and leads better than independent draws do, by more than
  the 5% coherence margin on the variogram score, without changing the marginal CRPS. A Schaake
  shuffle from past observed paths does at least as well as ECC.
prediction: >-
  Written before any score is read. P1. All four orderings share the same 51 values at each
  Primary, half-hour, and run, so the marginal CRPS is identical across them (a check, not a
  contrast), and the 51-value CRPS is within 0.03 points of capacity of the CRPS the stored 11
  quantiles give. P2. ECC by temperature (ECC-T) lowers the joint variogram score against the
  independent draws (IND) by 5% or more. P3. The Schaake shuffle (SCH) lowers it by more than ECC-T
  does, because experiment 304 found that members correlate 0.95 to 0.97 across farms while errors
  correlate 0.54 to 0.78, so member ranks may overstate dependence. P4. Ranking by shortwave
  radiation (ECC-SW) beats ECC-T in the daylight slice and loses it in the cold-night slice.
decision_rule: >-
  Scores in points of capacity (the series' effective capacity, 1 point = 1%), over the 12 monthly
  folds of experiment 425, runs scored at their `lead_days_1_10` half-hours and at the 12 to 36 hour
  band. The coherence scores are the energy score and the variogram score (exponent 0.5, equal
  weights) of the 51 paths in three blocks: across the 16 Primaries at each half-hour (spatial),
  across the 2-hourly leads of days 1 to 10 for each Primary (temporal), and across both at 6-hourly
  leads (joint). A contrast is the relative difference of the mean score, paired by fold, with a
  month-block bootstrap (20,000 resamples) at 95% Bonferroni-corrected over the four tested
  contrasts (98.75%). The registered margins are the maintainer's proposals: 0.03 points of CRPS and
  5% of the coherence score. C1 ECC-T against IND, C2 SCH against IND, C3 SCH against ECC-T, C4 ECC-
  SW against ECC-T. An ordering counts as coherent if its joint variogram interval lies wholly below
  -5% against IND. C3 or C4 count as a difference only if the interval lies wholly beyond 5%. The
  gating block is the joint variogram score; the spatial and temporal blocks, the energy score, and
  the 12 to 36 hour band are reported beside it as secondary and interim, without a verdict. If an
  interval straddles -5% or +5%, the page says "no difference at the margin" and gives the direction
  of the point estimate; P3 and P4 are directional predictions that fail unless their interval lies
  wholly beyond the margin. The marginal CRPS needs no contrast. The comonotone ordering (CO, all
  paths ranked alike) checks that the scores can see coherence: if CO and IND score within 5% of
  each other on the joint variogram, the scores are blind and the page says so. No verdict is sought
  on the member-wise route, which is step 2.
benchmark: >-
  IND, the independent random draw, scored on the same half-hours; the stored 11-quantile CRPS of
  arm A at the 1.15 widening (experiment 425) for the marginal check; CO as the dependence upper
  bound.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Marginal CRPS of the 51 values (energy form), energy score and variogram score (exponent 0.5) of
  the 51 paths in three blocks, daylight and cold-night slices, and the share of actuals beyond the
  smallest and largest of the 51 values, in points of effective capacity; leads 12 to 36 hours and
  days 1 to 10.
verdict: not supported
best_arm: >-
  None. No ordering of the 51 values beat independent draws by the 5% margin on the joint variogram
  score (days 1 to 10): the Schaake shuffle (SCH) scored 1.7669 and ECC-T 1.7752 against 1.7657 for
  independent draws (IND), and the comonotone ordering (CO) 1.9667. The ordering that helped most
  was SCH across leads (1.5757 against 1.6335 for IND).
headline: >-
  Ranking the 51 values by each member's temperature (ECC-T) made them 6.7% less coherent across
  the 16 Primaries (variogram, +4.7% to +9.1%) and 12.9% more coherent across leads at 12 to 36
  hours (-17.0% to -9.0%), so the joint score moved by +0.5% (+0.3% to +0.8%). The Schaake shuffle
  matched independent draws across Primaries (-0.4%, -1.7% to +1.0%) and beat them across leads
  (-3.5%, -6.3% to -0.9% at days 1 to 10; -15.3%, -20.9% to -9.2% at 12 to 36 hours).
lessons: >-
  The 16 Primaries' forecast errors are close to independent once each Primary has its own
  quantiles, so imposing the members' shared temperature order across Primaries adds dependence the
  observations do not show. Across leads the observed errors are persistent, and both temperature
  ranks and past paths capture that. A joint score of 640 components is dominated by the spatial
  and temporal parts cancelling, so it cannot tell the orderings apart (all within 1%) although it
  does tell the comonotone ordering from independent draws (+11.4%). Ranking by shortwave radiation
  adds nothing, because the radiation is zero for every member at night.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Re-ordering the stored demand forecast's 51 values by member temperature or by past observed
  paths does not give clearly better sample paths overall. Ordering by temperature made the paths
  about 7% less coherent across the 16 Primaries and 13% more coherent across leads at 12 to 36
  hours, and the two effects cancel in the combined score. A Schaake shuffle from past paths was no
  worse than independent draws across Primaries and 3% to 15% better across leads, so it is the
  reference to beat when step 2 runs the 51 members through the network.
advice:
  - >-
    Use the Schaake shuffle, not temperature-rank coupling, as the coherence reference for the
    member-wise comparison, because it matched independent draws across Primaries (-0.4%) and beat
    them across leads (-3.5% at days 1 to 10), while temperature ranks added 6.7% to the variogram
    score across Primaries (C1, C2).
  - >-
    Gate any member-route comparison on the spatial and temporal variogram scores separately, because
    the joint score of 640 components moved by less than 1% between every ordering except the
    comonotone one (C1 to C4).
  - >-
    Do not rank members by shortwave radiation, because the radiation is zero for every member
    at night and the daylight gain over temperature was -0.2% to -0.8% (C4).
  - >-
    Score the 51 values with the energy form of the CRPS, and do not compare the result with the
    11-level CRPS of earlier experiments, because on one run the two estimators differed by 31% on
    the same quantiles (the estimator check).
code: ml-experiments/00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/
---

**Re-ordering the 51 values by temperature or by past paths does not make the paths clearly more
coherent overall, because ordering by temperature helps across leads and hurts across Primaries.**
The stored forecast of experiment 425's arm A gives 11 quantiles per Primary and half-hour. This
experiment drew 51 values from them and assigned the values to 51 sample paths in five ways, then
scored how well the paths match the observed dependence, with no fit. The 16 Primaries are
substations, so they are named freely. The joint variogram score (the headline block) moved by 1%
or less between independent draws, temperature ranks, shortwave ranks, and the Schaake shuffle. It
moved by 11.4% for the comonotone ordering, so the score can see dependence. The registered
verdict is `not supported`: P2, P3, and P4 failed and P1 failed in its second half.

![Variogram score of each ordering relative to independent draws, per block and lead window](figure_1_variogram_contrasts.svg)

![Twelve of the 51 sample paths of one Primary under three orderings](figure_3_paths.svg)

![Day-ahead forecast of one Primary in winter](figure_2_example_winter_day-ahead.svg)

![Fourteen-day forecast of one Primary in winter](figure_2_example_winter_14_days.svg)

![Day-ahead forecast of one Primary in summer](figure_2_example_summer_day-ahead.svg)

![Fourteen-day forecast of one Primary in summer](figure_2_example_summer_14_days.svg)

- **Arms.** ECC-T (coldest member gets the highest value), ECC-SW (darkest member), SCH (Schaake
  shuffle from the 51 past observed paths nearest in day of year), IND (independent draws), and CO
  (comonotone). Every arm holds the same 51 values at each Primary, half-hour, and run.
- **Rows.** 12 monthly folds (2025-07 to 2026-06), 16 Primaries, daily runs at 09:00 UTC, scored on
  the half-hours with a cleaned actual (4,004,653 rows); 1,179 flagged exact-zero actuals are
  dropped from the whole scoring cache. Forecast: experiment 425's arm A, seeds 0 to 4 averaged in
  quantile space and widened by 1.15, stored in
  `data/ml-experiments/00425-best-mdn-architecture-assembly-with-ablations/work/seed_preds/A/`.
  Code reused by copying: experiments 87, 377, and 381 (scoring context); the page's scripts import
  those three folders.
- **Joint variogram score, days 1 to 10** (mean of 12 folds, points of capacity): IND 1.7657, ECC-T
  1.7752, ECC-SW 1.7679, SCH 1.7669, CO 1.9667. Against IND: ECC-T +0.5% (+0.3% to +0.8%), SCH
  +0.1% (-0.3% to +0.5%), CO +11.4% (+10.4% to +12.2%). At 12 to 36 hours the joint contrasts
  were +0.7% (-0.0% to +1.4%) for ECC-T, -0.4% (-1.0% to +0.1%) for SCH, and +7.0% (+5.9% to
  +7.9%) for CO. No ordering other than CO is within reach of the 5% margin.
- **Across Primaries (spatial), days 1 to 10:** IND 1.6045, ECC-T 1.7115, ECC-SW 1.6694, SCH
  1.5984, CO 1.7440. ECC-T against IND +6.7% (+4.7% to +9.1%), so it lies beyond the margin only
  in its point estimate. SCH against IND -0.4% (-1.7% to +1.0%); SCH against ECC-T -6.6% (-7.6% to
  -5.5%), beyond the margin. The energy score agrees (ECC-T +6.2%, +5.5% to +6.8%).
- **Across leads (temporal), days 1 to 10:** IND 1.6335, ECC-T 1.6050, ECC-SW 1.6268, SCH 1.5757,
  CO 2.1539. ECC-T -1.7% (-2.8% to -0.7%), SCH -3.5% (-6.3% to -0.9%), and neither reaches the
  margin. At 12 to 36 hours: IND 1.3914, ECC-T 1.2122, ECC-SW 1.3278, SCH 1.1778; ECC-T -12.9%
  (-17.0% to -9.0%), SCH -15.3% (-20.9% to -9.2%), SCH against ECC-T -2.8% (-5.7% to +0.2%).
- **Shortwave ranks (C4).** ECC-SW against ECC-T: -0.4% (-0.6% to -0.2%) jointly, -2.5% (-4.3% to
  -1.1%) across Primaries, and +1.4% (+0.5% to +2.3%) across leads at days 1 to 10. In the
  half-hours scored as daylight, ECC-SW beat ECC-T by -0.2% (-0.7% to +0.2%) across Primaries and
  -0.8% (-1.6% to +0.0%) across leads. In cold-night half-hours across Primaries ECC-SW beat ECC-T
  by -6.4% (-8.4% to -3.6%), the opposite of P4, because the radiation is zero at night and the
  ranks there are random, which is independent draws.
- **Marginal check.** The 51 values give the same CRPS for every ordering, so the marginal contrast
  is zero by construction and the 0.03-point margin applies only in step 2. The CRPS of the 51
  values was 3.597, 4.608, and 10.604 at the signed-MW six, the MVA-only eight, and D15 and D18
  (days 1 to 10), against 2.829, 3.598, and 8.431 from experiment 425's 11-level estimator, so the
  second half of P1 failed. The gap is the estimator: on one run (2025-10, fourth run) the same
  11 quantiles scored 3.487 with the 11-level formula and 4.558 with 51 even levels
  (`e00434_estimator_check.py`). The two numbers are not comparable, and the page makes no claim
  about discretisation loss beyond that.
- **Tails.** The share of actuals above the largest of the 51 values (days 1 to 10) was 2.43%, 0.90%,
  and 3.66% at the signed-MW six, the MVA-only eight, and D15 and D18; below the smallest it was
  0.70%, 0.69%, and 0.30% (a calibrated 51-value ensemble gives 1.92% each way). The ordering does
  not change these shares.
- **Slices.** The cold-night slice has actuals in only 5 to 7 of the 12 folds (the other months have
  no cold nights), so those contrasts are interim. The spatial daylight and cold-night slices keep
  the half-hours where at least half of the Primaries with an actual are flagged; the temporal
  slices use each Primary's own flags. Daylight is the local mid-day band, not the sun's elevation.
- **Caveats.** The Schaake pool starts on 2025-03-01 (so that D15 and D18 have data), holds 89 to
  346 dates per run, and uses the 51 dates nearest in day of year, which are mostly consecutive
  days. Experiment 425 gave the 2025-07 fold only March to mid-June dates. A past path of 14.6 days
  overlaps its neighbours, so the 51 paths are less varied than 51 independent years would be. The
  Primaries are scored together, not by group. The temperature sign (colder member, higher demand)
  was fixed in the plan and could not be checked against an observation per member. No per-member
  quantiles exist in the stored forecasts: arm A ran one pass on the ensemble mean and spread, so the
  member-wise comparison is not scored here.
- **Checks.** No model was fitted, so median-minus-mean and mean-error checks do not apply. Every
  arm was verified to hold the same 51 values at every run (the script stops otherwise), and the
  score code matches a brute-force version on random inputs. One audit trigger fired in spirit: the
  second half of P1 failed because the estimators differ, which the estimator check explains.
- **What to try next (the manager decides).** Step 2 runs the 51 members through the network. Compare
  it against SCH and ECC-T on the spatial and temporal variogram scores, because the joint score
  cannot separate them. A longer Schaake pool would need Primary data from before 2025-03-01 for D15
  and D18, or the pool could use the 14 Primaries with long histories.
- **Problems caused by the move.** None so far.
- **Reviews.** One Sonnet plan review (10 findings; the tails, verdict rules, pool, ties,
  missing-actuals rule, and scaling were added to the plan), one Sonnet code review before the first
  run (no score-changing bugs; NaN and permutation checks added, wording of the spatial slice and
  `n_dim` documented). Results and write-up reviews: see the pull request.
