---
id: "00416"
slug: mdn-explicit-solar-term-better-powered-follow-up
title: >-
  Does an explicit solar term inside the pooled mixture density network match the free irradiance
  input on sunny middays and in the upper tail, with five seeds instead of three?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: fitted networks, predictions, and the learned solar contribution per row, in this
  experiment's private folder. Production serving path: not touched (this repository holds none).
  Degradation rule: not touched. Designs: one defensible (experiment 409's network, four arms).
  Callers: the code imports experiments 87, 129, 397, 405, and 409.
hypothesis: >-
  The pooled plain mixture density network can keep its flexible irradiance input and gain an
  interpretable explicit solar term at no cost in skill: on sunny-midday half-hours at the six
  signed-direction Primaries, the network with the explicit solar term (C3), and the hybrid that
  also keeps the free irradiance input (H1), score within a stated margin of the network with the
  free irradiance input alone (A0), and they lower the 99th-percentile pinball loss.
prediction: >-
  Written before any fit. P1: H1 minus A0 on sunny-midday CRPS lies between -0.10 and 0.00 points.
  P2: C3 minus A0 on sunny-midday CRPS lies between -0.05 and +0.15 points. P3: C3 minus A0 on
  q99 pinball loss is negative, by 3% to 12% of A0's loss. P4: the learned capacity of C3 and H1
  rises towards the studies page at 3 of the 5 Primaries where experiment 409's C2 sat at 0.4 to
  0.5 of it.
decision_rule: >-
  See "Pre-registered rules". The primary family is two metrics (sunny-midday CRPS, q99 pinball
  loss at the signed-MW six) times three contrasts against A0 (C3, H1, A0-noirr-doy), six tests,
  Bonferroni-corrected. A contrast is a gain only if its interval excludes zero in the predicted
  direction and the point estimate exceeds the contrast's minimum detectable effect; it is
  non-inferior if the interval's upper bound is below the margin (+0.10 CRPS points, +5% q99
  pinball loss).
benchmark: >-
  Arm A0-doy (the pooled plain network with the free irradiance input and the day-of-year term,
  refitted here), arm A0-noirr-doy (no irradiance, day-of-year term), and arm C3 (experiment 409),
  with the stored XGBoost, BG2, and GAM-stack forecasts read as references and never refitted.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10; and pinball
  loss at q99.
date: 2026-10-30
code: ml-experiments/00416-mdn-explicit-solar-term-better-powered-follow-up/
---

**Status: planned. No fit has run.** The brief is `briefs/mdn_explicit_solar_followup2_brief.md`,
which supersedes `briefs/mdn_explicit_solar_powered_brief.md` where they differ. The sections up to
"Pre-registered rules" are the plan.

## What is asked

**The question is whether the explicit solar term can replace the free irradiance input without
losing skill, and whether it improves the tails.** [Experiment 409](../00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/result.md)
found the explicit solar term (C3) 0.064 CRPS points better than the same network with no
irradiance (interval -0.131 to +0.028), but its minimum detectable effect was 0.397, so the test
could not tell. The network with a free irradiance input (A0) still scored best, and C3 had a 7%
lower q99 pinball loss. This experiment adds the hybrid H1, uses 5 seeds instead of 3, and
measures where a solar term can show an effect: sunny middays.

## Review of the plan against the latest evidence

- **Merged since the brief was written:** experiment 409 (the predecessor; the adapter, the prior,
  and the capacity penalties changed nothing, so their arms are dropped). Experiments 388, 392,
  394, and 411 concern XGBoost lags and counterfactual re-scores and do not bear on this question.
  Experiment 413 (open, plan only) asks about shared learned shapes and keeps its own arms, so the
  solar question stays here.
- **Standing rules in `common.md` followed:** cleaned data with flagged `substation_zero`
  half-hours dropped and the count reported per fold, the D15 and D18 start cut at 2025-03-01
  (applied), monthly refits, saved models, tails per Primary, example forecast graphs, the
  seven-step review process, Sonnet reviews, draft pull request.
- **Departure from the brief:** the brief names A0 as arm of "the same network ... day-of-year term
  as in experiment 409". Experiment 403's stored A0 has no day-of-year term, so A0 is refitted
  here with the term (A0-doy). That gives a matched reference at the cost of 25 fits.
- **No change of scope.**

## Arms

All arms are experiment 409's pooled plain network: 8-wide Primary embedding, 3 Gaussian
components, 2 hidden layers of 64, ECMWF ensemble mean and spread through one pass, and the 20
day-of-year by hour inputs.

| Arm | Solar term | Irradiance and geometry inputs | Source |
|---|---|---|---|
| A0-doy | none | kept | refit (new) |
| C3 | explicit, adapter, prior | removed | seeds 0 to 2 from experiment 409, seeds 3 and 4 refit |
| H1 | explicit, adapter, prior | kept | refit (new) |
| A0-noirr-doy | none | removed | seeds 0 to 2 from experiment 409, seeds 3 and 4 refit |

**H1 is C3's solar module added to A0-doy's demand module.** The network can correct the explicit
term through the free irradiance input.

## Rows, folds, and cadence

- **Folds.** The five screen months of experiments 397 and 409: 2025-10, 2025-12, 2026-01,
  2026-03, 2026-06. Monthly refits on a 730-day window.
- **Seeds.** 5 (0 to 4).
- **Rows.** Cleaned data, flagged `substation_zero` half-hours dropped (count per fold reported),
  D15 and D18 cut before 2025-03-01 and report-only, 16 Primaries trained pooled, the six
  signed-direction Primaries (D5, D9, D10, D11, D14, D16) scored in the primary contrasts, the
  MVA-only eight reported.
- **Weather at lag times.** The lags are the Primary's own power, so no weather is used at lag
  times.

## Reuse of stored forecasts

- **Reused:** experiment 409's `seed_preds` for C3 and A0-noirr-doy, seeds 0 to 2. The folds, the
  window, the cleaning, the start cut, the code, and the seeds are the same, because this
  experiment imports experiment 409's modules. A check before scoring confirms that the stored
  rows' keys match the keys of the new arms. The 1.15 widening is not applied (raw).
- **Fitted:** C3 and A0-noirr-doy seeds 3 and 4 (20 fits), A0-doy (25 fits), H1 (25 fits). Total
  70 fits instead of 100, a saving of 30 fits (about 40 minutes).

## Pre-registered rules

1. **Primary slice.** Sunny-midday half-hours: realised CAMS clear-sky index of 0.7 or more, local
   time 10:00 to 15:00, within days 1 to 10, at the signed-MW six. The clear-sky index is the
   ratio of CAMS global horizontal irradiance to CAMS clear-sky irradiance from experiment 39's
   frame (a realised value, used only to select scored rows, never as an input). The slice is
   defined by realised weather, so every arm sees the same rows.
2. **Primary family (six tests).** The paired difference of each of C3, H1, and A0-noirr-doy from
   A0-doy, on (a) sunny-midday CRPS and (b) q99 pinball loss (co-primary), month-block bootstrap
   with 20,000 resamples, Bonferroni-corrected across the six (99.17% per interval).
3. **Minimum detectable effect.** For each contrast and metric, twice the standard deviation over
   the 5 folds of the seed-to-seed difference of the contrast. It is computed from the seed spread
   of experiment 409's stored seeds 0 to 2 for C3 and A0-noirr-doy before any new fit, and again
   from the 5 seeds after the fits. The pre-fit values are committed in an amendment to this page
   before the fits start. The page says plainly if the design is still underpowered.
4. **A gain** needs the interval to exclude zero in the predicted direction and the point estimate
   to exceed the minimum detectable effect. **Non-inferior** means the interval's upper bound is
   below the margin: +0.10 CRPS points (about 3% of A0's 3.35) and +5% of A0's q99 pinball loss. The
   margin is the loss the maintainer would accept for an interpretable physical term; it is a
   judgement and is stated before any score.
5. **Verdict.**
    - *Supported:* C3 or H1 is a gain on sunny-midday CRPS or on q99 pinball loss, and neither
      metric shows a harm (interval above zero beyond the margin).
    - *Partly supported:* C3 or H1 is non-inferior on both metrics and is a gain on neither, or is a
      gain on one metric and not non-inferior on the other.
    - *Not supported:* neither C3 nor H1 is non-inferior on sunny-midday CRPS.
    - *Inconclusive:* a minimum detectable effect exceeds the margin, so non-inferiority cannot be
      shown.
6. **Secondary contrasts (reported, same three-part rule):** C3 minus A0-noirr-doy, H1 minus C3,
   and all contrasts on headline-band CRPS (days 1 to 10, all half-hours) and the sunniest third
   slice of experiment 409.
7. **Tails.** q95 and q99 coverage and pinball loss per Primary and per metering group, with
   intervals from the same bootstrap; the q01 and q05 and threshold-weighted CRPS are reported.
8. **Capacity.** The learned capacity of C3 and H1 per Primary and season-year, with the seed
   range, beside the studies page. The prior resists falls and allows rises (the maintainer's
   rule). The 0.4 to 0.5 shortfall seen at 5 of 6 Primaries in experiment 409 is checked.
9. **Dropped from experiment 409:** the known-answer anomaly variant (untestable: injection about
   1 MW against seed spread about 0.35 MW), the adapter-free arm, the no-prior arm, and the
   capacity-path penalties.

## Amendment 1: pre-fit minimum detectable effect (written before any new fit)

Computed by `e00416_score.py --pre-fit` from experiment 409's stored seeds 0 to 2 of C3 and
A0-noirr-doy (twice the standard deviation over the 5 folds of the seed-to-seed difference of the
contrast, signed-MW six, days 1 to 10):

| Contrast, C3 minus A0-noirr-doy | CRPS (points) | q99 pinball loss |
|---|---|---|
| Sunny-midday slice (138,776 shared half-hours) | 0.969 | 0.110 |
| All half-hours | 0.397 (matches experiment 409) | 0.294 |

- **The design is expected to be underpowered for non-inferiority on CRPS.** The single-seed
  minimum detectable effect on sunny middays (0.97) is nearly 10 times the +0.10 margin. Scoring the
  seed-average of 5 seeds shrinks the seed noise by roughly a factor of 2, which leaves about 0.4,
  still about 4 times the margin. The page reports the post-fit value from the 5 seeds and applies
  the "Inconclusive" verdict if it exceeds the margin.
- **Only the month-to-month spread within a slice of 5 test months limits the interval.** More
  seeds do not fix that, and more test months would.

## Amendment 2: pre-registered second stage, more test months (manager, 2026-10-30)

The first stage runs as planned (5 test months, 70 fits) with the margin unchanged. **If the
post-fit minimum detectable effect exceeds the margin, the verdict is "inconclusive" until the
extension below runs.** The extension does not start without the manager's word.

- **Months available.** The ECMWF ensemble archive starts on 2024-04-01 and no valid time on or
  after 2026-07-01 may be scored. With at least 12 months of training history, the test months run
  from 2025-04 to 2026-06, 15 months. Ten are unused: 2025-04 to 2025-09, 2025-11, 2026-02,
  2026-04, and 2026-05. Months before 2026-04 have a training window shorter than 730 days (12 to
  23 months), which the page states as a caveat.
- **Size.** 8 to 12 test months means adding 3 to 7 months. Each added month needs all 4 arms at 5
  seeds, 20 fits, because experiment 409 stored no seeds there. That is 60 to 140 fits, about 1.7
  to 4 hours at the first stage's rate.
- **Priority order of added months.** 2025-07, 2025-08, 2025-05 (the sunny months, where the
  sunny-midday slice has most rows), then 2025-09, 2026-04, 2026-05, 2025-06.
- **Decision rule.** Unchanged, with the margin of +0.10 CRPS points and +5% q99 pinball loss, the
  Bonferroni level for six tests, and the post-fit minimum detectable effect recomputed on all
  months.

## Reports

- **Figures.** Contrasts with intervals first; sunny-midday CRPS and q99 pinball loss per Primary;
  the learned solar contribution by hour and month beside the clear-sky shape; learned capacity
  beside the studies page; a day-ahead and a 14-day example forecast (a winter and a summer
  month, one signed-MW Primary and one MVA-only Primary).
- **Checks.** Median minus mean forecast per Primary, the mean-forecast error beside each contrast,
  the flagged half-hours dropped per fold, the audit triggers that fired, and the saved-network
  path.

## Compute

- **Fits.** 70 fits at about 1.5 minutes each on the GPU, about 2 hours with feeder overhead. One fit
  is timed first.
- **Resources.** One GPU process under 8 GB, one feeder core, `MemoryMax=8G`,
  `OMP_NUM_THREADS=1`. No fit starts until the manager grants the fit slot.

## Earlier advice followed and departed from

- **Followed:** test the solar term on CRPS directly; compare on at least 3 test months per slice
  (the sunny-midday slice holds all 5 test months); report absolute skill beside contrasts.
- **Departed from:** none.

## Problems caused by the move

None so far.

## Reviews

Plan review, code review (before the first fit), results check, and write-up review are recorded
here as they run.
