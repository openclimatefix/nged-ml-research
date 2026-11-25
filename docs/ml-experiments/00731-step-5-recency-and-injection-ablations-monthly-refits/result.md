---
id: "00731"
slug: step-5-recency-and-injection-ablations-monthly-refits
title: >-
  In the Primaries' step-5 demand network refitted monthly, do the 90-day recency weights and the
  injected switching events of its final fine-tune each improve the forecast, and does a shorter or
  longer half-life do better?
theme: retraining-and-training-history
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
kind: other
size: >-
  Anything else. (1) It stores about 250 fitted neural networks (7 new arms, 12 monthly refits, 3
  seeds), each with per-row forecasts, all private; (2) more than one design defensibly tests the
  hypothesis (ablations, other half-lives, placebos), so every design is an arm with a pre-registered
  contrast; (3) it reuses the training, injection, and scoring code of experiments 677, 574, 581,
  and 591, whose callers cannot be named without searching; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Reviews, in order: a Sonnet plan review, Sonnet
  code, an Opus code review before each script's first run, a smoke test, an Opus result review, a
  Sonnet numbers review, and a Sonnet prose review.
hypothesis: >-
  The step-5 demand network of experiment 677 (arm 9a) already fine-tunes with training-row weights
  that halve every 90 days of age and with synthetic switching events injected into its lag inputs.
  At monthly refits, taking either of these two training choices out of the final fine-tune, or
  replacing the 90-day half-life with 30 or 270 days, changes the one-year backtest error of the
  median forecast. The weights help (removing them raises the error) and the injected events cost a
  small amount of error on ordinary rows while lowering the share of a switching event that the
  forecast copies.
prediction: >-
  Written before any fit. Normalised MAE of the median, arm minus arm 9a, in points of capacity.
  C1 (equal weights, W0): +0.02 to +0.08 at the six signed-MW Primaries and 0.00 to +0.05 at the
  eight MVA-only Primaries, because experiment 710 found a recency gain of 0.03 to 0.07 on a
  different network. C2 (30-day half-life, W30): between -0.03 and +0.05, with no gain predicted.
  C3 (270-day half-life, W270): between -0.02 and +0.04. C4 (no injection, the stored arm `5wo`):
  -0.03 to -0.08 (the stored quarterly contrast is -0.066 and the stored monthly arms differ by
  -0.063), with the copied share rising from about 0.07 to about 0.11. C5 (both removed, B0) is
  roughly the sum of C1 and C4. The placebos C6 (age-permuted weights, WP) and C7 (events with no
  level shift, IP) are within 0.02 of arm W0 and of C4 respectively. The likeliest miss is C1 at the
  eight, where experiment 710's recency gain was not significant.
decision_rule: >-
  Contrasts are paired, on the rows that the 31-system joint pass of experiment 677 shares, with a
  7-day-block bootstrap of origin days (10,000 draws), at the 99.5% level (Bonferroni over 5
  contrasts and 2 Primary groups, so 10 readings). The headline metric is the normalised MAE of the
  median, with CRPS beside it, read separately for the six signed-MW and the eight MVA-only
  Primaries, and also per Primary. The threshold is zero for every contrast, because a half-life and
  an injection switch cost nothing in production. Recency claim: the 90-day weight helps when C1's
  lower limit is above zero at a group and the placebo C6 is above zero there; it is `supported`
  when both groups hold, `partly supported` when one holds, and `not supported` when neither holds.
  An arm replaces arm 9a only when its MAE interval is wholly below zero at both groups, its CRPS
  upper limit is below +0.02, and its copied share of injected shifts (677's held-out bank) is no
  more than 0.02 above arm 9a's. Under that rule the stored injection-free arm is expected to fail
  the share guard. The verdict maps to the recency claim, and the injection result is reported in
  the headline beside it. Interim numbers before the final scoring are labelled interim.
benchmark: >-
  Arm 9a of experiment 677 (`k1_5w_sp1`, monthly refits, seeds 0 to 2), stored; its replay R0; the
  stored injection-free arm `k1_5wo_sp1`; the manual heuristic and climatology from the 31-system
  joint pass.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  Normalised MAE of the median forecast and CRPS, in points of capacity, at lead 0.5 hours to 14
  days, with the copied share of injected shifts and the four load-shift scenarios beside them.
---

**Both changes the maintainer named are already in arm 9a, so this experiment ablates them.** The
maintainer asked, in the brief of 2026-11-23, for a follow-up to experiment 710 on the step-5
network at monthly refits, testing a final fine-tune with weights that halve every 90 days of age
and injected switching events, against the current step-5 recipe. Reading the code shows that the
current recipe, arm 9a of [experiment
677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md), already
has both. This page therefore asks what each costs or gains by taking it out, and whether another
half-life is better.

## What arm 9a already contains

- **Injection:** arm 9a passes `--synthetic 1`, so stage 3 only sees the 581 injected shifts (5% of
  rows, shifts of 0.15, 0.30, or 0.50 of the series level, lasting 3, 7, 15, or 30 days, applied to
  the lag inputs; targets are never shifted). One injected world per refit month is shared by the three
  seeds. Test rows are never shifted. The stored arm `5wo` is the same recipe without injection.
- **Recency weights:** stage 3 weights each training row by `0.5 ** (age_days / 90)`, drops the newest
  7 days (the embargo), and halves the weight of the following 7 days (experiment 599's E3). Stages
  1 and 2 use weight 1.
- **The stage-3 fine-tune is short** (about 10 epochs, about 10 s per fit) and selects on an
  unweighted validation week.
- Details and file references: `data/ml-experiments/briefs/step5_recency_injection/findings.md`.

## Arms

All arms refit monthly at the 12 refits of 2025-11-01 to 2026-10-01, with seeds 0, 1, and 2, and
reuse experiment 677's stored stage-1 and stage-2 weights, so only stage 3 and prediction run. Each
arm differs from arm 9a in exactly one stated way.

| Arm | Stage-3 change from arm 9a | Role |
|---|---|---|
| 9a (stored) | none | reference |
| R0 | none (replay of arm 9a) | reproduction and seed-noise control |
| W0 | equal weights (E3 embargo kept) | C1: do the 90-day weights help? |
| W30 | half-life 30 days | C2 |
| W270 | half-life 270 days | C3 |
| WP | 90-day weights with each row's age randomly permuted within the fold (same total weight) | C6: placebo for C1 |
| 5wo (stored) | no injection | C4: does injection help? |
| IP | events placed as in arm 9a, with zero level shift | C7: placebo for injection |
| B0 | equal weights and no injection | C5: both removed |

## Stop rule and approvals

**Recency has four `not supported` verdicts and injection has three, so this experiment is the
fourth on recency and needs the maintainer's approval.** Recency: experiments 331, 407, 419, and 466.
Injection as the treatment: experiments 486, 495, and 603. The maintainer asked for this experiment by
name on 2026-11-23 (quoted in the brief) and approved starting it in chat on 2026-11-25 ("Approve"),
as the ML manager reported. Per the protocol's stop rule, every contrast is scored at every Primary.

## Method

### Architecture

- The network is arm 9a's: one pooled Gaussian-mixture neural network (3 components) over the 16
  Primaries, with a shared lag encoder (24-hour and 1-week lags), weather and date-time encoders,
  60 input columns, a D5 output offset, and 11 quantiles read from the mixture's CDF.
- The weather input of each row is the ECMWF ensemble run (00Z) issued 9 hours before the native
  09:00 UTC origin, so it is known at the forecast origin. Experiment 677's code asserts this per
  origin (`nwp_init_time <= power_fcst_init_time`), and the new code adds a test that prints one example
  row with the run time, the origin, and the oldest lag time. The network has no lag-free head.
- Stages 1 and 2 are frozen at 677's stored weights for each month and seed.

### Training regime

- Stage 3 fine-tunes from the stage-2 weights with AdamW (rate 2e-4, weight decay 1e-3), batch 8,192,
  up to 60 epochs with patience 8, and the loss is the weighted mixture negative log-likelihood.
- Training rows come from the 730 days before each refit, cleaned (the `substation_zero` half-hours
  dropped) with the noisy start of the short-history Primaries cut, as in experiment 677.
- Each arm refits monthly, because monthly refits are the research practice.
- Arms W0, W30, W270, and WP replace the weight rule through the `select_rows` hook of experiment
  510. IP and B0 replace the injection world's shift depth with zero or remove it.
- The GPU budget is projected at about 6 GPU-process-hours (7 arms x 36 fits x about 3 minutes with the
  month's data build), and the stop line is 12.
- Every fitted network is saved as a state file, with its loss history, in the experiment's data
  folder.

### Measurement

- Scored rows are experiment 677's (`backtest_year_2025_11_to_2026_10_daily_12utc`), 365 daily 12:00
  UTC origins, 672 half-hours each, 16 Primaries, at lead 0.5 hours to 14 days, on cleaned actuals.
  **Reproduction gate:** before any contrast is read, the joint pass over experiment 677's 31 systems
  must give arm 9a 7.39 (95% interval 7.14 to 7.67). R0's seed-mean MAE must lie within the largest
  deviation of one arm 9a seed from the seed mean.
- The new arms are scored in a joint pass with arm 9a, the stored injection-free arm, the manual
  heuristic, and climatology. The absolute numbers of that pass are reported beside the original 7.39.
- The year includes valid times from 2026-07 to 2026-10, so every score is tuning-adjacent and no
  selection is made from it beyond the pre-registered rule.
- The load-shift scenarios are experiment 677's four slices (all rows, outside events and at least 7
  days after one, inside events, the 7 days after an event ends).

## Checks planned

- A known-answer test of the weights: ages 0, 90, and 180 days give 1, 0.5, and 0.25, and the placebo
  keeps the multiset of weights, so the total weight is equal.
- A known-answer test of the injection: zero depth leaves the lag block unchanged, a depth of 0.3
  shifts the lag block by 0.3 of the series level, and a mid-day onset is placed correctly.
- A smoke test of one month and one seed per arm before the first quoted fit.
- The audit triggers of the `ml-experiment` skill (failed controls, contradicted prior, MAE and
  CRPS disagreeing, leave-one-Primary-out).

## Files

- `ml-experiments/00731-step-5-recency-and-injection-ablations-monthly-refits/`: scripts to be
  written (the stage-3 variant runner, the arm driver, the scorer, the figure script).
- `data/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time/`: read
  only (stage-1 and stage-2 weights, stored arm 9a and `5wo`).
- `data/ml-experiments/00731-step-5-recency-and-injection-ablations-monthly-refits/`: predictions,
  models, and loss histories written by this experiment.
