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
protocol: demand-v2
size: >-
  Anything else. (1) It stores about 250 fitted neural networks (7 new arms, 12 monthly refits, 3
  seeds), each with per-row forecasts, all private; (2) more than one design defensibly tests the
  hypothesis (ablations, other half-lives, placebos), so every design is an arm with a
  pre-registered
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
  -0.063), with the copied share rising from 0.071 to about 0.126 (the stored monthly values). C5
  (both removed, B0) is
  roughly the sum of C1 and C4. The placebos C6 (age-permuted weights, WP, minus R0) and C7 (events
  with no
  level shift, IP, minus R0) are within 0.02 of C1 and of C4 respectively, and the control C8 (IP
  minus the stored injection-free arm) is within 0.02 of zero. The likeliest miss is C1 at the
  eight, where experiment 710's recency gain was not significant.
decision_rule: >-
  The reference of every contrast is R0, the replay of arm 9a, because R0 shares the code state,
  environment, and run date of the new arms; contrasts against the stored arm 9a are secondary. A
  contrast is a paired difference (arm minus reference) of the normalised MAE of the median, on the
  rows that the 31-system joint pass of experiment 677 shares, with a 7-day-block bootstrap of
  origin
  days (10,000 draws), read separately for the six signed-MW and the eight MVA-only Primaries, and
  also per Primary, with D15 and D18 reported only and the eight also read without D8. Experiment
  677's 7-day blocks and MAE headline depart from the demand protocol's month blocks and CRPS
  headline, because the reference's published 7.39 is a 7-day-block MAE; CRPS is read beside every
  MAE contrast. The family is C1 to C5 (5 contrasts, 2 groups, 10 readings, 99.5% intervals).
  Placebos and controls C6 to C8 are read at the same level but sit outside the family. The
  threshold is zero for every contrast, because a half-life and an injection switch cost nothing in
  production, with a point estimate of at least 0.02 required for an arm to replace R0 so that a
  0.01-point gain does not churn the recipe. Each contrast must also have the same sign for all
  three
  seeds. Recency claim, per group: the weights help when C1's lower limit is above zero and C6's
  lower limit is above zero there. The verdict is `supported` when both groups hold, `partly
  supported` when one holds, `refuted` when C1's upper limit is below zero at either group, `not
  supported` when C1's interval lies within -0.02 to +0.02 at both groups, and `inconclusive`
  otherwise, including when C1 holds but the control C6 fails. The front-matter verdict covers
  recency only; the half-life (C2, C3) and injection (C4, C5, C7, C8) results are stated in the
  headline. An arm replaces R0 only when its MAE interval is wholly below zero at both groups, its
  CRPS upper limit is below +0.02, and its copied share of injected shifts (experiment 677's
  held-out bank, 0 to 14 days) is no more than 0.02 above R0's. Under that rule the injection-free
  arm is expected to fail the share guard. Numbers before the final scoring are labelled interim.
benchmark: >-
  Its replay R0 (the reference of every contrast) and arm 9a of experiment 677 (`k1_5w_sp1`, monthly
  refits, seeds 0 to 2), stored; the
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

- **Injection:** arm 9a passes `--synthetic 1`, so stage 3 only sees the 581 injected shifts (events
  covering 5% of each Primary's half-hours, which leaves 8.5% of rows with a shifted lag; shifts of
  0.15, 0.30, or 0.50 of capacity, lasting 3, 7, 15, or 30 days, applied to the lag inputs; targets
  are never shifted). The early-stopping week is shifted too. One injected world per refit month is
  shared by the three
  seeds. Test rows are never shifted. The stored arm `5wo` is the same recipe without injection.
- **Recency weights:** stage 3 weights each training row by `0.5 ** (age_days / 90)`, drops the
  newest
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
| WP | 90-day weights, ages permuted by experiment 710's `permuted_age_days` (calendar months permuted, one seed shared by the three seeds); E3 acts on the true ages | C6: placebo for C1 |
| 5wo (stored) | no injection (training rows and the early-stopping week both unshifted) | C4: does injection help? |
| IP | events placed as in arm 9a, with depths set to zero | C7: placebo for injection; C8 is IP minus 5wo |
| B0 | equal weights and no injection | C5: both removed |

## Stop rule and approvals

**Recency has four `not supported` verdicts and injection has three, so this experiment is the
fourth on recency and needs the maintainer's approval.** Recency: experiments 331, 407, 419, and
466.
Injection, as the ML manager grouped it: experiments 486, 495, and 603, which test injection rate
and schedule, a second pass, and perturbation in pre-training. The maintainer asked for this
experiment by
name on 2026-11-23 (quoted in the brief) and approved starting it in chat on 2026-11-25 ("Approve"),
as the ML manager reported. Per the protocol's stop rule, every contrast is scored at every Primary.

## Method

### Architecture

- The network is arm 9a's: one pooled Gaussian-mixture neural network (3 components) over the 16
  Primaries, with a shared lag encoder (24-hour and 1-week lags), weather and date-time encoders,
  60 input columns, a D5 output offset, and 11 quantiles read from the mixture's CDF.
- The weather input of each row is the ECMWF ensemble run (00Z) issued 9 hours before the native
  09:00 UTC origin, so it is known at the forecast origin. Experiment 677's assert of `nwp_init_time
<= power_fcst_init_time` holds by construction after 2026-06-30, where stub demand frames are used,
so the new code adds an independent assertion on the ensemble summary files of every origin from
2026-07 to 2026-10 (run time at or before origin minus 9 hours) and prints one example row with the
run time, the origin, and the oldest lag time. The lag-age check of 677 samples 200,000 rows, not
every row. The network has no lag-free head.
- Stages 1 and 2 are frozen at 677's stored weights for each month and seed.

### Training regime

- Stage 3 fine-tunes from the stage-2 weights with AdamW (rate 2e-4, weight decay 1e-3), batch
  8,192,
  up to 60 epochs with patience 8, and the loss is the weighted mixture negative log-likelihood.
- Training rows come from the 730 days before each refit, cleaned (the `substation_zero` half-hours
  dropped) with the noisy start of the short-history Primaries cut, as in experiment 677.
- Each arm refits monthly, because monthly refits are the research practice.
- The runner is a copy of experiment 677's `e00677_k1.py` (experiment 677 is immutable), with a test
  that the unpatched copy equals the original apart from the added block. W0, W30, W270, and B0 set
  `half_life` through `dataclasses.replace(e.SPEC, ...)` (infinite for equal weights), which keeps
  the E3 embargo and halving. WP wraps `select_rows` inside the E3 wrapper. IP sets `DEPTHS` of
  `e00581_bank` to zeros, so onsets, lengths, and signs equal arm 9a's. B0 also uses `--synthetic
0`.
- Every script sets `E00677_ROOT` to the experiment's own data folder and asserts that the path
  contains `00731`, so nothing writes into experiment 677's folder; 677's scores are linked
  read-only. Before each prediction phase, a script asserts that all 36 `.pt` files exist and that
  every train-phase log says "CHECKS PASSED", so a failed fit cannot be refitted on data that
  includes the scored month.
- The projected cost is 12 months x 7 arms x about 2.5 minutes (train about 84 s, predict about 63
  s) = about 3.5 GPU-process-hours, plus about 1.4 hours of held-out-bank passes (7 arms x 12 months
x
  about 1 minute), so about 5 GPU-process-hours against a stop line of 12. Disk is about 15 GB.
  Memory caps start at 10 GB per process and are reset from the measured peak.
- Every fitted network is saved as a state file, with its loss history, in the experiment's data
  folder.

### Measurement

- Scored rows are experiment 677's (`backtest_year_2025_11_to_2026_10_daily_12utc`), 365 daily 12:00
  UTC origins, 672 half-hours each, 16 Primaries, at lead 0.5 hours to 14 days, on cleaned actuals.
  **Reproduction gate:** before any contrast is read, the joint pass over experiment 677's 31
systems
  must give arm 9a 7.3946 (95% interval 7.14 to 7.67) on 3,879,313 rows with 2,000 draws (pass A,
  unchanged). R0's seed-mean MAE in pass B must be within 0.05 of arm 9a's; otherwise the months are
  compared one by one before any other reading (folds 2025-11 and 2025-12 ran before a head change).
- Pass B is the same 31 systems plus the new arms, so the shared rows and 9a's 7.3946 stay unchanged
  (both asserted); it uses 10,000 draws. The scorer is new code (677 has no group, per-Primary, or
  99.5% scoring), and 677's own slice files are never overwritten.
- The year includes valid times from 2026-07 to 2026-10, which the protocol reserves for finalists
  ("no experiment scores a valid time on or after 2026-07-01"). The row set is fixed by the brief,
so
  every score is tuning-adjacent, and no selection is made from it beyond the pre-registered rule.
- The copied share of injected shifts comes from experiment 677's held-out bank pass (`bank`) for
  R0, W0, W30, W270, WP, IP, and B0; the stored banks cover arm 9a and `5wo`.
- Also reported: the tails (q01, q05, q95, q99 coverage and pinball loss), cold winter nights, the
  avoidable-curtailment cost, a day-ahead and a 14-day forecast example, and at least 6 event plots
  with the unlogged-event caveat (the clean-lag rows may hold events no label covers).
- The load-shift scenarios are experiment 677's four slices (all rows, outside events and at least 7
  days after one, inside events, the 7 days after an event ends).

## Checks planned

- A known-answer test of the weights: ages 0, 90, and 180 days give 1, 0.5, and 0.25, and the
  placebo
  keeps the multiset of weights, so the total weight is equal.
- A known-answer test of the injection: zero depth leaves the lag block unchanged, a depth of 0.3
  shifts the lag block by 0.3 of the series level, and a mid-day onset is placed correctly.
- A smoke test of one month and one seed per arm before the first quoted fit, and a "fits starting"
  line to the ML manager with the fit count, the projected GPU-process-hours, the memory caps, and
  the cores.
- A known-answer test that IP's lag block equals the plain unshifted block on every row and column,
  and that WP's weights keep the multiset of weights apart from the clipping of experiment 710's
  placebo (the total weight is printed).
- A phases script that checks the review marker before each phase, and a per-phase time log with
  device, start, and end.
- Plots of the training and validation loss against the epoch for one fold and one seed per arm,
  with the stored epoch marked (per-epoch histories are saved by the new runner), and a gallery of
  injected worlds (lag block with and without the shift, for two refit months).
- Checks list: median minus mean forecast per Primary, the MAE contrast beside every CRPS contrast,
  and leave-one-Primary-out for every contrast.
- The audit triggers of the `ml-experiment` skill (failed controls, contradicted prior, MAE and
  CRPS disagreeing, leave-one-Primary-out).

## Files

- `ml-experiments/00731-step-5-recency-and-injection-ablations-monthly-refits/`: scripts to be
  written (the stage-3 variant runner, the arm driver, the scorer, the figure script).
- `data/ml-experiments/00677-building-up-the-candidate-demand-network-one-component-at-a-time/`:
  read
  only (stage-1 and stage-2 weights, stored arm 9a and `5wo`).
- `data/ml-experiments/00731-step-5-recency-and-injection-ablations-monthly-refits/`: predictions,
  models, and loss histories written by this experiment.
