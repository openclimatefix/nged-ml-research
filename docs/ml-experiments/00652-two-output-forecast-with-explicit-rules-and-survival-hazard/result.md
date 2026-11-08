---
id: "00652"
slug: two-output-forecast-with-explicit-rules-and-survival-hazard
title: >-
  Does a departure head with an explicit survival output beat the best forecast that does not know
  when a shift ends, once each forecast is scored under its own rules?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) It stores per-row scores and small head weights (a few thousand parameters each) and
  no new K1 weights; (2) more than one design would defensibly test the hypothesis (a fixed
  duration prior, a fitted survival curve, a learned survival output), so it runs them side by
  side; (3) it reuses experiment 647's head and inputs and experiment 650's banks, K1 rebuild, and
  stored heads, whose callers are named below; (4) it touches no serving path or degradation rule;
  (5) it changes no contract.
hypothesis: >-
  A prevailing-conditions head that outputs a signed depth and a survival probability per lead day,
  and is trained on a stated mixture of shift durations, scores a lower CRPS than K1 shifted by its
  own decayed recent error (R0 at 168 h) and than a no-fit hazard baseline (K1 shifted by its recent
  error and decayed by the best single survival curve), at every origin state in which a shift is in
  force, once every forecast is scored under its own rules.
prediction: >-
  Written before any head is fitted. Scaled CRPS of R-haz is below R0-168 by more than 0.05 points
  of capacity at origins far from a shift's end, and below the no-fit hazard baseline by more than
  0.05 points at origins within 24 hours of a shift's end. The loss of every prevailing forecast
  after a shift ends is larger than K1's, and the loss of R-haz there is no larger than that of the
  no-fit hazard baseline. R-haz's predicted survival probability is within 0.10 of the observed
  share of shifts still on at every lead day. The placebo scores within 0.05 of K1 after equal width
  scaling.
decision_rule: >-
  R-haz beats a baseline in a stratum when the month-block 95% interval (percentile bootstrap over
  the 5 development months, seeds averaged within a month) of the scaled CRPS difference, averaged
  over lead days 1 to 10, excludes zero in the predicted direction and the point estimate exceeds
  0.05 points. The 0.05 floor is the brief's minimum detectable effect, set before any score; no
  power calculation backs it, and R0-168 needs no fit, so a second neural network must beat it
  clearly. The one veto is on event-free real rows: R-haz must not exceed K1 by more than 0.03
  points of CRPS. A score after a shift ends that is worse than K1's is an expected cost of any
  prevailing forecast and enters no veto, provided R-haz is no worse there than the no-fit hazard
  baseline. The sensitivity runs (short-heavy and long-heavy duration mixtures) must keep the sign
  of every contrast; an R-haz that wins only under its own training mixture is reported as
  mixture-dependent. The first two months (2025-10 and 2025-12) are a feasibility gate, not a test.
benchmark: >-
  R0-168 (K1's quantiles shifted by K1's own mean residual over valid days 2 to 8 before the origin,
  times exp(-lead/168 h)), a no-fit hazard baseline (the same residual decayed by the best single
  survival curve fitted to the training durations), R-old and R-new (experiment 650's stored
  heads), and K1 as context only.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity after
  equal width scaling, by lead day (1 to 10) and by the state of the shift at the forecast origin;
  q05-q95 coverage; q01, q05, q95, and q99 pinball loss; survival reliability.
production_status: not-for-production
adopted_in: none
---

Plan only: this page is committed before any head is fitted, and the sections below are filled in
once the experiment has run.

## Two definitions and their scoring rules

A load shift is a period when the network operator moves part of one Primary's demand onto or off a
neighbouring Primary. K1 is the neural network of experiments 612 and 613. The two forecasts below
answer different questions, so each is scored on its own truth and its own rows.

- **The normal-running-arrangement (NRA) forecast ignores shifts.** Its target is the demand the
  Primary would draw with no shift. In the injected bank that truth is known exactly (the unshifted
  real power). On real rows it is scored only outside detected or labelled shifts. NGED's labels are
  a one-off, incomplete dump, so the counts of real shifts are lower bounds and clean real rows may
  hold unlabelled shifts. Scoring the NRA forecast on rows that contain a shift would blame it for
  riding through the shift, which is its job. The NRA output in this experiment is K1, unchanged.
- **The prevailing forecast continues the arrangement now in force.** Its target is the observed
  power, and it is scored on all rows. A prevailing forecast issued during a shift will carry the
  shift past its end, because shift ends are random and cannot be known at the origin. The only
  legitimate hedge is a survival probability that decays the shift over the forecast horizon. A bad
  score after a shift ends is therefore an expected cost, and the question is whether the forecast
  is worse than the best forecast with no knowledge of the end time.

## Scoring by the state at the forecast origin

The row is assigned to a stratum by the state of the shift at the origin, not at the valid time. The
strata, on the injected bank, are:

- **Event-free origin:** no shift in force at the origin, none ended in the previous 7 days, and
  none starting before the row's valid time. Rows whose horizon contains an unseen onset are
  excluded from all strata and counted, because no forecast can see an onset.
- **Far from the end:** a shift is in force and remains for more than 10 days.
- **Middle:** a shift is in force and remains for 1 to 10 days.
- **Near the end:** a shift is in force and remains for at most 24 hours.
- **Just ended (context):** the origin is within 7 days after the end.

Every table and figure reports the loss by lead day, 1 to 10. Real rows give the event-free veto and
a descriptive row for the verified D5 and D16 pair.

## Arms

All arms use the same rows, the same equal width scaling, and 2 seeds (the fitted ones).

- **K1:** the NRA output; context only for the prevailing scoring.
- **R0-168:** K1's quantiles shifted by K1's recent residual times exp(-lead/168 h). No fit.
- **B-haz:** the same residual decayed by S(age + lead) / S(age), where S is the best single
  parametric survival curve (exponential, Weibull, log-normal, or log-logistic, chosen by
  likelihood) fitted to the training shift durations. No neural network. Refitted under each
  duration mixture.
- **R-old and R-new:** experiment 650's stored heads (fixed duration prior), reused.
- **R-haz:** R-new plus an explicit survival output (a non-negative hazard per lead day, so that
  survival falls monotonically with lead) trained with a binary cross-entropy against the observed
  share of shifts still on, added to the pinball loss. Training rows are weighted so that the
  durations follow the stated mixture. The prevailing quantiles are the NRA quantiles plus the depth
  times the survival probability, in the mixture form of experiment 647, so the band widens with the
  uncertainty about whether the shift survives.
- **Placebo for R-haz:** the same head refitted with each training row's target, survival label, and
  in-force flag permuted among the rows of the same Primary, so that the target is real but unlinked
  to the inputs. It is scored on the same rows and reported beside every contrast.

## Duration mixture

The generator draws a duration of 3 to 14 days with probability 0.5 and 28 to 112 days otherwise.
The mixture is an assumption: the only measured durations are one verified 15-day pair and one
unconfirmed step of about 16 weeks, so no real distribution exists. The sensitivity runs reweight
training and scoring to 0.8 short and 0.2 long, and to 0.2 short and 0.8 long.

## Standing practices

CPU only (each head fit takes about 40 seconds); the GPU is not used. Heads are one per scored month,
trained on the other 4 development months, not refitted on any cadence. No row on or after
2026-07-01 is read. Flagged `substation_zero` half-hours are excluded as in experiment 650, and the
D15 and D18 rows before 2025-03-01 are cut as in experiment 650's inputs. Stored experiment 650
heads and K1 rows are reused; R-haz, B-haz, and the placebo are fitted here.
