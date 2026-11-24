---
id: "00722"
slug: msa-mva-bsp-closure-and-fast-hint
title: >-
  Does the meter of a group of Primaries, minus the other Primaries and the outside generators,
  show the sign of the flow through one Primary that reports apparent power?
theme: solar-disaggregation
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
kind: other
size: >-
  Anything else. (1) It stores topology tables, regression tables, and verdict files, all private;
  (2) more than one design defensibly tests the hypothesis (the meter above the group, or the
  within-day change of the reading against the sky), so the design note orders them as separate
  rungs; (3) it reads the site map of experiment 521, the engine state of experiment 697, and the
  frame of experiment 39, and the callers of those files cannot be named without searching the
  repository; (4) it touches no serving path or degradation rule; (5) it changes no contract. Full
  set of reviews: plan, code (Opus, before the first run), results, write-up, prose.
hypothesis: >-
  The signed meter above a group of Primaries (a bulk supply point, BSP) shows the physical flow of
  each child with the right sign and size. Subtracting every other child and every outside
  generator from the meter therefore leaves a residual that moves with the left-out child, by a
  slope of at least +1 when the child imports and of at most -1 when the child exports, for every
  reactive floor. The residual's 2-hour change follows an independent 44-day record of one
  generator below D8 with a slope near -1, and follows the hidden signed flow of D15 when D15's
  signed series is folded into a magnitude. Where those two known answers pass, the same residual
  separates export from import at D12 in a joint regression with a term for the sky.
prediction: >-
  Written before any fit. The generator slope lies within -1 +/- 0.3. D15's closure slope lies
  within 0.7 to 1.3, and D15's sign is recovered with an area under the curve above 0.9 on large
  hourly changes, of which under half are large enough to score. At D12 the calibration
  coefficient is positive and larger than 1, the test coefficient is below -1 with a wide interval,
  and D3's coefficient stays near +1. The experiment can fail on any of these.
decision_rule: >-
  Job 2 (rung U2a) is `not supported` if any readable test fails its bar, `supported` if at least
  one test is readable and every readable test and its placebo pass, and `inconclusive` if no test
  is readable. Only `not supported` stops job 3. Job 3 (rung U2b) is `supported` when the
  calibration bar, the reverse-flow test bar, and the three controls pass, `not supported` when the
  calibration bar or a control fails or the test refutes reverse flow, and `inconclusive` when a
  fit is void or the test is unresolved. Every bar and every threshold is a named `Final` constant
  of `e00722_core.py`, set before any real data are read. The experiment ports nothing, so no
  smallest gain applies. A failed rung is closed with no variant.
benchmark: >-
  Two known answers at BSP A (an independent 44-day generator record, and D15's own signed series
  folded to a magnitude), one placebo for each (a circular shift of the record by 7 and 14 days, and
  D15's reading shifted by 7 days), and D3 as a closure control.
row_set_id: >-
  not applicable (no forecast is scored): the jobs read past half-hours, and no valid time on or
  after 2026-07-01 is read.
metric: >-
  Slopes of the residual's change on the change of a child's series, with 95% intervals from 1,000
  bootstrap replicates that resample whole days; the area under the receiver operating characteristic
  curve for the sign; and detection limits of 2.8 bootstrap standard errors.
---

**We ask whether a meter that sits above a group of substations can tell us which way power flows
through one of them.** Eight of the 16 Primary substations report apparent power (MVA), which is a
magnitude, so a Primary that exports solar output shows a rise in its reading and not a negative
value. The signed meter at the bulk supply point (BSP) above a group of Primaries does carry the
sign. We subtract the readings of the other Primaries and the output of the generators that sit
outside every Primary, and ask whether the remainder moves with the Primary we left out. We test
this against two answers we know independently: a 44-day record of one generator below D8, and
D15, a Primary whose signed series we fold into a magnitude ourselves. If both tests pass, we ask
the same question at D12, where nobody knows the answer. Nothing here forecasts, and this page is
the plan: no data have been read for it.

**The honest expectation is that the generator test is readable and the D15 test is the one most
likely to be void,** because the two transformers that start in 2025 give a short span, and the
bars demand at least 200 scorable hours. A void test is reported as void and never read as a pass.

## Why this runs

**The design note for the untangling of the MVA-metered Primaries puts this experiment first,
because it needs no fit and costs under 1 core-hour a job.** The design (`mva_untangling_design_v2.md`),
its second Opus review, and its first amendment are in the private
`data/ml-experiments/briefs/msa_architecture/` folder. The review's section 8 is the specification
of this experiment, and it governs where it differs from the design. Four earlier uses of a BSP
meter for the sign were inconclusive, so each rung here runs once, and a failed rung closes with no
variant. Experiment 720 owns rung 0b of the earlier design, and the within-day concordance rung
is a separate experiment.

## What runs

**Job 1 (rung U0) has no hypothesis and holds no fit.** It joins each Primary to its BSP, classes
every metered generator by a regression of the Primary's half-hour change on the generator's change,
checks the engine record's coverage, the stamp alignment of the sources, the gaps, the switching-log
windows, the era-split table, and the holdout, runs the cloud-class audit from the sky alone, and
counts the same-sky pairs of hosts and farms. A failed audit or fewer than four pairs stops later
rungs and is reported.

**Job 2 (rung U2a) tests the BSP pathway against the two known answers.**

| Test | Bar |
|---|---|
| Detection limit, engine | 2.8 bootstrap standard errors under 0.5; otherwise void |
| Detection limit, D15 closure | 2.8 bootstrap standard errors under 0.3; otherwise void |
| Engine record | Slope interval excludes 0 and lies within −1.3 to −0.7; circular shifts by 7 and 14 days give an interval that includes 0 |
| D15 closure | Slope of `ΔR⁽⁻¹⁵⁾` on the true `ΔX_15`, hourly: interval within 0.7 to 1.3 |
| D15 sign | Area under the curve at least 0.85, with the ratio `ΔR⁽⁻¹⁵⁾ / Δm_15` as the score, on hours with `\|Δm_15\|` at least twice the noise scale of `ΔR⁽⁻¹⁵⁾`, crossing hours excluded and counted, at least 200 hours |
| D15 placebo | D15's de-meaned reading shifted by 7 days: interval of the area includes 0.5 (S3) |
| Reach | Share of D15's hours over the size threshold, no bar |
| Report-only | D10 folded under BSP B |

**Job 3 (rung U2b) is one joint regression at BSP A.** The response is the half-hour change of the
BSP meter minus the signed children's physical flows and the outside generators. The regressors are
the half-hour changes of the readings of D3, D8, D12, and D22, and the fast sky change `Δũ`. Every
series is de-meaned at matched quiet times first, and rows within two half-hours of a change in the
engine state of experiment 697 are dropped and counted. Days are counted by their number of
post-bounce stretches, and the bars bind on days with exactly one.

| Test | Bar |
|---|---|
| Detection limit | 2.8 bootstrap standard errors (whole-day blocks) under 1; otherwise void |
| D12, calibration fit | Lower end of the interval above +0.5; the point estimate is reported as the common-mode inflation |
| D12, test fit, days with one post-bounce stretch | Upper end below −0.5 supports reverse flow; lower end above +0.5 refutes it; otherwise unresolved |
| D3, both fits | Interval above 0 |
| Shifted date | D12's de-meaned reading from 7 days earlier: interval includes 0 |
| BSP B | D2, D4, D19, and D21: each interval above 0 |

## Method

- **Architecture:** no neural network and no hidden-state model: every test is an ordinary least-squares
  regression through the origin on series de-meaned at matched quiet times, with the regressor's
  noise left in.
- **Architecture:** the residual that leaves Primary `i` out is the BSP meter minus the readings of
  every other child and plus the output of each generator that sits outside every Primary or is
  added back, and it never subtracts the reading of Primary `i`.
- **Architecture:** a metered generator's output is subtracted only where the add-back regression of
  job 1 puts the generator outside every Primary or in a Primary's added-back series, and never twice.
- **Training regime:** nothing is trained. The seed of every bootstrap is fixed, and each job runs
  once on one core.
- **Measurement:** matched quiet times are half-hours with the same local half-hour, calendar month,
  and day type (weekday against weekend or bank holiday) that lie at least 6 half-hours from a change
  in the engine state.
- **Measurement:** every interval is the 2.5th to 97.5th percentile of 1,000 bootstrap replicates
  that resample whole days, with the same resampled days shared by every child of one BSP.
- **Measurement:** the detection limit of a test is 2.8 bootstrap standard errors, and a test over
  its limit is void.
- **Measurement:** no row on or after 2026-07-01 is read, and the stdout of every job holds ratios,
  counts, shares, and intervals only.

## Files

- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_core.py`: the named constants, the
  bootstrap, the matched-time de-meaning, and the decision rules (written, not yet run).
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/e00722_u0.py`, `e00722_u2a.py`, and
  `e00722_u2b.py`: jobs 1, 2, and 3 (written, not yet run).
- `ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/run.sh`: the launch wrapper, which runs
  no job without `--go`.
- `data/ml-experiments/00722-msa-mva-bsp-closure-and-fast-hint/`: every output, with dated detail
  under `PRIVATE/` (written by the jobs).
