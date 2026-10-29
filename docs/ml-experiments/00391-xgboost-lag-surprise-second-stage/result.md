---
id: "00391"
slug: xgboost-lag-surprise-second-stage
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  complex. Stored: per-arm prediction folders, a stage 1 surprise table per Primary (kept for
  experiment 392's arm R5), and detector alarm tables, all private. Designs: more than one
  defensible
  (surprise from walk-forward or cross-fitted stage 1; ENS-driven or ERA5-fed residual; which
  neighbour summaries). Callers: imports experiment 87's fit modules and experiment 383's folder
  (interim until its pull request merges; copied if it has not merged). Production serving path and
  degradation rule: not touched.
hypothesis: >-
  Giving the weekly XGBoost demand model a measure of how surprising each of its three lagged power
  readings is, at its own Primary and at the neighbouring Primaries, lets the model handle
  load-shift events as well as an explicit event detector's open-alarm state does. The detector
  state adds nothing that surprise at the Primary and its neighbours cannot supply.
prediction: >-
  Written before any score is read. P1. On rows whose lags lie inside a labelled event, own
  surprise (N0) cuts the continuous ranked probability score (CRPS) by 10% or more of the gap
  between the baseline and the oracle (O). P2. Neighbour surprise (N1) cuts it by a further 5% of
  that gap at the Primaries with a labelled transfer. P3. The detector state (N2) adds less than 5%
  of the gap beyond N1, and N3 (detector only) recovers less than N1. P4. On rows outside events
  and the 7 days after them, no surprise arm is worse than the baseline by more than 0.02 points of
  effective capacity. P5. The ENS-driven and ERA5-fed lagless forecasts give surprise with a
  correlation of at least 0.9 per Primary.
decision_rule: >-
  Seven planned contrasts (C1 to C7 below), each read at the 99.29% level (0.36% per tail, a
  Bonferroni correction across 7 at a 95% family level), with whole-event blocks on event slices and
  month blocks
  elsewhere. A surprise arm counts as helping when its interval on its primary slice (rows whose
  lags lie inside a labelled event, days 1 to 10) excludes zero in the favourable direction and its
  mean change on the rows outside events and the 7 days after is not worse than +0.02 points. Every
  other slice is descriptive.
  The headline C3 reads "surprise gives the benefit of the detector" as supported when its
  interval on the event-lag rows includes zero and has both bounds within ±0.03 points of effective
  capacity. The
  porting threshold is set by cost: an extra column set computed from the same stage 1 model
  (N0, N1) needs a gain whose interval excludes zero; the detector (N2, N3) adds a state machine
  and a veto, so it needs a gain of at least 0.03 points of CRPS on the event-lag rows to be worth
  porting.
benchmark: >-
  The weekly XGBoost member of experiments 379 and 381 (stored, for the reproduction gate and as a
  reference row), the same recipe refitted monthly (B0, the control for every contrast), the
  lagless fallback c2 of experiment 383 (interim), BG2, `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, mean absolute error of the median, pinball loss and
  coverage at q01, q05, q95, and q99, and avoidable-curtailment cost, in points of effective
  capacity (cost in GBP), from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days
  1 to 10.
code: ml-experiments/00391-xgboost-lag-surprise-second-stage/
production_status: not-for-production
adopted_in: none
---

# Does a "surprise" input let XGBoost tell when its lagged power is inside a load-shift event?

**Status: planned.** No fit has run and no score has been read.

## Question, and what is new

**We ask whether the weekly XGBoost demand model needs an explicit load-shift detector, or whether
a measure of how surprising its own recent readings are, at its Primary and at the neighbouring
Primaries, gives it the same help.** A load shift moves demand between substations, so a lagged
reading that lies inside one misleads the model. The detector of
[experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md) and the
distillation design flag such periods from the lagless forecast's residuals.
A surprise input hands the model the same residuals without the state machine.

**Prior work (figures to be checked against the pages before the write-up relies on them).**
Experiments 5 and 18 tried a staged chain with lag anomalies on old rows: in 18 the chain minus the
single-stage model N1 was +0.08 (99.17% interval -0.11 to +0.25) over all leads and -0.27 (-0.62
to +0.03) up to 48 hours, with no neighbours and no detector features. Experiment 8 found 24-hour
summaries added nothing (-0.02) to a calibrator with raw lags. **New here:** the `demand-v1` rows
at all 16 Primaries; a second-stage XGBoost model refitted monthly; neighbour surprise from NGED's
own interconnection list; detector features from the distilled design; and scoring on labelled
events with the avoidable-curtailment cost.

**Advice followed and departed from.** To be filled in after running the `advice` listing in the
skill (read before the code step); departures are recorded here before any score is read.

## Design

**Cadence.** Every arm, the baseline control, and the stage 1 lagless model refit monthly on the
rolling recipe of experiment 87's `n1h` configuration (500 rounds, half-life 0.5 years),
`colsample_bytree` and `colsample_bynode` fixed at 1.0 in every arm, seed 0. Weekly refits are run
only for the reproduction gate. A winning arm is re-checked weekly before any release claim, and
that re-check is outside this experiment.

**Widening.** The manager reports that experiment 381's stored bundle forecasts carry no 1.15
widening of the band, while experiment 379's `only_W` member has it (mean q95 minus q50 of 0.1360
against 0.1554; to be verified in the code step). **Every arm here, B0 included, is refitted by this
experiment with no widening, so the contrasts compare like with like.** Every score table is given
twice for every arm and reference row: raw, and with the band widened by 1.15 about the median. The
stored references are labelled with the widening they carry, and a stored row is rescored both ways
by undoing or applying the factor. The headline reads from the raw table; a contrast whose sign or
verdict differs between the two tables is reported as widening-dependent.

**Triage of the plan review (2026-10-29).** Accepted: stage 1 is fitted with no label filter
(experiment 383's `lagless` arm filters event targets, which would build every surprise from
hindsight labels; that filtered version is not run); N1 holds no detector state; the timing of
"the origin" is defined below; rows without surprise are kept with missing values, not dropped; the
Bonferroni and equivalence figures are corrected; the event counts, the signed-MW-six restriction of
the curtailment cost, and the scope check of the past-weather pages are added. P1 to P3 are
restated as absolute changes in points of CRPS, with the O minus B0 gap reported beside them.

**Timing.** The telemetry delay is 3.5 hours. Z in N0 is the mean of u over the 24 half-hours ending
at the origin minus 3.5 hours. The neighbour columns and the detector state come from the latest
6-hourly detector origin (00, 06, 12, 18 UTC) at or before the origin minus 3.5 hours. Stage 1's
quantiles for a half-hour come from the control member only: the training history holds one member
per run, so one member is used everywhere, and the run for half-hour s is the one with the latest
forecast origin before s (lead 0.5 to 24 hours; daily runs from 2024-04-01 cover every half-hour).
The first stage 1 refit is 2024-10-01, with 6 months of training rows. The noise units for Z~
use the expanding history up to the detector origin, so no row sees later medians. The threshold k
and the known-answer gate use the 2024-04 to 2025-06 labels as the distillation specifies, so rows
inside that span carry a labelled-tuning shortcut, reported in the write-up. The ERA5-fed stage 1
uses the same monthly walk-forward as the ENS-driven one. The ENS run frames must hold steps at
leads 5.5 to 29.5 hours; the code step checks this first and builds them from the ENS store if not.

**Reproduction gate (before any other arm is read).** Refit the baseline weekly at every fourth
refit with experiment 383's gate (`d` arm) and reproduce the stored weekly member within that gate's
tolerance. If the gate fails the experiment stops and reports the failure.

**Stage 1: the lagless quantile model, built twice.** It is experiment 383's arm `lagless` (no
origin lags, quantile output), refitted monthly. **ENS-driven (main, causal):** the same weather
inputs as the baseline, taken from the ECMWF ensemble run that was latest when the half-hour
happened. **ERA5-fed (labelled a hindsight ceiling):** ERA5 and CAMS weather for the half-hour
itself, as in experiment 371. ERA5 arrives about 5 days late and CAMS radiation 1 day late, so the
ERA5-fed arm cannot be served for the two recent lags.

**What counts as surprise at serving time.** For a past half-hour s, the ENS-driven residual uses
the latest ENS run with init time plus the 9-hour publication delay at or before s plus 3.5 hours
(the telemetry delay): the newest lag is s = origin minus 3.5 hours, so this run is the latest one
available at the earliest origin that uses s. The surprise table is therefore keyed by Primary and
half-hour alone, and the same table serves training and test rows. **The experiment trains and tests
surprise built the same way.** To be confirmed against experiment 87's run frames: the lead of s in
that run is 5.5 to 29.5 hours.

**Leakage rule for stage 1.** Surprise for half-hour s comes from the stage 1 refit with the latest
train end at or before s, so no stage 1 fit has trained on s (walk-forward, no cross-fitting).
Rows whose lags predate the first stage 1 refit plus one month keep missing surprise values in
every arm; the write-up states the first stage 1 train end and the rows affected per Primary. The
code review checks this rule,
and the stage 1 fits' stopping at `train_end`.

**Surprise columns.** The normal score u is the probability integral transform (PIT) of the
reading against stage 1's 11 quantiles, interpolated linearly, clipped to 0.005 to 0.995, then
converted by the normal quantile function.

- **N0, own surprise.** B0 plus u at each of the three origin lags, plus Z, the mean of u over the
  24 half-hours before the origin (at least 18 present).
- **N1, own plus neighbour surprise.** N0 plus three fixed columns summarising the neighbours'
  Z~ at the origin, using `nged_neighbours()` of experiment 383: the largest |Z~| among metered
  neighbours, the signed mean of Z~, and the number of neighbours with Z~ of the opposite sign and
  |Z~| above 2. A Primary with no metered neighbour gets missing values in all three. Unmetered
  interconnections have no series and enter only as a count column in the detector's veto
  eligibility, not as surprise.
- **N2, N1 plus the detector's open-alarm state.** The detector of
  `briefs/opus_detector_distillation_answer.md`
  (Z at 00, 06, 12, and 18 UTC, noise units Z~ by median and 1.4826 times the median absolute
  deviation, alarm on two consecutive same-sign origins above k, end below k/2 at two origins,
  14-day
  cap with re-opening), built from the ENS-driven residuals, with the neighbour veto (experiment
  315,
  changed so a load split across several neighbours protects the alarm). Causal columns only: open
  or closed, onset stamp, signed Z~, R (the mean residual in MW), the days since onset, and the veto
  tag. An alarm's end stamp and final magnitude are never used. **N2 without the veto** is one extra
  comparison row.
- N2 also adds the count of neighbours with an open alarm of the opposite sign. The veto uses
  experiment 298's neighbour list, as the distillation specifies, while N1 uses NGED's list; the
  write-up reports where they differ. **N2 without the veto** uses the same k as N2.
- **N3, detector features only.** B0 plus the N2 detector columns, with no surprise columns.
- **O, oracle ceiling.** B0 plus a 0/1 flag for the target being inside a hand-labelled transfer
  and one such flag per lag. Labelled
  hindsight, never a candidate.
- **N1-ERA5.** N1 with the ERA5-fed surprise, labelled a hindsight ceiling.
- Threshold k: the smallest value on the grid 1.5 to 6.0 in steps of 0.05 giving at most 2.0 false
  alarms per Primary-year after the veto, pooled over 2024-04 to 2025-06, on that period's labels
  only. The noise-unit median and scale use the same months, which post-date some 2024 cuts; the
  write-up reports that shortcut, as experiment 327 did.

**Detector evidence (before any real score).** The known-answer gate of the distillation, section
5, in full (injected transfers at the 10 Primaries of experiment 298's injection set, recall and
veto and placebo thresholds); the experiment stops and reports the detector alone if it fails. The
write-up reports the veto's false alarms removed and labelled transfers lost, and what the veto does
at Primaries with no metered neighbour.

**ENS against ERA5 for demand.** Per Primary, the correlation and the mean absolute difference
between the ENS-driven and ERA5-fed surprise, and the CRPS of the two stage 1 forecasts on the
same half-hours. If they agree closely (prediction P5) the ERA5-fed arm is called an equivalent
version, otherwise it keeps the label "hindsight ceiling".

**The surprise tables for experiment 392.** The ENS-driven stage 1 u, Z, and Z~ per Primary and
half-hour, written to `data/ml-experiments/00391-xgboost-lag-surprise-second-stage/surprise_tables/`
with a README stating how they were built and which stage 1 fits did not train on each row. The
manager is told the path as soon as they exist.

## Planned contrasts

- C1 N0 minus B0 (own surprise).
- C2 N1 minus N0 (neighbours).
- C3 N2 minus N1 (does the detector state add beyond surprise). **Headline.**
- C4 N3 minus N1.
- C5 O minus N2 (room left).
- C6 N1-ERA5 minus N1 (ENS-driven against ERA5-fed).
- C7 N2 minus N2 without the veto.

## Scoring

All 16 Primaries to 14 days under `demand-v1`, by lead band (0 to 24 hours, 24 to 72, 72 to 168,
168 to 240) and the days 1 to 10 headline. Slices: during labelled events and for 7 days after;
rows whose lags lie inside an event; rows outside both (surprise columns must cost nothing there,
and the detector's false alarms per Primary-year); per Primary, worst Primary, leave-one-out, per
group (signed-MW six, MVA-only eight, D15 and D18 report-only), cold nights, sunny half-hours.
D2 and D4 are the placebos. The avoidable-curtailment cost is reported in the three row sets
(during each event, 7 days after, all rows) with experiment 383's definition (limit 0.9 of the
earlier maximum, 12 to 36 hours leads; D9 and D11 have no event rows), copied into this folder.
Tail calibration at q01, q05, q95, q99 and the interval width and coverage on nulled-or-not rows
are reported beside CRPS. Importance of the surprise columns follows the `study` skill's recipe.

## Checks, reviews, and compute

Opus reviews the plan once; a Sonnet reviewer checks the code for leakage and the `train_end`
stops; one Sonnet audit of the scored numbers; then the write-up and prose reviews. The run uses 3
threads and `MemoryMax=16G` under `systemd-run --user --scope -p OOMPolicy=continue`.

## Compute estimate

Stage 1 refits monthly over the whole stage 2 training history for two versions (ENS-driven and
ERA5-fed) at 16 Primaries; stage 2 then fits 8 arms (B0, N0, N1, N2, N2 without the veto, N3, O,
N1-ERA5) at 16 Primaries for each test month. The exact fit count and wall-clock estimate are
written
here from the first timed fit, before the full run. Stage 1 predictions stream the daily run frames
one at a time and are cached per Primary. Event counts per slice are reported in every table.

## Design target and lag-encroachment sweep

**NGED would like a normal-running-arrangement (NRA) forecast, and it may be enough that only the
median is NRA while the outer quantiles span the NRA level and the abnormal level.** Each arm is
judged on three properties together, because CRPS on event rows rewards a forecast that follows the
event level and cannot tell the two apart. Tolerances, fixed before any score is read:

- **Median is NRA:** the median's share of the step depth D in the sweep is at most 0.2 at every
  encroachment level k (0 means NRA, 1 means the event level).
- **Tails span both levels:** at least 90% of event-level readings (rows whose target is in an
  event, and synthetic rows at the event level) lie inside the 1% to 99% band at k of 1 or more,
  and q95 and q99 (q05 and q01 for a negative event) reach the event level.
- **The price of spanning:** the 5% to 95% band on rows outside events and the 7 days after them
  widens by at most 10%, and CRPS there changes by at most +0.02 points of effective capacity, at
  all 16 Primaries.

**The sweep is prediction-only.** The fitted boosters of every arm are saved (`.ubj`) at every
monthly refit (12 per arm and Primary) under
`data/ml-experiments/00391-xgboost-lag-surprise-second-stage/work/boosters/<arm>/`; the sweep uses
all 12. Clean rows (target outside any event, all lags clean) get a step
of depth D added to the k most recent lagged inputs, in order: the newest reading (origin minus
3.5 hours), the same time on the latest day, the same time in the latest week. Levels: k of 0
(control), 1, 2, and 3. Depths: 2%, 5%, and 10% of effective capacity, plus the label table's
quartile depths (0.71, 1.1, and 2.3 MW) as shares of capacity, both signs. Two target cases are
reported separately: the event is still under way (reading is the clean value plus D), and it ended
before the valid time (reading is the clean value). For arms with surprise or detector columns,
the shifted readings replace the stored readings at the lag times: `u` is recomputed from the
stored stage 1 quantiles, the window mean and Z~ follow, the neighbours carry the opposite step
(shared equally over the listed neighbours, so load is conserved), and the detector is re-run on
the shifted statistics. Reported per arm and k, by lead band (0 to 24, 24 to 72, 72 to 168, 168 to
240 hours), group, and Primary: the median shift and its share of D; the ratio of the 5% to 95% and
1% to 99% band widths to their k of 0 values; the coverage of the clean and of the event-level
value by the 5% to 95% band and their quantile positions; and the k at which the band first
reaches the event level. The lagless stage 1 model is the control (share 0, no widening). A few
real labelled events, chosen by a rule fixed before looking (the three longest events with a
metered neighbour), are compared with the synthetic ones. The sweep is also run on experiment 392's
request for arm N1 (its arm R5) at the same refits.
