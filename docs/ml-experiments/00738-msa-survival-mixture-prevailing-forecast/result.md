---
id: "00738"
slug: msa-survival-mixture-prevailing-forecast
title: >-
  Does an axial-attention demand forecaster forecast better while a load shift is in force when it
  carries the shift forward with a survival curve, instead of carrying it forward in full or not
  at all?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - switching-labels-in-the-forecast
status: planned
date: 2026-11-28
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores about 120 fitted neural networks or heads (one control and four level-term
  arms, 8 monthly refits, 3 seeds), their per-row forecasts on injected worlds and on the real
  record, and private score tables; (2) more than one design defensibly tests the hypothesis (a
  frozen or a fine-tuned base forecast, a mean-path decay or a mixture, a survival curve fitted or
  learned, and training on injected shifts or on the real record), so the plan fixes one and gives
  each choice its reason; (3) it copies the training loop of experiment 713, the shift generator of
  experiment 650, and the strata and scaling of experiments 652 and 654, whose callers cannot be
  named without searching; (4) it touches no serving path or degradation rule; (5) it changes no
  contract. Reviews, in order: a Sonnet plan review (done, 22 findings, all triaged), Sonnet
  code, a fresh Sonnet code review before the first fit, a real-environment smoke test, a Sonnet
  numbers review, and a prose review.
hypothesis: >-
  The axial-attention demand forecaster `M1` of experiment 713, kept frozen, forecasts a Primary
  substation's demand better, in continuous ranked probability score (CRPS), at origins where an
  injected load shift is in force, when a small trained head adds a level term (an estimate of the
  shift's size in megawatts, read from the last 14 days of power) and the forecast carries that
  term forward with a survival curve (the chance that the shift is still on at each lead). The
  survival version beats a version that carries the level term forward in full at every lead. The
  survival structure, and not only a training regime that shows a forecaster shifted targets, is what
  lowers CRPS, and the structure does not raise CRPS where no shift is in force.
prediction: >-
  Written before any fit. Every difference is the first arm minus the second, in unscaled points of
  capacity, over leads of 24 h up to but not including 240 h, over 14 scored Primaries, 8 monthly
  folds, and 3 seeds. The arms are defined in "The arms": `M1`, W0 (a fine-tuned control with no
  level term), LP (level term, full persistence), LD, LM, and LR (survival versions), and S*, the
  survival version with the lowest in-force CRPS on the screen. In-force CRPS pools origins where a
  shift is still on. C1 (S* minus `M1`, in force) lies between -4.5 and -1.5. C2 (S* minus LP, in
  force) lies between -1.2 and -0.2, because half the injected shifts last 3 to 14 days and end
  inside the horizon. C3 (S* minus W0, in force) lies between -0.8 and +0.6, with no direction
  predicted, because W0 can follow a shift through its own history rows and S* cannot follow
  a shift older than the 14 history rows. V1 (S* minus `M1`, event-free injected origins) lies
  between -0.05 and +0.05. The net all-rows effect N1 = 0.05 C1 + 0.95 V1 then lies between -0.27 and
  -0.03. S* is LD or LM, and LR beats it by under 0.3. V2 (S* minus `M1` on the real record, all rows,
  no mask) lies between -0.03 and +0.12. Maintainer-side estimate of the verdicts: `partly
  supported` 40%, `supported` 12%, `not supported` 20%, `small gain` 10%, `inconclusive` 15%,
  `refuted` 3%. The prediction fails if any of C1, C2, C3, or V1 falls outside its range, or if
  S* is LR.
decision_rule: >-
  Fixed before any fit. S* is the arm of LD, LM, and LR with the lowest in-force unscaled CRPS on the
  4 screen folds at seed 0 (an arm within 0.10 loses to the simpler arm, in the order LD, LM, LR).
  The unscaled CRPS governs every reading, and the scaled figure is report-only. N1 = q C1 + (1 - q)
  V1 is the net all-rows effect of the bundle, with q = 0.05 the assumed share of real rows in a
  shift, and N3 = q C3 + (1 - q) V3 is the net effect of the survival structure alone, with V3 the
  event-free difference S* minus W0. The porting threshold is on the net effect: N1 at or below
  -0.05 (the protocol's 0.03 for a new head plus 0.02 for a new fitted step) and N3 at or below
  -0.03 (a new head). A net effect is a gain when the upper limit of its 98.33% t interval on the 8
  monthly differences (Bonferroni across N1, N3, and C2) and of the 98.33% month-block bootstrap
  interval are both below zero, and each of the 3 seeds has the same sign. C2 is a gain on the same
  rule with no threshold. V1 fails when its point estimate is above +0.02, the protocol's harm tier.
  V2 is a report-only harm check, flagged when its point estimate is above +0.10. The verdict is
  the first row of the "Verdict mapping" table that holds, and every outcome maps to one row. A
  known-answer gate that no worker may waive must pass before any screen score is read.
benchmark: >-
  `M1`, the stored stage 3 forecasts of experiment 713 for seeds 0 to 2 on the 8 folds, re-run on the
  same injected worlds and on the real record; W0 as the matched control; LP as the proposal's
  contract as written. The climatology is a context row.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels) in points of capacity, unscaled and
  after equal width scaling, on leads of 24 h up to but not including 240 h after the 12:00 UTC
  contract origin, by the state of the shift at the origin, with the normalised mean absolute error
  of the median beside it.
---

**We ask whether a demand forecaster should carry a recent load shift forward with a survival
curve.** A load shift is a period in which the electricity network operator moves part of one
primary substation's demand (a Primary) onto, or off, a neighbouring Primary. The forecaster is the
axial-attention neural network of experiments 713 to 733, which reads the last 14 days of a
Primary's power, weather, and calendar as a matrix of days by half-hours. We freeze that neural
network and train a small head that estimates the size of the shift in megawatts (the level term)
from the 14 days. We compare four ways to carry the level term over the next 10 days: not at all, in
full, decayed by one survival curve fitted to shift durations, and as a mixture whose chance of
"the shift continues" is a survival curve, either fitted or learned. A survival curve gives the
chance that a shift of a given age is still on after a given number of days. We score each way on
real Primaries with load shifts of known start, depth, and end added to their power, and on the real
record with nothing added. **The readings are on injected shifts from a generator that matches the
training family, so they say what the structure can do, not what it does on real shifts.** This
page is the plan, committed before any code runs. No result exists yet.

## Why the experiment runs

**An assessment argues that "carry the shift across the whole horizon" is the wrong design
contract, and that the contract should read "carry each shift forward with a survival curve".** The
argument rests on counts from the earlier line of experiments on the Candidate neural network K1. Of
the 66 hand-filtered real transfers on the 16 earlier series, 31 lasted under a day and 17 lasted 1
to 3 days, so 48 of 66 ended within 3 days
([experiment 101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md)). On
the injected shifts of [experiment
652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md), a forecast
that carried the shift forward scored 5.8 points of capacity where K1 scored 9.9, and the same
forecasts scored 3.5 (a learned survival output) and 4.6 (a fitted curve) points worse than K1 in
the last 24 hours before a shift ended. No experiment has put a level term and a survival curve on
the axial-attention forecaster, which follows shifts more than the Candidate does ([experiment
726](../00726-msa-candidate-follow-share-on-injected-shift-bank/result.md): +0.225 of the depth, 7
to 28 days after the onset, at the six signed-MW Primaries).

**The experiment ports no change to production, and the result settles a design question for the
forecast product.** A result in which the survival structure adds nothing over a training regime
that shows shifted targets would say that the survival head is not the part that matters.

## Facts about the earlier experiments that fix the design

**Four facts about the earlier pages change what the experiment can measure.** Each was checked on
the page and code named.

- **The shift bank of experiment 726 cannot score CRPS where a shift is in force.** The bank is
  experiment 581's. A shift of 3, 7, 15, or 30 days enters the power inputs only, and "the targets
  are never shifted" ([experiment
  726](../00726-msa-candidate-follow-share-on-injected-shift-bank/result.md)). This experiment draws
  shifts with the generator of [experiment
  650](../00650-extended-shift-departure-head-against-decayed-recent-error/result.md), which adds
  the shift to the inputs and to the target, with random depth, shape, onset, and end. The strata of
  experiments 652 and 654 are defined on that generator. The generator's mirrored shifts at
  neighbouring Primaries are not used, because the forecaster reads only its own Primary's power.
- **The stored `M1` was trained never to follow a shift in its target.** Stage 3 of experiment 713
  adds experiment 581's lag-only shifts to the history power and never to a target. Any gain of a
  level-term arm over `M1` therefore mixes two changes, the training worlds and the survival
  structure, so the plan adds the control W0 and reads the changes as separate contrasts.
- **B-haz and R-haz of experiment 652 differ in two ways.** B-haz is K1's quantiles plus the recent
  residual times a survival ratio, a decay of the whole forecast. R-haz is a mixture of K1 and K1
  shifted by a depth, weighted by the survival probability. The arm LM, a mixture with the fixed
  fitted curve, makes the two differences separable.
- **The forecaster scores target rows 2 to 10 after the 12:00 UTC contract origin, which are leads
  of 24 h up to but not including 240 h.** These are lead days 1 to 9 of experiments 652 and 654.
  The 16 Primaries of the forecaster are the same 16 as the neighbour lists of experiment 650's
  `world()`. Six report signed megawatts (D5, D9, D10, D11, D14, D16), eight report only
  megavolt-amperes (D2, D3, D4, D8, D12, D19, D21, D22), and D15 and D18 are report-only. The 14
  Primaries of the two groups are the scored Primaries on this page.

**"Scaled" means the equal width scaling of experiments 652 and 654.** One point of capacity is 1% of
a Primary's capacity. Each arm's quantile spread about its median is multiplied by one factor per
arm, fold month, and seed, chosen so that the q05 to q95 band covers 0.90 of that month's
training-family scoring rows pooled across strata. The scaling is a calibration fitted to the scored
month's actuals, so the scaled figure is report-only. The unscaled figure governs every reading,
and the printed report gives the fitted factors by arm.

## Design

### The arms

**Every level-term arm keeps the stored stage 3 state of `M1` frozen as its base forecast, and trains
only a small head on top of it, as experiment 652 trained R-haz on a frozen K1.** The base forecast
is `M1`'s own forecast, so with the level term at zero an arm equals `M1`, and the harm on rows with
no shift comes only from the head. The base is not blind to shifts, because `M1` follows 0.14 or
more of an injected step through its history rows (experiment 713, known-answer check KA3), and
training taught `M1` to follow little. The head therefore learns the depth that remains after the
base's own following.

| Arm | What the arm is | Fitted or stored | Seeds |
|---|---|---|---|
| `M1` | The stored forecaster of experiment 713, run on the new worlds from the stored states | Stored states | 0 to 2 |
| W0 | `M1`'s architecture, fine-tuned in stages 2 and 3 on the new worlds (shift in history and target), with no level term and no frozen base | Fitted | 0 to 2 |
| LP | Frozen `M1` plus a level term added to the mixture means at every lead (full persistence) | Head only | 0 to 2 |
| LD | Frozen `M1` plus the level term times the survival ratio, so that every quantile moves together (a decay, B-haz style) | Head only | 0 to 2 |
| LM | As LD, but a mixture: the shifted branch has weight p, the other branch weight 1 - p | Head only | 0 to 2 |
| LR | As LM, with p read from learned daily hazards (R-haz style) | Head only | 0 to 2 |

**C3 compares a frozen base plus a survival head against a neural network that can follow shifts
itself.** W0 reads the shifted history rows through all its layers and learns whatever it needs,
including the persistence of a shift older than the history. LD, LM, and LR change only a head. A
C3 reading near zero therefore says that the head matches a fine-tuned forecaster, and a positive C3
says that fine-tuning the whole forecaster is better, which is not evidence against survival. C3 is
read together with C2, which compares two heads on the same frozen base and isolates survival from
full persistence. LM and LR also have six mixture components against three, and the page names
that difference.

**The survival ratio is one formula, in days.** With `a` the age of the shift in days at the origin
(the age rule below) and `h` the lead in days, p(a, h) = S(a + h) / S(a), where S is the survival
function of shift durations. LD multiplies the depth by p(a, h). LM uses p(a, h) as the weight of the
shifted branch. LR replaces S with the product of daily survival terms from 10 non-negative hazards,
as in experiment 652's head, and gives the same floor as the fitted curve. A unit test asserts that
LD, LM, and LR agree when the hazards are set to reproduce the fitted curve.

**The seed count is 3, so the seed rule of the `study` skill has 3 seeds to check.** The in-force
seed noise is not measured yet. Phase 0 prints the seed spread of one screen fold, and the cut list
and the tie margin of 0.10 read that figure.

### The level term and its inputs

**The level head reads only the 14 history rows, the target row's weather and calendar, and the fold's
training window.** The newest power reading is stamped 24 hours before the native origin (09:00
UTC), and a history row of age `a` covers the 24 hours that start at 12:00 UTC `a` days before the
contract origin. No value after that limit enters a feature. The head's input is the frozen `M1`'s
pooled summary of the target row, the read-out's weighted mean and spread of similar days, the
one-hot lead day, and the 14 row means of the standardised history power (zero and a presence flag
where a row has no visible half-hour). The head has two hidden layers of 64 units. It outputs a
signed depth in standardised units, limited by a tanh to 0.7 of capacity divided by the Primary's
target standard deviation, which covers experiment 650's held-out depth of 0.65.

**The age rule A1 gives the age of a shift from the history rows, and its accuracy is measured
before any fit.** The reference level is the mean of the oldest 7 history rows. A1 counts the newest
consecutive rows whose mean differs from the reference by more than 2 noise units, where one noise
unit is 1.4826 times the median absolute deviation of the daily differences of row means over the
fold's training window. The age is the count plus 2, because the newest usable row is 2 days old. A
count of 0 means that no shift is seen, the level term is zero, and the arm equals `M1`.

**A shift older than the 14 history rows is invisible to A1, and every level-term arm reduces to `M1`
on such rows.** Such a shift fills the history, so the reference level is shifted as well, the count
is 0, and no level term is added. In the scoring worlds about 33% of the in-force origins are older
than 15 days (my simulation of experiment 650's generator), and the rows of a long shift are most of
the in-force time. The base `M1` can still follow such a shift through its history rows, and W0 can
follow it more. The page therefore reads C1 to C3 by age band (under 10 days, over 10 days, and
over 15 days) beside the pooled reading, so that a reader can see where the survival structure acts.

**Stage 0 prints an accuracy table for A1 before any fit.** For each band of the true age (0 to 2, 3
to 5, 6 to 10, 11 to 15, and over 15 days) the table gives the share of in-force origins on which A1
returns 0, the mean absolute age error in days, and A1's false-alarm share (the share of origins
where A1 sees a shift that is not in force) on injected event-free rows and on real event-free rows.
A1 is tuned only on the training worlds' true ages, never on a scoring world or a forecast score. The
bar is a false-alarm share of at most 0.05 on real event-free rows. A1 that misses the bar is
revised before the review marker is written.

**Nothing in the level head can read the forecast period.** A unit test perturbs every power value
after the visibility limit, every shift that starts after the origin, the noise unit, and the curve
parameters, and asserts that the features, the depth, and the hazards of every arm are bit-identical.
The test also covers the 3 hours of the age-2 row that lie after the 09:00 limit. The target row's
weather is the 00Z ECMWF ensemble run of the origin day, as in experiment 713, and
`e00713_weather_lookahead.py` prints a row to show it. The history rows use ERA5 and CAMS analyses,
which a live service could not build for the 2 to 5 days before the origin, so every reading is a
ceiling for a live forecast.

**Pre-fit diagnostics test that the depth head has not collapsed to zero.** On the screen fits the
report prints the head's median depth divided by the true depth by age band (3 to 5, 6 to 10, and 11
to 15 days). A median ratio under 0.7 on shifts of 3 to 10 days voids the C2 and C3 readings and goes
to the gate below. The head is trained on the likelihood alone, with no auxiliary loss on the true
depth. An auxiliary loss would make the head's task easier than the task of a production head that
has no true depth, and the frozen base removes the usual reason for a collapse (the base cannot
explain a shift that the head leaves out of the level).

### The training worlds

**The new arms train on worlds where an injected shift sits in the history rows and in the target.**
A world adds, for each Primary, a sequence of shifts to the cleaned power over the stage window. The
generator copies experiment 650's draw: depth log-uniform from 3 to 50 points of capacity, a random
sign, a constant or proportional shape (constant only at the MVA-only Primaries), a step or a ramp of
6 to 48 hours, and a duration of 3 to 14 days with probability 0.5 and 28 to 112 days otherwise. A
renewal process spaces the shifts with gaps drawn from an exponential distribution with a mean of
118 days, which puts about 25% of training origin days inside a shift (39.25 days of mean shift
length divided by 39.25 + 118). The real share is 0.025 of rows inside logged events, so the
training prior is high and the page names that. Each training pass draws a new world from the seed of
the fit and the pass, and the seed space is disjoint from the scoring seeds (a test asserts it). A
fixed validation world, drawn once per fold, serves early stopping.

**The duration mixture is an assumption, and the page says so.** The only measured durations we
found are the 66 hand-filtered real transfers, of which 31 lasted a day or less, shorter than any
injected shift. The generator keeps experiment 652's mixture so that this result can be compared
with 652. Two robustness worlds are report-only: experiment 650's held-out family (durations of 5,
10, 21, and 45 days, depths up to 0.65, ramps of 72 to 120 hours), and a world whose durations are
resampled from the 66 real transfers, sub-day durations included.

**The survival curve of LD and LM is the empirical survival function of the training worlds' drawn
durations, and nothing else.** The curve takes every shift's drawn duration, clipped or not, because
the generator knows each drawn duration and excluding clipped shifts would bias the curve towards
short shifts. The curve is rebuilt at each monthly refit from that fold's training worlds, which lie
in the training window only. The true survival function is flat between 14 and 28 days, which an
exponential, Weibull, log-normal, or log-logistic distribution cannot reproduce (Amendment 2 gives
the error). The best of those four families, chosen by likelihood, is still fitted and recorded
beside the empirical curve, for the report only, and no arm reads it. A pre-fit criterion bounds
the empirical curve: the largest absolute error against the generator's true S at 1, 3, 7, 14, 28,
and 56 days must be under 0.05, or Phase 0 stops. The 66 real transfers do not enter the
counted curve. Their dates fall inside the scored months, and fewer than 66 would remain before
each fold's cutoff. Half of them last under a day, a range the generator never draws. The counted
worlds' own durations come from the generator.

### The scoring worlds and the strata

**The scoring worlds hold one shift per Primary per pass, drawn by experiment 650's generator on
the 8 fold months, with 12 training-family passes and 4 held-out passes per month.** The onset falls
between 10 days before the month starts and day 20 of the month. A scoring shift lies in the scored
month and the 16 days before it, and the fold's training window ends before the month starts. The
seed space of the scoring worlds is disjoint from the training seeds. The forecaster reads the
injected panel as its history power, and the actuals are the injected power.

**The strata are the state of the shift at the forecast origin, each a half-open interval, as in
experiments 652 and 654.** A row belongs to the stratum of its origin, except for the event-free
rule, which also looks at the valid time.

- **Far from the end:** a shift is in force and more than 10 days remain.
- **Middle:** a shift is in force and the remaining time is above 1 day and at most 10 days.
- **Near the end:** a shift is in force and the remaining time is above 0 and at most 1 day.
- **Ended in the last 24 hours:** the shift ended at most 24 hours before the origin (an origin at
  the half-hour of the end counts).
- **Event-free:** no shift in force, none ended in the previous 7 days, and none starting before the
  row's valid time.
- **Context strata, report-only:** onset ahead (the shift starts after the origin and before the
  valid time), and ended above 1 and at most 7 days before the origin.

**The expected stratum shares of the scoring worlds come from a simulation of the generator, and the
data smoke check must match them within 3 points of share (1 point for the near and ended strata).**
The rows span leads of 1 to 10 days, and the event-free and onset-ahead shares depend on the lead,
because the valid time moves with it. Over origins on days 0 to 29 of a month, pooled over leads of
1 to 10 days, the simulated shares are: far 0.38, middle 0.10, near 0.015, ended in the last 24
hours 0.015, ended 1 to 7 days before 0.09, event-free 0.29, and onset ahead 0.10. With every row
at a lead of 10 days, event-free is 0.225 and onset ahead 0.17. In both cases 0.50 of origins are in
force and 0.33 of the in-force origins are older than 15 days. The clipping of shifts at the window
edge moves the shares by under 1 point. "In force" pools far, middle, and near, and the strata
are built from the shift as the generator drew it (its drawn end and drawn depth). No in-force row is
dropped because the window would have cut the shift short, because that would drop rows on
information that lies after the forecast. Every scored valid time lies before 2026-07-01, so the
drawn end is exact for every scored row. An origin at the half-hour of the shift's end counts as
ended in the
last 24 hours. The scaling factor pools all strata of a month, so
the factor cannot absorb a stratum. The event-free stratum can hold a history with a shift that
ended 8 to 15 days before the origin, and the page reports that share.

### Contrasts, thresholds, and gates

**Three counted contrasts test the hypothesis.** Every contrast is the first arm minus the second, in
unscaled CRPS, averaged over the rows of a month and then over months, so a negative number favours
the first arm.

| Id | Contrast | Question | Threshold |
|---|---|---|---|
| C1 | S* minus `M1`, in force | Does the bundle of a survival-carried level term and its training worlds beat the stored forecaster where a shift is in force? | Net N1 at or below -0.05 |
| C2 | S* minus LP, in force | Does survival beat full persistence, the proposal's contract as written? | Interval below zero |
| C3 | S* minus W0, in force | Does the survival head match a forecaster fine-tuned on the same worlds? | Net N3 at or below -0.03 |

**The porting threshold is on the net unscaled effect over all rows.** With q the share of real rows
in a shift, N1 = q C1 + (1 - q) V1 and N3 = q C3 + (1 - q) V3, where V1 is S* minus `M1` on
injected event-free rows and V3 is S* minus W0 on the same rows. Porting the bundle adds a head
(0.03 in the protocol's tier table) and a fitted step (0.02), so N1 must be at or below -0.05. Porting
the survival head over a training world that the forecaster already uses adds a head, so N3 must be
at or below -0.03. The in-force share q is 0.05: logged events cover 0.025 of the 5.9 million rows
of experiment 713's protocol file (all arms pooled), 0.19 at the Primary with the most logged events
(D5), and one final-test window of experiment 654 had 0.14 of its rows with the end detector's gate
flag on, a figure that includes false alarms. NGED's log misses events, so q is twice the logged
share. The printed report gives N1 and N3 at q of 0.025, 0.05, and 0.14, which are report-only apart
from 0.05.

**One gate and one check guard the rows where no shift is in force.** V1 reads injected event-free
origins and fails when its point estimate is above +0.02, the protocol's tolerance for a group
headline. V2 is the real record, every row with no mask, pooled over the 14 scored Primaries, with
D15 and D18 beside them. V2 is a report-only harm check, flagged when its point estimate is above
+0.10. V2 cannot be a gate, because the monthly differences between two seeds of the same forecaster
`M1` have a standard deviation of 0.18 to 0.33 over the 8 months (0.21 to 0.38 without D15 and
D18, computed from the stored forecasts of experiment 713), so the half-width of a 95% t interval on
8 months is 0.15 to 0.32, and an upper limit at or below +0.10 cannot be reached. The report prints
the share of real event-free rows on which S*'s median departs from `M1`'s by more than 0.5 points
of capacity, because a gain after an ending can be offset by departures elsewhere (experiment 663).

**Report-only readings.** They are W0 minus `M1` (the training worlds alone), LR minus LM (a learned
curve against a fitted curve, on the training family and on the held-out durations), LM minus LD (a
mixture against a decay), scaled in-force CRPS, the readings at each q, the per-group and
per-Primary versions of C1 to C3 with D15 and D18 beside the groups, and the contrasts of LD and LR
beside S*. S* is chosen on 4 of the 8 folds that the counted readings also use. The selection bias is
small, because the three candidates share the structure, and the page prints the reading on the 4
unscreened folds, which must carry the sign of the 8-fold reading.

### Verdict mapping

**The verdict is the first row that holds, fixed before any fit, and every outcome maps to exactly one
row.** "Gain" for a net effect or for C2 means that the upper limits of the 98.33% t interval and of
the 98.33% month-block bootstrap interval are both below zero. "Seeds agree" means that each of the 3
seeds has the sign of the point estimate. "Held-out agrees" means that the point estimates of C2 and
C3 on the held-out-duration world have the sign of the training-family point estimates. A test
enumerates every combination of the 14 boolean states (16,384) and checks that each maps to one
verdict.

| Order | Verdict | Condition |
|---|---|---|
| 1 | `invalid` | A leakage test, a generator oracle, the reproduction of stored `M1`, or the known-answer gate fails and no fix is accepted |
| 2 | `refuted` | The lower limit of the 98.33% t interval of N1 is above 0 (S* is worse than `M1` on the net effect) |
| 3 | `not supported` | V1 fails; or the 98.33% t interval of N1 lies inside -0.05 to +0.05 |
| 4 | `inconclusive` | A gain in N1, N3, or C2 has an interval below zero but the seeds do not agree |
| 5 | `supported` | N1 gain at or below -0.05, N3 gain at or below -0.03 with the 98.33% t interval of in-force C3 (unscaled CRPS, S* minus W0) below zero, C2 gain, V1 passes, seeds agree, and held-out agrees |
| 6 | `partly supported` | N1 gain at or below -0.05, V1 passes, seeds agree, and N3 does not pass (a gain at -0.03 with C3 below zero), or C2 is not a gain, or held-out disagrees (the reading is then labelled "duration-dependent") |
| 7 | `small gain` | N1 gain with a point estimate above -0.05, V1 passes, and seeds agree |
| 8 | `inconclusive` | Anything else |

**A `partly supported` verdict because N3 is not a gain means that the training worlds, and not the
survival head, would carry the gain over `M1`.** A V2 flag is printed in the first sentence of the
verdict, because V2 is not a gate. A result in which the bundle's net effect is zero because V1
rises is read as a harm of the training worlds on rows with no shift.

## Standing practices

**The plan follows these practices and departs from none.**

- **Refit monthly.** Each of the 8 folds (2025-11 to 2026-06) has one refit with its cutoff at
  00:00 UTC on the first day of the month and no refit inside the month.
- **Cleaned data everywhere.** Training rows, history rows, and actuals drop the half-hours that the
  `substation_zero` rule flags, and the page reports the count per fold. Injected shifts are added
  after the drop.
- **The noisy start cut.** Every D15 and D18 row before 2025-03-01 is dropped from training, history
  rows, and actuals. D15 and D18 are report-only.
- **Saved fits.** Every fitted state is saved under this experiment's data folder, with its loss
  history, and never committed.
- **Stored forecasts first.** `M1`'s stored forecasts on the real record are the control. The run of
  the stored `M1` states on the new worlds is a new forecast, not a stored one. A check re-runs the
  stored `M1` on the real record and reproduces experiment 713's stored forecasts to 1e-5.
- **Screen first.** 4 folds (2025-11, 2026-01, 2026-03, 2026-06), seed 0, 5 arms, with the choice of
  S* made from the screen alone and every screen number labelled interim.
- **Tails and cost.** Coverage and pinball loss at q01, q05, q95, and q99, the cold winter nights,
  and the avoidable-curtailment cost, through experiment 713's scorer, beside CRPS.
- **Forecast plots.** A day-ahead forecast and the full 14-day forecast of one origin, each against
  what happened, in a winter and a summer month.
- **Four load-shift scenarios.** All rows; rows outside events and 7 days or more after one; rows
  inside events; and the 7 days after an event ends, for each arm, on the
  injected worlds (events known exactly) and on the real record (NGED's logged events, the
  `event_rows`, `outside_events`, and `event_following_7_days` flags of experiment 713's protocol
  rows). One example time series of each scenario is drawn.
- **At least 6 load-shift time series plots,** across both metering groups, marking each event's
  start and end and showing each arm's forecast from 14 days before to at least 7 days after: 3
  injected worlds and 3 real logged events, at each group, none at D8. The page states the
  unlogged-event caveat: NGED's logs may miss events, so the real event-free rows may hold events no
  label covers, the harm table is optimistic, and the real-event slices undercount.
- **Loss plots.** The training and validation loss against the epoch for each arm, one fold and one
  seed, with the stored epoch marked.
- **Origin-time inputs.** The level features, the depth, and the hazards read nothing after the
  visibility limit, by the perturbation test above.
- **Streaming scoring.** The scorer reads one fold and one seed at a time. A forecast file holds
  16 Primaries by about 30 origins by 672 half-hours (322,560 rows) by 11 quantiles, about 15 MB, so
  one fold and seed of 7 arms and 16 world passes peaks near 2 GB. The cap is 16 GB, and the
  measured peak goes into the page.
- **Time log and stop line.** A per-phase log of each job's device, start, and end, summed into
  GPU-process-hours after each phase, with a stop at 7.0 GPU-process-hours and at 12 CPU-hours.
- **Review gates.** No stage runs before the review marker of the stage before it exists, and every
  edited reviewed script gets a new hash and a re-check limited to the diff.
- **Mid-day onsets and an oracle.** The generator's tests place shifts at mid-day onsets, and an
  oracle that adds the true shift to the median returns a follow share of 1 through the scorer.
- **The synthetic world gallery.** At least 8 plots of the generator's output, committed before the
  first fit (see "The gallery").
- **Anonymisation.** The experiment uses no metered generator's series. The real series of D8 is
  never drawn in a published plot, and every published number is an aggregate. D8 stays in the
  pooled scores, and the page reports the eight MVA-only Primaries with and without D8.

**The plan follows these items of earlier advice.**

- Fit a survival curve to durations with no neural network, as the baseline for a learned survival
  output ([experiment
  652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md); LD and LM).
- Score by the state of the shift at the origin and by lead day, and read the far and near strata
  together (652).
- Do not fit equal width scaling per stratum, and report unscaled CRPS beside it (652).
- Report survival calibration on durations the training mixture did not contain (652; the held-out
  world).
- Score a prevailing forecast on the real record before judging a gain on injected shifts, and
  report the share of event-free rows on which the forecast departs from the normal forecast
  ([experiment 663](../00663-end-detector-for-the-prevailing-forecast-followup-to-654/result.md);
  V2 and its departure share).
- Judge a curriculum arm against a placebo that receives the same training rows ([experiment
  709](../00709-era-signal-in-the-candidate-part-1/result.md); W0).
- Read the MVA-only Primaries per Primary before pooling, and score D15 and D18 beside the groups
  ([experiment 713](../00713-msa-axial-attention-demand-network-part-1/result.md)).

**The plan departs from these items, with the reason.**

- Experiment 650 advises comparing a departure head with R0 at 168 hours. R0 needs the forecaster's
  own forecast of each history day, which the forecaster does not produce, so no R0 arm runs. LD is
  the no-fit comparator.
- Experiments 713 and 715 advise a longer pass budget for known-answer fits. The real fits keep
  experiment 713's cap of 60 passes with a patience of 8. The known-answer gate has one
  pre-registered remedy, below.

## The known-answer gate

**The gate runs after the screen fits and before any screen score is read.** The gate needs the
trained LP and W0 fits of the 4 screen folds at seed 0, so the order is smoke test, screen fits, gate,
screen scoring, full fits. Three checks must pass. An oracle that adds the true shift times the true
survival to `M1`'s median forecast must return a follow share of at least 0.95 through the scorer, at
a mid-day onset as well as at midnight. LP must add at least 0.75 of the depth that `M1` leaves
unfollowed (Amendment 2 gives the derivation) of a shift that is in force and far from its end, to
the median at lead days 1 to 2, beyond the median of `M1`, on the rows where A1 sees the shift (A1's
age is above zero). The record also holds the same share over all far
rows, and the share of far rows that A1 sees, as diagnostics. The depth head's median depth divided
by the true depth must be at least 0.7 of the same remaining depth on shifts of 3 to 10 days.
Checks 2 and 3 read only the rows
where A1 sees the shift, so a head can pass while A1 misses many shifts. The Phase 0 table of A1
reports the share of shifts that A1 misses by age band, and the gate record holds the share of far
rows that A1 sees.

**One remedy is pre-registered, and it runs once before the maintainer is called.** Experiment 713's
equivalent gate failed four times at a cap of 60 passes and a patience of 8, and passed in runs
without early stopping. The remedy is a cap of 120 passes with a patience of 15 for the head fits,
which applies to every head arm alike. A first failure therefore refits all 16 screen head fits (LP,
LD, LM, and LR on 4 folds) under the remedy and predicts them again, so that S* is chosen, and C2 is
compared, between fits that share one recipe. The 4 screen fits of W0 are not refitted, because the
remedy does not apply to W0. A second failure stops the stage and goes to the maintainer. No worker
waives the gate. The gate and the remedy fits are in the cost table.

## Amendment 1 before any fit

**Amendment 1 (2026-11-28) changes the second check of the gate, the remedy, and the member-path
check, after a code review and before any fit.** No fit, prediction, or score had run. The changes
below are already written into the sections above, and this section gives the reason for each.

**Gate check 2 judges only the rows where A1 sees the shift.** A1 reads a history of 14 daily rows,
so it cannot see a shift that began more than about 15 days before the origin. The head then reads
age zero, and LP adds exactly zero. In a simulation of the scoring generator (the training family, 8
months, 12 passes, 16 Primaries, lead days 1 and 2), 41% of the far-stratum rows lie on origins where
the shift is older than the 15-day history. Even a perfect head with a perfect A1 then reaches a
share of 0.59 to 0.62 over all far rows (the figure depends on the age at which A1 stops seeing),
below the bar of 0.75, and the remedy cannot change that. The threshold stays at 0.75. The check now
uses the far rows with A1's age above zero, and the share over all far rows is a diagnostic beside
it.

**A1's age may overstate the true age, and the report says so.** A1 returns the count n of differing
history rows plus 2, while the onset may lie between n and n + 1 days before the origin, so A1's age
may be 1 to 2 days too high. The age sets every survival ratio. The Phase 0 table of A1 now reports
the mean signed error of A1's age against the true age, by band and over all seen rows. A mean error
beyond 1 day in either direction is recorded in the gate file, beside the readings that rest on the
survival ratio. The definition of A1 is not changed.

**The remedy refits every head arm.** The screen fits of LD, LM, and LR would otherwise keep a cap of
60 passes while the LP fits and the 80 later head fits use a cap of 120, and S* would be chosen on a
different recipe from the one that produces most of its counted cells. The remedy now refits all 16
screen head fits. The cost table carries the extra fits.

**The member-path check gets a different construction and tolerances from simulation.** The first
construction gave path k the k-th lowest base level and the k-th longest end time, so the shift was
on only in the low half of the base distribution. For a Gaussian base forecast it returned a
quantile gap of 1.7 points against a tolerance of 0.5. The scorer's CRPS also weights the tails, so
its value sits about 22% below the fair CRPS that the check compared it with. The construction
above removes the coupling, and the CRPS comparison now uses the scorer's formula on both sides.

**The tolerances come from known-answer cases.** The base forecast is a normal distribution with a
standard deviation of 1 to 5 points of capacity, the shifted branch is the same distribution moved by
a depth of 3 to 20 points (either sign), and the stored forecast is the exact mixture quantiles at the
survival ratio of a Weibull curve (shape 0.7 to 1.5, scale 5 to 40 days) at an A1 age of 2 to 14 days
and a lead day of 1 to 9. Over 400 random cases of this kind, a right construction gives a mean gap of
0.005 in the ended share (largest 0.010), 0.32 points in the quantiles (95th percentile 0.58, largest
0.67), and 0.050 points in the CRPS (95th percentile 0.105, largest 0.224). The check reports a mean
over thousands of rows, so each tolerance sits above the mean of its gap: 0.02 for the ended share,
0.5 points for the quantiles, and 0.10 points for the CRPS. Two faults show that the tolerances
detect an error. A base level tied to the end time (the first construction) gave a mean quantile gap
of 1.74 points and a mean CRPS gap of 0.42 points. A stored forecast that follows a Weibull curve
with a scale 30% too short or too long gave a mean ended-share gap of 0.094 or 0.058, and exceeded the
ended-share tolerance in 99% or 91% of the cases. `members.tolerance_study` repeats the simulation.

**The bootstrap keeps every seed.** The first plan drew one seed in each resample, so the
bootstrap interval held seed noise that the t interval does not. The seed points of a reading were
also ratios of pooled sums, which differ from the mean of monthly means that the point estimate
uses when the months hold unequal row counts. Both now use the mean of the monthly means.

## Amendment 2 before any fit

**Amendment 2 (2026-11-28) changes A1, the survival curve, the level head of LP, the holdout, the
strata, the gate bars, the choice of S*, the verdict, and the run guards, after a second code review
and before any fit.** No fit, prediction, or score had run, and no real forecast number had been
seen. The sections above already describe the new design, and this section gives the reason for each
change and records each finding that was not taken.

**A1 has a tuned threshold, and Phase 0 stops if the tuning fails.** The code review measured that,
on real rows of the fold 2025-11's training window, the 2-noise-unit threshold flags a shift on
0.198 of the rows, four times the bar of 0.05. Phase 0 now tunes the threshold, over the grid 2.0 to
12.0 in steps of 0.25 noise units, to the smallest value whose share of flagged real rows is at most
0.05. The rows come from the earliest fold's training window, which lies inside the training window
of every fold, and never from a scored month. The rows include the 2.5% that sit in logged events,
which overstates the false alarms slightly, so the tuned threshold errs high. The table prints the
miss rate by true age band, and Phase 0 stops with a message to the coordinator if no threshold
meets the bar, or if A1 at the tuned threshold misses more than 0.6 of the shifts with a true age of
3 to 10 days. The tuned threshold is written to `a1/threshold.json`, every head fit reads it, and a
real stage that finds no file refuses to run.

**LD and LM use the empirical survival function, and the Weibull is report-only.** The best of the
four families was a Weibull with shape 1.03 and scale 40.6 days. Its error against the generator's
survival function is 0.216 at 14 days and 0.081 at 56 days, against the bar of 0.05, because the
generator's function is flat between 14 and 28 days. The fitted survival ratio was then nearly
constant across ages, so LD and LM would not have timed the end of a shift. The empirical function
of about 16,000 drawn durations per fold is within sampling error of the generator's.

**LP is full persistence.** The depth of LP read the lead-day one-hot, the base forecast's mean and
spread, and the pooled target summary, so its level term could change with the lead. LP now reads
only the 14 row means, the 14 presence flags, and the Primary, and a unit test asserts that its
level term is identical at every lead and for different target weather.

**No forecast, score, or injected world uses a valid time on or after 2026-07-01.** Experiment 713's
own pass filters those rows, and the forecast pass here did not, so the 2026-06 fold would have read
July actuals. The forecast pass now drops them, the scorer asserts that none is left, and the member
check leaves out the blocks that the cut shortens. Phase 0 writes every fold's scoring shifts to
disk, reads the files back, and checks that no shift ends after the holdout and that the filter
drops rows of the 2026-06 fold and of no other fold, which needs no fit.

**The strata use the drawn end.** Dropping in-force rows because the window clips the shift removed
0, 0, 0, 0, 0.01, 0.17, 0.52, and 0.70 of the in-force rows of the folds 2025-11 to 2026-06 in a
simulation, and the rows kept in 2026-06 were 0.81 short shifts. The strata now use the drawn end
and the drawn shift, and the stratum-share check reads the injected worlds that were written to
disk, not a simulation.

**The bars of gate checks 2 and 3 follow `M1`'s own follow share.** The gate record holds `M1`'s own
share f of the injected depth on the same rows. A perfect head adds the remaining depth 1 - f
(floored at 0.2), and the bars are 0.75 and 0.7 of that remaining depth. The gate prints the
derivation. With f = 0, the bars are the plan's 0.75 and 0.7, and the first run shows whether a
larger f is real.

**S* is chosen by unscaled CRPS, and N3 needs an in-force C3 below zero.** The scale factor is
fitted on the rows it scores and can hide a loss in the unscaled score, so the scaled figure is
report-only. N3 passes only if its net effect is at most -0.03, both intervals are below zero, and
the 98.33% t interval of the in-force unscaled C3 lies below zero. This adds one boolean to the
verdict table, which now has 16,384 states.

**The artefacts carry the code that wrote them.** Every state file and forecast file has a recipe
file that holds the hash of the fitting modules (the state) or the fitting and prediction modules
(the forecast). A stage that scores or reuses a file refuses one with another hash, and the
prediction job refuses to reuse or overwrite it. The diagnostics file is written before the forecast
file, so a forecast file implies its diagnostics.

**Smaller guards.** The remedy resumes after a crash: the first attempt moves aside once into
`gate_first_attempt/`, each finished remedy task is skipped, and only a complete remedy makes a
second failure final. GPU waits are printed and are not counted by the stop line. A GPU stage
refuses more than one worker. The smoke hours are copied into the real ledger. The prediction
asserts that the fold's cutoff is the first day of the month.

**New report-only readings.** The report prints the survival curve's error and the fitted family
beside the empirical curve, A1's table and tuned threshold, the gate's depth ratios by age band, the
mean CRPS of every arm in the four scenarios of the injected worlds (all rows, outside events and
the 7 days after, inside events, and the 7 days after an end), and C2 for LD, LM, and LR split into
the median's error and the 90% interval's coverage, which shows whether a mixture gains by spread.

**Not taken.** The scenario table has no climatology row, because no climatology forecast exists for
an injected world. The LM control with a constant survival ratio is not built, because the cost
table has no room for another arm, and the split of C2 into the median's error and the coverage
tests the same question.

## Amendment 3 before any fit

**Amendment 3 (2026-11-28) changes the gate bars, the smoke test, the A1 checks, the recipe link,
and the exit codes, after a re-review of Amendment 2 and before any fit.** No fit, prediction, or
score had run. The re-review listed ten findings (R1 to R10), and all ten hold.

**The gate bars carry the survival discount of a lead-free head (R1).** LP's depth is one number per
history, trained over 14 lead days, so its best value is the remaining depth times the survival
ratio averaged over those leads. From the generator's survival function the mean is 0.74 for shifts
of 3 to 10 days (check 3) and 0.82 for ages 2 to 16 days (check 2). The bars were 0.7 and 0.75 times
the remaining depth, so a correct head sat 6% above the bar of check 3 before any error. The code
now builds the empirical curve of the fold's training worlds, averages the ratio over the 14
training lead days and the ages of each check, and multiplies each bar by the result. The gate
prints the derivation, and a test feeds an ideal lead-free head to the gate and requires a pass.
Check 2 reads all ages from 2 to 16 days, and the far rows are older than that average, so its bar
is the more lenient of the two.

**The smoke test reads the predicted panels (R2).** `smoke-collect` reads the `pass00` panel of
every arm on the first screen fold, subtracts the real record's actuals, and requires the difference
to be exactly zero outside each shift and nonzero on at least 0.9 of the rows inside one (the MVA
floor leaves a few rows unchanged). It asserts that no valid time is on or after 2026-07-01 and
prints the rows per stratum. The Phase 0 world check still tests the shares and the holdout
arithmetic, and the page no longer claims that it tests the panels.

**Every filter at the holdout has a test that fails without it (R3, R8).** The concat-and-filter of
the forecast pass is a function, `frame_of`, and a test fails when its filter line is deleted. The
gate, the injection check, the scorer, and the member check each have such a test. The world check
leaves the rows on or after the holdout out of the shares, and a test shows that its "reaches the
holdout" fault fires on a file whose end was not clipped.

**A1's false-alarm share is checked on all 8 folds (R4).** After tuning, Phase 0 starts one child
process per fold, applies the tuned threshold to the real rows before that fold's cutoff in the
fold's own noise unit, and prints the share per fold and pooled. It stops with exit code 3 if any
fold is above 0.10 or the pooled share is above 0.05. The rows include logged events, so the shares
are slightly high. The limit of 0.6 on the miss rate for shifts of 3 to 10 days is kept as the
coordinator set it. At that limit A1 sees at least 0.4 of those shifts, and the gate reads only the
rows that A1 sees.

**The A1 stops are tested, and the gallery uses the tuned threshold (R5).** The decision to stop or
write `a1/threshold.json` is a function, and tests show that it raises and writes nothing when the
tuning fails or the miss rate is above 0.6.

**A forecast records the state it came from (R6).** The sidecar of a forecast file holds the digest
of the state file of its arm and fold (for `M1`, the stored stage 3 state of experiment 713).
Scoring and prediction refuse a forecast whose state has changed. The figures stage checks the
recipes too. Tests show that `fit_job` and `load_for_prediction` refuse a stale recipe.

**The drawn end of a sub-half-hour shift is one half-hour after the onset (R7).** The strata now use
the same clamp as the injection.

**The page promises no climatology row (R9).** The four-scenario table of the injected worlds has no
climatology row. It pools the (fold, seed) cells by rows, not by the mean of monthly means that the
counted readings use, and the report says so.

**Exit codes and the wait ceiling (R10).** A refused call, such as several workers on a GPU stage,
exits with 64. A crash inside `gate-remedy` exits with 7, so that it is not read as the gate's
second failure (code 5), and the stage can be rerun. A chain that has waited 2 hours in all for
another session's GPU process exits with 75.

## The gallery

**The gallery is committed in a plan update before the first fit.** Each panel shows one component of
the world separately, on a real Primary other than D8: the real cleaned power; one injected shift of
each family (constant step, proportional shape, ramp, short, long, and held-out); the same shift in
the history rows and in the target; the strata marked on one 40-day record; the fitted survival
curve against the generator's true survival; and A1's output on one shift. A smoke check on the data
files asserts that every injected shift is nonzero in the rows where the shift should act, and that
the stratum shares match the simulated shares above.

## Member paths

**The stored and scored forecast is the mixture's 11 quantiles at each lead, and the 51 member paths
are a check on it.** CRPS at one valid time depends only on the marginal distribution at that valid
time, so a mixture whose shifted branch has probability p at that lead scores the same as 51 paths
whose end times are drawn from the same survival curve. A path is a draw of the base level plus the
shift times the survival indicator. For each origin, the 51 end times T are the survival curve's
inverse at S(a) (k - 0.5) / 51 for k = 1 to 51, conditioned on A1's age a. The 51 base levels are
the same evenly spaced levels (j + 0.5) / 51, taken in the order of the stride-19 permutation of
j, so that the base level of a path is not tied to its end time. Path k returns the base forecast's
quantile at its base level, plus the depth while the path's lead is below T. The base forecast's
value at a level is a linear interpolation of its 11 stored quantiles.

**The check asserts three properties on the screen folds, with tolerances from known-answer cases.**
The share of paths that end inside 240 hours is within 0.02 of 1 - p at 240 hours. The 51-path
empirical quantiles at the 11 levels differ from the stored quantiles by at most 0.5 points of
capacity on average. The scorer's 11-level CRPS formula, applied to the paths' own 11 empirical
quantiles, differs from the same formula applied to the stored quantiles by at most 0.10 points. The
energy score and the variogram score of the 51 paths against the actual half-hours of the day, and
the fair CRPS of the paths, are report-only. The check runs on LM only: LR learns its hazards from
the features of each sample, which no stored file holds, and LD is not a mixture. The result says
whether S* is LM.

**The scored CRPS is a tail-weighted CRPS.** It is twice the mean pinball loss at 11 levels, and 4
of the 11 levels lie beyond q05 or q95. For a Gaussian forecast it is about 22% below the fair CRPS
(0.440 and 0.564 standard deviations). All arms use the same formula, so the contrasts are
unaffected, and the check compares the formula with itself.

## Checks

**These checks are in the printed report and the write-up.** They are the median forecast minus the
mean forecast per Primary, the normalised error of the median beside every CRPS contrast (audit
trigger 3), leave-one-Primary-out and leave-one-month-out versions of C1 to C3 including a line
without D5 (which holds 0.19 of the real event rows), the eight MVA-only Primaries with and without
D8, the reproduction of the stored `M1` on the real record (the tolerance is 1e-5 of capacity; if the
first smoke run misses it, check the launcher for a TF32 setting or a batch size that differs from
experiment 713's before touching the tolerance), the A1 accuracy table, the depth
diagnostics, the fitted-curve error, and the fitted scaling factors by arm.

## Cost

**The plan fits 120 heads or networks and projects about 6.3 GPU-process-hours, with a stop line of
7.0, and about 5.5 CPU-hours.** The head arms train a head on a frozen forecaster, which needs a
forward pass and no backward pass through the trunk, so each head fit is projected at 1 minute. W0
repeats stages 2 and 3 at 2.2 minutes, from the stored times of experiment 713 (stage 2 60 seconds
and stage 3 66 seconds on average over its 40 `M1` fits). A pass of 3,360 forecasts took 1.6 seconds
in experiment 726. The frozen-trunk fit time is not measured, and phase 0 times it.

| Phase | Jobs | GPU-process-hours | CPU-hours |
|---|---|---|---|
| 0. Generator, tests, A1 table, gallery, curve fits, pre-fit guards, timing of fits | 12 | 0.3 | 1.5 |
| 1. Real-environment smoke test, one epoch of each arm, scratch output | 6 | 0.2 | 0.2 |
| 2. Screen fits: 16 head fits and 4 W0 fits (4 folds, seed 0), then predict | 24 | 0.8 | 0.3 |
| 3. Gate, and its pre-registered remedy refit of the 16 screen head fits and their prediction | 24 | 0.8 | 0.1 |
| 4. Remaining fits: 80 head fits and 20 W0 fits | 100 | 2.1 | 0.2 |
| 5. Prediction on the worlds and the real record, one job per fold and seed | 20 | 1.8 | 0.2 |
| 6. Scoring, intervals, scenarios, member-path check | 8 | 0 | 2.5 |
| 7. Figures, loss plots, time series plots | 4 | 0.3 | 0.5 |
| **Total** | **about 200** | **about 6.3** | **about 5.5** |

**After the 20 screen fits, the projection is recomputed from the mean and the 90th percentile of the
fit times, not from one fit, and the cut list fires on that figure.** The cuts, in order: drop LD
(its contrast with LM is report-only), then fit seeds 0 and 1 only. No fold is cut from a counted
reading, because the 8 months are the resampled unit. The manager approves any cut before phase 4.
The launcher stops between jobs at the stop line and never inside a fit. A fit writes its state to a
temporary name and renames the file when complete. The ledger counts a crashed attempt. The waits under
contention are printed and booked as waits, and the stop line does not count them. A job treats the
GPU as busy when `nvidia-smi` lists a compute
process of another session, and backs off for 5 minutes. A chain that has waited 2 hours in all
exits with code 75. The run uses one GPU process at a time under
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue`, one thread
for each numerical library, and cores 8 to 15 for the CPU phases.

## Method

**Every value below is fixed before any fit.**

### Architecture

- The base forecaster is experiment 713's neural network, `M1`: width 64, two axial blocks, the
  similar-day read-out, and a three-component Gaussian mixture head, with the frozen weather and
  date-time encoders of the Candidate's pre-training. The level-term arms freeze its stage 3 state.
- The sample is one Primary, one origin day, and one lead day k (1 to 14): 15 rows of 48
  half-hours, the target row and 14 history rows of ages 2 to 15, with the visibility limit
  of 24 hours before the native origin.
- LP, LD, LM, and LR add a level head (two hidden layers of 64 units) that reads the frozen trunk's
  pooled target summary, the read-out's weighted mean and spread, the one-hot lead day, and 14
  history row means, and outputs a signed depth. LR adds 10 daily hazards.
- The shifted branch is a copy of the three mixture components shifted by the depth. LD shifts the
  three means by the depth times p(a, h). LP shifts them by the depth. W0 has no level head and is
  fine-tuned in full.
- The shift generator and the strata are experiment 650's and experiment 654's, copied into this
  experiment's folder with the changes that "The training worlds" and "The scoring worlds" state.

### Training regime

- Training data: 16 Primaries, cleaned power, ERA5 and CAMS weather in stage 1 (stored), ENS 00Z
  weather in stages 2 and 3, with an injected world in every pass of the new fits.
- W0 starts from `M1`'s stored stage 1 state and runs stages 2 and 3 on the worlds with the
  weights, learning rates, AdamW settings, batch of 256, gradient clipping at 1.0, cap of 60 passes,
  patience of 8, and 25.5% block dropout of history power of experiment 713's recipe.
- The head arms run one stage on the worlds with the same optimiser settings, a head learning rate
  of 1e-3, and the same cap and patience, with the 120-pass remedy of the gate if needed. The head
  learning rate is fixed here and not chosen on any score.
- The loss is the mixture negative log-likelihood on the standardised, shifted target, with the
  training worlds' shifts in the history rows and the target.
- Seeds 0 to 2. One refit per month. The GPU hours are 6.3 projected, and the measured figure goes
  into the finished page.

### Measurement

- Scored rows: the 8 monthly folds, daily 12:00 UTC contract origins, leads 24 h up to but not
  including 240 h, valid times before 2026-07-01, under the row set of the front matter. The
  injected worlds use the same origins and valid times.
- Truth: the cleaned power divided by capacity on the real record, and the injected power divided by
  capacity on the worlds.
- Metrics: unscaled and scaled CRPS, the normalised mean absolute error of the median, coverage and
  pinball loss at q01, q05, q95, and q99, and the follow share, by stratum, age band, and lead day.
- Baselines: `M1`, W0, LP, and the month-by-half-hour climatology as a context row.
- Intervals: a t interval on the 8 monthly differences of the seed-averaged forecasts at 98.33% for
  N1, N3, and C2 and at 95% for V1 and V2, and a month-block bootstrap of 100,000 resamples, which
  draws whole months with replacement and keeps all 3 seeds in every resample (the statistic is the
  mean of the drawn months' seed-averaged values, the statistic of the point estimate). A gain needs
  both intervals below zero, because
  8 monthly blocks share overlapping training windows and give fewer than 7 effective degrees of
  freedom.
- The decision rule, the gate, and the verdict mapping are above.

## Files

**The code lives in `ml-experiments/00738-msa-survival-mixture-prevailing-forecast/`.** Each script
carries the id as `e00738_`. The implementation plan names each file, its tests, and the run order.

- `e00738_common.py` holds every constant (arms, folds, thresholds, stop lines) as a named `Final`.
  `e00738_phases.py` holds the review markers, the hour ledgers, and the done markers.
  `e00738_run.py` is the one entry point. `run.sh` starts a stage inside a memory-capped
  `systemd-run` scope, and `run_workers.sh` starts the workers of a CPU stage inside that scope.
  `run.sh` takes `E00738_713_DIR`, the folder of experiment 713 in the frozen checkout (commit
  `7c29114a8`). The whole `ml-experiments` folder of that checkout goes first on `PYTHONPATH`,
  because the stored fits carry the code hash of that checkout, and that hash covers the recipe
  folders beside experiment 713's own. The two recipe files that changed since that commit
  (`e00513_slices.py` and `e00709_era_table.py`, which now need a private engine file) then come
  from the frozen commit. `run.sh` also sets `PYTHONDONTWRITEBYTECODE=1`, so the frozen checkout
  stays clean. A job stops before it can write into experiment 713's cache folder, because a missing
  dense array there would be built and written by experiment 713's pipeline.
- **Child processes.** Experiment 713's pipeline reads its month and phase once, when it is
  imported, so every fit job and every prediction job runs in a child process of its own
  (`e00738_run.py one-task`). The parent process keeps the ledger, the failure records, and the done
  markers. The parent also refuses to start a GPU stage when `nvidia-smi` lists a compute process of
  another session (exit code 75), and waits between jobs.
- **The smoke test and its markers.** The smoke stages write all their output to the scratch
  folder `E00738_OUT`. They read `phase0.done` and the review marker from the data folder, read
  only, and `smoke-collect` writes `smoke.ok` to both folders. That file is the one the smoke test
  writes into the data folder, and `screen-fits` needs it there. The pass cap of one pass reaches
  only the child processes of the smoke stages, and a real stage refuses to start when
  `E00738_PASS_CAP` is set.
- **The cut list.** `cuts.json` in the output folder holds `drop_ld` and `seeds01`. The fit,
  prediction, scoring, report, protocol, and member-path code all read it, so a cut run has no
  missing-file error. After `drop_ld`, LD exists in the 4 screen cells only, every reading that
  involves LD uses those cells, and the full fits refuse a cut list that drops the arm that the
  screen chose as S*.
- **Failure records and markers.** A task that failed once and then finished has its record moved to
  `failed/archive/`, so a rerun can finish the stage. With several workers, the stage marker is
  written by the worker that finds every task of the stage done.
- `e00738_world.py` draws the training worlds and the scoring worlds from experiment 650's
  generator, adds the shift to the inputs and the target, and assigns the strata.
- `e00738_survival.py` fits the four candidate curves to the drawn durations, returns p(a, h), and
  holds A1, the age rule.
- `e00738_model.py` extends experiment 713's neural network with the level head, the three level
  terms, and the mixture of six components.
- `e00738_train.py` and `e00738_fit.py` run the fits on the worlds and store the states and the loss
  histories. `e00738_panel.py` copies the panel-swap code of experiment 726 (the original runs
  `argparse` at import) with a provenance note, and asserts experiment 713's code hash.
- `e00738_predict.py` predicts the worlds and the real record, for the stored `M1` states and the
  new arms, and writes one parquet file per arm, fold, seed, and panel under `predictions/`, with a
  second file of diagnostics (age, depth share, survival ratio) beside it. It does not use
  `experiment_store.write_predictions`, because the gate and the member-path check read the
  diagnostics files.
- `e00738_score.py`, `e00738_report.py`, and `e00738_members.py` compute every contrast, interval,
  gate, scenario, check, and the member-path check, and print every number of the page.
- `e00738_protocol.py` adapts the protocol of experiment 713 to score the real record of every arm
  in the four load-shift scenarios (all rows, outside events, inside events, and the 7 days after an
  event). The `members` stage runs the 51-path check on `LM` and writes `checks/members.json`.
- **Exit codes.** A failed CPU task or stage exits with 3. A failed GPU chain exits with 5, as does
  the known-answer gate's second failure. The gate's first failure exits with 6, and means that the
  remedy flag is set and `gate-remedy` runs next. A crash inside `gate-remedy` exits with 7, and the
  stage is rerun. A refused call exits with 64. A launcher that finds the GPU busy, or a chain that
  has waited 2 hours for it, exits with 75. The plan states only the code 5 and "a different code"
  for the busy GPU.
- `e00738_worldcheck.py` writes every fold's scoring shifts to disk and checks the strata and the
  holdout from the files. `e00738_gate.py` runs the known-answer gate. `e00738_figures.py` and
  `e00738_gallery.py` draw the
  figures and the gallery.
- The tests are `test_e00738_world.py`, `test_e00738_survival.py`, `test_e00738_score.py`,
  `test_e00738_panel.py`, `test_e00738_run.py`, `test_e00738_review_fixes.py`,
  `test_e00738_opus_fixes.py`, `test_e00738_rereview_fixes.py`, `test_e00738_model.py`,
  `test_e00738_attributes.py`, and `test_e00738_dryrun.py`. The last three need `torch` and skip
  without it. The tests cover every pure function, the visibility limit of the level features, the
  keyword names of `run_task`, every state of the verdict table, the guards, and the existence of
  every attribute that the scripts take from an imported module, including lazy imports inside
  functions. The dry run fits every arm, predicts four panels, scores them, and runs the member-path
  check on a tiny synthetic fold, on the CPU, in a scratch folder, with every read of the data
  folder replaced. It also perturbs the power after the origin, and adds a shift that starts after
  the origin, in the real `Source` class, and asserts that the level head's features, age, depth,
  and hazards stay bit-identical. It runs for about 3 minutes. The known-answer tests of the
  member-path check use a Gaussian forecast, and the gate test uses site identifiers that are not in
  the order of the site table.
- Read from experiment 713 (`ml-experiments/00713-msa-axial-attention-demand-network-part-1/`): the
  network, inputs, batch, train, and score modules, and the stored stage 1 and stage 3 states and
  forecasts of `M1`, under
  `data/ml-experiments/00713-msa-axial-attention-demand-network-part-1/work/`.
- Read from experiments 650, 652, and 654: the generator (`e00650_bank.py`), the survival head
  (`e00652_head.py`), and the strata and scaling (`e00652_score.py`, `e00654_score.py`).
- Written, never committed, under
  `data/ml-experiments/00738-msa-survival-mixture-prevailing-forecast/`: `worlds/` (the scoring
  worlds per fold), `networks/` (every fitted state), `fits/` (loss histories, seconds, and peak
  memory), `predictions/` (per-row forecasts), and `scores/`.
