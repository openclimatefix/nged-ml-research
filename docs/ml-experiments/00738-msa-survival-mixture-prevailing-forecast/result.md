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
  Complex. (1) It stores about 120 fitted neural networks (5 new arms, 8 monthly refits, 3 seeds),
  their per-row forecasts on injected worlds and on the real record, and private score tables;
  (2) more than one design defensibly tests the hypothesis (a level term built by a fixed rule or
  learned, a mean-path decay or a mixture, a survival curve fitted or learned, and training on
  injected shifts or on the real record), so the plan fixes one and gives each choice its reason;
  (3) it copies the training loop of experiment 713, the shift generator of experiment 650, and the
  strata and scaling of experiments 652 and 654, whose callers cannot be named without searching;
  (4) it touches no serving path or degradation rule; (5) it changes no contract. Reviews, in order:
  a Sonnet plan review, Sonnet code, a fresh Sonnet code review before the first fit, a
  real-environment smoke test, a Sonnet numbers review, and a prose review. An Opus plan review
  is requested because the design departs from the brief that proposed it.
hypothesis: >-
  The axial-attention forecaster `M1` of experiment 713 forecasts a Primary's demand better, in
  continuous ranked probability score (CRPS), at origins where an injected load shift is in force,
  when a level term is carried forward by a survival curve and the forecast is a mixture of "the
  shift continues" and "the shift has ended", than when it is not. It also forecasts better than a
  level term carried forward in full at every lead (full persistence). The survival structure,
  and not only a training regime that shows the network shifted targets, is what lowers CRPS, and
  the structure does not raise CRPS where no shift is in force.
prediction: >-
  Written before any fit. Every difference is the first arm minus the second, in points of capacity
  after equal width scaling, averaged over lead days 1 to 9 (24 h up to but not including 240 h),
  where a shift is in force (far, middle, and near strata pooled), 14 counted Primaries, 8 monthly
  folds, 3 seeds. W0 (M1's architecture trained on injected shifts in inputs and targets) minus `M1`
  lies between -4.0 and -1.5, because the stored `M1` was trained never to follow a shift in its
  targets. C1 (the survival arm S* minus `M1`) lies between -4.5 and -2.0. C2 (S* minus LP, full
  persistence) lies between -1.2 and -0.2, because half the injected shifts last 3 to 14 days and
  end inside the horizon. C3 (S* minus W0) lies between -0.6 and +0.1, so the likeliest reading is
  that the survival structure adds under the porting threshold on top of the training regime. S* is
  one of the two arms with the fixed fitted curve (LD or LM), and the learned curve LR beats it by
  under 0.3, as in experiment 652. The harm on injected event-free origins, S* minus `M1` in
  unscaled CRPS, lies between -0.05 and +0.05. The harm on the real record, all rows with no mask,
  lies between -0.03 and +0.12, and the likeliest miss is that the real-record veto fails. Maintainer-side
  estimate of the verdicts: `partly supported` about 40% (the regime gains and the survival
  structure does not clear its threshold), `supported` about 15%, `not supported` about 25% (a
  veto fails), `small gain` about 10%, `inconclusive` about 10%, `refuted` under 2%. The prediction
  fails if C1 is above -1.0, if C3 is below -0.6, or if C2 is above 0.
decision_rule: >-
  Fixed before any fit. S*, the survival arm, is the arm of LD, LM, and LR with the lowest
  in-force scaled CRPS on the 4 screen folds at seed 0 (an arm within 0.10 of the best loses to
  the simpler arm, in the order LD, LM, LR). Three counted contrasts, each read as a t interval on
  the 8 monthly differences of the seed-averaged forecasts at 98.33% (Bonferroni across 3, 7 degrees
  of freedom), with the month-block bootstrap beside it: C1 (S* minus `M1`), C2 (S* minus LP), and
  C3 (S* minus W0). Thresholds follow the cost of porting, at an assumed in-force share q of 0.05
  of the real rows (logged events cover 0.025 of the rows, and 0.14 of the rows of one final-test
  window sat in detected shifts). Porting the bundle adds a level head, a survival fit at each
  refit, and a training world to the forecaster, a pipeline step, so it needs the 0.05 points of
  all-rows CRPS that experiment 713 set for replacing the network, which is 0.05 / 0.05 = 1.0
  in-force points (tau1). Porting the survival structure alone on top of the training regime adds a
  head and a curve fit, so it needs the 0.03 points of the protocol's new-head tier, which is
  0.03 / 0.05 = 0.6 in-force points (tau3). Two vetoes, both read on unscaled CRPS, S* minus `M1`:
  V1 on injected event-free origins and V2 on the real record, all rows, no mask. A veto passes
  when the point estimate is at most +0.03 and the upper limit of its 95% t interval is at most
  +0.10, reads "cannot tell" when only the upper limit is above +0.10, and fails when the point
  estimate is above +0.03. The harm of 0.03 on 0.95 of the rows cancels 57% of the gain at tau1, so
  the page prints the net all-rows effect q * (C1) + (1 - q) * (V1) at q of 0.025, 0.05, and 0.14.
  Each counted contrast needs the same sign for each of the 3 seeds. Held-out durations (5, 10,
  21, and 45 days) are a report-only world whose sign must agree for C2 and C3, or the gain is
  reported as duration-dependent. A known-answer gate on injected worlds must pass before any real
  fold is scored. The verdict table follows the order in "Verdict mapping".
benchmark: >-
  `M1`, the stored stage 3 forecasts of experiment 713 for seeds 0 to 2 on the 8 folds, run on the
  same injected worlds and on the real record; W0 (the same network trained on the injected
  worlds, no level term) as the matched control; LP (full persistence) as the proposal's contract
  as written. The Candidate and the climatology are context rows only.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity,
  after equal width scaling and unscaled, on leads of 24 h up to but not including 240 h after the
  12:00 UTC contract origin, by the state of the shift at the origin, with the normalised mean
  absolute error of the median beside it.
---

**We ask whether an axial-attention demand forecaster should carry a recent load shift forward with
a survival curve.** A load shift is a period in which the electricity network operator moves part of
one primary substation's demand (a Primary) onto, or off, a neighbouring Primary. The forecaster is
the one of experiments 713 to 733: a neural network that reads the last 14 days of a Primary's
power, weather, and calendar as a matrix of days by half-hours, with attention along both axes. We
add a level term that estimates the size of the shift from the 14 days. We compare four ways to
carry the term forward over the next 10 days: not at all, in full, decayed by one survival curve
fitted to shift durations, and as a mixture whose chance of "the shift continues" is a survival
curve (fitted or learned). We score each way on real Primaries with load shifts of known start,
depth, and end added to their power, and on the real record with no shift added. This page is the
plan, committed before any code runs. **No result exists yet.**

## Why the experiment runs

**The assessment that proposed it argues that "carry the shift across the whole horizon" is the
wrong design contract, and that the contract should read "carry each shift forward with a survival
curve".** The argument rests on counts from the earlier line of experiments on the Candidate network
K1. Of the 66 hand-filtered real transfers on the 16 earlier series, 48 ended within 3 days
([experiment
101](../00101-ro-nc-detector-on-real-hand-filtered-switching-events/result.md) holds the counts: 31
lasted under a day, and the count of 17 lasting 1 to 3 days is taken from the assessment, not
re-checked here). On the injected shifts of [experiment
652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md), a forecast
that carried the shift forward scored 5.8 points of capacity where K1 scored 9.9, and the same
forecasts scored 3.5 (a learned survival output) and 4.6 (a fitted curve) points worse than K1 in
the last 24 hours before a shift ended. No experiment has put the level term and the survival curve
inside the axial-attention forecaster, which is the forecaster that follows shifts more than the
Candidate ([experiment
726](../00726-msa-candidate-follow-share-on-injected-shift-bank/result.md): +0.225 of the depth, 7
to 28 days after the onset, at the six signed-MW Primaries).

**The experiment would not port a change to production, and it settles a design question for the
product framing.** The forecaster's own scores on shifted data would say whether the product
"prevailing forecast with a survival curve" is worth building on the forecaster. A result that
the survival structure adds nothing over a training regime that shows shifted targets would send the
next experiment to the training regime alone.

## What the brief got wrong, and the design that follows

**Four facts about the earlier pages change the design.** Each was checked on the pages and code
named.

- **The shift bank of experiment 726 cannot score CRPS where a shift is in force.** The bank is
  experiment 581's. A shift of 3, 7, 15, or 30 days enters the power inputs only, and "the targets
  are never shifted" ([experiment
  726](../00726-msa-candidate-follow-share-on-injected-shift-bank/result.md)). Every shift is
  temporary against its own truth, so the bank scores a follow share and has no in-force CRPS and
  no strata. This experiment draws shifts with the generator of [experiment
  650](../00650-extended-shift-departure-head-against-decayed-recent-error/result.md), which adds
  the shift to the lag inputs and to the target, with random depth, shape, onset, and end. The
  strata of experiment 654 are defined on that generator. Experiment 726's code supplies the
  plumbing that swaps an injected panel into the forecaster's history power, and its follow share is
  a report-only reading here.
- **The stored `M1` was trained never to follow a shift in its target.** Stage 3 of experiment 713
  adds experiment 581's lag-only shifts to the history power and never to a target, so the
  fine-tune teaches the network to ignore a shift, and the network still follows 0.14 to 0.4 of
  the depth. Any gain of a level-term arm over `M1` therefore mixes two changes: a training regime
  that shows shifted targets, and the survival structure. The plan adds the matched control W0, the
  same network trained on the new worlds with no level term, and reads the regime and the structure
  as separate contrasts.
- **B-haz is a mean-path decay and R-haz is a mixture, so the brief's "B-haz style" and "R-haz
  style" differ in two ways at once.** In experiment 652, B-haz is K1's quantiles plus the recent
  residual times a survival ratio, so every quantile moves together. R-haz is a mixture of
  K1 and K1 shifted by a depth, with the survival probability as the weight of the shifted
  component. A contrast between them mixes "fitted or learned curve" with "decay or mixture". The
  plan adds the arm LM, a mixture with the fixed fitted curve, which makes the two differences
  separable.
- **The scored leads run from 24 h to under 240 h, and the groups are 6 + 8 counted Primaries.** The
  forecaster scores target rows 2 to 10 after the 12:00 UTC contract origin, which is lead days 1
  to 9 of experiments 652 and 654. The 16 Primaries of the forecaster are the same 16 as the
  neighbour lists of experiment 650's `world()`: six signed-MW Primaries (D5, D9, D10, D11, D14,
  D16), eight MVA-only Primaries (D2, D3, D4, D8, D12, D19, D21, D22), and D15 and D18 as
  report-only. The assessment's "14 history rows" and "14-day horizon" are the history length, not
  a forecast horizon.

**"Scaled point of capacity" means the equal width scaling of experiments 652 and 654, applied to
points of capacity.** One point is 1% of a Primary's capacity. Each arm's quantile spread about its
median is multiplied by one factor per arm, fold month, and seed, chosen so that the q05 to q95 band
covers 0.90 of that month's scored rows pooled across strata. The scaling removes the part of a gain
that a wider band alone buys. Unscaled CRPS is printed beside every scaled figure, and the two
vetoes use unscaled CRPS.

## Design

### The arms

**Every new arm starts from the stored stage 1 state of `M1` for the same fold and seed, and
reruns stages 2 and 3 of experiment 713's recipe on the new worlds.** The stored stage 1 state is
trained on ERA5 and CAMS weather and clean power, so the arms share it and differ only in what
follows.

| Arm | What it is | Fitted or stored | Seeds |
|---|---|---|---|
| `M1` | The axial-attention network of experiment 713, stage 3 as stored (lag-only shifts, clean targets) | Stored forecasts, run on the new worlds from stored states | 0 to 2 |
| W0 | `M1`'s architecture, stages 2 and 3 on the new worlds (shifts in history and target), no level term | Fitted | 0 to 2 |
| LP | W0 plus a level term added to every mixture mean at every lead (full persistence, S = 1) | Fitted | 0 to 2 |
| LD | W0 plus the level term decayed by one survival curve fitted to training durations (mean-path decay, B-haz style) | Fitted | 0 to 2 |
| LM | As LD, but a mixture: the shifted branch carries the chance S(age + lead) / S(age), the other branch does not | Fitted | 0 to 2 |
| LR | As LM, with the chance read from learned daily hazards (R-haz style) | Fitted | 0 to 2 |

**The five new arms cost 120 fits, and the cuts below name which fit goes first.** The seed count is
3, so the seed rule of the `study` skill (an interval over repeated seeds that excludes zero) has
three seeds to check. Experiment 713's seed-noise standard error at 5 seeds was 0.017 points of
all-rows CRPS at the six signed-MW Primaries. The in-force seed noise is not measured yet, so
the seed guard (the same sign for each of the 3 seeds) stands in for a noise floor, and phase 0 prints
the seed spread of one screen fold before phase 3 starts.

### The level term and its inputs

**The level term is built only from the 14 history rows, the target row's weather and calendar, and
constants fitted on the fold's training window.** The history rows carry power up to the
visibility limit of experiment 713: the newest reading is stamped 24 hours before the native origin
(09:00 UTC), and a history row of age a covers the 24 hours that start at 12:00 UTC a days before
the contract origin. No value after that limit enters a feature. Three features feed the level
head.

- **Row means.** The mean over the visible half-hours of each of the 14 history rows of the
  standardised power (zero and a presence flag where a row has no visible half-hour). The head
  reads the trunk's pooled summary of the target row, the read-out's weighted mean and spread of
  similar days, the one-hot lead day, and these 14 means.
- **The age rule A1.** A fixed rule gives the age of the shift in days. Take the mean of the
  oldest 7 history rows as the reference level. Count the newest consecutive history rows whose
  mean differs from the reference by more than 2 noise units. A noise unit is 1.4826 times the
  median absolute deviation of the daily differences of row means over the fold's training window.
  The age is that count plus 2, because the newest usable row is 2 days old, and a count of 0 gives
  no shift (the level term is zero, as in experiment 647's age floor). The rule is not learned and
  not tuned on any score.
- **The depth.** The head outputs a signed depth in standardised units, limited by a tanh to 0.5 of
  capacity divided by the Primary's target standard deviation, the range of the generator.

**The level term adds to the head's three mixture means, which keeps the forecast a mixture of
Gaussians with 11 stored quantiles.** In W0 no level term exists. In LP the three means each add the
depth. In LD the three means each add the depth times S(age + 1 + lead / 24) / S(age + 1), with S
the fixed curve of the next subsection, so every quantile moves together. In LM and LR the head
outputs six components: the three base components keep their weights times (1 - p), and three
copies shifted by the depth take their weights times p, where p is the probability that the shift
is still on at the valid time. LM sets p to the same survival ratio as LD. LR reads p from 10
non-negative daily hazards, as experiment 652's head does (cumulative hazard summed over the days of
the lead, with p forced to zero below A1's age floor). The mixture negative log-likelihood is
exact for all five arms, and the 11 quantiles come from the mixture by the same routine as
experiment 713.

**Nothing in the level term can read the forecast period.** A unit test perturbs every power value
after the visibility limit, and every shift that starts after the origin, and asserts that the
features, the depth, and the hazards of every arm are bit-identical. The origin-time weather of
the target row is the 00Z ECMWF ensemble run of the origin day, as in experiment 713, and
`e00713_weather_lookahead.py` prints a row to show it. The history rows use ERA5 and CAMS
analyses, which a live service could not build for the 2 to 5 days before the origin, so every
reading is a ceiling for a live forecast, as in experiment 713.

### The training worlds

**The new arms train on worlds where an injected shift sits in the history rows and in the target,
and the stored `M1` does not.** A world adds, for each Primary, a sequence of shifts to the cleaned
power over the stage window. The generator copies experiment 650's draw: depth log-uniform from 3
to 50 points of capacity, a random sign, a constant or proportional shape (constant only at the
MVA-only Primaries), a step or a ramp of 6 to 48 hours, and a duration that is 3 to 14 days with
probability 0.5 and 28 to 112 days otherwise. Shifts of one Primary are separated by gaps drawn from
an exponential distribution with a mean of 45 days, which puts about 46% of origin days inside a
shift. Each training pass draws a new world from the seed of the fit and the pass. A fixed
validation world, drawn once per fold from a seed no training pass uses, serves early stopping, so
the validation loss does not move with the draw.

**The duration mixture is an assumption, and the page says so.** The only measured durations are
the 66 hand-filtered real transfers on the earlier 16 series, of which 31 lasted a day or less,
shorter than any injected shift. The generator's mixture keeps experiment 652's, so this
experiment's result bears on 652's assumption and can be compared with it. Two robustness worlds
cover the mismatch, both report-only: experiment 650's held-out family (durations of 5, 10, 21, and
45 days, depths up to 0.65, ramps of 72 to 120 hours), and a world whose durations are resampled
from the 66 real transfers.

**The fitted survival curve uses the training worlds' durations and nothing else.** The best of an
exponential, Weibull, log-normal, and log-logistic distribution, chosen by likelihood, is fitted to
the durations of the shifts that end inside the training window, with shifts clipped at the
window edge excluded from the fit (the false-end rule of experiment 652). The curve is refitted at
each monthly refit from that fold's training worlds. The 66 real transfers do not enter the
counted curve, for four reasons. Their dates fall inside the scored months, so a pooled fit would
read the test period. Restricted to transfers that ended before each fold's cutoff, the fit would
use fewer than 66 durations and some folds would differ in kind. Half of them last under a day, a
range the generator never draws, so a curve fitted to them would be misspecified for the counted
worlds by construction. The counted worlds' own durations come from the generator, so the curve
fitted to the generator's training durations is the correct curve for them, and LR learns from
the same durations.

### The scoring worlds and the strata

**The scoring worlds hold one shift per Primary per pass, drawn by experiment 650's generator on
the 8 fold months, 12 training-family passes and 4 held-out passes per month.** The onset falls
between 10 days before the month starts and day 20, so the ends fall before, inside, and after the
horizon. A scoring shift lies in the scored month and the 16 days before it, and the fold's
training window ends before the month starts, so no scoring shift enters training. The forecaster
reads the injected panel as its history power, through the swap of experiment 713's `era` module,
and the actuals are the injected power.

**The strata are the state of the shift at the forecast origin, as in experiments 652 and 654.**

- **Far from the end:** a shift is in force and more than 10 days remain.
- **Middle:** a shift is in force and 1 to 10 days remain.
- **Near the end:** a shift is in force and at most 24 hours remain.
- **Ended in the last 24 hours:** the shift ended within 24 hours before the origin.
- **Event-free:** no shift in force, none ended in the previous 7 days, and none starting before
  the row's valid time.
- **Context strata, report-only:** onset ahead (the shift starts after the origin and before the
  valid time), and ended 1 to 7 days before the origin.

**"In force" pools far, middle, and near, and shifts clipped at the window edge are left out of the
three strata that use the end.** The strata use the future end, so they serve scoring only. The
scaling factor uses all strata of the month pooled, so the factor cannot absorb a stratum.

### Contrasts, thresholds, and vetoes

**Three counted contrasts test the hypothesis, and each has its own threshold.** Every contrast is
the first arm minus the second, in scaled in-force CRPS, so a negative number favours the first
arm.

| Id | Contrast | Question | Threshold |
|---|---|---|---|
| C1 | S* minus `M1` | Does the bundle of training worlds and a survival-carried level term beat the stored forecaster where a shift is in force? | tau1 = 1.0 |
| C2 | S* minus LP | Does survival beat full persistence, the proposal's contract as written? | 0 (the interval excludes zero) |
| C3 | S* minus W0 | Does the survival structure add anything to the training worlds alone? | tau3 = 0.6 |

**The thresholds come from the cost of porting, at an assumed in-force share q of 0.05.** Real rows
inside logged events are 0.025 of the 5.9 million rows of experiment 713's protocol file (all arms
pooled), and 0.19 at the Primary with the most logged events. One final-test window of experiment
654 had 0.14 of its rows inside detected shifts. NGED's log misses events, so the plan takes a q of
0.05, twice the logged share and well under the detected share. C1 ports a level head, a survival
fit at each monthly refit, and a world generator in training, which is a pipeline step. It needs
the 0.05 points of all-rows CRPS that experiment 713 set for replacing a network, which at q of 0.05
is 1.0 in-force points. C3 ports the head and the curve fit over a training world that the
experiment already accepts, which is a new head, so it needs the protocol's 0.03 points, which is
0.6 in-force points. C2 has no threshold of its own, because C1 and C3 carry the cost of porting, and
C2 tests the design contract (survival, not full persistence). A reader who takes q as 0.025 or
0.14 can move the thresholds by the same ratio, and the page prints them.

**Two vetoes guard the rows where no shift is in force, both on unscaled CRPS of S* minus `M1`.**
V1 reads injected event-free origins. V2 reads the real record, every row with no mask, which is the
harm check that includes unlogged shifts. A veto passes when the point estimate is at most +0.03
and the upper limit of the 95% t interval on the 8 monthly differences is at most +0.10, and the
limit of 0.10 is above 0.06, the detectable difference that experiment 713 found at the six
signed-MW Primaries, because 8 monthly blocks cannot show an upper limit below 0.03 (experiment 652
measured plus or minus 0.3 on 5 months). A veto reads "cannot tell" when the point estimate is at
most +0.03 and the upper limit is above +0.10, and fails when the point estimate is above +0.03.
Each veto also reports the share of real event-free rows on which S*'s median departs from W0's by
more than 0.5 points of capacity, because a gain after an ending can be bought with departures
elsewhere (experiment 663).

**Four readings stay report-only.** They are the contrast of W0 minus `M1` (the training regime
alone), LR minus LM (a learned curve against a fitted curve, on the training family and on the
held-out durations), LM minus LD (a mixture against a decay), and the per-group and per-Primary
versions of C1 to C3, with D15 and D18 beside the groups.

### Verdict mapping

**The verdict is the first row that holds, fixed before any fit.** "Gain" for a contrast means its
98.33% t interval lies below zero and its seed signs agree. Both vetoes must pass for any verdict
that is a gain.

| Order | Verdict | Condition |
|---|---|---|
| 1 | `invalid` | A leakage test, a generator oracle, or the known-answer gate fails and no fix is accepted |
| 2 | `refuted` | C1's lower limit is above 0, so S* is worse than `M1` where a shift is in force |
| 3 | `not supported` | A veto fails; or C1's interval includes zero or its point estimate is above -1.0 with the interval including zero |
| 4 | `supported` | C1 gain with a point estimate at or below -1.0, C2 gain, C3 gain with a point estimate at or below -0.6, both vetoes pass, and the held-out durations agree in sign on C2 and C3 |
| 5 | `partly supported` | C1 gain at or below -1.0 and both vetoes pass, but C3 is not a gain at -0.6 (the gain comes from the training worlds), or C2 is not a gain |
| 6 | `small gain` | C1 gain with a point estimate between -1.0 and 0, and both vetoes pass |
| 7 | `inconclusive` | anything else, including a veto that reads "cannot tell" |

**A `partly supported` verdict means: port the training worlds, and not the survival head.** The
take-home message would say so. A `not supported` verdict because of V2 would say that the world
regime raises the error on the real record, and the next experiment would test a world that mixes
clean and shifted targets.

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
- **Stored forecasts first.** `M1`'s stored forecasts are the control, run on the new worlds from the
  stored states, and no fit of `M1` repeats.
- **Screen first.** 4 folds (2025-11, 2026-01, 2026-03, 2026-06), seed 0, 5 arms, with the choice
  of S* made from the screen alone and every screen number labelled interim.
- **Tails and cost.** Coverage and pinball loss at q01, q05, q95, and q99, the cold winter nights,
  and the avoidable-curtailment cost, through experiment 713's scorer, beside CRPS.
- **Forecast plots.** A day-ahead forecast and the full 14-day forecast of one origin, each against
  what happened, in a winter and a summer month.
- **Four load-shift scenarios.** All rows; rows outside events and 7 days or more after one; rows
  inside events; and the 7 days after an event ends, for each arm and for the climatology row, on the
  injected worlds (events known exactly) and on the real record (NGED's logged events, the
  `event_rows`, `outside_events`, and `event_following_7_days` flags of experiment 713's protocol
  rows).
- **At least 6 load-shift time series plots,** across both metering groups, marking each event's
  start and end and showing each arm's forecast: 3 injected worlds and 3 real logged events at each
  group, none at D8. The page states the unlogged-event caveat: NGED's logs may miss events, so
  the real event-free rows may hold events no label covers, the harm table is optimistic, and the
  real-event slices undercount.
- **Loss plots.** The training and validation loss against the epoch for each arm, one fold and one
  seed, with the stored epoch marked.
- **Origin-time inputs.** The level features, the depth, and the hazards read nothing after the
  visibility limit, by the perturbation test above.
- **Streaming scoring.** The scorer reads one fold and one seed at a time. A forecast file holds
  16 Primaries by about 30 origins by 672 half-hours (322,560 rows) by 11 quantiles, about 15 MB, so
  one fold and seed of 7 arms and 16 world passes peaks near 2 GB. The cap is 16 GB, and the
  measured peak goes into the page.
- **Time log and stop line.** A per-phase log of each job's device, start, and end, summed into
  GPU-process-hours after each phase, with a stop at 7.5 GPU-process-hours and at 12 CPU-hours.
- **Review gates.** No stage runs before the review marker of the stage before it exists, and every
  edited reviewed script gets a new hash and a re-check limited to the diff.
- **Mid-day onsets and an oracle.** The generator's tests place shifts at mid-day onsets, and an
  oracle that adds the true shift to the median returns a follow share of 1 through the scorer.
- **The synthetic world gallery.** At least 8 plots of the generator's output, drawn before the first
  fit (see "The gallery").
- **Anonymisation.** The experiment uses no metered generator's series. The real series of D8 is
  never drawn in a published plot, and every published number is an aggregate.

**The plan follows these items of earlier advice.**

- Fit a survival curve to durations with no network, as the baseline for a learned survival output
  ([experiment 652](../00652-two-output-forecast-with-explicit-rules-and-survival-hazard/result.md);
  LD and LM).
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
- Experiment 713 and 715 advise a longer pass budget for known-answer fits of the forecaster. The
  real folds keep experiment 713's cap of 60 passes with a patience of 8, because the stages start
  from stored states. The known-answer gate (below) fits on the screen folds at the same budget and
  is diagnosed, not waived, if it fails.

## The known-answer gate

**Before any real fold is scored, two checks must pass on the screen folds at seed 0.** An oracle
that adds the true shift times the true survival to `M1`'s median forecast must return a follow
share of at least 0.95 through the scorer, at a mid-day onset as well as at midnight. LP, trained,
must follow at least 0.75 of the depth of a shift that is in force and far from its end, at lead
days 1 to 2. Experiment 713's network failed an equivalent check four times and the maintainer
waived it. Here a failure stops the stage and goes to the maintainer, and no worker waives it.

## The gallery

**The gallery is drawn and shown in the plan update before the first fit.** Each panel shows one
component of the world separately, on a real Primary other than D8: the real cleaned power; one
injected shift of each family (constant step, proportional shape, ramp, short, long, and held-out);
the same shift in the history rows and in the target; the strata marked on one 40-day record; the
fitted survival curve against the generator's true survival; and the age rule A1's output on one
shift. A smoke check on the data files asserts that every injected shift is nonzero in the rows where
it should act, and that the share of origin days in each stratum matches the generator's design to
within 3 points.

## Member paths

**The stored and scored forecast is the mixture's 11 quantiles at each lead, and the 51 member paths
are a check on it.** CRPS at one valid time depends only on the marginal distribution at that valid
time, so a mixture whose shifted branch has probability p at that lead scores the same as 51 paths
whose end times are drawn from the same survival curve. The check draws, for each origin, 51 end
times T = the survival curve's inverse at (k - 0.5) / 51 for k = 1 to 51, conditioned on A1's age,
and a shared quantile level for the base mixture, so that a path keeps the shift until T and returns
to the base after it. The check asserts three properties on the screen folds: the share of paths
that end inside 240 hours is within 0.02 of 1 - p at 240 hours, the 51-path empirical quantiles at
each lead differ from the stored 11 quantiles by at most 0.5 points of capacity on average, and the
51-path ensemble CRPS (with the finite-sample correction) is within 0.05 points of the quantile
CRPS. The energy score and the variogram score of the 51 paths against independent draws are
report-only, because an earlier check found no measurable gain from coupling member paths.

## Cost

**The plan fits 120 networks and projects about 6.7 GPU-process-hours, with a stop line of 7.5, and
about 5 CPU-hours.** The estimate comes from stored times: stage 2 took 60 seconds and stage 3
took 66 seconds on average over the 40 fits of `M1` in experiment 713, and a pass of 3,360 forecasts
took 1.6 seconds in experiment 726. The estimate has not been measured on the new code.

| Phase | Jobs | GPU-process-hours | CPU-hours |
|---|---|---|---|
| 0. Generator, tests, gallery, curve fits, pre-fit guards, timing of one fit | 12 | 0.3 | 1.0 |
| 1. Real-environment smoke test, one epoch of each arm, scratch output | 6 | 0.2 | 0.2 |
| 2. Screen: 5 arms, 4 folds, seed 0, fit and predict on the worlds and the real record | 24 | 0.95 | 0.5 |
| 3. Remaining fits: 5 arms, the other 20 fold-seed pairs of 24 | 100 | 3.7 | 0.2 |
| 4. Prediction on the worlds and the real record, one job per fold and seed | 20 | 1.2 | 0.2 |
| 5. Scoring, intervals, scenarios, member-path check | 8 | 0 | 2.5 |
| 6. Figures, loss plots, and time series plots | 4 | 0.3 | 0.5 |
| **Total** | **about 175** | **about 6.65** | **about 5.1** |

**The order of cuts if the projection after phase 0 is above 7.5 GPU-process-hours:** drop LD (its
contrast with LM is report-only), then fit seeds 0 and 1 only, which keeps every fold and arm. No
fold is cut from a counted reading, because the 8 months are the resampled unit. The manager
approves any cut before phase 2. The run uses one GPU process at a time under `systemd-run --user
--scope -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue`, one thread for each numerical
library, and cores 8 to 15 for the CPU phases.

## Method

**Every setting below is fixed before any fit.**

### Architecture

- The forecaster is experiment 713's network: width 64, two axial blocks, the similar-day
  read-out, and a three-component Gaussian mixture head, with the frozen weather and date-time
  encoders of the Candidate's pre-training (arm `M1` of experiment 713).
- The sample is one Primary, one origin day, and one lead day k (1 to 14): 15 rows of 48
  half-hours, the target row and 14 history rows of ages 2 to 15, with the visibility limit
  of 24 hours before the native origin.
- LP, LD, LM, and LR add a level head (two hidden layers of 64 units, as the R-haz head has) that
  reads the pooled target row summary, the read-out's weighted mean and spread, the one-hot lead day,
  and 14 history row means, and outputs a signed depth. LR adds 10 daily hazards.
- The shifted branch is a copy of the three mixture components shifted by the depth. LD shifts
  the three means by the depth times the survival ratio. LP shifts them by the depth. W0 has
  no level head.
- The shift generator and the strata are experiment 650's and experiment 654's, copied into this
  experiment's folder with the changes that "The training worlds" and "The scoring worlds" state.

### Training regime

- Training data: 16 Primaries, cleaned power, ERA5 and CAMS weather in stage 1 (stored), ENS 00Z
  weather in stages 2 and 3, with an injected world in every pass of stages 2 and 3 and no
  injected shift in stage 1.
- Loss: the mixture negative log-likelihood on the standardised, shifted target, with the weights,
  learning rates, AdamW settings, batch of 256, gradient clipping at 1.0, passes cap of 60 and
  patience of 8 of experiment 713's recipe, and the 25.5% block dropout of history power.
- The level head and the hazards train with learning rates 10 times the trunk's, because the lead-day
  parameters of experiment 713 trained slowly, and the multiplier is fixed here and not chosen on any
  score.
- Seeds 0 to 2. One refit per month. The GPU hours are 6.65 projected and the measured figure goes
  into the finished page.

### Measurement

- Scored rows: the 8 monthly folds, daily 12:00 UTC contract origins, leads 24 h up to but not
  including 240 h, valid times before 2026-07-01, under the row set of the front matter. The injected
  worlds use the same origins and valid times.
- Truth: the cleaned power divided by capacity on the real record, and the injected power divided by
  capacity on the worlds.
- Metrics: CRPS scaled and unscaled, the normalised mean absolute error of the median, coverage and
  pinball loss at q01, q05, q95, and q99, and the follow share, by stratum and lead day.
- Baselines: `M1`, W0, LP, and the month-by-half-hour climatology as a context row.
- Intervals: a t interval on the 8 monthly differences of the seed-averaged forecasts at 98.33%
  for C1 to C3 and 95% for the vetoes, with a month-block bootstrap of 100,000 resamples beside it.
- The decision rule, the vetoes, the gate, and the verdict mapping are above.

## Files

**The code sketch lives in `ml-experiments/00738-msa-survival-mixture-prevailing-forecast/`.** Each
script carries the id as `e00738_`. The implementation plan names each file, its tests, and the run
order.

- `e00738_world.py` draws the training worlds and the scoring worlds from experiment 650's
  generator, adds the shift to the inputs and the target, and assigns the strata.
- `e00738_survival.py` fits the four candidate curves to the training durations and returns the
  survival ratio, and holds A1, the age rule.
- `e00738_model.py` extends experiment 713's network with the level head, the three level terms, and
  the mixture of six components.
- `e00738_train.py` and `e00738_fit.py` run stages 2 and 3 of experiment 713's recipe on the worlds
  and store the states and the loss histories.
- `e00738_predict.py` predicts the worlds and the real record, for the stored `M1` states and the
  new arms, and writes forecasts with `experiment_store.write_predictions`.
- `e00738_score.py`, `e00738_report.py`, and `e00738_members.py` compute every contrast, interval,
  veto, scenario, and the member-path check, and print every number of the page.
- `e00738_gate.py` runs the known-answer gate. `e00738_figures.py` and `e00738_gallery.py` draw the
  figures and the gallery.
- Read from experiment 713 (`ml-experiments/00713-msa-axial-attention-demand-network-part-1/`):
  the network, inputs, batch, train, and score modules, and the stored stage 1 states and stage 3
  forecasts of `M1`, under `data/ml-experiments/00713-msa-axial-attention-demand-network-part-1/work/`.
- Read from experiments 650, 652, and 654: the generator (`e00650_bank.py`), the survival head and
  curve fit (`e00652_head.py`, `e00652_fit.py`), and the strata and scaling (`e00652_score.py`,
  `e00654_score.py`). Read from experiment 726 the panel plumbing of `e00726_msa_panel.py`.
- Written, never committed, under `data/ml-experiments/00738-msa-survival-mixture-prevailing-forecast/`:
  `worlds/` (the scoring worlds per fold), `networks/` (every fitted state), `fits/` (loss histories,
  seconds, and peak memory), `predictions/` (per-row forecasts), and `scores/`.
