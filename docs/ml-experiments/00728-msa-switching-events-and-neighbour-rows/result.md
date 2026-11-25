---
id: "00728"
slug: msa-switching-events-and-neighbour-rows
title: >-
  Does training the axial-attention demand network on synthetic switching events, with and without
  the rows of its electrical neighbours, keep its forecast at the normal running arrangement while a
  switching event sits in the history?
theme: lagged-power-and-error-correction
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts and network weights for about 34 fits at the first level
  and up to 74 with both options; (2) more than one design defensibly tests the hypothesis, and the
  page fixes one; (3) it extends the synthetic-world code of experiment 715, whose callers cannot be
  listed without searching the repository; (4) it touches no serving path or degradation rule; (5)
  it changes no contract. Full set of reviews: plan, code (Opus, each new script, twice), results,
  write-up, prose.
hypothesis: >-
  A metered mirror at an electrical neighbour tells the axial-attention demand network that a step
  in a Primary's recent power is temporary. A network trained on a synthetic world of both kinds of
  event (temporary moves of load between Primaries, and permanent level shifts), and given the rows
  of its electrical neighbours, forecasts the unswitched truth during a switching event with lower
  CRPS than the same network without the neighbour rows.
prediction: >-
  Written before any fit. A1 (`Rbn` minus `Rb`) lies between -0.15 and +0.05 points of capacity. The
  sign of T1 (`Rb` minus `M*`) is not predicted. T2 (`Rbn` minus `M*`) lies between 0.15 points
  below T1 and 0.05 points above T1. Guard P for T1 is at or below zero. The predicted verdict is
  `small gain` or `inconclusive`. A prediction with a range fails if the point estimate falls
  outside the range.
decision_rule: >-
  Three counted contrasts (T1, T2, A1), each with one pooled reading on the counted slice of `g2`'s
  temporary moves of visible depth with a metered mirror, each read at 98.33% from experiment 714's
  percentile bootstrap. The verdict is read from A1 alone, by the tables of the section "Contrasts
  and verdict", with two non-inferiority guards and a negative control. The smallest effect worth
  acting on is 0.10 points of capacity. Seeds 2 to 4 run only by the staging rule quoted verbatim in
  the section "Order of work and the staging rule".
benchmark: >-
  `M*` (experiment 713's selected arm, `M1`, stored stage 3 states at seeds 0 to 4); not refitted.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity against the unswitched truth on the counted slice, with the mean
  absolute error of the median forecast normalised by capacity beside it; the follow share as a
  secondary reading; the stopping pass of every fit.
---

**We ask whether the axial-attention demand network of experiment 713 (the "MSA") can tell a
switching event from a permanent level shift.** A switching event is a temporary move of load from
one Primary substation to another. The forecast should hold the power the Primary would have drawn
under the normal running arrangement while the move lasts, and should follow a permanent change. We
will train the network on a synthetic world that holds both kinds of event, once with the history
rows of the Primary alone and once with the rows of its electrical neighbours added, and score both
on a second synthetic world that the training never saw. This page is the plan: no neural network
has been fitted, no GPU has been used, no code has been written, and every number below is a design
value or a figure from an earlier experiment.

## Decisions of the maintainer

**The maintainer has answered decisions (a), (b), and (c), and all three are closed.** The three
decisions are stated below as the design states them, each followed by the answer. The ceiling of 18
GPU-process-hours stands.

- **(a) What happens if the reference-error term takes the pooled count below the floor of 60.**
  The count of usable switching events is 81 before the reference-error term of the visible-depth
  definition, and the count after the term is not known until the harness rebuilds the reference.
  The design stops the experiment before any GPU job if the count falls below 60. The alternative
  is to define the counted set of `g2` by the floor of 0.10 of capacity alone, which keeps all 81
  events and counts events smaller than 2 times the reference error. **Closed: keep as designed.**
  The counted set is defined by the 0.10-of-capacity floor alone only if the floor of 60 still holds
  after the reference-error term. Otherwise the experiment stops as written, and the floor is not
  loosened after the fact.
- **(b) Whether to accept that the experiment probably ends at 2 seeds.** T1 can be above zero, and
  the futility rule then fires unless T2 is below zero in a group. The verdict on A1 would then
  rest on 2 seeds and at most 81 events. **Closed: accepted.**
- **(c) Whether the three values set in the latest revision of the design stand.** The values are
  GH4's floors of 60 pooled events and 8 events per fold month, and the cap of two permanent
  changes per Primary within +-0.50 of capacity. No measurement stands behind any of the three.
  **Closed: the three values stand as written.**

## Corrections to the plan made before any gate number was seen

**Five rules of the gate and the training world were corrected before any count or test result on
real data was printed.** The corrections are recorded here so that the commit order shows they came
first.

- **The span of the reference error follows the design text.** The design defines the reference
  error over the fold's training span. The first code read the days `[md - 400, md)` of the stored
  window, where `md` is the first day of the fold month. The corrected code reads
  `[md - 758, md - 28)`, the 730 days on which experiment 714's reference is fitted. The reason is
  that the design names the training span, and the stored window is a different and shorter span.
- **GH5 pools its tests and uses a Bonferroni level.** The first rule ran three chi-squared tests at
  the 1% level on each of 40 training draws, which is 120 tests with no correction. A correct
  generator would fail at least one of the 120 tests with a chance of about 70%. The corrected rule
  pools the 5 draws of each of the 8 fold months into one table, runs 5 tests per pooled fold (kind
  by sign, kind by Primary, kind by sign within each group, and the balance of unmirrored rises and
  falls), and reads the 40 tests at the level 0.01 divided by 40.
- **The training world applies one placement rule to both kinds of event.** The design states the
  floor and the redraw for permanent changes only, and experiment 715 clips the steps at recipients
  only. The first code also clipped a temporary move at its MVA-only host, and tested a temporary
  move over its own length but a permanent change over the whole grid after its onset. Synthetic
  panels showed that these differences let the group and the sign of an event predict its kind. The
  corrected rule tests every event at an MVA-only Primary over `[onset, end of grid)` of the
  current history, so a rise and a fall of equal depth pass or fail the same way for both kinds. The
  limits on stacking (a summed shift within 0.50, and a target at or above zero as if the step were
  a fall) ignore the sign and apply to both kinds; only the count of two permanent changes per
  Primary applies to permanent changes alone, because it depends on the host alone. A temporary
  move is no longer clipped at its host, and a step that fails is redrawn at another Primary or
  onset with its sign and depth kept.
- **A permanent change is first placed in the first 28 days after its onset.** The onset of a
  permanent change is drawn over the grid as if the change lasted 28 days (the cover that the
  schedule counts), so that its onset distribution is uniform and does not crowd the end of the
  schedule; the change itself never ends.
- **GH5 adds a z-test on the fall share, which counts toward the verdict.** The chi-squared tests of
  a single fold cannot see a gap of about 0.05 between the fall share of permanent changes and that
  of temporary moves, and a gap of that size lets a network read the kind from the group and the
  sign. The gate therefore pools all 40 draws and runs, in each group (MVA-only and signed-MW), a
  z-test on the difference of the two fall shares with a pooled variance. It is read at the same
  Bonferroni level as the other tests, 0.01 divided by 42 tests (5 per pooled fold times 8 folds,
  plus these 2). A failure stops the line and no waiver applies. A group with no event of one kind
  is reported with a notice, and the test then has nothing to compare.
- **GH5 stops on a training world that lacks events.** In each group (MVA-only and signed-MW) and
  for each kind, the 40 training draws must hold at least 60 events, and the 5 draws of each fold
  month at least 8, the floors of GH4. A group with no event of a kind stops the line, with no
  waiver and no fallback.

**The placement rule changes three things in the design, and they are design changes.**

- **Temporary moves now face the limit on the stack and the mirrored target floor.** Experiment 714
  and experiment 715 apply neither to a temporary move. The corrected training world applies them
  to both kinds, so a temporary move of a given depth is dropped at an MVA-only Primary whose power
  could not carry a fall of that depth.
- **The training draws are smaller.** On synthetic panel A a draw holds 96 events on average, down
  from 105; on panel B it holds 86, down from 104.
- **The z-test excludes only a strong cue.** With about 400 permanent changes and 650 to 2,800
  temporary moves in a group, the smallest gap that the test finds with 80% power is 0.10 to 0.14.
  A gap of 0.05 would pass. The gate prints the smallest gap from the real event counts.

**Three rules were run on two synthetic panels of 40 draws each, at the gate level 2.4e-4.** The
MVA-only power of panel A runs from 0.25 to 0.65. The MVA-only power of panel B runs from about 0 to
0.6, clipped at zero. The first rule is the generator before the correction (commit `cf4a29d91`).
The second rule is the first form of the correction, in which the count, the stack, and the target
floor were limits for permanent changes only and a change over a limit was dropped without a redraw.
That form was proposed first and it was wrong, because the limits depend on the depth and the floor
test passes falls of small depth only, so the limits changed the mix of signs of the permanent changes
alone. The third rule is the placement rule in force. The gap is the fall share of permanent changes
minus the fall share of temporary moves.

| Rule | Panel | Group | Permanent: n, fall share | Temporary: n, fall share | Gap | z | Result |
|---|---|---|---|---|---|---|---|
| Before | A | MVA-only | 560, 0.364 | 1351, 0.413 | -0.049 | -1.98 | pass |
| Before | A | signed-MW | 630, 0.608 | 1661, 0.587 | +0.021 | +0.91 | pass |
| Before | B | MVA-only | 523, 0.216 | 1041, 0.251 | -0.035 | -1.52 | pass |
| Before | B | signed-MW | 629, 0.665 | 1956, 0.653 | +0.012 | +0.54 | pass |
| Permanent-only limits | A | MVA-only | 211, 0.483 | 1046, 0.263 | +0.221 | +6.38 | fail |
| Permanent-only limits | A | signed-MW | 483, 0.617 | 2024, 0.636 | -0.019 | -0.77 | pass |
| In force | A | MVA-only | 398, 0.492 | 657, 0.463 | +0.030 | +0.94 | pass |
| In force | A | signed-MW | 569, 0.506 | 2223, 0.519 | -0.013 | -0.53 | pass |
| In force | B | MVA-only | 0 | 0 | none | none | stop: no events |
| In force | B | signed-MW | 601, 0.506 | 2855, 0.507 | -0.001 | -0.06 | pass |

**The z-test does not flag the generator before the correction, so the correction rests on the
reasoning and not on that test.** The gaps before the correction (-0.049 and -0.035 among MVA-only
Primaries) lie inside the smallest gap that the test finds. Panel B under the rule in force shows the
coverage stop at work: the MVA-only Primaries hold no event of either kind, so the gate would stop.

**Unconfirmed assumptions, stated as the design states them.** The implementer confirms each one
before the plan commit.

- **Every code line, stored-score figure, and file count quoted from the design review** (the era
  bank figures, the stage 1 and stage 3 states of `M1` at seeds 0 to 4, the 65 files of experiment
  715's phase 4, `age_emb` and `beta_age`, and the line numbers) was confirmed by the re-check, and
  is confirmed again by the implementer before the plan commit.
- **`g2` holds 512 events with the mirror classes, shapes, and size grid of the section "The
  worlds",** and the world seeds of `g2` can be read from the stored files.
- **The pooled number of counted moves after the visible-depth filter is at least 60, with at least
  8 in each injected fold month.** The harness check GH4 prints the counts. The count is 81 before
  the reference-error term, and the term can only lower the 81.
- **Every one of the 81 moves has target half-hours in the counted slice at a scored origin.** The
  shortest move of `g2` lasts 1 day, so the counted slice of that move ends 8 days after the onset
  and the first scored origin is 7 days after the onset. GH4 prints the target half-hours per move.
- **`e00714_common.build_reference` runs on the CPU for the 4 guard folds,** for which the function
  has never been run, and within the 9G memory limit.
- **Experiment 715's floor on the power of an MVA-only Primary exists as the training world
  assumes,** and the cap of two permanent changes per Primary still lets a training draw hold 30%
  permanent changes.
- **The factor 3.48 is experiment 713's convention for the smallest detectable difference `δ`,** and
  the spread of `F4` minus `F4-R` is close to the spread of A1.
- **The `stage3_mirror` mode runs with host rows alone** for `Rb`, with no change to experiment
  715's network code.
- **A recipient's target stays clean in `mirror_world`** when a move has two or three recipients.
- **The frozen checkout `frozen-713-7c29114a8` still exists and is clean** when the fits start.
- **The GPU-process-hours per fit of a host-only arm on the training world, the 0.26 per arm and
  seed of predictions, the 0.8 of the `M*` predictions at seeds 2 to 4, and the 0.8 of the held-out
  draw** are estimates.
- **The chi-squared threshold of GH5, GH4's floors of 60 and 8, the cap on stacked permanent
  changes, the 0.01-nat flag of the training-budget check, and the 0.15-point statement on `δ`**
  have no measurement behind them.

## Names used on this page

- **Experiment 713** is the axial-attention ("MSA") demand network. **`M*`** is experiment 713's
  selected arm, `M1`. **`C`** is the Candidate of
  [experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md).
  **Experiment 715** (F4) added neighbour rows to `M*`: arm **`F4`** reads the rows of electrical
  neighbours, and arm **`F4-R`** reads the rows of rewired (wrong) neighbours.
  [Experiment 714](../00714-msa-era-identifiability-audit-and-frozen-probe/result.md) is the
  identifiability audit. [Experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md)
  built the bank of injected worlds.
- **A switching event** is a temporary move of load between Primary substations. **The donor** is
  the Primary that the load leaves, and **a recipient** is a Primary that the load arrives at. **A
  permanent change** is a level shift that does not end.
- **The unswitched truth** is the power a Primary would have metered under the normal running
  arrangement, with no temporary move. **The injected truth** is the power with the injected change
  included.
- **`g2`** is experiment 714's second generator of injected worlds. **The training world** is the
  two-kind synthetic world that stage 3 trains on.
- **A metered mirror** is the opposite-signed step at a recipient that has readings in the panel.
  **A hidden mirror** is a step at a substation with no readings in the panel.
- **The follow share** is the share of an injected step that the forecast median takes up.
  **CRPS** is the continuous ranked probability score, in points of capacity. A negative contrast
  favours the treatment.
- **The injected folds** are the four folds that `g2` covers: 2025-11, 2026-01, 2026-04, and
  2026-06. **The guard folds** are the other four folds of experiment 713: 2025-12, 2026-02,
  2026-03, and 2026-05.
- **The two groups** are the six signed-MW Primaries and the eight MVA-only Primaries.
- **The counted slice** of a temporary move holds the target half-hours from the onset of the move
  to 7 days after the move ends.

## Why this experiment runs

**No stored score answers the switching question.** Experiment 715's two counted contrasts were
overall CRPS on real folds. Its arms trained on a mirror world were never fitted, and its
injected-world predictions were written for one fold and one seed of `F4` alone.

**The earlier F4 CRPS result (experiment 715) is not a reason against this experiment.** That result
is overall CRPS on real folds, and it says nothing about CRPS during a switching event against the
unswitched truth. No page of this experiment cites the result as a reason to expect a null.

**The stored `M*` follows lasting steps strongly, and follows temporary moves too often.** Experiment
713's era bank gives `M1` minus `C` of -3.23 [-3.74, -2.72] points of CRPS on lasting eras at 7 to
28 days, and +1.28 [+1.08, +1.49] on the switching slice scored against the unswitched truth. Stage
3 of `M*` already trains on clean targets, so clean targets alone do not stop `M*` from following a
temporary move. The training world differs from experiment 581's shifts in three ways that reach a
host-only network: the event cover is 15% and not 5%, the lengths run from 1 hour to 120 days, and
30% of events are permanent changes that stay in the target.

**The maintainer's note on what can separate the two kinds is the design's context.** In the
maintainer's words, "in general, it's hard to discriminate between a temporary step versus a
permanent step; the only two hints are what a primary substation's electrical neighbours do, and
the rough shape of the step (although the shape is probably less informative)". Arm `Rb` reads host
rows alone, so the shape and the length of a step are its only cues. Arm `Rbn` adds the neighbours'
rows, so `Rbn` minus `Rb` isolates the first hint.

## Arms

| Arm | Neighbour rows | Stage 3 trains on | Fitted | Seeds | Role |
|---|---|---|---|---|---|
| `M*` | None | Experiment 581's host-only shifts | Stored | 0 to 4 | Baseline of every counted treatment contrast |
| `Rb` | None | The training world | Stages 2 and 3, from the stored stage 1 state of `M*` | 0 and 1; 2 to 4 by the staging rule | Treatment: training on both kinds of event, host rows alone |
| `Rbn` | Electrical neighbours | The training world, same draw as `Rb` | As `Rb`, through experiment 715's `stage3_mirror` mode | As `Rb` | Treatment: experiment 715's arm `F4-W` |
| `F4`, `F4-R` | Electrical; rewired | Experiment 581's host-only shifts | Stored | 0 and 1 | Report-only: networks never shown a mirror |
| `Rb10`, `Rbn10` | As `Rb`, `Rbn` | As `Rb`, `Rbn` | As `Rb`, `Rbn`, with the lead-day learning rate ten times higher | 0 and 1 | Optional second level, run only if the learning check fails |

## The worlds

**The training world is experiment 715's `mirror_world` with seven changes.** Each event is a
constant step in megawatts with an abrupt onset. Seventy per cent of events are temporary moves and
30% are permanent changes. A temporary move lasts from 1 hour to 120 days (log-uniform), and a
permanent change never ends. A Primary takes at most two permanent changes per draw, the summed
permanent shift stays within +-0.50 of capacity, and an MVA-only Primary stays above experiment
715's floor on power. A third of temporary moves have one metered recipient, a third have two or
three, and a third have none. The sign at the donor is drawn independently of the kind and of the
mirror class. Events cover 15% of each Primary's half-hours. Three quarters of events are of
visible depth. A temporary move sits in the history and never in a target, and a permanent change
sits in the history and in the target. `Rb` and `Rbn` read one shared draw per fold and seed, and
training seeds and test seeds are disjoint by an assertion.

**The counted test world is `g2`, and the counted event set is its temporary moves of visible depth
with a metered mirror.** `g2` holds 512 events on 4 folds: 256 permanent changes and 256 temporary
moves. Of the 256 temporary moves, 160 have a metered mirror, and 81 of those have a donor step of
at least 0.10 of capacity. Of the 81, 43 last 7 days or more. A step is of visible depth at a
Primary if the step is at least 2 times that Primary's daily-mean reference error and at least 0.10
of capacity, so the 81 can only fall. The 96 hidden-mirror moves are the negative control. Experiment
709's bank, and a held-out draw of the training generator, are report-only. The page discloses that
three quarters of `g2`'s permanent changes are not flat steps, which the training world never shows,
and that a permanent fall of visible depth is always 0.15 of capacity and a rise always 0.30.

## The gate

**The gate has two halves, G-H and G-L, and a fail of either half stops the experiment with no
waiver.** G-H is nine checks on the CPU: oracle forecasters through the scorer (GH1, GH2),
reproduction of the stored `M*` forecasts (GH3), event counts with a floor of 60 pooled moves and 8
per injected fold month (GH4), balance of sign and site across the two kinds (GH5), disjoint seeds
(GH6), clean targets and the cap on stacked permanent changes (GH7), permanence of `g2`'s
permanent changes in the scored month (GH8), and the shared draw of `Rb` and `Rbn` (GH9).

**G-L fits `Rb` and `Rbn` on fold 2025-11, seed 0, exactly as the counted fits run.** G-L prints one
number per arm: the follow share on that fold's permanent changes of visible depth at origins 14 and
21 days after the onset. The threshold, written now, is at least 0.30 at the best-validation state.
A fail stops the first level. The only permitted next step is the second level, `Rb10` and `Rbn10`,
which runs G-L once with the same threshold. If the second level fails, the experiment ends.

## Contrasts and verdict

| Id | Contrast | Role |
|---|---|---|
| T1 | `Rb` minus `M*` | Counted treatment contrast against the baseline |
| T2 | `Rbn` minus `M*` | Counted treatment contrast against the baseline |
| A1 | `Rbn` minus `Rb` | Ablation; never opens a seed |

**Each contrast has one counted reading, pooled over both groups on the counted slice, and the three
intervals are read at 98.33%.** Each event has equal weight, and every contrast pairs seed s of one
arm with seed s of the other. The verdict is read from A1 alone. A1 reads `refuted` on a clear
loss, `not supported` on a loss or no detectable difference, `supported` on a gain when T2 has a
point estimate below zero and guards P and O and the negative control hold, `partly supported` on a
gain when any of those fails, `small gain` on a small gain, and `inconclusive` otherwise. The smallest
effect worth acting on is 0.10 points. The smallest detectable difference `δ` is 3.48 times the
standard error of `F4` minus `F4-R` on the counted reading, and is computed from stored networks
before G-L.

**Two non-inferiority guards stand beside the counted contrasts, each read for T1 and for A1.**
Guard P is CRPS against the injected truth 7 to 28 days after the onset of `g2`'s permanent changes
of visible depth, with a margin of +0.30 points. Guard O is overall CRPS on the 8 real folds, with
margins of +0.094 points at the six signed-MW Primaries and +0.25 points at the eight MVA-only
Primaries. Guard O penalises a correct forecast during unlogged real switching events, and the page
says so. The negative control is A1 on the hidden-mirror moves of visible depth, which should be
near zero.

## Order of work and the staging rule

**The staging rule and the futility rule are fixed before any fit.** The rule is quoted verbatim
from the design:

> After the 8-fold (or 4-fold g2), 2-seed reading of the counted contrasts against M*, seeds 2 to 4
> run only if, for at least one counted treatment contrast against the baseline, the point estimate
> is below zero in at least one group; ablation contrasts such as Rbn minus Rb never open seeds; if
> every counted point estimate is at or above zero in both groups the experiment is written up at 2
> seeds and the page states that the futility rule fired. This rule was added before any treatment
> score was read.

**The futility rule is likely to fire, and the page says so before any fit.** T1 can be above zero
in both groups, because the permanent changes of the training world push `Rb` towards following
every step. Seeds 2 to 4 then open only if A1 is negative enough in one group to take T2 below zero.

**The order is fixed.** The eight CPU checks of G-H run first. Phase 0 writes `g2` predictions of the
stored `M*`, `F4`, and `F4-R`, with GH3 on its first fold. The value of `δ` is added to the plan
commit. G-L runs next. The 4 injected folds follow, then the counted reading and the staging rule,
then the 4 guard folds, then seeds 2 to 4 if the staging rule opens them. A stage-1 refit runs only
if a review requires it.

## Compute

**The plan needs about 8.8 GPU-process-hours, and at most 17.4 with the optional second level and
seeds 2 to 4, against a ceiling of 18.**

| Phase | Fits | GPU-process-hours |
|---|---|---|
| G-H, the harness gate | 0 | 0 (CPU) |
| 0. `g2` predictions of the stored `M*`, `F4`, and `F4-R`, seeds 0 and 1 | 0 | 1.6 |
| 1. G-L: `Rb` and `Rbn`, fold 2025-11, seed 0, with `g2` predictions | 2 | 0.5 |
| 2. `Rb` and `Rbn` on the 4 injected folds, seeds 0 and 1 | 16 | 2.1 |
| 3. `g2` and bank predictions, 4 arm-seeds; the counted reading and the staging rule | 0 | 1.0 |
| 4. `Rb` and `Rbn` on the 4 guard folds, seeds 0 and 1 | 16 | 2.1 |
| Reserve for reruns of stages that hit the pass cap | | 1.5 |
| **Subtotal** | **34** | **8.8** |
| Optional second level: `Rb10` and `Rbn10` on the 4 injected folds, seeds 0 and 1 | 16 | 3.1 |
| Seeds 2 to 4 of `Rb` and `Rbn` on the 4 injected folds, with predictions | 24 | 4.7 |
| `g2` predictions of the stored `M*` at seeds 2 to 4, run only with the row above | 0 | 0.8 |
| **Total with both options (upper bound)** | **74** | **17.4** |

**Stage 1 is never refitted by the plan, and a stage-1 refit runs only if a review requires it.**
The implementer reports a projection to the manager after phase 1 and after the first 3 fits of
phase 2, and stops and asks before crossing the ceiling. The order of cuts is the held-out draw, then
the bank predictions, then the guard folds cut to seed 0.

## Standing practices

This experiment follows the standing practices of the `ml-experiment` skill that apply to a
network warm-started from stored states: cleaned data everywhere, the noisy start of the short-history
Primaries cut, every fitted network saved outside git, stored predictions reused as controls, and
the normalised mean absolute error of the median forecast beside CRPS. It departs from the monthly
refit practice only in that it refits no stage 1 and fits stages 2 and 3 once per fold and seed. It
follows no earlier `advice` item that conflicts with the design.

## Disclosures

- **The known-answer gates of experiments 713 and 715 failed and were waived,** and this experiment
  builds on both networks. G-H and G-L replace those gates here, and neither is waived.
- **The G-L threshold was written after experiment 713's diagnostics,** and the learning-rate
  multiplier was chosen on a stage-1 synthetic fit at one training seed.
- **`g2` is a synthetic world,** and A1 is an upper reading of what the neighbour rows would do on
  real load transfers.
- **Each counted contrast has one pooled reading on at most 81 events,** and every split by group
  or by slice is report-only. The interval rests on 4 fold months and is conservative.
- **Stage 3 of `M*` already trained on clean targets,** so T1 reads the event cover, the lengths, and
  the permanent changes of the training world, and T1 can be above zero.
- **The unlogged-event caveat applies:** NGED's logs may miss events, so the real folds may hold
  switching events that no label covers.
- **Every score is of a Primary substation.** No page names a generator, and no page plots or
  tabulates a metered generator's time series.

## Method

- **Architecture:** every arm is the network of experiment 713's `M*`, the axial-attention demand
  network, warm-started from its stored stage 1 state of `M1`.
- **Architecture:** `Rb` reads the host Primary's history rows alone, and `Rbn` adds the rows of
  its electrical neighbours through experiment 715's `stage3_mirror` mode.
- **Architecture:** the forecast carries no information after the forecast origin, as in experiment
  713.
- **Training regime:** stages 2 and 3 use experiment 713's recipe, with a fine-tune rate of 2e-4, a
  cap of 60 passes, and patience 8, and the scored state is the best-validation state.
- **Training regime:** stage 3 trains on the training world, drawn once per fold and seed, shared by
  `Rb` and `Rbn`.
- **Training regime:** seeds 0 and 1 for every counted arm, seeds 2 to 4 only by the staging rule,
  and the GPU hours are those of the compute table.
- **Measurement:** CRPS against the unswitched truth at the donor and at every metered recipient of
  `g2`'s counted temporary moves, on the counted slice, scored at origins 7, 10, 14, and 21 days
  after the onset.
- **Measurement:** the baseline is `M*`, the intervals are experiment 714's percentile bootstrap with
  100,000 draws, and the decision rule is the one above.

## Files

Nothing has been written yet. The planned files are listed here, and the folder
`ml-experiments/00728-msa-switching-events-and-neighbour-rows/` does not exist at plan time.

- `e00728_worlds.py`: the training world as an extension of experiment 715's `mirror_world`, the seed
  lists, the visible depth, and the held-out draw.
- `e00728_train.py`, `e00728_fit.py`, `run.sh`: the warm start, the `stage3_mirror` mode for `Rb` and
  `Rbn`, and the stopping-pass record.
- `e00728_predict.py`: the prediction path for stored and new states on `g2`, the bank, and the
  held-out draw.
- `e00728_gate.py`: G-H checks GH1 to GH9, and G-L.
- `e00728_score.py`, `e00728_figures.py`, `e00728_report.py`: the contrasts, guards, negative
  control, `δ`, the staging rule, the plots, and every number of the page.
- `data/ml-experiments/00728-msa-switching-events-and-neighbour-rows/`: this experiment's private
  data folder, which holds its predictions and fitted states, never committed.
- `data/ml-experiments/briefs/msa_architecture/switching_design.md`: the private design note that this
  plan summarises, with its reviews in `switching_followup_review.md`.
