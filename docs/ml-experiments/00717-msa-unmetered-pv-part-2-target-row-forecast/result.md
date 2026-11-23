---
id: "00717"
slug: msa-unmetered-pv-part-2-target-row-forecast
title: >-
  Does an explicit forecast of unmetered solar output in the target row help the axial-attention network forecast a Primary's demand?
theme: solar-disaggregation
also_relevant_to:
  - lagged-power-and-error-correction
  - demand-inputs
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts and network weights for about 100 fits; (2) more than one
  design defensibly tests the hypothesis; (3) it imports and extends the code of experiment 716,
  which copies training code from experiment 713; (4) it touches no serving path or degradation
  rule; (5) it changes no contract. Full set of reviews: plan, code (Opus, each new script,
  twice), results, write-up, prose.
hypothesis: >-
  An explicit forecast of a Primary's unmetered solar output in the target row (an as-of capacity
  times a yield curve driven by the ENS shortwave of the 00Z run of the origin day), added to the
  axial-attention network that already reads the historical estimate of experiment 716, lowers the
  continuous ranked probability score (CRPS) at leads of 24 h to 240 h, because the forecast
  states the target day's solar dip in the units of the power. The gain comes from the forecast sky
  of the target day, and not from the capacity or the added parameters.
prediction: >-
  Written before any fit. B1 (HT minus H) lies between -0.06 and +0.06 points of capacity at the
  six signed-MW Primaries and between -0.16 and +0.16 at the eight MVA-only Primaries, and the
  interval of B2 (HT minus HT-S) includes zero in both groups. The predicted verdict is `not
  supported` (no detectable difference) or `inconclusive`. The prediction fails if B1 shows a gain
  or a loss beyond those ranges, or if the interval of B2 excludes zero.
decision_rule: >-
  Two counted contrasts, B1 and B2, each at the six and at the eight, so four readings, each read
  at 98.75% (1 - 0.05/4) from experiment 713's month-block bootstrap, with 713's three guards (a t
  interval on the 8 monthly differences, the same sign for each seed alone, and a point estimate of
  at least 0.034 at the six and 0.090 at the eight). The smallest gain worth porting is 0.02 points
  of capacity, the demand protocol's tier for a new input. The group readings and the six verdict
  rows are those of experiment 716, with B1 in place of A1 and B2 in place of A2. Row 5 (`small
  gain`) needs the placebo arm HT-S not to be a "loss" against H in either group; this guard was
  added before any fit, as a design fix (the second Opus design review, finding S7), and it can only
  turn a `small gain` into `inconclusive`. A `small gain` whose placebo contrast has a point
  estimate at or above zero reads `inconclusive`. A verdict of `not supported` or `refuted` adds
  one to the stop-rule count of explicit PV forecasts in a net-demand forecast and closes the line.
  Every verdict, headline, and take-home message states that the warm-start states of experiment 713
  come from a waived known-answer gate, and that the verdict concerns a forecast added to a network
  that already reads the historical estimate.
benchmark: >-
  H (experiment 716's arm of the same folds and seeds); M* (experiment 713's selected arm); the
  Candidate (P7+D50 of experiment 667); the month-by-half-hour climatology; the manual heuristic
  and BG2 where stored rows cover the scored keys.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity on leads from 24 h up to but not including 240 h after the 12:00 UTC
  origin, with the mean absolute error of the median forecast normalised by capacity beside it;
  q01, q05, q95, and q99 pinball loss and coverage; cold winter nights; avoidable-curtailment cost;
  the four load-shift scenarios; secondary readings S1, S2, and S4 on the solar block; the table
  of all 16 Primaries.
---

**We ask whether the axial-attention network forecasts a Primary's demand better when its target
row also carries an explicit forecast of the Primary's unmetered solar output, on top of the
historical estimate that [experiment 716](../00716-msa-unmetered-pv-part-1-history-estimate/result.md)
tests.** The forecast is an as-of capacity times a yield curve, driven by the ENS-mean and
ENS-spread shortwave irradiance of the 00Z run of the origin day. This page is part 2 of
follow-up F7. It is the plan: no network has been trained, and every number below is a design
value or a figure from an earlier experiment. The design is condensed from the private design note
`f7_design.md`, and the details that this part shares with part 1 (the PV estimate, the as-of
capacity, the network paths, the training recipe, the known-answer worlds, and the leakage rules)
are on the page of experiment 716.

**The honest expectation is a tie.** Six earlier experiments found that an explicit PV term built
on forecast weather did not help net demand (the table below), and the target tokens already carry
ENS shortwave. With 5 seeds the design detects, with 80% power, a CRPS difference of 0.06 points
of capacity at the six signed-MW Primaries and 0.16 at the eight MVA-only Primaries.

## Why this runs although the stop rule counts earlier verdicts

**The maintainer asked for follow-up F7 by name and confirmed the scope on 2026-11-23 ("I
confirm."), relayed through the experiment manager. That instruction is the approval that the
demand protocol requires, and part 2 is at least the seventh counted experiment on its idea.** The
stop rule counts the verdicts `not supported` and `refuted` on one idea, resets the count at
`supported` or `small gain`, and from the fourth experiment requires the maintainer's approval and
scoring at every Primary. The last `supported` or `small gain` verdict on this idea is experiment
325, in the GAM stack. Part 2 is scored at every one of the 16 Primaries.

| Experiment | What the experiment gave the forecast | Verdict | Counted |
|---|---|---|---|
| 409 | An explicit solar term in the pooled mixture density network | `not supported` | Yes |
| 477 | An explicit PV module in the net demand neural network | `not supported` | Yes |
| 501 | An explicit solar-output column from an as-of capacity | `refuted` | Yes |
| 644 | An explicit solar-capacity term with ENS weather | `refuted` | Yes |
| 655 | A member-level solar curve with cell temperature | `refuted` | Yes |
| 664 | Net demand as gross demand minus a sampled ENS solar curve | `refuted` | Yes |
| 661 | Whether the ENS time step explains experiment 644, and a bias adapter | `refuted` | Borderline: gives 7 |
| 350, 351 | A capacity-free irradiance curve and a capacity read-out in the GAM stack | `not supported`, twice | Arguably: gives 9 |
| 329, 419 | A 3-member solar distribution; a solar summary in the mixture density network | `not supported`, twice | Borderline: gives 11 |
| 031, 349, 405, 416 | PV features in XGBoost and the GAM stack; explicit term with learned capacity or 5 seeds | `inconclusive` | No |
| 352, 639, 643 | Explicit terms scored with recorded weather | `partly supported` | No: neither counts nor resets |

**This part departs from the advice of experiments 644 and 655 (no explicit solar term in the ENS
demand network) on the maintainer's instruction.** It follows experiment 661's advice to keep
production's window-mean resampler for ENS shortwave and experiment 501's advice to judge a
capacity by a placebo test. The new element over the counted experiments is the paired history
estimate of experiment 716. Experiment 713's screen stop rule binds this part: if it fires, or
if part 1 stops at its screen, part 2 stops and asks the manager. Experiment 713's known-answer
gate failed KA3 and KA12 and was waived by the maintainer on 2026-11-23, so every verdict here
makes no claim about era recognition or level shifts.

## What is compared

| Arm | PV input | Seeds | Role |
|---|---|---|---|
| `H` | CAMS `PV̂` in the 14 history rows | 0 to 4 | Reference of B1 (experiment 716's fit) |
| `HT` | As `H`, plus the ENS PV forecast in the target row | 0 to 4 | Treatment |
| `HT-S` | As `HT`, with the target-row forecast from the ENS run 28 days earlier | 0 to 4 | Counted placebo |
| `T` | The ENS PV forecast in the target row only | 0 and 1 | Report-only: scope (b) without scope (a) |

**B1 is read against `H` and not against `M*`, so that part 2 is not credited with a gain that
part 1 made.** `HT` with its target paths at zero is `H` exactly, and both start from the same
state of `M*`. `HT` minus `M*` and `T` minus `M*` are report-only. `HT-S` reads the ENS run issued 28 days before the origin day
at the same lead day, with the same clear-sky-index transfer on the mean and the spread. For
origins in the first 28 days of the ENS record the donor run is 28 days later, which is issued after the sample's
origin and says nothing about the sample's target day; the check counts those origins in training
and in held-out days and the page lists them.

## The target-row forecast

**The target row gets two numbers per half-hour, `p0_mean` and `p0_sd`, from the yield curve's
mean and standard deviation over three Gauss-Hermite nodes of the ENS shortwave.** The shortwave
and its spread are columns 0 and 7 of the Candidate's non-lag layout, un-standardised with the
fold's standardiser from float32 source rows, with the ENS-mean 2 m temperature. A second token
path `γ_t ⊙ (W2_t GELU(W1_t [p0_mean·ok, p0_sd·ok, ok]) / r_t)` enters the target tokens, a table
`c[primary]` multiplies `p0_mean` in the mixture means, and the head map reads `p0_mean` and
`p0_sd`. All of them start at zero with a random partner. Arm `HT` has 4,800 new parameters
(134,799 in all, 3.7% more than `M*`). With `a = 1` and `c = -1` the mean is the similar-day mean of
gross demand minus the forecast PV, the decomposition of experiment 064; `p0_sd` lets the head
widen the mixture when the ENS members disagree about the sky. The network, training recipe,
optimiser groups (1e-2 for `a` and `c`, 3e-3 for `γ_h` and `γ_t`), and known-answer checks are those
of experiment 716, with world WT and checks KF11 and KF12 for part 2.

## Counted contrasts and the decision rule

| Id | Contrast | Question |
|---|---|---|
| B1 | `HT` minus `H` | Does an explicit PV forecast in the target row add to the network that reads the historical estimate? |
| B2 | `HT` minus `HT-S` | Does the gain come from the forecast sky of the target day? |

Each part has four counted readings, so each interval is read at 98.75%. Because the two parts
share arm `H`, each page also reports its contrasts at 99.375% (eight readings across both parts) as a
labelled sensitivity reading that changes no verdict. **Report-only, at 95%:** `HT` minus `M*`, `T`
minus `M*` (the reading that compares directly with experiments 501 and 644), `HT` minus `T`, every
arm minus the Candidate, every contrast at all 16 Primaries, by lead band, by Primary, with one
Primary left out, and with one month left out. The `supported` guards are those of experiment 716,
except the mask guard, which belongs to part 1. Secondary readings S1, S2, and S4 (the solar block of the sunny
folds, and no harm in the dark folds) apply to B1 and B2 and change no verdict.

## Leakage, compute, and plots

**The target-row forecast is buildable live.** The ENS run of the origin day is issued 9 hours
before the 09:00 UTC native origin, and the experiment asserts `nwp_init_time <= origin - 9 h` for
every row and for the PV table. A poisoned-input check sets every target-row value from a run
issued after the native origin minus 9 hours to 1e6 and to NaN and requires every assembled input
to stay unchanged. **Because `H` carries the lab shortcut on history weather, every score is a
ceiling for a live forecast.** Part 2 needs about 8 GPU-process-hours (`HT` and `HT-S` on 8 folds
with 2 seeds, 2.6; seeds 2 to 4, about 3.8; `T`, 1.3), inside a reporting ceiling of 24 across
both parts. Part 2 has no screen, because the stop rule's requirement of scoring at every Primary
replaces any screening arm. The plots are those of experiment 716, with `HT` added to the
day-ahead and 14-day forecasts, and the learned `c` per Primary. No page names or plots a metered
generator's time series.

## Risks

- **The ENS cloud error defeated the explicit term in experiments 644, 655, and 661.** `HT` reads the
  same forecast. `p0_sd` and the zero-initialised gain let the network discount it.
- **A nested arm beating its container** (`H` beating `HT`) is audit trigger 5 of experiment 713.
- **The design cannot resolve a gain under 0.06 at the six**, although the smallest gain worth
  porting is 0.02.

## Method

The experiment is planned and has no fitted result yet, so these bullets describe the design.

- **Architecture and training regime:** experiment 716's, with the target-row paths of this page.
  Every arm starts stages 2 and 3 from the stored stage 1 state of `M*` of the same fold and seed.
- **Measurement:** as experiment 716, on 8 folds and 16 Primaries, with no screen.
- **Standing practices followed:** as experiment 716.

## Reviews

Two Opus design reviews preceded this page, and the second review's finding S7 changed one verdict
rule before any fit, as the decision rule states. Still to come: an Opus code review of every new
script before any run, a second Opus review of the fixed code, the known-answer gate, a result
review, an audit if a trigger fires, a write-up review, and a prose review.

## Files

- `ml-experiments/00717-msa-unmetered-pv-part-2-target-row-forecast/`: the scoring, figure, and
  report scripts of part 2. The network, batches, training, and predictions are in
  `ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/`, which part 2 imports.
- `data/ml-experiments/briefs/msa_architecture/f7_design.md` and its reviews (read, private).
- Written by this experiment (to be created): fitted states under
  `data/ml-experiments/00717-msa-unmetered-pv-part-2-target-row-forecast/work/`, never committed.
