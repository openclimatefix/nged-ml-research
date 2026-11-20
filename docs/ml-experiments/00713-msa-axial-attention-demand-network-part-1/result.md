---
id: "00713"
slug: msa-axial-attention-demand-network-part-1
title: >-
  Does an axial-attention neural network over a matrix of the last 14 days by 48 half-hours forecast
  Primary demand better than the Candidate, and does the gain come from the recent days?
theme: lagged-power-and-error-correction
also_relevant_to:
  - demand-inputs
  - production-reference-and-best-guess
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It plans about 160 fitted neural networks and 8 refitted Candidate chains, and stores
  their per-row forecasts; (2) more than one design defensibly tests the hypothesis, and the plan
  fixes one; (3) it copies training code from experiments 397 to 709, whose callers cannot be named
  without searching; (4) it touches no serving path or degradation rule; (5) it changes no
  contract. Full set of reviews: plan (an Opus design review ran before this commit), code before
  the first run, results, write-up, prose, and the audit triggers.
hypothesis: >-
  An axial-attention neural network over the 14 most recent days of power, weather, and date-time
  (a matrix of days by half-hours, with attention along both axes) forecasts Primary demand 2 to 10
  days ahead with a lower CRPS than the Candidate, because across-day attention can weight similar
  recent days and within-day attention can carry the shape of a day.
prediction: >-
  Written before any fit. H2 (the MSA network minus the placebo whose 14 history rows are random
  past days) is at most -0.10 points of CRPS at both groups of Primaries. H1 (the MSA network minus
  the Candidate) lies between -0.06 and +0.06 at the six signed-MW Primaries and between -0.16 and
  +0.16 at the eight MVA-only Primaries, so the predicted verdict is not supported (no detectable
  difference) or inconclusive. The prediction fails if H1 shows a gain or a loss beyond those
  ranges, or if H2 is above -0.10. Secondary prediction for the era hypothesis: the MSA network
  reads a lasting change and is vetoed on temporary moves.
decision_rule: >-
  Two counted readings, H1 (arm M* minus the Candidate C) and H2 (M* minus the placebo P), each at
  the six and the eight Primaries, so four readings, each read at 98.75% (Bonferroni across the
  four), with 5 seeds per arm and whole origin months as bootstrap blocks. Three guards stand beside
  the interval: a t interval on the 8 monthly differences at 98.75% with 7 degrees of freedom, the
  same sign for each of the 5 seeds, and a point estimate of at least twice the seed-noise standard
  error (0.034 at the six, 0.090 at the eight). The smallest gain worth porting is tau = 0.05
  points of CRPS, set above the protocol's 0.03 because the network replaces the whole network and
  its input pipeline. A group reads, in order: clear loss (lower limit above +tau, all guards),
  loss (lower limit above 0, all guards), gain (upper limit below 0, point estimate at or below
  -tau, all guards), small gain (upper limit below 0 and the seed guard holds for 4 of 5 seeds),
  no detectable difference (interval inside +-delta, delta 0.06 at the six and 0.16 at the eight),
  cannot tell. The verdict is the first that holds of: refuted (clear loss in either group), not
  supported (loss in either group, or no detectable difference in both), supported (gain in both
  groups, H2's upper limit below 0 in both, and the supported guards hold), partly supported (gain
  in one group with no loss in the other, or gain in both with H2 or a guard failing), small gain
  (small gain in a group with no gain and no loss), inconclusive. The era readings R1 to R3 are
  secondary and change no verdict. A not supported or refuted verdict closes the last-14-days
  axial line.
benchmark: >-
  The Candidate of experiment 667 (P7+D50): stored forecasts for seeds 0 and 1 of every fold, and
  seeds 2 to 4 refitted into a new output root; and the placebo network P.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS (twice the mean pinball loss over 11 quantile levels, in points of capacity) on leads of 24
  h up to but not including 240 h after the 12:00 UTC contract origin, with the normalised mean
  absolute error of the median beside it.
---

**Status: planned. No network has been fitted and no number below is a result.**

## What this experiment asks

**This experiment asks whether a neural network that reads the last 14 days of power as a grid of
days by half-hours forecasts a Primary substation's demand better than the Candidate demand network,
and whether any gain comes from the recent days themselves.** A Primary substation is one of 16
NGED substations whose demand the Candidate forecasts. The Candidate is the demand network of
[experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md)
(its recipe is called `P7+D50`). The Candidate
reads two lags of power (the newest same-half-hour reading and the newest same-weekday reading),
summaries of the neighbouring substations, and weather. The new network, called MSA after the
multiple sequence alignment that AlphaFold 2 reads, instead lays out the last 14 days as 14 rows of
48 half-hours, adds one row for the day being forecast, and lets the half-hours attend to each other
along both axes ([Jumper et al. (2021)](https://doi.org/10.1038/s41586-021-03819-2) introduced the
axial attention of AlphaFold 2). We train it with the Candidate's three-stage recipe, and we compare
it with the Candidate and with a placebo network whose 14 history rows are random past days.

**The maintainer named this experiment on 2026-11-20, and the idea has already failed several
times in pieces.** The ingredients of the idea have 18 earlier verdicts of `not supported` and 3 of
`refuted`. Attention or sequence networks over the power history alone have 5 verdicts of `not
supported` (experiments 334, 421, 428, 471, and 568) and 1 of `refuted` (experiment 632). The brief
that governs this work says to stop and count before proposing a fourth attempt. The count is
exceeded, so the experiment runs on the maintainer's decision, with the differences from the failed
attempts listed under "Why this is not a repeat".

**The honest expectation is a tie with the Candidate.** Experiments 585, 581, and 632 each found
that more history than a day lag and a week lag added little. A tie, a `not supported`, or a
`refuted` verdict closes the line of work on the last 14 days with axial attention.

**The history rows read ERA5 and CAMS weather, which the Candidate does not.** ERA5 is the European
Centre for Medium-Range Weather Forecasts (ECMWF) reanalysis of past weather, and CAMS is the
Copernicus Atmosphere Monitoring Service's solar estimate. ERA5 arrives about 5 days late, so 4 of
the 14 history rows could not be built in a live service. The maintainer chose this as an explicit
lab shortcut. Every comparison with the Candidate is therefore a ceiling for a live forecast, and
three prediction-only checks bracket the effect.

## Verdict

**There is no verdict yet.** The finished page replaces this section with the verdict word and its
numbers, after the take-home message.

## Pre-registered contrasts

**There are two counted contrasts, each at two groups of Primaries, so four counted readings, each
read at 100 × (1 − 0.05 / 4) = 98.75%.** Each contrast is the treatment minus the reference in CRPS
on the headline band, with seeds 0 to 4 of both arms. The six signed-MW Primaries are D5, D9, D10,
D11, D14, and D16. The eight MVA-only Primaries are D2, D3, D4, D8, D12, D19, D21, and D22, and
every number for them carries the label "reactive power assumed zero".

| Id | Contrast | Question |
|---|---|---|
| H1 | `M*` minus `C` | Does the MSA network beat the Candidate? |
| H2 | `M*` minus `P` | Do the 14 most recent days carry the gain (real minus placebo)? |

**Report-only readings, at 95%, on seeds 0 and 1 of both arms:**

- `M*` minus `A` (named H3: does the within-day attention matter?).
- `M*` minus `K`, `K` minus `N`, and `M*` minus `N`.
- `E` minus `M*`, and `G` minus `M*` when arm `G` is fitted.
- The pairwise differences among `M1`, `M2`, and `M3` on the screen folds.
- Every contrast for all 16 Primaries pooled, and for D15 with D18.
- Every contrast by lead band, by Primary, with one Primary left out, and with one month left out.
- The normalised mean absolute error (MAE) of the median beside every CRPS contrast.

## Arms

| Arm | What the arm is | Fitted or stored | Seeds |
|---|---|---|---|
| `C` | The Candidate `P7+D50` of experiment 667: two power lags, neighbour summaries, table T, mixture head | Seeds 0 and 1 stored; seeds 2 to 4 refitted into a new root | 0 to 4 |
| `M1` | MSA network; weather and date-time encoders loaded from the Candidate chain's pre-training and frozen in every stage | Fitted | 0, 1 (screen folds) |
| `M2` | As `M1`, with the loaded encoders trained in every stage | Fitted | 0, 1 (screen folds) |
| `M3` | As `M1`, with the encoders initialised at random and trained in every stage | Fitted | 0, 1 (screen folds) |
| `M*` | The arm among `M1`, `M2`, and `M3` with the lowest held-out validation loss | Selected, then fitted on all 8 folds | 0 to 4 |
| `P` | `M*` with its 14 history rows replaced by 14 random past days of the same Primary (the placebo) | Fitted | 0 to 4 |
| `A` | `M*` without the within-day attention (across-day attention and read-out kept) | Fitted | 0, 1 |
| `K` | `M*` with both trunk attentions replaced by the identity: the read-out and the head only, a learned kernel over the 14 same-half-hour readings | Fitted | 0, 1 |
| `N` | `M*` with the top row only: no history rows, no power | Fitted (about 2 minutes) | 0, 1 |
| `E` | `M*` with a 1-D convolution stem over the 48 half-hours of each history row; secondary | Fitted | 0, 1 |
| `G` | `M*` with a recurrent stem in place of the convolution stem; secondary and optional | Fitted only if the compute plan allows | 0, 1 |

**This page says "within-day attention" and "across-day attention" and not AlphaFold's "row
attention" and "column attention", because the two names are easy to swap.** Within-day attention
lets the 48 tokens of one day attend to each other. Across-day attention lets the 15 tokens at one
half-hour attend to each other across days.

**Removing the trunk's across-day attention does not give arm `N`.** The similar-day read-out
(Method, architecture) is itself a path across days, so a network whose trunk does nothing is still
a learned kernel over 14 same-half-hour readings with a weekday prior. Arm `K` is that network.
`M*` minus `K` says what the axial trunk adds to the read-out, and `K` minus `N` says what the
read-out adds to the top row alone.

**Reference rows, never refitted:** the month-by-half-hour climatology
(`baseline_forecasters.climatology.ClimatologyForecaster`, trained on each fold's training window),
and the manual heuristic and BG2 where stored rows cover the scored keys. The scorer states which
reference rows are missing.

## Decision rule

**Each group is read on H1, with D the point estimate and [L, U] the 98.75% bootstrap interval. The
group's reading is the first row that holds.** The three guards are the t guard (the t interval on
the 8 monthly differences of the seed-averaged forecasts, at 98.75% with 7 degrees of freedom,
excludes zero on the side the bootstrap interval does), the seed guard (the contrast has the same
sign for each of the 5 seeds taken alone), and the noise-floor guard (the point estimate is at least
twice the seed-noise standard error at 5 seeds, 0.034 at the six and 0.090 at the eight).

| Order | Group reading | Condition |
|---|---|---|
| 1 | clear loss | L > +τ, and all three guards hold |
| 2 | loss | L > 0, and all three guards hold |
| 3 | gain | U < 0, D ≤ −τ, and all three guards hold |
| 4 | small gain | U < 0, and the seed guard holds for at least 4 of the 5 seeds |
| 5 | no detectable difference | L > −δ and U < +δ, with δ = 0.06 at the six and 0.16 at the eight |
| 6 | cannot tell | anything else |

**The smallest gain worth porting is τ = 0.05 points of CRPS.** The protocol's highest tier is 0.03,
for a new head. The MSA network replaces the whole network and its input pipeline, and a live
service would need a replacement for ERA5 on the 4 newest history rows, so the threshold is set
above that tier. The experiment is `kind: other`, because the Candidate is not the Primary demand
research default, so the threshold belongs to this plan.

**The experiment's verdict is the first row that holds.**

| Order | Verdict | Condition |
|---|---|---|
| 1 | `refuted` | "clear loss" in either group |
| 2 | `not supported` | "loss" in either group; or "no detectable difference" in both groups |
| 3 | `supported` | "gain" in both groups, H2's upper limit below 0 in both groups, and every `supported` guard holds |
| 4 | `partly supported` | "gain" in one group and no "loss" in the other; or "gain" in both groups with H2's upper limit at or above 0 in a group, or with a `supported` guard failing |
| 5 | `small gain` | "small gain" in at least one group, with no "gain" and no "loss" in either group |
| 6 | `inconclusive` | anything else |

**Reading 4 follows the `ml-experiment` skill:** an interval that excludes zero in the predicted
direction but falls under the threshold, or is not secured at the threshold, is `small gain` and
never `inconclusive`. **Reading 5 uses the detectable difference δ and not τ**, because an interval
inside ±0.05 is out of reach with 5 seeds and 8 months. The write-up words reading 5 as "a
difference larger than 0.06 (0.16 at the eight) is ruled out; a gain of 0.05 is not".

**The guards for `supported`** are these four: the sign of H1 holds with each Primary left out in
turn and with each month left out in turn; the q01 and q99 pinball loss of `M*` is within 0.05 of
the Candidate's in each group (one-sided 95%); the normalised MAE of the median moves the same way
as CRPS; and H1 keeps its sign in swap (a) of "Three checks on history weather".

**H3, `M*` minus `K`, and `K` minus `N` change no verdict.** They say which part of the network did
the work, and they choose the follow-up.

**Stop rule for the line.** A verdict of `not supported` or `refuted` closes the last-14-days axial
line, so the 28-row layout (a follow-up with 14 target rows and 14 history rows) does not run. A
retrieval follow-up, which picks the 14 history rows by similarity to the target day, may still run
after `not supported`, but only when H2 is clearly negative and `M*` is not clearly worse than `C`,
because retrieval tests a different source of information.

**What the design cannot do.** τ is below the detectable difference δ in both groups. A true gain
of exactly τ will most often read `small gain` or `inconclusive`, and a tie can be read as "no
difference larger than δ", never as "no difference larger than τ".

## Detectable effect

**With 5 seeds per arm, the design can detect a difference from the Candidate of about 0.06 points of
CRPS at the six signed-MW Primaries and 0.07 to 0.16 at the eight MVA-only Primaries, and no
smaller.** The two stored seeds of the Candidate alone would resolve only 0.09 at the six and 0.11
to 0.25 at the eight, which is why the Candidate is refitted for seeds 2 to 4. A gain of the size
that experiments 699, 709, and 711 failed to resolve (0.03 to 0.07) may still end `inconclusive` or
`small gain` here.

**The calculation scores the Candidate's two stored seeds over the 8 folds, on the headline band,
with valid times from 2026-07-01 dropped.** The seed-noise standard error of a contrast between two
arms with n seeds each, over 8 months, is the monthly standard deviation of seed 0 minus seed 1
divided by √(8n). The detectable difference δ is 3.48 standard errors (99.17% two-sided, 80%
power). It assumes the MSA arm is as noisy as the Candidate.

| Group | CRPS of the seed-mean forecast, pooled over rows | The same, mean of the 8 monthly means | Mean over months of seed 0 minus seed 1 | Standard deviation of that difference over 8 months | Half-hours |
|---|---|---|---|---|---|
| Six signed-MW | 3.203 | 3.175 | +0.001 | 0.108 | 611,055 |
| Eight MVA-only | 3.704 | 3.694 | +0.059 | 0.285 (0.136 without 2025-11) | 816,768 |
| D15 and D18 | 8.106 | 8.002 | +0.127 | 0.974 | 203,040 |

| Seeds per arm | Standard error, six | Standard error, eight | Detectable δ, six | Detectable δ, eight |
|---|---|---|---|---|
| 2 | 0.027 | 0.071 (0.034) | 0.094 | 0.25 (0.12) |
| 5 | 0.017 | 0.045 (0.022) | 0.059 | 0.16 (0.07) |

**One seed of the Candidate failed in fold 2025-11 at the eight** (CRPS 4.19 against 3.43 for the
other seed), which doubles the eight's seed noise. The stored arm is used as it is. The figures are
lower bounds, because they leave out the month-to-month variation of the true difference, which the
month-block bootstrap and the t guard add. The decision rule uses δ = 0.06 at the six and 0.16 at
the eight. Known-answer check KA8 requires the implementer's scorer to reproduce the first table.

## Screen, and the choice of `M*`

**The screen follows the standing practice: four folds first, every number labelled interim.** The
screen folds are 2025-11, 2026-01, 2026-03, and 2026-06, with seeds 0 and 1, for `M1`, `M2`, and
`M3`. The Candidate's stored forecasts match the screen's folds and recipe exactly.

- **Choice of `M*`, without reading any scored forecast.** `M*` is the arm with the lowest mean
  held-out validation loss, where the loss is the unweighted mixture negative log-likelihood of the
  stage 3 state on the stage 3 validation samples (held-out origin days, all before the fold's
  cutoff, at lead days 1, 2, 4, 7, 10, and 14), averaged over the 4 screen folds and 2 seeds. An
  arm within 0.005 of the best loses to the simpler arm, in the order `M1`, `M2`, `M3`. The
  selection script reads the fit logs only and runs before any MSA forecast is scored.
- **Caveat on the selection.** A later fold's validation days fall inside earlier scored months
  (fold 2026-03 holds out days in 2025-11 to 2026-02). Those days are training-window data of the
  later fold, used as every early-stopping set in the recipe uses them. No scored forecast of any
  arm enters the selection.
- **Order of work after the selection.** Fit `P` on the screen folds; apply the sanity rule and the
  stop rule; then fit `M*` on the other four folds (2025-12, 2026-02, 2026-04, 2026-05), `P` on
  those four, `A`, `K`, `N`, and `E` on all 8, seeds 2 to 4 of `M*` and `P`, and arm `G` last if the
  compute plan allows.
- **Sanity rule.** If `M*` minus `P` is positive in both groups on the screen folds, stop and audit
  (audit trigger 1), because a network that gains nothing from real recent days over random days is
  either broken or indifferent to its history rows.
- **Stop rule.** Stop if `M*` minus `C` (seeds 0 and 1) on the screen folds is above +0.15 in both
  groups, or above +0.10 in both groups and positive in at least 3 of 4 folds in both groups. The
  result is then written up from the screen as `not supported`, labelled as a screen. The scorer
  prints the reading a second time with fold 2025-11 left out, because the Candidate's failed seed
  sits in that fold; the rule itself uses all four folds.
- **No tuning on the scored folds.** Width, depth, heads, and learning rates are fixed in the Method
  section. A change after the screen is labelled post hoc and leaves the pre-registered contrasts
  unchanged.

## Why this is not a repeat

| Earlier experiment | What it tested | What differs here |
|---|---|---|
| 632 (`refuted`) | A 1-D sequence of 672 tokens: two gated recurrent units and three joint self-attention layers over 7 days of history and 7 days of future, scored on ERA5 only, never trained on ENS | A 2-D matrix with the half-hours of each day aligned; attention split by axis; a read-out that starts as a weighted mean of similar days; the full 3-stage recipe on ENS; a placebo for the history; the Candidate as control |
| 428 (`not supported`) | A convolution-and-attention encoder over days of raw power, feeding a per-row network, built to stop the forecast following load shifts | The whole forecast day is one row of tokens; history rows carry weather and date-time as well as power, so attention can match days on weather |
| 568 (`not supported`) | Self-attention over the encoder outputs of one row (tokens are encoders, not times) | Tokens are half-hours of days |
| 709 (`inconclusive`) | An explicit era signal fed to the Candidate | No era signal is built: the 14 history rows are the only source of era information |

**The overlap with experiment 632 is large.** Experiment 632 also gave history tokens their weather
and date-time embeddings, also masked the newest 24 h of power, and found that three weeks of
history did no better than one. The differences that could change the result are the aligned
across-day attention, the similar-day starting point, and the ENS stages.

## Secondary: the era hypothesis

**Everything in this section is secondary. No reading here is counted among the four counted
readings, and no reading here can change the verdict.** Each interval is read at 95%. The set holds
about ten readings, so one false positive among them is likely, and a secondary result is a pointer
to a follow-up experiment and no more.

**The era hypothesis (maintainer, 2026-11-20): 14 full days at every half-hour, with attention
across days, let the neural network recognise the era a Primary is in and calibrate the forecast to
that era.** An era is a lasting state of a Primary's demand. At D8 the era is set by whether a gas
generator runs and whether a solar farm's output is present. Experiment 709 fed the Candidate an
explicit era signal and ended `inconclusive`. Its injected bank showed the trap: the arm that
lowered CRPS most after injected permanent changes (−2.9 points) raised CRPS by 3.9 points at the
donor Primary after injected temporary moves, because the arm carried a temporary move forward.

**Two arms serve the hypothesis, both fitted on `M*` with seeds 0 and 1 on all 8 folds.** Arm `E`
adds a 1-D convolution stem over the 48 half-hours of each history row, to detect rapid changes such
as the steps of a generator turning on. Arm `G` (optional) adds a small recurrent stem in its place
and is the first item dropped when the compute plan is tight.

| Id | Rows | Contrasts | Interval |
|---|---|---|---|
| R1 | D8 alone, real rows, headline band, 8 folds, seeds 0 and 1 | `E` − `M*`, `E` − `C`, `M*` − `C` (and the same three with `G` when fitted) | Month-block bootstrap at 95%, and the t interval on the 8 monthly differences at 95% with 7 degrees of freedom |
| R2 | Injected permanent changes (experiment 709's "eras"), scored against the injected power, forecast origins 7 to 28 days after the onset | `M*` − `C`, `M*` − `P`, `E` − `M*`, `E` − `P`; also `E` − `M*` for origins 0 to 7 days after the onset | Percentile bootstrap at 95% that resamples whole passes (a fold month and a pass index) |
| R3 | Injected temporary moves (experiment 709's "switching events"), at the donor Primary, scored against the power without the move, from the onset to 7 days after the end or the month's end | `M*` − `C`, `E` − `C`, `M*` − `P`, `E` − `M*` | As R2 |

**The injected worlds are experiment 709's stored bank, reused without a new draw.** The bank holds
16 passes for each of the fold months 2025-11, 2026-01, 2026-04, and 2026-06. Each pass injects two
permanent changes and two temporary moves on the first day of the scored month, so 64 passes give
128 events of each kind. The permanent kinds are a level step of −10% of capacity, −25%, and +15%,
and a solar-shaped reduction with a peak of 25% of capacity. A temporary move is a transfer built
with the generator of experiment 660, mirrored at an electrical neighbour, reverting after a
duration drawn from the verified events. D8, D15, and D18 are never hosts.

- **Prediction only.** The onset is the fold's cutoff day, so every fitted state was trained on the
  real power. The era script loads the stored states of `M*`, `E`, `P`, and `K` (and `G` when
  fitted), replaces the history power with the injected power, and predicts the host and donor
  Primaries of each pass. Nothing is refitted.
- **Reference `C` on the bank** is experiment 709's stored arm `none`, the Candidate without an era
  signal. The implementer first checks that `none` equals experiment 667's stored Candidate on the
  real rows of the four months. If the two differ, the bank reference is labelled "experiment 709's
  `none`" and the difference is printed.
- **Scoring** copies the slices of experiment 709's event losses: the event's CRPS is the mean over
  its rows and over seeds 0 and 1, against the injected share for a permanent change and the clean
  share for a temporary move. All 14 lead days are scored.
- **Real minus placebo.** Arm `P` reads history rows of ages 16 to 379 days, so for origins up to 15
  days after the onset no placebo row holds injected power, and `M*` − `P` is the effect of reading
  the injected days. For origins 16 to 28 days after the onset, at most 12 of the 364 pool days are
  injected, which the write-up states. A report-only slice, origins 17 to 28 days after the onset,
  is the slice in which all 14 history rows of `M*` lie after the onset.

**Thresholds, fixed now.** Experiment 709 gives the scale: after injected permanent changes its arm
`none` scores 10.75, its oracle lowers that by 7.4 points, and its placebo moves by 0.06. At D8 the
Candidate's stored seeds give a CRPS of 7.18 on the headline band and a seed-noise standard error of
0.064 for a contrast with 2 seeds per arm.

| Reading | Holds when |
|---|---|
| "Reads the era" (R2) | `M*` − `C` ≤ −1.0 and `M*` − `P` ≤ −1.0 at origins 7 to 28 days after the onset, both intervals below zero |
| "The stem adds after a permanent change" (R2) | `E` − `M*` ≤ −0.30 with the interval below zero, at 0 to 7 days or at 7 to 28 days; both are printed |
| "Harm after a temporary move" (R3) | Arm − `C` above +0.10, or the interval's upper end above +0.20 |
| "The stem worsens the harm" (R3) | `E` − `M*` above +0.10 with the interval above zero |
| "Gain at D8" (R1) | The contrast ≤ −0.25, both intervals below zero, the same sign for seed 0 and seed 1, and the same sign with fold 2026-01 left out |
| "Loss at D8" (R1) | The contrast ≥ +0.25, both intervals above zero |

**Prediction for the era set, written before any fit:** `M*` reads the era and is vetoed on
temporary moves, because the similar-day read-out copies recent levels by construction and no input
tells the network that a neighbour gained what the donor lost. `E` − `M*` is inside ±0.30 in R2.

**What each outcome says.** If `M*` or `E` reads the era and neither is vetoed on temporary moves,
the injected evidence supports the hypothesis, and the next step is a confirmation with 5 seeds and
new passes. If the same arm is vetoed on temporary moves, the network follows any level change and
does not tell a lasting change from a temporary move; the write-up then prints the mean of the two
changes, as experiment 709 advises. If neither arm reads the era, the hypothesis is not supported
as stated. If no reading at D8 passes, the design says nothing about D8, because both of D8's
register connections fall before 2025-11, so no scored origin sees an era begin there.

**Limits.** D8's scored rows contain no era onset. The injected kinds exclude a generator that
steps on and off within a day. The bank has four fold months and 2 seeds. A veto on temporary moves
is a reading on the donor Primary only.

## Known-answer checks and the placebo

**Every threshold below is fixed now.** The check script writes `work/known_answers_passed.json`,
and the fitting script refuses to run a real fold without that file. The synthetic-world fits call
the training loop directly and bypass the gate. A failed check fires audit trigger 1.

**The synthetic world keeps the real grid, the real ERA5 weather, and the real calendar of four
Primaries, and replaces the power with a known function.** In shares of capacity,
`y(p, t) = 0.5 + 0.02 × max(0, 15.5 − T(p, t)) + w(day type of t, half-hour of t) + shift(p, t) +
noise`. `T` is the ERA5 2 m temperature in °C. `w` is a fixed table of 8 × 48 values drawn once from
a uniform distribution on −0.10 to +0.10 (seed 0), keyed on the per-token local day type and the
local half-hour. The noise is Gaussian with standard deviation 0.01. `shift` is zero in world W0. In
world W1 each Primary has level steps of +0.30 or −0.30 lasting 10 days, starting at random days
that cover 10% of the history, in the power and so in the inputs and the targets. The training uses
stage 1 with at most 15 passes and ERA5 as the target row's weather.

| Check | What must hold | Threshold |
|---|---|---|
| KA1, untrained read-out | The median of an untrained neural network equals `base` (the weighted mean of similar days) computed independently in NumPy from the same batch, for `M1`, `E`, and `G` | Maximum absolute difference ≤ 1e-5 |
| KA2, weather and day-type function | In W0, arm `N` after training has a mean absolute error of the median on held-out origins | ≤ 0.02 of capacity, twice the noise |
| KA3, real history recovers a shift | In W1, the follow share on target tokens inside a step that began at least 3 days before the origin (the trained median on the W1 inputs, minus the trained median on the same inputs with the shift removed from the history power, divided by the step, averaged over the qualifying tokens) | `M` ≥ 0.75. Wiring checks that cannot fail on a correct build: `P` ≤ 0.25, `N` exactly 0 |
| KA4, no harm from history | In W0, `M` minus `N`, mean absolute error of the median | ≤ +0.005 |
| KA5, poisoned power | Setting every power reading with grid index above `origin_idx − 48` to 1e6, and in a second run to NaN, in a copy of the arrays leaves every model output unchanged, for 64 samples per fold and every test origin, for every arm with history rows | Bit-identical; and the negative control (cap 0) must change the output or raise |
| KA6, future weather canary | A target row built from the ENS run of day D + 1 must fail the origin-time assertion | The assertion raises |
| KA7, row layout | The grid indices and per-token day types of 1,000 random samples equal an independent calendar computation with `datetime` and `zoneinfo`; the masked columns of the age-2 row are exactly 43 to 47 | Exact |
| KA8, stored Candidate | The scorer reproduces the detectable-effect table and the Candidate's export from the per-seed files | 3 decimals; 1e-4 MW or MVA |
| KA9, placebo draw | No placebo row is younger than 16 days before the origin, the draw is identical on a second call, and no placebo row repeats inside a sample | Exact |
| KA10, no recency signal where none exists | In W0, the absolute difference between `M` and `P` in mean absolute error of the median. A leak of target-time information through the history rows would make `M` beat `P` | ≤ 0.005 |
| KA11, injected bank | The injected power equals the cleaned power outside the injected hosts and windows; experiment 709's `none` event losses are reproduced | 1e-6 of capacity; 3 decimals |
| KA12, stem on a step | In W1 with steps that begin at a random half-hour, arm `E` (and `G`) has a follow share at least that of `M` minus 0.05 | Report the value; a lower value fires audit trigger 1 |

**In the placebo `P`, the 14 history rows of each sample are 14 different 24-hour blocks of the same
Primary, drawn uniformly from ages 16 to 379 days before the origin.** Each placebo row keeps the
12:00 UTC start and carries its own power, its own ERA5 and CAMS weather, its own date-time columns,
and its own per-token day types, so the row is a true past day. The rows are sorted newest first and
take the age embedding of their slot (2 to 15), so `P` has the parameters of `M*` exactly. The draw
is a fixed function of (Primary, origin day, placebo seed 0), the same in training and at
prediction, and the same for every network seed. H2 therefore measures what the 14 most recent days
add over 14 random days of the past year. The placebo still knows the Primary's level and daily
shape from a year of history.

## Leakage statement

**No power reading later than 27 h before the contract origin enters any arm.** The history rows
read ERA5 and CAMS weather at the token's own time, which is an explicit lab shortcut, and the
target row reads only ECMWF ensemble (ENS) forecasts from the run issued at or before the forecast
origin.

| Item | Source | Why no information after the origin reaches the forecast |
|---|---|---|
| Target row, weather mean and spread | ENS 00Z run of day D (51 members) | Issued 9 h before the native origin. The existing build asserts only `nwp_init_time ≤ power_fcst_init_time`, so this experiment adds its own assertion on every row: `nwp_init_time ≤ o − 9 h` |
| Target row, B1 columns | Trailing windows of the same ENS run, and CAMS of the last complete UTC day before the origin | The Candidate's own inputs |
| Target row, power | None: a learned token | No power enters the target row |
| Target row, date-time and day type | Calendar and solar geometry | Deterministic |
| History rows, power, and the stems of `E` and `G` | Cleaned power | Visible only at grid index ≤ `origin_idx − 48` (27 h before the contract origin); asserted on every batch and by KA5 |
| History rows, weather and B1 | ERA5 and CAMS at the token's own time | Lab shortcut (below) |
| History rows, date-time and day type | Calendar | Deterministic |
| Placebo rows | Blocks of ages 16 to 379 days | All before the origin; KA9 |
| Standardiser, target mean and standard deviation, B1 statistics | The Candidate chain's stored fold statistics | Computed on the fold's training region only |
| Loaded encoders (`M1`, `M2`) | Pre-training of the Candidate chain | Trained on rows before the fold cutoff |
| Choice of `M*` | Stage 3 validation loss in the fit logs | Validation days lie before each fold's cutoff |
| Injected bank | Experiment 709's stored worlds | Prediction only; the same 27 h rule applies to the injected power |

**The lab shortcut on history weather.** The history rows read ERA5 and CAMS, which arrive about 5
days late, so 4 of the 14 history rows (ages 2 to 5) could not be built in a live service. Rows of
ages 6 to 15 could. The target row is not affected. H2, H3, and the contrasts among MSA arms are not
affected either, because every MSA arm reads the same ERA5 history. Every H1 score is a ceiling for
a live forecast.

**The newest usable power is stamped 09:00 UTC on day D − 1, 27 h before the contract origin.** The
Candidate's lag code sets the newest visible grid index to the native origin minus 48 half-hours (a
24 h power cap), and the Candidate's script asserts the cap. The 24-hour block of age 1 therefore
holds no usable power and is not a row. In the newest history row (age 2), columns 0 to 42 are
visible and columns 43 to 47 are masked.

### Three checks on history weather

**No neural network is refitted for these checks; each one changes an input at prediction and
reports H1 again.**

- **Swap (a): ENS history weather, B1 kept.** The 7 weather columns of every history token come
  from the Candidate's expected-input table (the ENS run issued at or before the token's time). The
  14 B1 columns stay as ERA5 and CAMS give them.
- **Swap (b): ENS history weather, B1 zeroed.** As swap (a), with the 14 B1 columns of every history
  token set to zero, which is how the Candidate reads its lag times.
- **Swap (c): the Candidate with ERA5 at its lag times.** The stored Candidate states predict with
  the ERA5 table attached in place of the ENS table. The forecasts are written into this
  experiment's data folder, and nothing in experiment 667's folder is changed.

**The `supported` guard uses swap (a) alone**, because swap (a) changes one input. Swap (b) minus
swap (a) is the effect of the B1 columns, and swap (c) says how much the Candidate would gain from
the same shortcut. A neural network trained on one weather source and tested on another is outside
its training distribution in all three, so a small loss is expected and is reported, not tested. An
MSA arm trained with ENS history weather is a follow-up.

## Compute

**One fit is one chain of three stages for one arm, one fold, and one seed.** Stage 1 (ERA5 and
CAMS, 2019-09 to the cutoff, about 19,000 pairs of Primary and origin day per pass) takes 4 to 10
minutes on the GPU, stage 2 (ENS, the 730 days before the cutoff) 2 to 4 minutes, stage 3 1 to 2
minutes, and prediction (31 origins × 16 Primaries × 14 lead days) under 1 minute. The estimate
comes from experiment 632, whose sequence network read 672 tokens per sample at width 64 in 5 to 12
minutes and 2.3 GB of GPU memory. Axial attention compares 45,360 token pairs per sample and block,
against 451,584 for joint attention over 672 tokens.

| Phase | Fits | GPU-process-hours |
|---|---|---|
| 1. Screen: `M1`, `M2`, `M3` × 4 screen folds × seeds 0 and 1 | 24 | 4.8 |
| 2. `P` × 4 screen folds × seeds 0 and 1 (before the sanity rule and the stop rule) | 8 | 1.6 |
| 3. `M*` and `P` × the other 4 folds × seeds 0 and 1 | 16 | 3.2 |
| 4. `A` × 8 folds × seeds 0 and 1 | 16 | 3.2 |
| 5. `K` × 8 folds × seeds 0 and 1 (about 6 minutes each) | 16 | 1.6 |
| 6. `N` × 8 folds × seeds 0 and 1 (about 2 minutes each) | 16 | 0.5 |
| 7. `E` × 8 folds × seeds 0 and 1 | 16 | 3.2 |
| 8. The Candidate chain, seeds 2 to 4, 8 folds, into a new root | 8 chains | about 6 |
| 9. `M*` and `P` × 8 folds × seeds 2 to 4 | 48 | 9.6 |
| 10. Injected bank of the era set: 64 passes × 2 seeds × `M*`, `E`, `P`, `K`, host and donor Primaries only | 0 | about 2.5 |
| 11. Swaps (a), (b), and (c), the all-history-masked read-out, experiment 581's bank | 0 | about 1 |
| 12. Known-answer checks, timing runs, and reruns at the 120-pass cap | about 12 | about 2 |
| Total without arm `G` | 160 MSA fits and 8 Candidate chains | about 39 |
| 13. Optional: `G` × 8 folds × seeds 0 and 1, and its bank passes | 16 | about 4.4 |

**Ceiling: 42 GPU-process-hours without arm `G` and 46 with it; the session stops and asks the
manager before crossing either.** At 3 concurrent processes the plan is about 14 to 16 hours of
wall-clock time. When the plan is tight, the cuts run in this order: drop arm `G`; then run arm `G`
or arm `E` on the four bank months only, which keeps R2 and R3 and loses R1; then drop arm `K`'s
bank passes. Phases 1 to 3, 8, and 9 carry the counted readings and are never cut.

**The slot is slot B: 4 cores, 18 GB of GPU memory, and at most 3 concurrent GPU processes, each
under `MemoryMax=12G` with 1 thread.** The fourth core runs scoring and checks one at a time. Each
process runs through a `systemd-run` wrapper copied from experiment 710, with the BLAS and Polars
thread counts set to 1. The first fit prints its host and GPU peaks, and a GPU peak above 5 GB
halves the batch for every arm before any further fit. Each job has a fixed seed and writes its
output atomically, so a job whose output exists is skipped and the result does not depend on the
schedule. The session checks in with the manager before the first fit and before phase 8.

### What is reused, and what is refitted

- **The Candidate's forecasts for seeds 0 and 1 are reused.** They are the per-seed forecast files
  of the 8 folds in experiment 667's folder, 16 chains of about 0.25 GPU-hours each, so about 4
  GPU-hours are saved. The recipe matches: monthly cutoffs, cleaned power, the D15 and D18 cut, the
  ENS 00Z run, the 24 h lag cap, and no widening. KA8 checks the files against experiment 667's
  export before use, and the scorer prints each file's modification time and hash.
- **The Candidate chain is refitted for seeds 2 to 4, into a new output root (approved in principle
  by the manager).** Experiment 667's chain script hard-codes seeds 0 and 1, and a merged script is
  never edited, so a copy in this experiment's folder runs seeds 2, 3, and 4 with its output root
  set to this experiment's data folder and the memory cap at 12 GB. The copy states its source and
  the lines changed. Nothing under experiment 667's folder is written: the implementer lists the
  modification times of that folder's chain and fold directories before and after the first chain,
  and asserts they are unchanged. The first chain also reruns seed 0 of one fold into the new root
  and compares the forecast with the stored seed 0. A maximum absolute difference above 1e-3 of
  capacity fires audit trigger 1, because it would show that the copied wrapper does not reproduce
  the stored recipe. This refit costs about 6 GPU-hours and may run before any MSA fit, because it
  depends on none.
- **Other stored inputs are reused:** experiment 581's and experiment 709's injected-bank forecasts
  of the Candidate, experiment 709's injected worlds, the Candidate's pre-trained encoders and fold
  statistics, and the extended inputs of experiment 667.

## Checks

**These checks run when the experiment runs; none has run yet.** The finished page lists their
results here, with the median minus the mean forecast per Primary (the prediction files carry the
mixture mean), the MAE contrast beside each CRPS
contrast, and which of the seven audit triggers fired. Audit trigger 5 applies: `N`, `K`, or `A`
beating `M*`, `P` beating `M*`, or `M*` beating `E` or `G`, is a nested arm beating its container.

**A report-only token diagnostic runs on every fitted network.** For each fitted network, on 256
real samples of one fold at lead days 1 and 7, `e00713_token_diagnostic.py` records three measures
in its own file, `work/token_diagnostic/<arm>_<month>_s<seed>.json`. The first is the length of each
part of a token: weather, date-time, Primary, time of day, lead, ensemble spread, day age, and power.
The second is how far the forecast moves with history power, split into the part through the
read-out base and the part through the trunk. The third is the distance between the masked-token
vector and the vector of a token with the power at its Primary's mean level. The diagnostic changes no
training, no forecast, and no verdict, and it is declared here before the first fit.

**Reviews, in order:** an Opus design review ran before this plan was committed (triage in the pull
request). An Opus code review of the layout, model, adapter, batch, training, scoring, and Candidate
seed scripts follows, before any fit. After the runs come a result review, an audit if a trigger
fires, the write-up, and a prose review.

## Method

**This section describes the planned design, so it is written in the future tense where a fact
depends on a fit.** Every setting is fixed now and none is tuned on the scored folds.

### Architecture

- **The sample is one Primary, one origin day, and one lead day k (1 to 14).** The matrix has 15
  rows of 48 half-hours: row 0 is the target row (valid times `c + 24 (k − 1) h + 0.5 j h`, where
  `c` is the contract origin) and rows 1 to 14 are the history rows of ages 2 to 15 days, newest
  first (valid times `c − 24 a h + 0.5 j h`). That is 720 tokens per sample (48 for arm `N`).
- **Rows are 24-hour blocks that start at 12:00 UTC**, so 14 target rows tile the 672 contract
  half-hours exactly, and column `j` of every row has the same time of day. A half-hour is stamped
  with its period end. The contract origin is 12:00 UTC on day D, the native origin is 3 h earlier
  (09:00 UTC), and the weather run is the ECMWF ENS 00Z run of day D, available at 09:00 UTC.
- **A row holds two local calendar days**, so the day type is set per token, not per row: the local
  weekday (0 to 6, Europe/London) of the token's own half-hour, or 7 where that local day is a bank
  holiday in England. Columns are UTC half-hours, so a history row on the other side of a clock
  change is offset by one hour of local time, which the write-up states as a limitation.
- **Every target token has exactly two history tokens of its own weekday in its column**, because
  the history token of age a is 24 (a + k − 1) h before the target token, and that is a multiple of
  7 days for two ages in 2 to 15.
- **The inputs per batch of B samples** are: `top_weather` (B, 48, 21), 7 ENS-mean weather columns
  and 14 B1 columns (B1 is the set of 14 trailing-temperature and solar-gain columns of experiment
  569); `top_spread` (B, 48, 7), the ENS spread (zero in stage 1); `top_datetime` (B, 48, 21), 21
  solar-geometry, calendar, and holiday columns; `hist_weather` (B, 14, 48, 21) and `hist_datetime`
  (B, 14, 48, 21) at the token's own time; `hist_power` (B, 14, 48), power over capacity
  standardised per Primary with the fold's target mean and standard deviation, clipped at ±5, and 0
  where not visible; `hist_visible` (B, 14, 48); `row_exists` (B, 14); `day_type_top` (B, 48) and
  `day_type_hist` (B, 14, 48); the Primary index and `lead_day` (B,); and the `target` and `weight`
  (B, 48). The weather columns, B1 columns, and their standardisation are the Candidate's.
- **Token construction.** Width is 64. The weather encoder (29 → 64 → 32) and the date-time encoder
  (29 → 64 → 32) are the Candidate's, each reading an 8-wide Primary embedding. Every token is
  `P_w e_w + P_d e_d + tok_emb(primary) + col_emb(j) + row_part`, where `P_w` and `P_d` are linear
  maps 32 → 64 without bias, `tok_emb` is a new 16 × 64 Primary embedding, and `col_emb` is a
  learned 48 × 64 time-of-day embedding. The target row's `row_part` is a lead-day embedding (14 ×
  64), a map of the 7 spread columns (initialised at zero), and a learned masked-power token. A
  history row's `row_part` is an age embedding (14 × 64) plus `u z + b` where the token is visible
  (`z` is the standardised power) and a learned vector where it is not.
- **Two axial blocks**, pre-LayerNorm, residual, dropout 0.1. Each block applies within-day
  attention (4 heads, width 64, over the 48 tokens of each row), then across-day attention (4
  heads, over the 15 tokens of each column, with a learned bias per head and key row, and rows with
  `row_exists` False excluded as keys), then a transition network 64 → 128 → 64 with GELU (the
  Gaussian error linear unit). The output projection of each attention and the last layer of each
  transition start at zero, so an untrained trunk is the identity.
- **The similar-day read-out** is an addition to the maintainer's sketch, made so that an untrained
  network gives a weighted mean of similar days, and the maintainer kept it. For the target token at
  column `j`, a logit for each visible history row is a scaled dot product of two 64 → 32 maps of
  the layer-normalised tokens, plus a table `β_day[day type of the target token, day type of the
  history token]` (8 × 8) and a per-age bias `β_age`. A softmax over the visible rows gives weights
  `α`, a weighted mean `base`, and a weighted spread `s`; with no visible row, `base = 0`, `s = 0`,
  and a flag `no_history = 1`. The maps and `β_age` start at zero, and `β_day` starts at 3.0 for the
  same day type, 1.5 for two different Monday-to-Friday day types and for Sunday with a bank
  holiday, and 0 otherwise. A Monday-morning target token therefore starts with about 50% of the
  weight on the two Monday mornings, 45% on the eight other working-day mornings, and 5% on the
  four weekend mornings.
- **The head is the Candidate's mixture.** A network `[h_0 (64), base, s, no_history] → 64 → 9`
  gives 3 mixture weights, 3 means, and 3 log standard deviations, initialised as the Candidate's
  head is (output weights zero, means at −0.5, 0, and +0.5), with `base` added to the 3 means. The
  loss is the mixture negative log-likelihood on the standardised target, and the 11 quantiles
  (0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99) come from the mixture, then are
  rescaled by the target standard deviation and mean into shares of capacity. No widening factor is
  applied to any arm. KA1 asserts that an untrained network forecasts a 3-component mixture centred
  on the day-type-weighted mean of the last 14 days at the same half-hour.
- **Parameter count.** The built module is about 128,655 parameters (the encoders 8,128, the token
  maps and embeddings 10,688, two axial blocks 100,600, the read-out 4,302, and the head 4,937);
  `M1` trains 120,527 of them. Arm `A` has 33,536 fewer parameters than `M1` (33,280 attention
  weights and 256 in the two LayerNorms of the removed within-day sublayers), and arm `K` has 67,192
  fewer (66,560 attention weights, 120 key-bias values, and 512 in the removed LayerNorms). Both
  counts are those of the built modules.
  The Candidate's lag network has about 30,000 parameters. A check script asserts every count, and
  the count of the built module is the reference.
- **Arm differences.** `M1`, `M2`, and `M3` differ only in the encoders: `M1` and `M2` load the
  weather, date-time, and embedding weights from the Candidate chain's pre-training for the same
  fold and seed (trained on data before the fold cutoff), `M1` freezes them, and `M3` starts them at
  random. `A` replaces the within-day attention of both blocks with the identity, `K` replaces both
  attentions, `P` changes the batch only, and `N` builds the target row only (`base = 0`,
  `no_history = 1`).
- **Arms `E` and `G` add a stem** on the history rows, before the first block. The stem reads two
  channels per half-hour (the standardised power, 0 where not visible, and the visibility flag).
  Arm `E` uses two 1-D convolutions of kernel 5 with zero padding (2 → 16 → 16 channels) and a
  linear layer 16 → 64, 2,560 parameters, each output seeing 9 half-hours (4.5 h). Arm `G` uses a
  bidirectional gated recurrent unit of hidden width 16 per direction and a linear layer 32 → 64,
  4,032 parameters. The last layer of each stem starts at zero, so an untrained `E` or `G` equals an
  untrained `M*`. Both stems read only the history power and visibility, which KA5 poisons.
- **A free read-out:** every fitted `M*` is also scored with all history power masked at prediction,
  which shows how the forecast degrades when telemetry is missing.
- **No monthly offset knots.** The Candidate's offset tracks slow level changes; the history rows do
  that for the MSA network. Arm `N` has neither.

### Training regime

- **Data.** The 16 Primaries, with cleaned power (the half-hours the `substation_zero` rule flags
  are dropped and counted per fold) and the D15 and D18 rows before 2025-03-01 cut, everywhere:
  training rows, history rows, and actuals. Stage 1 uses ERA5 and CAMS weather estimates; stages 2
  and 3 use ENS forecasts. Calendar columns are present, as in the Candidate, which departs from the
  skill's advice of a control without them. Arm `N` and the climatology row show what the calendar
  and the weather give alone.
- **Three stages, as experiment 667.** Stage 1 trains on daily origins from 2019-09 to the cutoff
  with ERA5 and CAMS target weather, spread zero, targets cut at the cutoff minus 3.5 h, weight 1,
  and learning rate 5e-4 (2e-4 for the loaded encoders of `M2`). Stage 2 trains on daily origins in
  the 730 days before the cutoff (from 2024-04-01) with the ENS 00Z run, targets cut a further 3
  days, weight 1, and learning rate 2e-4 (5e-4 for the spread map). Stage 3 fine-tunes with
  targets cut a further 7 days (half weight for the 7 days before), weight 0.5 to the power of age
  in days over 90, learning rate 2e-4, and synthetic lag-only shifts of experiment 581 added to the
  history power and never to a target.
- **One stated departure: the trunk is kept from stage 1 into stage 2.** The Candidate attaches a
  fresh mixer in stage 2, because its fine-tuning mixer has more inputs. The MSA trunk has the same
  inputs in both stages apart from the spread map, so it keeps what 6 years of ERA5 taught it. A
  second departure is 300 warm-up steps before a constant rate, because of the attention layers.
- **A third departure: the MSA network trains on fewer target rows than the Candidate.** The matrix
  holds the target half-hours from the contract origin (12:00 UTC) to the 672nd half-hour after it.
  On fold 2026-03 the Candidate's training rows include 37,381 rows before the contract origin (leads
  0.5 to 2.5 hours) and 175,925 rows after the 672nd half-hour, and the MSA network never trains on
  them. The 730-day window of stages 2 and 3 is cut by origin day, where the Candidate cuts by valid
  time, which removes up to 14 origin days for the folds whose window starts after 2024-04-01 (2026-05
  and 2026-06). The scored rows are identical, so no scored number depends on this difference, and
  nothing leaks.
- **Shared by all stages.** AdamW with weight decay 1e-3 and gradient clipping at 1.0; batch of 256
  samples; at most 60 passes with patience 8; loss `(weight × nll).sum() / weight.sum()` over finite
  targets. A pass visits every (Primary, origin day) once, with one lead day drawn uniformly,
  seeded by (seed, pass). Block dropout masks every history power token on 25.5% of origin days and
  the newest history row on 4.5%.
- **Under-training is checked and not assumed away.** One lead day per pair means a pass shows one
  fourteenth of the target half-hours a pass of the Candidate shows. Every fit logs, per stage, the
  pass at which early stopping fired and whether the cap was hit. A stage that ends at the 60-pass
  cap is rerun from its starting state at 120 passes before any score of that arm is read, and the
  report lists every rerun. A rerun repeats the 60 passes of the first attempt, so a capped stage costs
  up to 180 passes of GPU time in total, and phase 12 of the compute plan covers those reruns.
  Drawing more lead days per pair is not adopted, because it changes the sample weights within an
  origin.
- **Embargo and validation.** Every training process runs with the cleaned power set to NaN from the
  fold month's start, and prediction runs in a separate process. Early stopping uses held-out
  origin days (7 days in every 42, with a purge of 15 days before and 9 days after), grouped by
  origin day so that the 14 lead days of one origin sit on one side. A training sample in the 6 days
  after a purge reads a held-out block's power as history; this touches early stopping and the
  choice of `M*` only, never a scored forecast.
- **Folds and refits.** One refit per month, with the cutoff at 00:00 UTC on the first day of the
  month and no refit inside the month. The synthetic shifts of stage 3 are drawn over a range
  that starts 16 days (768 half-hours) before the first target, so that the oldest history rows of
  the first targets also get shifts.
- **Seeds.** Seeds 0 to 4 as the arm table lists. A seed sets the initialisation, the pass order,
  the lead-day draws, and the dropout draws.
- **Saved.** The state after each stage, in this experiment's data folder, and the per-seed
  forecasts with the Candidate's columns plus `seed`, `actual`, `lead_hours`, and `nwp_init_time`.
- **GPU hours.** About 39 GPU-process-hours without arm `G` and about 44 with it (the Compute
  section). The first fit times stage 3 of one fold once on the GPU and once on one CPU thread, and
  the finished page records both.

### Measurement

- **Scored rows.** The 8 monthly folds 2025-11 to 2026-06, one contract origin per day at 12:00 UTC
  (672 half-hours per origin), under the row set id in the front matter. Valid times on or after
  2026-07-01 are dropped for every arm, because that period is kept for a finalist. The four folds
  2026-07 to 2026-10 of experiment 667 are not used.
- **Actuals.** Cleaned power divided by the capacity in the series-capacity table. A join assertion
  checks that every arm holds identical (series, forecast origin, valid time) keys before any score
  is read.
- **Headline band.** Leads from 24 h up to but not including 240 h after the 12:00 UTC origin
  (target rows 2 to 10). Day 1 (0 to 24 h), days 11 to 14, and the protocol's days 1 to 10 band
  (0 to 240 h) are reported beside it.
- **Groups.** The six signed-MW and the eight MVA-only Primaries, separately. All 16 pooled, and D15
  with D18, are report-only.
- **Metrics.** CRPS, the normalised MAE of the median, coverage and pinball loss at q01, q05, q95,
  and q99, q05 to q95 coverage and mean band width, threshold-weighted CRPS, and the avoidable
  curtailment cost (report-only), by lead band, by Primary, and in the cold winter night, winter,
  and time-of-day slices. The climatology, manual heuristic, and BG2 rows stand beside them.
- **Intervals.** Each contrast uses seed-averaged forecasts (seed mean, then sorted quantiles, as in
  experiment 667), a per-row loss frame paired on run and valid time, and a month-block bootstrap
  with whole origin months as blocks and 100,000 resamples. The bootstrap pairs arms on seed, so
  each seed-averaged arm enters with one pseudo-seed, and the three guards carry the seed noise. A
  percentile bootstrap over 8 month blocks is too narrow: its "98.75%" half-width is about 2.3
  standard errors of the monthly mean where a t distribution with 7 degrees of freedom needs 3.3,
  and experiment 710's 99.17% interval excluded zero while a t test on the same monthly differences
  gave p = 0.077. The t guard uses the t quantile directly.
- **Load shifts.** Because the MSA network reads 14 days of power, the page reports four scenarios
  (all rows; rows outside events and at least 7 days after one; inside events; the 7 days after an
  event) for `M*`, `E`, `P`, `K`, `N`, `C`, and the climatology, the share of an injected shift that
  the median follows on experiment 581's bank beside the Candidate's stored bank forecasts, at least
  6 event plots across both groups, and the caveat that NGED's logs may miss events. Experiment
  632's sequence network followed 0.40 of a shift against 0.27 for the lag network, so the
  expectation is that `M*` follows more than `C`.
- **Figures.** A day-ahead forecast and a full 14-day forecast of one origin, winter and summer, at a
  signed-MW and an MVA-only Primary, for `M*` and `C`; the read-out weights by age and day type for
  `M*` and `P`, and before and after an injected permanent change; and a training-log table of the
  mean stopping pass per stage and the count of reruns at the cap.
- **Anonymisation.** Scores and forecasts are of Primaries. The page may say that a gas generator and
  a solar farm connect at D8, as experiment 709's page does. It never names either generator and
  never plots or tabulates a metered generator's own time series.
- **Standing practices followed:** monthly refits, cleaned data, the D15 and D18 cut, saved models,
  reused stored predictions, a four-fold screen, tails, example plots, load-shift plots, and a
  draft pull request. **Earlier advice followed:** experiment 710's leave-one-Primary-out reading,
  experiment 632's injected-shift read-out for any network that reads lags, and experiment 709's
  advice to score injected permanent changes and temporary moves together and against a placebo.
  **Earlier advice departed from:** experiment 710 advises selecting the fine-tune on a
  recency-weighted held-out set; this design keeps experiment 667's unweighted selection so that the
  MSA arms and the Candidate share one recipe.

## Risks

- **The matrix may add little over the Candidate's lags**, which already read the newest
  same-half-hour reading and the newest same-weekday reading, with averages of the gap to a
  lag-free forecast over 1, 3, and 7 days. The most likely result is a tie.
- **The design cannot resolve a gain smaller than 0.06 at the six.**
- **The Candidate has inputs the MSA network lacks** (neighbour summaries and table T), and the MSA
  network has inputs the Candidate lacks (ERA5 history weather, and 14 days of power against 2
  lags). H1 cannot separate those from the architecture.
- **The MSA network may follow load shifts more than the Candidate.** The follow share, the four
  scenarios, and R3 measure it.
- **About 129,000 parameters on 16 Primaries.** Stage 2 has about 10,000 (Primary, origin) pairs.
  Stage 1's long history, dropout, and early stopping are the protections, and `M1` against `M3`
  shows whether the encoders overfit.
- **Band width can pass for skill.** The MAE of the median and the mean band width stand beside
  every CRPS contrast.
- **The era set is small.** R1 is one Primary. R2 and R3 are four fold months, 2 seeds, and injected
  events that may not resemble a real era.

## Follow-ups, specified and not run

- **An MSA arm trained with ENS history weather**, so that the contrast with the Candidate is like
  for like and the arm could run in a live service.
- **Similarity retrieval at the same token budget:** the 14 history rows are the 14 blocks nearest
  to the target day among the 90 blocks of ages 2 to 91, with their true ages kept.
- **The 28-row layout:** 14 target rows and 14 history rows, 1,344 tokens, in one pass per origin.
- **Era follow-ups,** if the secondary set points to one: neighbour rows in the matrix, training on
  injected permanent changes and temporary moves together, and the three folds 2025-04 to 2025-06,
  in which D8's solar connection of 2025-03-24 falls inside the history.

## Files

**Planned code, in `ml-experiments/00713-msa-axial-attention-demand-network-part-1/`** (each script
carries the id as `e00713_`; none exists yet):

- `e00713_layout.py` is pure index arithmetic: sample to grid indices of the 15 rows, the visibility
  mask, per-token day types, and the placebo draw.
- `e00713_model.py` holds the network, the axial blocks, the two stems, the read-out, the head, and
  the arm switches.
- `e00713_inputs.py` adapts the Candidate chain's inputs: cleaned power, fold statistics, dense
  ERA5 and CAMS arrays, ENS target-row arrays, and the validation split.
- `e00713_batch.py` assembles batches and runs the per-batch assertions and the poisoned-copy check.
- `e00713_arms.py`, `e00713_train.py`, `e00713_fit.py`, and `e00713_predict.py` define the arms and
  run the three stages and the prediction; `e00713_select.py` chooses `M*` from the fit logs.
- `e00713_bank.py`, `e00713_swap_c.py`, and `e00713_era.py` run the prediction-only read-outs: the
  injected banks, the three weather swaps, and the era set.
- `e00713_known_answer.py` builds the synthetic world and runs KA1 to KA12.
- `e00713_token_diagnostic.py` is the report-only token diagnostic described under Checks.
- `e00713_score.py` and `e00713_report.py` compute every contrast, interval, and guard, and every
  number and figure of this page.
- `run.sh` is the memory-capped wrapper, and `chain_seeds.sh` is the copy of experiment 667's chain
  script for seeds 2 to 4.

**Data, never committed, under `data/ml-experiments/00713-msa-axial-attention-demand-network-part-1/`:**

- `work/models/<arm>/` will hold the saved state after each stage, and `work/fits/<arm>/` the
  per-seed forecasts (written by this experiment).
- `candidate_seeds/` will hold the refitted Candidate chains for seeds 2 to 4 (written by this
  experiment).
- `predictions/` will hold the immutable predictions, written with the experiment store.

**Earlier experiments' files read:** the stored Candidate forecasts, pre-trained encoders, fold
statistics, and extended inputs of
[experiment 667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md);
the injected worlds and the
`none` bank of [experiment 709](../00709-era-signal-in-the-candidate-part-1/result.md); the
injected bank of experiment 581; and the code of experiments 397, 422, 428, 510, 558, 559, 560, 572,
574, 578, 581, 606, 610, 632, and 710, imported without edits.
