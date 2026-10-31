---
id: "00429"
slug: unified-forecaster-wind-ablations
title: >-
  Which parts of a neural wind-farm forecaster earn their place: passing each ensemble member
  through the network, a physics power curve inside it, and a head that puts probability mass at
  zero and at full output?
theme: generator-forecasting
also_relevant_to:
  - weather-products-and-blending
  - calibration-and-uncertainty
status: planned
date: 2026-10-31
kind: other
protocol: wind-v1
size: >-
  complex. Stored: network weights, per-seed quantile tables, and 51-member weather caches under
  data/ml-experiments/00429-unified-forecaster-wind-ablations/ only. Serving path: not touched, but
  the member route needs 51 network passes per forecast and the page says so. Designs: several are
  defensible (the encoder, the pooling statistics, where the plant sits), so the plan fixes one of
  each before any score. Callers: this folder's scripts import experiments 230, 303, 310, 326, 336,
  and 347 and call no reviewed module beyond `studies.*`. Sonnet only: one plan review, one code
  review, one results review, one write-up review, one prose review.
hypothesis: >-
  A single pooled neural network can forecast the three metered wind farms as well as Wind farms
  research default 2 (experiment 347's y2: a frozen ERA5-trained power curve run on each of the 51
  ensemble members, followed by a mixture-density head). Each difference between that design and
  the Primary-demand design (experiment 425's arm A: one pass on the ensemble mean and spread, no
  physics, an unbounded head) is kept only if an ablation shows it is needed.
prediction: >-
  Written before any score. G0: the shared code reproduces 347's y2 within 0.05 points on the kept
  rows. A1: the member route beats one pass on the mean and spread by at least 0.10 points at true
  days 3 to 10 and cuts the 0.99 pinball loss by at least 5% (the design note's prediction; I
  expect it to hold, because experiment 242 found that running a curve once on the ensemble mean
  raised its 0.99 pinball loss from 0.658 to 1.378). A4: a plant arm does not beat the plain
  encoder by 0.10 points (experiment 313's ERA5 result was a tie); the frozen plant is the more
  likely tie than the trained plant. A7: censoring the likelihood at zero and at capacity lowers
  the share of observations above q99 from about 2.3% to 1.5% or less, and raises the score by no
  more than 0.02 points.
decision_rule: >-
  Reproduction gate G0 first: the shared code's y2 repeat must land within 0.05 points of 347's
  11.081 on the kept rows over every lead, and within 0.05 of 10.518 at true days 1 to 10; if it
  misses, stop and report. The maintainer agreed the margins on 2026-10-31: 0.10 points for wind,
  and no tail harm above 0.02. A difference is kept only if its 99% month-block interval (Bonferroni
  across the planned contrasts) excludes zero and the point estimate beats the margin, and the
  pooled 0.01 and 0.99 pinball losses rise by no more than 0.02 points. A plant arm that only ties
  is dropped, and the plain network is chosen (maintainer, 2026-10-31). An arm with an upper-tail
  failure (more than 1.5% of observations above q99 at true days 1 to 10) does not advance. Seeds: 5
  per arm; a contrast smaller than the minimum detectable effect printed in the report is called
  inconclusive, not null.
benchmark: >-
  Wind farms research default 2 (experiment 347's y2, stored rows), Wind farms research default 1
  (experiment 339's default), climatology (experiment 324's W0), and the frozen XGBoost on the
  ensemble mean (experiment 343, arms M and MS).
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline). Training: the daily ENS hindcast 2024-04-01 to 2025-04-15, early stopping on
  2025-04-16 to 2025-06-16, exactly as experiment 347 (so the sixth-month-wide replicate is not
  repeated).
metric: >-
  Quantile score over 11 levels, in points of capacity, over every lead in the row set and at true
  days 1 to 10 (lead 24 to 240 hours), kept rows. Tail: 0.01, 0.05, 0.95, 0.99 pinball loss and
  coverage per farm.
code: ml-experiments/00429-unified-forecaster-wind-ablations/
---

**Status: planned. No score exists yet.** This page is the plan, and the write-up replaces it.

## Question

The maintainer wants one neural design for the demand, wind, and solar forecasters, with each
difference between the three justified by an ablation. The Opus review that opens this programme
(`briefs/unified_forecaster_design_answer.md`, private) proposes a shared skeleton: a pooled
mixture density network (MDN) with a site embedding, a weather encoder, an optional physics power
layer, and a head. This experiment is the wind worker's part: a reproduction gate (G0), then
ablations A1 (member handling), A4 (physics layer), and the wind arms of A7 (bounded head), on the
three metered wind farms W1 to W3. A second worker runs the solar farms in a separate experiment
with its own id.

## Arms

All arms share one code path (`e00429_model.py`): a pooled network with an 8-wide learned farm
embedding, two hidden layers of 64, a three-component Gaussian mixture on the farm's output
fraction, and the 11-level quantiles read from the mixture's cumulative distribution (experiment
347's recipe). Inputs per case are the lead (sine and cosine of the hour of day and the lead in
hundreds of hours, as in 347) and the weather columns of experiment 303's member table (wind speed
at 10 m and 100 m, direction sine and cosine at both heights, and temperature). Wind farms in this
repository use no lagged power in the 347 head, so none is added.

| Arm | Member handling (D1) | Physics layer (D2) | Head (D3) |
|---|---|---|---|
| `G0` | Fixed summary of the plant's 51 powers | Frozen stage-3 plant, as 347 | Clipped |
| `one_pass_ms` | One pass on the mean and standard deviation of each column | none | Clipped |
| `one_pass_m` | One pass on the mean of each column | none | Clipped |
| `member_plain` | Shared encoder per member; mean, standard deviation, p10, p90 pooled | none | Clipped |
| `member_frozen` | As `member_plain` | Frozen stage-3 plant power as an extra encoder input and as the mixture's centre | Clipped |
| `member_e2e` | As `member_plain` | The stage-3 plant as an `nn.Module`, trained end to end from 347's parameters | Clipped |
| `member_<best>_censored` | As the better of the three member arms | as that arm | Censored at 0 and 1 |
| `ref_347_y2`, `ref_default`, `ref_clim` | stored rows | | |

`one_pass_ms` is the demand design moved to wind. Arms without a plant centre the mixture on the
mean fraction of the training cases, so C3 changes two things at once: the plant power as an encoder
input and the plant median as the mixture's centre. A tie in C3 cannot say which of the two failed
to help. The trainable plant in `member_e2e` learns at 2e-4 (the rest of the network at 2e-3) and
has no weight decay, so it starts from experiment 347's parameters. The member encoder is a
7-to-16-to-16 network applied to every member; the 51 outputs are pooled and fed to the trunk. The
design note suggests training on one random member per row to match the one-pass cost, but pooled
statistics need all 51 members, so the member arms use all 51 on the GPU and the cost is stated in
the write-up.

**The censored head** (A7) adds the exact likelihood of an observation at zero or at capacity:
for an observed fraction at or below 0.001 the likelihood is the mixture's mass below zero, and at
or above 0.999 it is the mass above one. This places point masses at 0 and 1 equal to the Gaussian
tails (the design note names point masses fitted by likelihood; the censored mixture is the
simplest form with the same effect). The clipped head leaves the likelihood alone and clamps the
quantiles, as 347 does.

**Carry-forward.** The censored arm runs on whichever of `member_plain`, `member_frozen`, and
`member_e2e` scores best at true days 1 to 10 (if none beats `member_plain` by the margin, on
`member_plain`), and the arm name states the choice. If `one_pass_ms` is within the margin of
`member_plain` (A1 fails), the A4 arms still run on the member route because a power curve needs
members (experiment 242); the write-up labels this.

Only the censored arm for the selected base is fitted, so there is one C5. C1 is judged on true
days 3 to 10 (lead 72 to 240 hours, kept rows), where the design note's prediction sits; the other
contrasts are judged on true days 1 to 10. The tail test is the change in the pooled 0.01 and 0.99
pinball loss at true days 1 to 10. Per-farm tail tables are in the report; the example-forecast
figures come from a separate script before the write-up.

## Added by the maintainer's decisions of 2026-10-31

- **Lead ranges.** Every table reports true days 1 to 10, the 12 to 36 hour band, and true days 1
  to 14 (lead 24 to 336 hours, the part of the row set that exists) side by side.
- **Training data.** Wind may use the one year of ENS hindcasts (experiment 303 argued against it;
  this experiment follows experiment 347 and uses it).
- **Coherence (A1).** NGED runs power-flow models one ensemble member at a time, so the forecast
  must give 51 members that are coherent across time and across the three farms. The script
  `e00429_coherence.py` scores the 51 member paths with the energy score and the variogram score
  (order 0.5) across W1 to W3 jointly, in two kinds of block: one run and one calendar day (at least
  36 shared half-hours), and one run's every twelfth lead across all leads. Four path constructions
  are compared: **ECC** (ensemble copula coupling: member k takes the arm's quantile at the rank of
  its own 100 m wind speed among the 51 at that farm and valid time); the **Schaake shuffle** (the
  51 hindcast runs nearest in day of year supply observed paths, and one historical date orders all
  three farms and all leads); **member-shifted** (member k keeps its own stage-3 plant power and
  adds the arm's median minus the plant's median; this stands for the member route using each
  member's own path); and **independent** (no coherence, the lower reference). Experiment 304 found
  member forecasts correlate 0.95 to 0.97 across farms while errors correlate 0.54 to 0.78, so the
  Schaake shuffle may beat ECC. The design note's prediction is that the member route beats ECC on
  the variogram score by at least 5% and on the marginal score by at least 0.10; the 5% coherence
  margin is the Opus note's proposal, not the maintainer's. The member-shifted construction keeps
  the plant's own spread and uses no learned encoder, so comparisons across arms mix in marginal
  differences. It is my reading of "the member route using each member's path", and the write-up
  says so. The stored default 2 is not path-scored because its stored rows are marginal only.

## Pre-registered contrasts

Each is the difference in kept-row quantile score at true days 1 to 10 and, beside it, over every
lead. All intervals are month-block bootstrap intervals (experiment 347's resampler), 99% level
with a Bonferroni correction over these five (level 0.998).

1. **C1 (A1):** `member_plain` minus `one_pass_ms`.
2. **C2 (A1):** `member_plain` minus `one_pass_m` (experiment 343's input choice).
3. **C3 (A4):** `member_frozen` minus `member_plain`.
4. **C4 (A4):** `member_e2e` minus `member_frozen`.
5. **C5 (A7):** the censored arm minus the same arm clipped.

Reference contrasts, not corrected: each arm against `ref_347_y2` and `ref_default`.

## Standing practices this experiment follows

- **Cadence:** one fit per seed on the fixed hindcast, as 347 (not a refit schedule; the wind
  protocol has no refit loop).
- **Reuse:** experiment 347's case frames, stored plant parameters and plant-power summaries,
  experiment 339's default rows, experiment 324's climatology, and experiment 343's rows are read,
  not refitted. The scored rows match (the same 53 Tuesday runs and the kept mask).
- **Seeds:** 5 per arm (0 to 4); the minimum detectable effect of each contrast is printed from the
  seed spread and the interval half-width.
- **Models saved:** each fitted network's state file under the experiment's data folder.
- **Tails and curtailment:** q01, q05, q95, q99 pinball loss and coverage per farm beside the score.
  No wind-specific curtailment-cost metric exists in the wind protocol, so none is reported.
- **Examples:** a day-ahead forecast and a full 14-day forecast for one farm, anonymised.
- **Not applicable:** cleaned Primary data, the D15/D18 cut, load-shift events.
- **GPU:** one process under 8 GB, one feeder core, `MemoryMax=8G`, `OMP_NUM_THREADS=1`. The fit
  slot is requested from the manager before the first fit.

## Plan review

Earlier advice read and followed: 347's reproduction recipe, 339's head, 343's input comparison,
and 242's finding about member handling. Departure: the design note's random-member training trick
is not used (see Arms).
