---
id: '00430'
slug: unified-forecaster-solar-and-pretraining-ablations
title: >-
  Which of four design differences does a pooled mixture-density network for the six metered solar
  farms need: ensemble members passed one by one, a PyTorch plant inside the network, ERA5 and CAMS
  pretraining, and point masses at zero and at capacity?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
- production-reference-and-best-guess
status: planned
kind: change
size: >-
  medium. Stored: fitted networks, forecasts, and score tables in this experiment's private folder
  only. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Designs: the arms follow the design note of experiment 13 (`briefs/unified_forecaster_design_answer.md`),
  and the choices it leaves open (how the member route is trained, what the pretraining rows hold,
  how the point masses are defined) are fixed in this plan. Callers: this folder's scripts, which
  import experiment 310's feature and refit modules and read experiment 345's stored forecasts.
  Asset graph: not touched. Sonnet writes and reviews; no Opus.
hypothesis: >-
  A pooled mixture-density network for the six metered solar farms needs, beyond experiment 345's
  solar-only pooled network (the demand-style default: one pass on the ensemble mean and spread, no
  physics, no pretraining, a clipped Gaussian mixture), only those of four switches that an ablation
  shows to help: the 51 ensemble members passed through a shared encoder one by one (A2), a PyTorch
  fleet plant that adds a power channel (A5), ERA5 and CAMS pretraining followed by fine-tuning on
  the ensemble (A6), and point masses at zero and at capacity in the output (A7).
prediction: >-
  Written before any fit and before any score is read. Scores are quantile scores in points of
  capacity; "better" means lower. Percentage changes are of the farm-equal mean score. The margins
  (1% for the solar score, no tail harm above 0.02 points at the 0.01 and 0.99 pinball loss) were
  agreed by the maintainer on 2026-10-31. G0. The shared code with every
  switch off reproduces experiment 345's `solar_pooled` network (five seeds, own history) to within
  0.5% over all bands and within 1.0% in every band. A2. The member route beats one pass by at least
  2% at days 3 to 10; experiment 224 found 1.8% with the plant, so this may fail. Coherence: the
  member route beats one pass with ensemble copula coupling by at least 1% on the marginal score or
  by at least 5% on the variogram score of the member paths (the 5% coherence margin is the Opus
  design note's proposal, not the maintainer's); paths that keep each member's identity score a
  lower energy score than independent draws from the same marginals. A5. A plant
  channel is at least 1.5% better than the plain encoder at days 6 to 10 and within 0.5% at day 1
  (experiment 313 found 3.4% at days 8 to 10). A6. Pretraining followed by fine-tuning beats both
  the ensemble-only and the ERA5-and-CAMS-only arms by at least 1%; experiments 303 and 275 argue
  against this. A7. Point masses leave the 0.99 pinball loss no worse than the clipped mixture by more
  than 0.02 points and raise the coverage of the 0.01 and 0.99 quantiles towards 1% without raising
  the quantile score by more than 0.02 points. Each prediction can fail.
decision_rule: >-
  Contrast = percentage change of the farm-equal mean quantile score against the stated reference,
  month-block resampling of the 12 scored months, 10,000 draws, 99% interval. A difference is kept
  only if the interval excludes zero on the favourable side, the point estimate beats the margin of
  its prediction, and the 0.01 and 0.99 pinball losses are no more than 0.02 points worse. Where the
  point estimate sits within the minimum detectable effect of a margin, the arms are refitted with a
  second seed set (seeds 3 to 5) and the contrast is read on all six seeds. If A2 shows one pass
  within the margin, the later ablations drop the member arms; the winner of each stage is carried
  forward and named in the arm names of the next. A physics arm that only ties the plain network is
  dropped (maintainer, 2026-10-31: the first production version optimises for simple code). G0
  gates every ablation: if it misses, the run stops and reports. A2 also reads the coherence of the
  ensemble members (energy and variogram scores of 51 paths against the observed vector) for each
  coupling method, beside the marginal quantile score. No arm is adopted by this experiment.
benchmark: >-
  Experiment 345's `solar_pooled` forecasts (five seeds, own history) as the gate reference;
  Metered solar farms research default 1 (experiment 310's blend) as the absolute reference; the
  one-pass arm of this experiment as the reference of A2, A5, A6, and A7 (or the stage winner).
row_set_id: >-
  Experiment 310's rows: farms A to F, scored year 2025-09 to 2026-08, day bands d1, d2, d3, d4_5,
  d6_7, d8_10, quarterly refits, rows that every compared arm holds with finite quantiles.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99, median minus mean forecast, seed spread
code: ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations/
---

*Status: planned. No fit has run and no score has been read.*

## Question

**We test which of four design differences a pooled network for solar farms needs.** The Opus
design note of experiment 13 proposes one network design for demand, wind, and solar: a mixture
density network (MDN) pooled across sites with a learned site embedding, fed the ensemble mean and
spread once, and optionally pretrained on ERA5 and CAMS. Each difference between the three
forecasters must be justified by an ablation. This experiment is the solar worker's half; the wind
worker's half is experiment 429. Here a "farm" is one of the six metered solar farms, labelled A to F.

## Design

**The four switches are small changes to one network class, and every arm differs from its
reference in exactly one switch.** The network is experiment 310's `MixtureHead`: the standardised
inputs, two hidden layers of 64, and a three-component Gaussian mixture on the capacity-normalised
output. All arms are pooled across the six farms and the six day bands, with experiment 345's 22
input columns (calendar, sun position, ensemble mean and spread of shortwave, beam, diffuse and
temperature, the clear-sky index, the lead in hours, and eight further ensemble variables).

- **Member route (A2).** A shared encoder (4 channels to 32 to 16) runs on each of the 51 members'
  shortwave, beam, diffuse, and temperature. The 51 encodings are pooled by mean, standard
  deviation, 10th and 90th percentile, and joined to the one-pass inputs. The member arm therefore
  sees everything the one-pass arm sees. All 51 members are used in every training step, which is
  cheap at this size, so the design note's one-random-member-per-epoch device is not needed.
- **Physics layer (A5).** A PyTorch copy of experiment 310's fleet plant adds a power channel (on
  the member mean for one pass, per member for the member route). "Frozen" fixes tilt 20 degrees
  (experiment 282 found 15 to 20), azimuth 180 degrees, and a scale fitted by least squares on the
  training rows; "end to end" also learns tilt, azimuth, scale, and clip. The plain arm adds the
  clear-sky irradiance as a column so that the arms differ only by the plant channel.
- **Pretraining (A6).** Pretrain on the farms' ERA5 and CAMS history before 2024-04-01 (about
  153,000 half-hours at five farms; farm E has none), then fine-tune all weights on the ensemble
  rows with half the learning rate. Columns the history lacks (eight ensemble variables and the
  spread) are 0 after standardisation, the lead is drawn uniformly from 12 to 252 hours, and every
  member is the same ERA5 value. One standardiser, fitted on ensemble rows before 2025-08-31, serves
  every refit of every A6 arm, including the ensemble-only control.
- **Coherent members (A2).** NGED's power-flow modelling runs one ensemble member at a time, so
  the forecast must give 51 member paths that are coherent across time and across the six farms.
  Four routes give paths. (i) The one-pass forecast with ensemble copula coupling (ECC): the
  member with the j-th lowest shortwave at a farm and half-hour takes the j-th of 51 quantile levels
  of that half-hour's forecast. (ii) The same marginals with a Schaake shuffle: the 51 members are
  ranked in each dimension by the ranks of 51 observed past paths of the same layout (the same
  farm and offsets from the run, on days within 25 days of the same date one year earlier; a
  dimension with no history is ranked at random). (iii) The member-wise route: each member is
  passed alone through the network, which is trained on one randomly drawn member per row per epoch
  with the spread input set to zero, and path m is the member's quantile function at one level
  that is fixed for the member across all times and farms (the 51 levels are (j - 0.5) / 51 in a
  fixed random order). (iv) The independent draw, which permutes the quantile levels at random in
  each dimension and is the floor. The coherence scores are the energy score and the variogram score
  of order 0.5 (over same-farm half-hour pairs up to 6 hours apart and same-time pairs across farms)
  of the 51 paths against the observed path, for the six farms and all leads of one run jointly, and
  for each day band alone.
- **Point masses (A7).** The head adds two point masses, at zero and at the farm's output ceiling
  (the 0.995 quantile of its training output, as experiment 310), and truncates the Gaussians to the
  interval between them. The likelihood is exact, and quantiles invert the truncated mixture.

**The stages run in this order, and each uses the winners of the stages before it.** G0, then A2,
then A6, then A5, then A7. The wind arms of A6 follow in a second stage once the wind worker's shared
code exists (the ENS-hindcast training rows for wind come from experiment 303's pipeline), and this
page is extended in the same pull request before those fits start.

| Stage | Arms (all three seeds unless stated) | Reference |
|---|---|---|
| G0 | `g0_onepass`, five seeds | experiment 345 `solar_pooled` |
| A2 | `a2_mean_only` (spread dropped), `a2_member` (pooled encoder), `a2_memberwise`; paths by ECC, Schaake shuffle, independent draw, member-wise | `g0_onepass` (seeds 0 to 2), ECC |
| A6 | `a6_ens_only`, `a6_finetune`, `a6_pretrain_only` (on the A2 winner) | `a6_ens_only` |
| A5 | `a5_plain`, `a5_frozen`, `a5_e2e` (on the A2 and A6 winners) | `a5_plain` |
| A7 | `a7_atoms` (on the winners) | the same network with the clipped head |

## Pre-registered reading

- **G0.** Over all bands within 0.5% and in each band within 1.0% of experiment 345's score, five
  seeds each. Both networks are the same recipe with the same seeds, so the allowed gap is seed and
  GPU noise.
- **A2.** Row set: days 3 to 10. A member arm (pooled encoder or member-wise) is kept if it beats
  the better of the one-pass and mean-only arms by at least 2% (interval excluding zero). Otherwise
  one pass is adopted for the marginal forecast. Coherence is read separately: the energy score
  and variogram score of each route (independent, ECC and Schaake on the one-pass and
  pooled-encoder marginals, member-wise paths), jointly and per band, with month-block intervals
  over the 12 months of runs. A route that wins the marginal contrast but is worse than the best
  route on either coherence score by more than 5% does not advance (the Opus note's proposal).
- **A6.** Row set: all bands, and days 6 to 10 separately. Pretraining with fine-tuning is kept if it
  beats the ensemble-only arm and the ERA5-and-CAMS-only arm by at least 1%.
- **A5.** Row set: days 6 to 10 for the gain and day 1 for the harm. A plant arm is kept if it beats
  `a5_plain` by at least 1.5% at days 6 to 10 and is within 0.5% at day 1. The end-to-end arm
  needs to beat the frozen arm by a further 0.5% to be preferred.
- **A7.** Row set: all bands, tail metrics. Kept if the coverage of the 0.01 and 0.99 quantiles is
  closer to 1% and the score and the 0.99 pinball loss are no worse than 0.02 points.

## Lead range

**Every contrast is read at days 1 to 10 and, beside it, in the 12 to 36 hour band (day 1).** Days
11 to 14 are not scored, because experiment 310's tables stop at day 10 and rebuilding them was
judged not worth it for an optional extension.

## Recipe

- **Rows and refits.** As experiment 345: training rows have a valid time at least one day before
  the refit; the last 61 days are the validation set; four quarterly refits from 2025-09; own
  history only. Monthly refits are not used because the reference forecasts are quarterly.
- **Fit.** AdamW, learning rate 2e-3, weight decay 1e-4, 20 epochs, batch 2048, the best epoch by
  validation likelihood. Seeds 0 to 4 for G0 and 0 to 2 for ablations.
- **Weather at forecast time.** Every input is ensemble data of the 00 UTC run, the same as
  experiment 310. ERA5 and CAMS appear only in pretraining rows, which end before the ensemble
  store starts. No lagged power enters (experiment 345's solar-only network has none).
- **Reuse.** Experiment 345's forecasts are the gate reference and Metered solar farms research
  default 1's blend (experiment 310) is the absolute reference. Experiment 310's tables are read
  and never written. This experiment's one-pass arm is fitted here.

## Compute

About 135 fits: 20 for G0, 36 for A2, 27 for A6, 36 for A5, 12 for A7, plus second seed sets where
needed. The smallest network trains in about a minute and the member route in a few minutes on one
GPU process under 8 GB with one feeder core, so the whole programme is a few GPU-hours.

## Reviews

- Plan and code: one fresh Sonnet review before the first fit (leakage, the fit stopping at the
  refit cutoff, the scoring, row selection, agreement with experiment 345's recipe).
- Results, write-up numbers, and write-up prose: a fresh Sonnet review each.
- Unit tests: the PyTorch plant against pvlib and experiment 310's NumPy plant; the head's density
  and point masses integrate to 1; quantiles stay within bounds.
