---
id: "00418"
slug: mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens
title: >-
  Does pretraining the pooled mixture density network on estimates of past weather (ERA5 and CAMS)
  before fine-tuning on ENS forecasts lower the Primary demand score?
theme: retraining-and-training-history
also_relevant_to:
  - hierarchy-and-pooling
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: forecasts, pretrained and fine-tuned networks, and small score tables in this
  experiment's private folder only; experiments 397 and 406 are read and never written. Production
  serving path: not touched (this repository holds none). Degradation rule: not touched. Designs:
  more than one defensible (what to pretrain on, how much of the network to fine-tune, whether to
  add forecast-like noise), so all four are registered as arms. Callers: the scripts import
  experiments 87, 48, and 397 and the `studies` evaluation code, whose callers sit in those folders.
hypothesis: >-
  The pooled plain mixture density network of experiment 397, pretrained on the Primaries' own
  power and ERA5 and CAMS weather from 2019-09 and then fine-tuned on the ENS rows, scores lower
  CRPS at days 1 to 10 than the same network trained on the ENS rows alone, because the ENS archive
  starts on 2024-04-01 and the pooled network sees at most 18 to 24 months of it.
prediction: >-
  Written before any fit. P1: pretraining and fine-tuning all weights (arm P1) lowers CRPS at days
  1 to 10 by 0.05 to 0.20 points at the signed-MW six and by less at the MVA-only eight, because the
  extra years add winters and summers the ENS window lacks, but ERA5 differs from an ENS forecast
  and recent power carries most of the skill at short lead (experiment 1 found an ERA5-trained
  weather-to-power model better than an ENS-trained one only up to day 6). P2: freezing the
  backbone (arm P2) lands within 0.05 points of P1, because the backbone has one 64-by-64 layer.
  P3: adding forecast-like noise to the ERA5 inputs (arm P3) beats P1 at days 4 to 10 and loses to
  it on day 1 by under 0.03 points, and its q95 and q99 coverage is closer to nominal than P1's.
  P4: the gain of the best pretrained arm over the ENS-only network is at least twice as large when
  fine-tuning uses only the first 6 months of ENS as with the full window.
decision_rule: >-
  Contrasts, all read per group (signed-MW six; MVA-only eight with and without D8; D15 and D18
  report-only) on CRPS at days 1 to 10 (days 11 to 14 descriptive), paired by fold and seed with the
  month-block bootstrap of `studies.eval_protocol`: C1 P1 minus the ENS-only reference; C2 P2 minus
  the reference; C3 P3 minus the reference; C4 P3 minus P1; C5 P2 minus P1; C6 low-data check, the
  best of P1 to P3 minus the ENS-only network, both fine-tuned on the first 6 months of ENS, minus
  the same contrast on the full window. The screen has four monthly folds (2025-10, 2026-01,
  2026-03, 2026-06), so every screen number is interim, carries no interval, and cannot support a
  claim of benefit. An arm advances to the other 8 monthly folds of 2025-07 to 2026-06 if, at the
  signed-MW six, it beats the reference in at least 3 of 4 folds and its four-fold mean gain is at
  least 0.03 points; its MVA-only eight mean is no worse than +0.02 points; and its q95 and q99
  coverage at the signed-MW six are no further from nominal than the reference's. The smallest gain
  worth porting is 0.03 points (about 1% of the reference score), because pretraining adds a second
  training stage, two more weather sources to maintain, and a network refreshed on its own
  schedule. At the full folds, an arm is `supported` if its difference from the reference (C1, C2, or
  C3) has a 95% interval wholly below -0.03 at the signed-MW six; it is a `small gain` if the
  interval lies wholly below zero but not below -0.03; otherwise it is `not supported`. No adoption is sought from this experiment.
benchmark: >-
  The pooled plain mixture density network of experiment 397 (summary scheme, setting g09, 8-wide
  Primary embedding, three seeds) on the same four folds; BG2, `manual_heuristic`, and
  `climatology` as absolute-skill references. At the full folds the ENS-only network is refitted
  for the months experiment 397 did not store.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and the avoidable-curtailment cost, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/
---

**Plan: pretrain the pooled mixture density network on ERA5 and CAMS weather, fine-tune it on ENS.**
The status is `planned`. Nothing below has been run.

## Design note

**Every arm is the experiment 397 pooled plain network with fixed carried components.** Names carry
the components, so a reader sees what each arm inherits:

- `pooled-emb`: one network across the 16 Primaries with an 8-wide Primary embedding, because
  [experiment 406][e406] found the embedding carries the pooling gain.
- `summary`: the inputs are the ensemble mean and ensemble standard deviation of each of the seven
  weather variables (experiment 397's summary scheme), so the network is applied once per run.
- `g09`: 64 units, 2 hidden layers, 3 mixture components, dropout 0.1, the setting experiment 397
  chose on its validation block. No setting is tuned here.
- Not carried: the attention gate (inert in experiment 397), the explicit solar term, the lag-trust
  component, and the D15 and D18 start cut. Each is added in the assembly if it wins elsewhere.

**Arms** (arm names are `<components>_<training>`):

| Arm | Name | Training |
|---|---|---|
| Reference | `pooled-emb-summary-g09_ens-only` | Stored forecasts of experiment 397 (`plain_pooled`, summary scheme), three seeds |
| P1 | `pooled-emb-summary-g09_era5cams-pretrain_ft-all` | Pretrain on ERA5 and CAMS, then fine-tune every weight on the ENS rows |
| P2 | `pooled-emb-summary-g09_era5cams-pretrain_ft-input-head` | Same pretraining; the middle hidden layer is frozen, and the input layer, the Primary embedding, and the output layer are fine-tuned |
| P3 | `pooled-emb-summary-g09_era5cams-noisy-pretrain_ft-all` | As P1, with forecast-like noise added to the pretraining weather means |

**Pretraining rows.** A pseudo-run starts every second day from 2019-09-01 at 00:00 UTC, with the
origin 9 hours later (the publication delay of experiment 87). Each pseudo-run has one row per
Primary and half-hour for leads 0.5 to 336 hours. Weather is the Primary's ERA5 hourly value
(temperature, dew point, surface pressure, longwave, precipitation, 10 m wind speed) held over both
half-hours, with CAMS global horizontal irradiance in place of the ERA5 shortwave. Power and the
three origin lags come from the dense power grid with experiment 87's `origin_lags`, so the lags
obey the same newest-usable-observation rule as an ENS row. Rows whose power is missing or flagged
are dropped, as in experiment 397. The pretraining set holds an estimated 8 to 13 million rows
depending on the fold; the build prints the count per fold.

**Spread inputs for pretraining rows.** ERA5 has no ensemble spread, so the `std_*` input columns of
a pretraining row take the median of the ENS `std_*` column at that lead band (24-hour bands) over
the fold's own ENS training rows. The mean-input and standard-deviation-input distributions then
match what the network sees at fine-tuning. This table uses only months before the fold's test
month.

**The noise of arm P3 is the ENS spread.** P3 adds independent Gaussian noise to each
`mean_*` input of a pretraining row, in the transformed input space, with standard deviation equal
to the median ENS `std_*` input at that lead band: the noise grows with lead because the ENS spread
does. The draw is made once per fold with a fixed seed and stored. The spread of an ensemble mean is
close to its error only for a well-calibrated ensemble, so the build also prints, per variable and
lead band, the ratio of the root-mean-square difference between ENS mean and ERA5 to that spread, as
a check that the noise is neither far too small nor far too large. The rule stays "noise equals
the median spread" whatever the ratio, and the ratio is reported.

**Fine-tuning rows.** These are exactly the rows of experiment 397's `build_train` for the fold:
ENS control-member rows with the summary columns, the 24 months before the refit within the ENS
archive (from 2024-04-01), one week in 12 held out for early stopping, runs near the held-out weeks
purged, and recency weights with a half-life of 0.5 years. The standardiser and the per-Primary
target mean and standard deviation of the fold are fitted on those rows and reused unchanged for the
pretraining rows, so both stages see the same input scale and the same targets.

**Leakage rule.** A fold with test month M and refit time R (00:00 UTC on the first day of M) sees a
pretraining row only if its valid time is at least 3.5 hours before R and its origin is at or before
R (experiment 87's rule). The same rule bounds the ENS rows, the median-spread table, the
standardiser, and the CAMS and ERA5 values used. The early-stopping weeks of the pretraining stage
are the same absolute weeks as the fine-tuning stage (week index of the origin day, one in 12), and
the runs near them are purged from both stages, so a pretraining row never trains on a week the
fine-tuning stage validates on. Inference uses ENS only and the same standardiser. The build asserts
the valid-time and origin bounds on every array it hands to a network.

**Training recipe.** Pretraining: AdamW at 5e-4, weight decay 1e-3, batch 16,384, at most 40
epochs, patience 6, no recency weights (the purpose is the long history), early stopping on the
pretraining held-out weeks. Fine-tuning: AdamW at 2e-4, the same weight decay, batch size, epoch
limit, and patience, the experiment 397 recency weights, early stopping on the ENS held-out weeks,
starting from the best pretrained weights. Three seeds (0, 1, 2) as in experiment 397; seed `s`
seeds both stages. One pretrained network per fold and seed is shared by P1 and P2.

**Scale of the work.** The screen is 4 folds, 3 seeds, 2 pretrainings (clean and noisy), and 3
fine-tunings: 24 pretrainings and 36 fine-tunings. Each fine-tune and each pretraining is a
7,305-parameter-scale network on up to 13 million rows, so one fit runs for minutes on the GPU. The
resource limits are one GPU process under 8 GB, one feeder core, `MemoryMax=8G`, and a stop line of
12 GB available memory.

## Planned checks the write-up reports

- **Row counts:** pretraining and fine-tuning rows per fold, and the number of Primaries and years
  each contributes.
- **Absolute skill:** CRPS at days 1 to 10 for every arm beside BG2, `manual_heuristic`, and
  `climatology`, at the signed-MW six and the MVA-only eight.
- **Tails:** q95 and q99 coverage at both groups, with the lead-band breakdown.
- **Low-data check (C6):** the arm with the lowest signed-MW-six screen CRPS among P1 to P3 and the
  ENS-only network, both fine-tuned (or trained) on only the first 6 months of ENS
  (2024-04-01 to 2024-09-30), three seeds, on the four screen folds.
- **Production cost:** one pretrained network refreshed rarely (the build and fit time measured
  here), plus a cheap monthly fine-tune (its measured time), and the two weather feeds it needs
  (ERA5 and CAMS history).
- **Figures first:** the headline chart of CRPS against lead for all arms, the gain by Primary, and
  the coverage by lead band, each with plain captions.

## What would stop the experiment

If a build assertion fails on leakage, the experiment is `invalid` and stops. If the pretrained
network's held-out NLL on ERA5 inputs is no better than the untrained network's, pretraining has
failed and the verdict is `failed` for that arm.

## Evidence this plan follows

- Follows [experiment 403](../00403-pooled-versus-per-primary-mixture-density-network/result.md) and
  experiment 406: one pooled network with a Primary embedding.
- Follows [experiment 397][e397]: no attention gate, summary inputs, setting g09, and its tails flagged as the open problem.
- Departs from the experiment 412 capacity-prior arms, which do not apply here because the
  pretraining target is the same signed-MW or MVA power as the fine-tuning target.
- Experiment 413 (differentiable-physics network) did not advance, and experiment 406 found the
  pooled gain grows with short histories, which motivates the low-data check C6.

[e397]: ../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md
[e406]: ../00406-pooled-mixture-density-network-without-embedding-and-with-short-history/result.md
