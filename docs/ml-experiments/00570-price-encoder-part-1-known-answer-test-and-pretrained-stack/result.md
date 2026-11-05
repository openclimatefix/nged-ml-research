---
id: "00570"
slug: price-encoder-part-1-known-answer-test-and-pretrained-stack
title: >-
  Does training a price encoder on the true electricity price give the weather and datetime encoders
  of the modular lag-free demand network cleaner supervision, even though price is absent at
  forecast time?
theme: demand-inputs
also_relevant_to:
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted network weights for about 8 pre-trainings and the gated fine-tuned arms
  under this experiment's private data folder; no earlier experiment's files are written. Serving
  path: not touched (this repository holds none). Degradation rule: not touched. More than one
  design is defensible (residualised or raw price features, price encoder kept or discarded in
  fine-tuning), so the plan fixes each before any score is read. Callers: this folder's scripts
  import experiments 397, 454, 470, 511, 545, 554, and 562. One Sonnet review of the plan and code
  before the first fit, one Sonnet review of the write-up against the printed report, and one Sonnet
  prose review.
hypothesis: >-
  Price-following assets such as batteries and gas engines put price-driven variation into a
  series' power that weather and calendar cannot explain. A price encoder that sees the true price
  while the modular lag-free network is pre-trained can absorb that variation, so the weather
  encoder and the datetime encoder learn cleaner patterns from the same targets. The price encoder
  is discarded before fine-tuning, so serving needs no price feed.
prediction: >-
  Written before any score is read. The known-answer test (K1 to K4) passes on the six MW Primaries
  and the four BSP and GSP series. A6-1: the residualised price encoder (A6-res) pre-trained stack
  beats M0's by at least 1% of CRPS at the Primaries and 3% at the BSP and GSP series (fails if
  within 0.5% at either). A6-2: a linear probe on the weather and datetime encoder outputs predicts
  the residualised price features with an R-squared at least 20% lower (relative) in A6-res than in
  M0. A6-3 (base rate null): fine-tuned A6-res minus its dummy-noise control is within the minimum
  detectable effect at both Primary groups. A6-4: raw price features are worse than residualised
  ones by at least 0.03 points. A6-5: the typical-shape placebo recovers at least half of A6's
  pre-trained-stack gain. Step 0: the price window alone explains 2% to 6% of one metered battery's
  variance.
decision_rule: >-
  Step 0 is report-only and the known-answer test checks that the encoder can learn a stated
  rule; neither gates the real-data arms (maintainer decision, 2026-11-05). A fine-tuned contrast counts only if its point estimate exceeds the minimum detectable effect of its
  group and at least 5 of 5 folds agree in sign (six counted tests, Bonferroni factor 6). The
  pre-trained-stack contrasts A6-1, A6-2, and probe CRPS are a separate counted family of 6. Fine-tuned
  scoring runs only for arms that pass A6-1 or A6-2. The threshold is the minimum detectable effect,
  because a price encoder adds a pipeline step only if it is used at forecast time; A6 itself
  discards it, so its production cost is zero and any resolvable gain would be worth porting.
benchmark: >-
  New monthly-refit M0 and M0-R controls with the same seeds (never the stored daily-refit M0), and
  each price arm's dummy-noise, dummy-constant, shuffled-day, and typical-shape controls. Experiment
  554 and 564 stored scores supply minimum detectable effects. L-1d, A-pool, and manual_heuristic
  are report-only.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only Primaries and the six signed-MW
  Primaries, with the leave-D8-out score beside each, D15 and D18 report-only, and the four BSP and
  GSP series as case studies.
---

# Experiment 570: does a price encoder trained on the true price clean up the other encoders?

Planned. The full design is in the private brief `price_encoder_p1_brief.md`; this page is filled
in as the steps finish. Order of work: step 0 (CPU diagnostics), the known-answer test, the
pre-trained-stack diagnostics, a gate, then fine-tuned scoring for arms that pass the gate.

## Price data and licence

- **Licence:** "Copyright: Licence to use BMRS open data", Elexon Limited
  (<https://www.elexon.co.uk/bsc/operations-settlement/bsc-central-services/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/>),
  read verbatim on 2026-11-05. It grants "a worldwide, royalty-free, perpetual, non-exclusive
  licence to Use the BMRS Data", and the licence text lists copying, publishing, distributing, and
  adapting among the permitted uses. It sets no research-specific term and no limit on publishing
  derived results. Attribution line, required on every chart of price data: "Contains BMRS data ©
  Elexon Limited copyright and database right 2026."
- **Fetch:** market index price (`APXMIDP` provider) and system price with net imbalance volume, for
  2019-09-01 to 2025-08-31, through the public keyless API with 2 threads, about 4,300 small
  requests and 18 MB, no cost. Saved under this experiment's private data folder (`market_early/`),
  never committed. The longer series is `*_2019-09_to_2026-09.parquet` there.
- **Gaps:** 72 whole days of market index price and 9 whole days of system price are not published
  (listed by the fetch script); the price encoder treats them as missing.
- **Definition is unchanged across the span:** the system sell and buy prices are identical in every
  row from 2019 to 2026 (single imbalance price), and every zero market index price coincides with
  zero traded volume, so zero-volume rows are masked. The market index mean is 37 GBP per MWh in
  2019, 198 in 2022 (the gas shock), and 72 to 103 in 2024 to 2026, which is why the transform
  normalises by trailing 28-day statistics.

## Maintainer decisions

- **Step 0 is report-only.** Step 0(a), run on the 13 months then on disk, found an in-sample R-squared
  of 0.2% to 4.1% at the 16 Primaries and a negative leave-one-month-out R-squared at all 16
  (-0.3% to -16%). That shows no linear rule on same-time price features transfers across months. It
  does not rule out a nonlinear rule, which a learnt encoder may find, so the real-data arms run
  regardless.
- **The known-answer test is a check, not a gate.** It shows whether the encoder can learn the
  threshold, rank, and cheapest-and-dearest-4-hours rules. A failure is reported and the real-data
  arms still run, with their results read in that light.
- **Ceiling:** 6 GPU-hours; the maintainer is asked before it is exceeded.
