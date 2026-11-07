---
id: "00631"
slug: bsp-group-sum-with-metered-generators-against-direct-bsp-forecast
title: >-
  Once the metered wind and solar farms beneath a bulk supply point are added to the sum of its
  Primaries, does a coherent scenario sum give a usable tail for the flow through that point?
theme: calibration-and-uncertainty
also_relevant_to:
  - hierarchy-and-pooling
  - generator-forecasting
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  defensible (which remainder treatment, which scenario source), so the plan registers each.
  Callers: this folder's scripts only, which import experiments 434, 436, 615, 625, and 629 and
  `studies.*`. One Sonnet code review before the first run, one Sonnet results check, one Sonnet
  write-up review.
hypothesis: >-
  Experiment 629 found that sums of the 16 Primaries beneath a bulk supply point (BSP) put 31% to
  79% of actual flows below their q05 (nominal 5%), and that the metered wind and solar series
  beneath the BSP explained much of the gap (R squared 0.94 at Skegness). The claim here is that,
  once those metered farms are added to the sum together with a fitted remainder, a coupled
  scenario set (J3, the Schaake shuffle, or J5, the fitted copula) puts the q05 and q95 exceedance
  of the BSP flow within the binomial error of nominal at both BSPs, and beats independent draws
  (J0) on CRPS.
prediction: >-
  Written before any score is read. P1. With the farms and no remainder the q05 exceedance falls
  from 44% to 79% (experiment 629) to under 30% at both BSPs, but stays above 10% because a
  battery, a biofuel plant, other generators, and one photovoltaic farm with too little history
  are still missing. P2. A time-of-day remainder brings J3 and J5 within two binomial errors of 5%
  at Skegness and not at Boston, because Boston has more unmetered generation. P3. Coupling beats
  independent draws on CRPS by under 2% at either BSP, because experiment 436 found the Primaries'
  errors nearly uncorrelated (0.29) and the farms are a small part of the sum. P4. The direct
  forecast of each BSP's meter stays better calibrated than every sum.
decision_rule: >-
  Fixed before any score is read. Row set A: every daily run (09:00 UTC) of the development months
  2025-10, 2025-12, 2026-01, 2026-03, and 2026-06 (the 153 runs of experiment 629), valid times
  before 2026-07-01 only, leads 0 to 240 hours, BSP-meter half-hours that are not flagged, and only
  half-hours where every source scored has a value. Demand is the stored K1 network, as in
  experiment 629. Farms are the stored development-month backtests of experiment 615 (wind: the
  physics plant with the mixture-density head; solar: the quarterly-refitted network), converted
  to MW by each series' scoring capacity. Row set B: the Tuesday runs of the same months in the
  stored scenario sets of experiment 436 (XGBoost demand, all four couplings including the
  ensemble copula coupling J2 with the farms), reported beside row set A and never used to change
  its verdict. The sum is BSP flow = Primaries minus metered farm output (power flows in at the
  BSP are positive) plus a remainder. Three remainders: none; a constant, the mean of the observed
  meter minus the observed (Primaries minus farms) over every half-hour from 2025-03-01 to 16 days
  before the test month's first run; and the same mean by half-hour of the day. Couplings on row
  set A: J0 (independent draws), J3 (Schaake shuffle), J5 (Gaussian copula on probability-integral-
  transform scores, no ensemble mixing, fitted leaving the test month out), each with and without
  the farms; J2 (ensemble copula coupling) without the farms only, because the daily runs hold no
  member weather at the farms. Quantiles of the 51 sums use the Weibull definition. Export side is
  the share of actual flows below the q05 of the sum; import side is the share above the q95. The
  independent unit is the valid calendar day. Threshold-weighted CRPS (twCRPS) uses the chaining
  function max(x, L) with L the 90th percentile of the positive exports of the BSP meter in the
  scored rows (post hoc, because experiment 629 found that the scaled limit, 0.9 times the largest
  export before 2025-10-01, is never reached); the page states if it is reached. A contrast is the
  relative difference of the monthly mean CRPS, paired by month, with a month-block bootstrap
  (20,000 resamples) and a 95% interval. `sum usable` if, with the farms and a fitted remainder
  (constant or time of day, each judged separately), J3 or J5 puts both the q05 export exceedance
  and the q95 import exceedance of the 0 to 240 hour band within two binomial standard errors of 5%
  at both BSPs, and has a CRPS below J0's with the interval wholly below zero at both BSPs.
  `direct needed` otherwise. The page reports absolute scores beside every contrast and states that
  two BSPs is thin evidence. No rescue tuning. Stop rule: a farm series beneath either BSP with no
  stored forecast (reported, not a reason to stop, if the missing series is a short-history farm
  or not a wind or solar farm), or any read of a valid time on or after 2026-07-01.
benchmark: >-
  J0 (independent draws) for coherence; the direct BSP forecast of experiments 611, 619, and 622
  for the sum.
row_set_id: bsp_daily_devmonths_v1
metric: >-
  Exceedance rate at q05 (export) and q95 (import) of the BSP flow by lead band; CRPS; twCRPS at
  the 90th percentile of observed export; mean error in MW at summer midday and winter evening.
production_status: not-for-production
adopted_in: none
---

**Planned.** This page holds the question, the predictions, and the decision rule, written before
any score exists. The finished page replaces this text.
