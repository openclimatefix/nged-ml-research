---
id: "00451"
slug: farm-a-upper-tail-exceedances-and-capacity-state
title: >-
  Why does Farm A's upper tail stay above the 1.5% limit after table P, and does a Farm A-specific
  table fix it?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: finished
kind: other
protocol: wind-v1
date: 2026-11-01
size: >-
  simple. No fit and no stored model: the scripts re-score the stored forecasts of experiment 429's
  `member_plain` and read experiment 341's capacity-state table. Serving path: not touched. The
  scripts import the folders of experiments 230, 303, 310, 326, 336, 347, 429, and 450 and
  `studies.*`. Sonnet only:
  one code and numbers review, one write-up review.
hypothesis: >-
  Farm A's share of observations above q99 after table P (1.76%, experiment 450) comes either from
  a table that is too coarse for Farm A, or from a capacity step that puts output above a q99
  scaled to the old capacity.
prediction: >-
  Written before the scripts ran. A finer Farm A table (by forecast median or wind speed) lowers the
  share to 1.5% or less. Experiment 341's capacity-state table shows a reduced or out state, or a
  step, during the scored months.
decision_rule: >-
  Step 1: Farm A passes if its share above q99 (kept rows, true days 1 to 10, 11 scored months,
  every rescaled quantile clipped, each cell fitted on earlier months only) is at most 1.5%. Step 2
  (only if step 1 fails): split Farm A's exceedances of the table P q99 by month, hour, wind-speed
  bin, forecast-median band, and capacity state. If they cluster in a capacity-step period, say so
  and stop. Step 3 (a per-farm tail weight in the loss, 3 seeds, about 9 fits) runs only if steps 1
  and 2 leave Farm A failing.
benchmark: >-
  Experiment 450's table P (farm x lead band, clipped to [0, 1]) on `member_plain`.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Share of observations above q99 (kept rows, true days 1 to 10), by farm; Farm A's exceedance rate
  by month, hour, wind-speed bin, forecast-median band, and capacity state.
verdict: partly supported
best_arm: >-
  Table P with its upper clip raised from 1.0 to the farm's largest earlier power over capacity:
  Farm A 1.17%, Farm B 0.71%, Farm C 1.00%, pooled 0.96%.
headline: >-
  Farm A's upper tail fails because its output reaches 1.00 to 1.035 of its effective capacity and
  table P clips every quantile at 1.0, not because of a capacity step or a coarse table. Finer
  tables make it worse (1.98% by forecast-median band, 2.08% by wind-speed tercile). Raising the
  clip to the largest power over capacity seen in earlier months gives Farm A 1.17%.
lessons: >-
  A quantile forecast of power over effective capacity must not be clipped below the output the
  farm has been seen to reach. Where output exceeds the effective capacity by a few per cent, a
  clip at 1.0 turns every high-output half-hour into a counted exceedance, and no recalibration
  table can remove it.
take_home_message: >-
  Farm A's upper-tail failure after table P is mostly a clipping artefact: 1.04 of its 1.76 points
  are half-hours where the forecast q99 is clipped at 1.0 and the farm produced 1.00 to 1.035 of
  its effective capacity. Raising the clip to the largest power over capacity seen in earlier
  months brings Farm A to 1.17% and all three farms under the 1.5% limit (pooled 0.96%). No capacity
  step explains it, and a finer table does not help, so no tail-weight fits were needed.
advice:
  - >-
    Clip a rescaled quantile at the farm's largest observed power over effective capacity, not at
    1.0, before counting coverage of the upper tail.
  - >-
    Check whether the effective capacity used to normalise Farm A is slightly too low, because 1.2%
    of its half-hours exceed it.
  - >-
    Do not add finer cells (forecast-median band or wind-speed tercile) to table P for the wind
    farms; they raised Farm A's share above q99.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00451-farm-a-upper-tail-exceedances-and-capacity-state/
---

**Take-home (interim, unaudited): Farm A's upper-tail failure after table P is mostly a clipping
artefact.** Of Farm A's 1.76% of observations above q99, 1.04 points are half-hours where the
forecast q99 is clipped at 1.0 and the farm produced 1.00 to 1.035 of its effective capacity.
Raising
the clip to the largest power over capacity seen in earlier months brings Farm A to 1.17% and all
three farms under the 1.5% limit (pooled 0.96%). Experiment 341's capacity state is "full" at Farm A
for every scored half-hour, so no capacity step is involved, and no fits were needed.

**Terms.** *Table P* (experiment 435) rescales each quantile forecast per farm and lead band, fitted
on earlier months only. *Effective capacity* is the capacity that normalises the farm's power.
*Kept rows* are the half-hours outside experiment 210's idle-band mask. *True days 1 to 10* means
forecast lead times of 24 to 240 hours.

![Share of observations above q99 by farm, for each table variant](figure_1_variants.svg)

![Farm A's share above q99 by month](figure_2_by_month.svg)

![Farm A's share above q99 by ensemble-mean wind speed](figure_3_by_wind_speed.svg)

![Farm A's output at the half-hours that exceed q99](figure_4_exceedance_output.svg)

## Results

- **Finer tables do not help (step 1).** Farm A's share above q99: unrecalibrated 2.90%; table P
  (farm x lead band) 1.76%; plus forecast-median band 1.98%; plus ensemble-mean wind-speed tercile
  2.08%. Farm B and Farm C move the same way (Farm B 1.13%, 1.33%, 1.32%; Farm C 1.31%, 1.54%,
  1.62%).
- **The clip at 1.0 causes most of the failure.** At Farm A, 1.17% of kept half-hours at true days
  1 to 10 have power over capacity above 1.0 (maximum 1.035; Farm B 0.80%, Farm C 0.74%). Table P's
  q99 is clipped at 1.0 for 38.6% of Farm A's half-hours, and 1.04 of its 1.76 points of exceedance
  are half-hours with q99 at 1.0, against 0.73 points where q99 is below 1.0. Half the exceedances
  have an output above 1.008.
- **Raising the clip fixes Farm A.** With the upper bound of every rescaled quantile set to the
  farm's largest power over capacity in earlier months (at least 1.0): Farm A 1.17%, Farm B 0.71%,
  Farm C 1.00%, pooled 0.96%, against 1.76%, 1.13%, 1.31%, and 1.41% with the clip at 1.0.
- **Where the exceedances sit (step 2, table P with the clip at 1.0, 332 exceedances in 18,831
  half-hours).**
    - By wind speed: 19.9% above q99 at 12 to 16 m/s ensemble-mean 100 m wind (418 half-hours) and
      19 of 21 above 16 m/s, against 0.9% at 4 to 8 m/s and 1.9% at 8 to 12 m/s.
    - By forecast median: 19.4% above q99 where the forecast median is at least 0.7 of capacity
      (581 half-hours), against 0.5% to 2.4% in the three lower bands.
    - By month: 5.1% in 2026-03, 3.5% in 2025-11, 3.3% in 2025-09, 2.6% in 2026-04, and 0.1% or
      less in 2025-10 and 2026-01.
    - By hour: 0.3% to 1.0% from 04:00 to 06:00 and 2% to 3.3% from 16:00 to 22:00.
    - By lead band: 1.4% to 2.6%, with no pattern.
- **No capacity step.** Experiment 341's capacity state is "full" for every scored half-hour at
  Farm A, with a constant reduced share of 0.061.

## Caveats

- This is a screen on one year of data with 11 scored months. The 95% month-block interval of Farm
  A's 1.76% in experiment 450 was 0.85% to 2.78%, so 1.17% against 1.5% is a point estimate.
- The raised clip uses the largest power over capacity in earlier months, a single extreme value.
  The largest value in the scored months is 1.035.
- The cause (an effective capacity set slightly too low) is a reading of the data, not a test: no
  one has checked the capacity figure.
- Every number is interim until the maintainer has read this page; one fresh reviewer reproduced the
  headline numbers.

## Problems caused by the move

None.
