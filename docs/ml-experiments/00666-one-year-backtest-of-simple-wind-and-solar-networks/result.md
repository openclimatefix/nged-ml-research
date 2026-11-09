---
id: "00666"
slug: one-year-backtest-of-simple-wind-and-solar-networks
title: >-
  What does a one-year backtest of the simple shared wind network S and the simple shared solar
  network PLP forecast for the 3 metered wind farms and the 6 metered solar farms?
theme: generator-forecasting
also_relevant_to:
  - production-reference-and-best-guess
status: finished
kind: other
protocol: wind-v1
date: 2026-11-09
size: >-
  Medium. (1) Stores private Parquet forecasts and fitted-network state files in the data folder
  only. (2) The refit cadence and the lead range were the open design choices; the Discuss session
  fixed them (below). (3) Spans code from experiments 659, 615, 430, and 259, which this folder
  copies. (4) Touches no serving path and no degradation rule. (5) Changes no contract.
hypothesis: >-
  Descriptive. Over daily forecast origins from 2025-11-01 to 2026-10-31, the networks of
  experiment 659 (wind S, solar PLP), retrained monthly (wind) or quarterly (solar) on data
  available before each refit, produce calibrated 11-quantile forecasts out to 14 days, and solar
  PLP trained on leads out to 336 h scores no worse at days 10 to 14 than climatology.
prediction: >-
  Written before any score. (1) Wind S over days 0 to 10 scores within 0.15 points of capacity of
  its experiment 659 score (3.684) on the overlapping origin months. (2) Solar PLP at days 10 to
  14 scores better than `climatology` but by less than at days 1 to 3. (3) The q05-q95 coverage of
  both stays between 0.85 and 0.95 in every lead band, except that very sunny days still exceed
  q99 on 10% or more of rows.
decision_rule: >-
  Descriptive; nothing is ranked or adopted. The file is a deliverable for NGED. No contrast is
  tested.
benchmark: >-
  `climatology` (the farm's own past power) for the 10 to 14 day solar band; the scores of
  experiment 659 for the overlapping months.
row_set_id: year_backtest_2025-11_to_2026-10_v1
metric: >-
  CRPS (11-level quantile score) and normalised MAE of the median in points of capacity, by lead
  band and by month; q05-q95 coverage; tail shares above q99 and below q01.
verdict: inconclusive
best_arm: none (descriptive backtest)
headline: >-
  Over 365 daily origins, wind S scores 4.06 points of capacity (continuous ranked probability
  score, CRPS) and solar PLP scores 2.91
  points on daylight half-hours, with q05-q95 coverage 0.91 and 0.885. At days 10 to 14 solar PLP
  scores 3.24 against 3.39 for climatology, but at days 10 to 14 it is worse than climatology from
  November to March.
take_home_message: >-
  The two simple networks produce complete, sorted 11-quantile forecasts to 14 days for 3 wind farms
  and 6 solar farms, and wind S matches experiment 659 on the rows both cover. Solar PLP at days 10
  to 14 beats climatology from April to September but not from November to March. Its median is
  high in spring and low in autumn (seasonal or caused by refit age; this experiment cannot
  separate them), its lower tail is about three times too frequent, and its upper tail on very
  sunny days is still too low, so use the 10 to 14 day solar band with care.
production_status: not-for-production
---
**The year-long backtest is complete.** `wind.parquet` and `solar.parquet` hold forecasts for 365
daily origins. Both neural networks score sensibly at every lead out to 14 days, and solar at days
10 to 14 is only slightly better than climatology. The files are private (they hold the time series
of
NGED's metered generators, labelled by their original `time_series_id`). The backtest is
descriptive,
not a ranking: the year overlaps the development months of
[experiment
659](../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md),
and origins from 2026-07-01 to 2026-10-31 lie in the window earlier experiments kept as a final
test, so those months are no longer a held-out test for choosing between the networks and the
defaults. No figures are included on this page; the scores below come from `scores/*.csv`.

**Scores fall with lead for both neural networks, and solar PLP stays below climatology in every
band.** CRPS (continuous ranked probability score) is the mean pinball loss over 11 quantile levels,
in points of capacity (smaller is
better). Capacity is the production effective capacity, the same table as the shared folder's
`series_capacity.parquet`. Bands are half-open ranges of days from the 12:00 UTC origin to the valid
time; the "7-14 d" and "10-14 d" rows overlap the other rows, so the table is not a partition. Wind
scores
count idle readings (experiment 210's idle mask) as zero until 2026-09-13; solar is scored on
daylight half-hours only.

| Lead band | Wind S CRPS | Wind coverage q05-q95 | Solar PLP CRPS | Solar coverage q05-q95 | Solar climatology CRPS |
|---|---|---|---|---|---|
| 0-1 d | 2.29 | 0.901 | 2.19 | 0.894 | 3.38 |
| 1-3 d | 2.73 | 0.904 | 2.41 | 0.889 | 3.38 |
| 3-7 d | 3.84 | 0.909 | 2.87 | 0.882 | 3.38 |
| 7-14 d | 4.82 | 0.919 | 3.18 | 0.884 | 3.38 |
| 10-14 d | 4.93 | 0.922 | 3.24 | 0.886 | 3.39 |
| All | 4.06 | 0.913 | 2.91 | 0.885 | 3.38 |

Wind S (raw ensemble weather only) and solar PLP (physics-lite power plus raw weather) are the
networks of experiment 659. Wind S median mean absolute error (MAE, % of capacity) runs from 8.2 at
0-1 d to 18.3 at 10-14 d, and solar PLP from 7.9
to 12.8 (climatology: 13.3). Share of observations above q99 (ideal 0.01): wind 0.016 at 0-1 d and
0.023 at 10-14 d; solar 0.012 and 0.011. Share below q01: wind 0.011 and 0.007; solar 0.028 and
0.032 (most solar rows below q01 have an observed power of zero or less). By origin month, wind S
scores best in July and August (2.67 and 2.77) and worst in November (5.52); solar PLP scores best
in December (1.27) and worst in March (3.95) and September (3.74).

## Checks and caveats

- **Wind S matches experiment 659 on matched rows.** On the rows both cover (December, January,
  March, and June origins, 231,000 rows), this experiment scores 4.174 and experiment 659 scores
  4.179 when both use experiment 659's scoring capacity. The absolute numbers on this page are about
  5% higher than experiment 659's because the production effective capacity is 2.3% to 8.7% below
  experiment 659's wind capacity. Solar is 0.13 points worse than experiment 659 on 120,000 matched
  rows (3.350 against 3.224), possibly because each refit here is one month older for the scored
  months and the solar neural network here trains on leads up to 348 h.
- **Prediction (1)** (within 0.15 points of 3.684 on overlapping months) cannot be judged on that
  basis: the overlap-month score here is 4.011 because of the capacity difference and because the
  3.684 includes October 2025. On matched rows and matched capacity the two agree. **Prediction
  (2)** holds
  overall (3.24 against 3.39) but not from November to March, when solar PLP at 10-14 d scores
  worse than climatology by 0.03 to 0.42 points; it is better by 0.17 to 0.48 points from April to
  September. **Prediction (3)** holds for wind and solar coverage in every band (0.88 to 0.92; solar
  is just under 0.90). The share of very sunny-day observations above q99 is 2.8% on this
  experiment's own definition (each farm's top tenth of days by energy in its month), which is not
  comparable with experiment 659's 10.6% to 12.7%; the weak upper tail on very sunny days is not
  resolved.
- **Seasonal bias.** The solar median is high in spring (+6.5 points of capacity in March) and low
  in autumn (-6.9 in September, -8.5 in October, all 6 farms), and the solar neural network has no
  calendar input; one year with quarterly refits cannot separate a seasonal effect from refit age.
  The wind median at 10-14 d is low in November and December (only 28% and 32% of
  observations fall below q50). Wide bands at long leads may explain why coverage stays near 0.90.
- **Wind idle mask ends on 2026-09-13.** From that date idle readings count as observed in scoring
  and in the 2026-10-01 refit's training target, so September and October are scored on a different
  target (one farm's share below q01 rises from 1.1% to 5.4%).
- **Capacity scale.** The networks scale by the 99.9th percentile of cleaned power over the 730 days
  before each refit, which is 1.02 to 1.12 times the stored effective capacity, so some quantiles
  exceed the stored capacity (the largest q99 is 1.12 times, at solar series 22). Observed solar
  power exceeds the stored capacity on 5% to 9% of April daylight rows.
- **Solar night.** Night half-hours are zero. The daylight flag is the mean cosine of the zenith
  over
  the 3-hour or 6-hour ensemble period, so about 3,000 half-hours around dusk and dawn at leads of
  6 days or more carry small non-zero quantiles (q99 at most 0.7% of capacity), and some dawn and
  dusk half-hours in a mostly dark period are forecast as zero. The solar all-half-hours tables in
  `scores/` are diluted by night rows and by 3.8% negative night readings; use the
  `solar_daylight_*.csv` tables.
- **Scoring rows.** Rows without observed power are dropped, so origins after about 2026-10-26 are
  scored on part of their 14 days. Observed power is not in the files.
- **No break at any refit boundary in the median, band width, or q99:** consecutive-origin forecasts
  for the same valid times differ by
  no more than the ordinary day-to-day change at all 11 wind boundaries and all 3 solar boundaries.
  Median minus mean forecast per site was not computed (the mixture mean is not stored).
- **Reviews** (all fresh agents; every finding was triaged against the code):
    - Before the first fit, 2 Sonnet reviews (leakage and contract) found 8 minor points. The
      accepted fixes were assertions that no training valid time reaches the refit, coverage and
      archive checks that raise, and Int8 and origin-count checks.
    - Before the first fit, 1 Opus review of leakage and fold logic found 3 low or informational
      points and no leakage.
    - After the files were written, 1 Sonnet file audit found 4 low points. Both files were clean,
      one wind fold and one solar fold re-ran from scratch to 0.0 MW difference, and the scores
      reproduced.
    - 1 Opus review of the files and scores found 12 points, which are the caveats above.
    - 1 Sonnet prose and number review found no wrong number; its prose findings are applied.

## Method

- **Neural networks and weather inputs.** Wind S and solar PLP of
  [experiment
  659](../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md):
  a pooled network of two hidden layers of 64 units with a three-component Gaussian mixture over
  power as a fraction of capacity, 11 quantiles read from the mixture and capped at capacity, 3
  seeds averaged quantile by quantile and then sorted. Wind S reads the mean and standard deviation
  across the 51 ensemble members of 10 m and 100 m wind speed, the sine and cosine of the 100 m
  direction, and air temperature, plus the lead (sine and cosine of its hour of day, and hundreds of
  hours) and a one-hot wind-farm column. Solar PLP reads 18 inputs: mean and standard deviation of
  shortwave, air temperature, and 10 m wind speed, the ensemble means of relative humidity,
  longwave radiation, precipitation, and surface pressure, the mean and standard deviation of the
  fixed physics-lite power and of the Faiman cell temperature, the sun's elevation, the sine and
  cosine of its azimuth, and the lead; there is no farm column and no calendar column. No ERA5 or
  CAMS
  pre-training.
- **Origins and weather.** One origin per day at 12:00 UTC from 2025-11-01 to 2026-10-31, using the
  00 UTC ECMWF ENS run of that day (51 members, treated as available at 09:00 UTC); valid times
  run half-hourly for 672 half-hours from 12:00 (lead from the 00 UTC run 12 h to 347.5 h). Where
  the
  ensemble steps are 3-hourly or 6-hourly, the value is interpolated as in experiments 310 and 430;
  the share of daylight half-hours lacking their own stamp is 0% at every lead.
- **Power lags.** Neither network reads a power lag, so the 6-hour power delay changes no feature.
  The power delay sets only the training cut-off of each refit. The first forecast half-hour (valid
  12:00) has
  already ended at the origin; no power is an input, so this leaks nothing.
- **Refits.** Wind S refits on day 0 of each month (12 refits), training on runs and valid times
  before 62 days earlier and stopping early on the last 62 days, 40 epochs. Solar PLP refits on
  2025-11-01, 2026-02-01, 2026-05-01, and 2026-08-01 (each scores the next three months), training
  on valid times before one day earlier, with the last 61 days for early stopping, 20 epochs, and
  daylight rows with leads below 348 h from the 00 UTC run (experiments 430 and 659 stopped at 251.5
  h); solar trains on all daylight half-hours, including more low-sun half-hours than experiment 659
  kept. Wind training leads run to 351 h from the 09:00 stamp, as in experiment 659. Capacity is the
  99.9th percentile of cleaned power over the 730 days before each refit. Wind idle readings are
  zero in training and scoring until 2026-09-13.
- **Compute time.** About 20 minutes of GPU time in 3 parallel processes (36 wind fits, 12 solar
  fits).
- **Scoring.** The 11-level pinball loss times 100 and the normalised median MAE, by origin-to-valid
  lead band and origin month, against cleaned observed power (outside the files). Solar climatology
  is the farm's own power at the same month and half-hour before each refit.

## Files

- Forecasts (private): `data/ml-experiments/backtest_year_2025-11_to_2026-10/wind.parquet`
  (735,840 rows = 3 series x 365 origins x 672 half-hours; ids 4, 5, and 6; MW for 4 and MVA for 5
  and 6) and `solar.parquet` (1,471,680 rows; ids 21, 22, 23, 29, 30, and 31; MW). Columns:
  `time_series_id` (Int8), `valid_time`, `nwp_init_time`, `power_fcst_init_time`, then `q01` to
  `q99`
  in the same unit. Checks passed: no duplicate keys, nulls, non-finite values, or crossings; UTC.
- Per-seed fit tables, saved networks, and scores: `data/ml-experiments/00666-*/fits/`, `models/`,
  `scores/`.
- Code, in `ml-experiments/00666-*/`: `e00666_wind_members.py`, `e00666_wind_features.py`,
  `e00666_solar_inputs.py`, `e00666_solar_gate.py`, `e00666_wind_fit.py`, `e00666_solar_fit.py`,
  `e00666_export.py`, `e00666_score.py`.
