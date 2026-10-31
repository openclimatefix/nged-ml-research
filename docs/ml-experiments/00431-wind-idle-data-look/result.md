---
id: "00431"
slug: wind-idle-data-look
title: >-
  Do idle half-hours at the three metered wind farms look different from genuine low output, and do
  they follow a weekly maintenance cycle?
theme: generator-forecasting
also_relevant_to: []
status: done
kind: other
size: >-
  simple. Stored: summary tables and a private gallery under
  data/ml-experiments/00431-wind-idle-data-look/ only. Serving path: not touched. Degradation rule:
  not touched. Designs: one defensible design (a descriptive look at three series, no fit).
  Callers: this folder's scripts only, which read experiment 210's stored `joined.parquet`.
hypothesis: >-
  H1: the idle floor of each metered wind farm has a texture (exact zeros, repeats, a quantised step,
  small first differences) that genuine low output at the same power level does not have. H2: idle
  half-hours at high wind (a whole-farm outage or maintenance) follow a weekly cycle, because
  maintenance is done on weekday daytimes and weather-driven idling has no weekly cycle.
prediction: >-
  Written before any number is read. H1 fails for a farm if the best single texture statistic gives
  an area under the receiver operating characteristic curve (AUC) below 0.65 for sure-idle rows
  against sure-generating low-output rows. H2 fails if the share of idle half-hours at high wind on
  weekday daytimes is under 1.5 times the share at weekends and at night.
decision_rule: >-
  Sure-idle rows: ERA5 100 m speed below 2 m/s and power inside experiment 210's idle band (5% to
  95% of power at ERA5 wind below 2 m/s, per farm). Sure-generating low-output rows: power between
  3% and 8% of capacity at ERA5 wind of 6 m/s or more. Texture statistics, computed per half-hour
  from the farm's own series only: exact-zero flag, repeat-of-previous-value flag, rolling standard
  deviation of first differences, and lag-1 and lag-2 autocorrelation of first differences, each
  over 3-hour and 24-hour trailing and centred windows, amplitude relative to the farm's idle-floor
  p95, and amplitude relative to capacity. The quantisation step is the smallest non-zero absolute
  first difference (reported, not scored). Each statistic's AUC is the larger of AUC and 1 minus
  AUC, with a bootstrap interval over calendar-year blocks (2,000 resamples). Stop the cheap texture
  detector D1 if the best statistic's AUC is below 0.65 at all three farms; otherwise keep it. For
  H2, "high wind" is ERA5 100 m speed above 1.5 times the farm's cut-in from experiment 210, "idle"
  is inside the idle band, "weekday daytime" is Monday to Friday 08:00 to 17:59 UTC, and "weekend or
  night" is all other hours. Keep the whole-farm outage class I2 and the weekday covariate if the
  ratio of the two idle shares is at least 1.5 at any farm (interval reported); otherwise merge I2
  into "unexplained outage".
  Amendment before any number was read (code review): the sure-generating set has power of at
  least 3% of capacity and the sure-idle set sits inside the idle band, so the level of power alone
  separates them at some farms. The gate therefore applies only to level-free statistics (the
  rolling standard deviation of first differences, the lag-1 and lag-2 autocorrelations, the
  repeat-of-previous-value flag and its rolling share), and to each of them as a level-matched
  AUC: the AUC within bins of 0.5% of capacity, averaged with weight equal to the product of the
  two class counts. The exact-zero flag, its rolling share, and the two amplitude statistics are
  level statistics: they are reported raw and never gate D1. A farm whose two sets share no
  power bin cannot test D1; that is reported as untestable and does not count as an AUC below
  0.65. The sign of each statistic is fixed once on the full sample, and the bootstrap resamples
  the signed AUC. A constant window's autocorrelation counts as 0. The high-wind speed for H2 also
  has a sensitivity run with the empirical cut-in of experiment 210. The two sure sets are defined
  by the same wind speed as the 210 mask, so the gate tests texture, not the mask.
  Second amendment, after the weekday result had been read: the maintainer corrected the I2 gate
  because maintenance and waits for components can disable turbines for weeks, so a weekday excess
  is supporting evidence only. I2 is kept if idle runs at high wind (ERA5 speed above 1.5 times the
  cut-in, power in the idle band, no gaps) last longer than 24 hours and longer than the longest
  spell of ERA5 wind below the cut-in in the farm's own series (the longest idle that low wind alone
  could produce). The run definitions were fixed before any run length was read; the post hoc
  variant (power at or below the top of the idle band, gaps of up to 3 half-hours allowed) was added
  after the strict definition returned short runs and is labelled post hoc.
benchmark: >-
  None. This is a descriptive precursor to the wind idle-noise detector design (Stage 0).
row_set_id: >-
  all half-hours of the three wind farms from 2019-09-17 to 2026-09-20 with a power reading, as in
  experiment 210; not comparable with the leaderboard
metric: >-
  AUC of each texture statistic with a year-block bootstrap interval; ratio of idle shares at high
  wind by weekday and hour.
code: ml-experiments/00431-wind-idle-data-look/
verdict: partly supported
production_status: not-for-production
adopted_in: none
best_arm: >-
  Not applicable (no forecast arms). Best level-free texture statistic: the rolling standard
  deviation of first differences over 3 hours (Farm B, level-matched AUC 0.79) and the share of
  repeated values over 24 hours (Farm C, 0.76).
headline: >-
  Idle half-hours differ in texture from genuine low output at the same power level at Farm B and
  Farm C (level-matched AUC 0.79 and 0.76, intervals above 0.70), so D1 stays. Farm A cannot test
  it, because the two sets share no power level. Idle half-hours at high wind are not more common
  on weekday daytimes. Runs of idle half-hours at high wind last up to 2 days (strict definition)
  or 5 days (post hoc) at Farm C and never reach 7 days at any farm.
lessons: >-
  Without level matching, the gate would have passed on power level alone (raw AUC 0.95 at Farm A,
  1.0 for the amplitude statistics), so any later texture detector must be scored within power
  bins. Farm B's negative readings do not depend on the hour of day, and began in 2020.
take_home_message: >-
  A cheap rule that looks at how jumpy a wind farm's recent readings are can tell idle readings
  from genuine low output at the same power level at two of the three farms, so the texture
  detector stays in the design. Idle readings at high wind do not follow a weekday-daytime cycle,
  and the stored series show no idle run of 7 days or more; Farm C has runs of several days that
  low wind cannot explain, so the whole-farm outage class stays for Farm C and is weak at Farm A
  and Farm B. Farm A cannot be tested for texture from its power alone.
advice:
  - >-
    Score any idle-versus-low-output detector within bins of power as a share of capacity, because
    the unmatched AUC reaches 0.95 at Farm A on level alone (AUC table).
  - >-
    Keep the whole-farm outage class I2 for Farm C (idle runs of up to 5 days at high wind, longer
    than low wind explains), treat it as weak at Farm A and Farm B, and make the weekday covariate
    optional (Run lengths, Weekday and hour).
  - >-
    Test the texture detector at Farm B and Farm C first, and at Farm A only on the rows where the
    two classes overlap in power, which the stored labels do not provide (AUC table).
---

**Texture separates idle half-hours from genuine low output at Farm B and Farm C, and idle runs at
high wind are short except at Farm C.** At the same power level, idle readings
have smaller and more repeated changes from one half-hour to the next. The level-matched area under
the receiver operating characteristic curve (AUC) is 0.79 at Farm B and 0.76 at Farm C, so the
texture detector D1 stays in the design. At Farm A the two comparison sets share no power level,
so texture cannot be tested from power alone. The idle share at high wind is lower on weekday
daytimes than at other hours at Farm B and Farm C, but a weekday-only result does not settle the
whole-farm outage class I2, because maintenance and waits for components can disable turbines for
weeks. The run-length test finds idle runs at high wind of up to 5 days at Farm C that low wind
cannot explain, and none of 7 days or more at any farm. Verdict: `partly supported`.

We looked at the three metered wind farms (Farm A and Farm C metered in megavolt-amperes, Farm B
in megawatts) with no fit. The "sure-idle" set is half-hours with ERA5 100 m wind below 2 m/s and
power inside the idle band of experiment 210. The "sure-generating low-output" set is half-hours
with ERA5 wind of at least 6 m/s and power between 3% and 8% of capacity. Both sets are defined by
the same wind speed as the experiment 210 mask, so the test is of texture, not of the mask.

![AUC of each level-free statistic, matched on power level, with year-block
intervals](figure_1_auc.svg)

![Share of high-wind half-hours that are idle, by weekday and
hour](figure_2_idle_share_weekday_hour.svg)

![Idle share on weekday daytimes over other hours, with year-block
intervals](figure_3_weekday_ratio.svg)

![Idle floor by year as a share of capacity](figure_4_floor_by_year.svg)

![Farm B negative readings by hour of day](figure_5_farm_b_negatives_by_hour.svg)

![Farm B against Farm C, which share one ERA5 cell](figure_6_farm_b_vs_c.svg)

![Share of high-wind idle half-hours by the length of the run they sit
in](figure_7_idle_run_lengths.svg)

## Results

- **Gate for D1 (texture):** best level-matched AUC of a level-free statistic is 0.79 (3-hour
  rolling standard deviation of first differences, interval 0.75 to 0.83) at Farm B and 0.76
  (24-hour centred share of repeated values, 0.71 to 0.80) at Farm C. Both are above 0.65, so D1
  is kept. Farm A is untestable: its sure-idle power (0 to 3% of capacity) and sure-generating
  power (3% to 8%) do not overlap in any 0.5%-of-capacity bin.
- **Level alone separates the sets:** unmatched AUCs reach 0.95 (Farm A), 0.93 (Farm B), and 0.85
  (Farm C) for level-free statistics, and 1.0, 0.99, and 0.91 for the amplitude statistics. The
  matched numbers above are the ones that count.
- **Exact zeros and repeats:** 43%, 21%, and 43% of sure-idle half-hours are exact zeros at Farm
  A, B, and C, against none in the sure-generating set (which starts at 3% of capacity). Repeats
  of the previous value are 36%, 19%, and 33% of sure-idle half-hours and 0.1%, 0.2%, and 1.3% of
  sure-generating ones.
- **Quantisation step:** the smallest non-zero change is 0.005%, 0.011%, and 0.037% of capacity
  at Farm A, B, and C.
- **Run lengths (I2 gate):** with the pre-registered definition (power in the idle band, ERA5 wind
  above 1.5 times the used cut-in, no gaps), the longest run is 24 hours at Farm A, 28.5 hours at
  Farm B, and 49 hours at Farm C; 0, 3, and 21 runs exceed 24 hours; none exceeds 7 days. Farm C's
  longest run is just above the longest spell of ERA5 wind below the cut-in (47.5 hours). With the
  post hoc definition (power at or below the top of the idle band, gaps of up to 3 half-hours), 1,
  6, and 81 runs exceed 24 hours, the longest is 36, 55, and 120.5 hours, and the share of
  high-wind idle half-hours that sit in runs of 1 to 7 days is 6%, 3%, and 19%; none reaches 7
  days. With the empirical cut-in as the high-wind speed, no run exceeds 44 hours. So multi-day
  runs that low wind cannot explain appear at Farm C, only marginally at Farm B, and not at Farm
  A. No multi-week run is visible in these series.
- **Weekday and hour (secondary check on H2):** the idle share of high-wind half-hours on weekday
  daytimes (08:00 to 17:59 UTC), over the share at other hours, is 1.01 (0.67 to 1.35) at Farm A,
  0.77 (0.70 to 0.83) at Farm B, and 0.82 (0.77 to 0.86) at Farm C. The ratio never reaches 1.5, so
  the weekday cycle gives no support for planned weekday maintenance. This result alone does not
  settle I2 (see run lengths). With the empirical cut-in of experiment 210 as the high-wind speed
  the ratios are 1.01, 0.91 (0.55 to 1.42), and 0.72 (0.65 to 1.47), all with intervals that include
  1 and none reaching 1.5.
- **Floor by year:** the share of exact zeros in the idle floor falls from over 80% in 2019 to
  about 10% in 2026 at all three farms, and the median floor rises from 0 to about 1.6% (Farm A),
  0.7% (Farm B), and 2.5% (Farm C) of capacity. The floor level is not stable across years.
- **Farm B negatives:** none before 2020. From 2021 on, 33% to 53% of sure-idle readings are
  negative, with no pattern by hour of day (36% to 55% in every hour).
- **Farm B against Farm C:** at high wind, 10,368 half-hours read idle at both farms, 2,634 only
  at Farm B, and 20,018 only at Farm C. Of Farm B's idle half-hours at high wind, 80% are also
  idle at Farm C. A shared ERA5 error or a shared cause (for example a grid event) fits this,
  and so would an idle band that is wide at Farm C. The data cannot say which.
- **Gallery:** 60 farm-days per farm (20 sure-idle, 20 sure-generating low-output, 20 where the
  experiment 210 mask and the texture rule disagree), kept private under the experiment's data
  folder and not published. The disagreement days show flat zero runs at moderate wind that the
  mask does not cover, and low, noisy readings at low wind inside the mask.
- **Checks:** no forecast is scored, so the median-minus-mean, tail, and curtailment checks do
  not apply. Audit triggers: none fired.

## Caveats and what the data cannot show

- **No ground truth.** No outage log, availability record, or reactive-power channel exists on
  disk, so the data cannot say why a half-hour is idle, nor whether the maintainer's reactive-power
  reading holds. Farm B is metered in megawatts and still has idle half-hours.
- **Window statistics include the half-hour's neighbours.** A 24-hour centred window around a
  sure row can include half-hours outside the set, which dilutes contrast. Constant windows count
  as autocorrelation 0.
- **Coarse bootstrap.** The intervals resample about 8 calendar-year blocks, 2019 and 2026 being
  partial, and each class-year is capped at 3,000 half-hours.
- **Run lengths depend on the definition.** One noisy half-hour breaks a strict run, which shortens
  runs; the post hoc gap tolerance lengthens them, and both definitions are shown. The data cannot
  separate a whole-farm outage from an ERA5 error that lasts days, and ERA5's 25 km cell may show
  high wind while the turbines see calm.
- **24 statistics were scored per farm, 17 of them level-free and gated, and the best is quoted.**
  The best value is optimistic, though several statistics agree at Farm B (three above 0.77).
- **The high-wind speed is 1.5 times the used cut-in** (4.5 m/s at Farm B and Farm C), which is
  below their empirical cut-ins and dilutes the idle share; the sensitivity run above shows the
  conclusion does not change.
- **Sure sets are narrow.** Both come from ERA5 wind and a power band, so they cover only the
  clearest cases; a detector must still be tested on the ambiguous rows.

## What to try next

- **Build D1 as a level-matched detector**, scored within power bins, at Farm B and Farm C, then
  decide how to test Farm A (its overlap in power is empty on these labels).
- **Decide the I2 class per farm** from the run lengths, then test it against any outage record
  NGED can supply.
- **Look at the disagreement days with the maintainer** before the Stage 1 hand labels are drawn.
- **Ask NGED** whether a reactive-power channel or an outage log exists, because only that can
  say why a half-hour is idle.

## Experiment details

- **Code:** `ml-experiments/00431-wind-idle-data-look/` (`e00431_analysis.py`, `e00431_gallery.py`,
  `e00431_runs.py`, `e00431_charts.py`), reading experiment 210's `joined.parquet` and
  `report.json`.
- **Review:** one fresh Sonnet code review before the run found a crash in the centred correlation,
  a level leak that would have passed the gate on power alone, NaN rows dropped in the most
  idle-like stretches, and an upward bias from folding the AUC; the fixes were applied and the plan
  was amended before any number was read. The run-length analysis was added after the manager
  relayed the maintainer's correction, once the weekday result had been read.
- **Cadence, cleaned data, and weather:** no fit; the customer-generator series are never flagged
  by the production cleaning rule; weather is ERA5 100 m wind (research-only, about 5 days late).
- **Data:** `data/ml-experiments/00431-wind-idle-data-look/` (private).
