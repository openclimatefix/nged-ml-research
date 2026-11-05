# The tail metrics of experiment 582, explained with one worked example

**This page explains every number in [experiment 582](result.md) for a reader who has not met
them.** All
figures use one made-up half-hour, so the arithmetic can be followed by hand. Power is in percent of
the
Primary's effective capacity.

## The words

**A load-shift is a logged transfer of load from one Primary to a neighbouring Primary for days or
weeks.** The Primary that gives load away sees its readings fall, and the neighbour sees them rise
by the
same amount. Experiment 580 found the load-shifts that show in the telemetry alone, so only those
are
used here.

**The normal running arrangement (NRA) is the power the Primary would have drawn without the load-
shift.**
Inside a load-shift, the NRA is the observed power minus the load-shift's fitted depth. Outside it,
the
NRA is the observed power. The NRA is what NGED want the forecast to cover, because the network
operator
must plan for the moment the load-shift ends.

**A quantile forecast gives several levels, and a band is the range between two of them.** The
q05-q95
band holds the readings that the network expects to fall between its 5th and 95th percentile, so a
well-calibrated forecast holds 90% of readings inside it. The q01-q99 band should hold 98%.

**A phase is a stretch of valid times measured from the event.** The five phases are the 7 days
before
onset, days 0 to 3, day 4 to the end, 0 to 7 days after the end, and 7 to 14 days after the end. The
forecasts are scored at two lead bands: forecasts issued 0 to 24 hours before the valid time, and 24
to
120 hours before it.

## The worked example

**One half-hour, one Primary.** A load-shift has lowered the Primary's load by a depth of 15 points.
The
NRA at this half-hour is 80, and the observed power is 65. The corrected-lag forecast, which never
saw the
load-shift in its lags, says q05 = 74, q50 = 79, q95 = 85, q99 = 88. The forecast being tested (the
arm)
has followed the load-shift and says q05 = 64, q50 = 69, q95 = 75, q99 = 78.

| Metric | Formula | Worked value |
|---|---|---|
| NRA coverage | 1 if q05 ≤ NRA ≤ q95, otherwise 0; averaged over rows | 80 is not between 64 and 75, so 0 |
| One-sided NRA coverage | The load-shift lowered load, so the NRA lies above the forecast: 1 if NRA ≤ q95, otherwise 0 | 80 ≤ 75 is false, so 0 |
| Tail follow ratio at q95 | (q95 of the arm − q95 of the corrected-lag forecast) × s ÷ depth, where s = −1 when the load-shift lowered load and +1 when it raised load | (75 − 85) × (−1) ÷ 15 = 0.67 |
| Coverage of the corrected-lag median | 1 if the arm's q05 ≤ the corrected-lag q50 ≤ the arm's q95 | 79 is not between 64 and 75, so 0 |
| Pinball loss at level τ | (τ × (NRA − q)) if NRA ≥ q, otherwise ((1 − τ) × (q − NRA)) | At q95: 0.95 × (80 − 75) = 4.75. At q05: 0.05 × (80 − 64) = 0.80 |
| Band width | q95 − q05 | 75 − 64 = 11 |
| Width inflation | Band width of the arm ÷ band width of the recalibrated control R | If R's width is 10, the inflation is 1.1 |

**How to read each one.**

- **NRA coverage** answers "did the band hold the power NGED must plan for?" A value of 0.90 is the
  target. A value far below 0.90 means the forecast has followed the load-shift away from the NRA.
- **The tail follow ratio** answers "how far did the edge of the band move towards the load-shift?"
A
  value of 0 means the edge did not move. A value of 1 means it moved by the whole depth. The worked
  value of 0.67 means the q95 edge dropped by two-thirds of the load-shift. The sign s makes a
positive
  value always mean "moved towards the load-shift", whichever way the load went.
- **The safe-side quantile** is q95 or q99 when the load-shift lowered load, and q05 or q01 when it
  raised load. This is the edge of the band that must stay put to keep the NRA covered.
- **Pinball loss** is the standard score of one quantile. Smaller is better. A quantile that is too
  high or too low is charged in proportion to the miss, and the charge is heavier on the side that
the
  level protects: at q95, a reading above the quantile costs 0.95 per point, and a reading below
costs
  0.05 per point.
- **CRPS** is the continuous ranked probability score. This page approximates it as twice the mean
  pinball loss over the 11 quantile levels.

## Comparing arms without fooling ourselves

**R is the recalibrated control, and every other arm is compared with it.** R is the v3+D5 forecast
with
each level's distance from the median multiplied by a factor, so that on earlier months' rows with
no
labelled event the share of readings below each quantile equals its level. A widening that only
fixes
calibration therefore shows no gain over R.

**The false-alarm cost is the pinball loss minus R's, on rows with no labelled event within 7
days.** A
positive value means the arm is worse than R when nothing is happening. The interval comes from
resampling whole months (5 months here, so close to the minimum and maximum of the monthly values).

**The calibration share is the share of observed readings at or below each quantile on those rows.**
The
target at q95 is 0.95. A share of 0.91 means the q95 edge is too low.

**An event-block bootstrap resamples whole events with replacement.** There are only 6 Primary-
events,
so each interval is descriptive. A leave-one-Primary-out range, which drops each Primary in turn,
shows
whether one Primary decides the result.
