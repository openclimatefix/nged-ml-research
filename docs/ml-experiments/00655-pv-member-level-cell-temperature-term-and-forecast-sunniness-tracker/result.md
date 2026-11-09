---
id: "00655"
slug: pv-member-level-cell-temperature-term-and-forecast-sunniness-tracker
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  complex. Stored: fine-tuned neural-network weights and test-window quantiles in this experiment's
  private folder, plus a per-member extract of the already-downloaded ensemble archive (about 1.5
  GB, under the 10 GB limit). Serving path: not touched. Degradation rule: not touched. Designs:
  more than one way to build a member-level solar term and to date the tracker is defensible, so the
  choices are fixed below before any result is read. Callers: the scripts import the modules of
  experiments 48, 87, 397, 405, 428, 510, 554, 634, 638, 639, and 644 and edit none of them. Asset
  graph: not touched. One GPU process at a time; ceiling 4 GPU-hours (estimate 0.6).
hypothesis: >-
  Experiment 644's explicit solar term lost to the free network, and its physics curve used the
  ensemble-mean shortwave irradiance (no spread), a 35-degree tilt, and a cell temperature equal to
  the air temperature plus 25 K times the output per unit capacity, with no wind term. A curve computed for each of the
  51 ensemble members and then averaged, with a cell temperature in place of the air temperature,
  beats the free network on sunny middays. Separately, the 14-day error correction of experiment 644
  keeps at least half of its gain when it is applied by the ensemble forecast's sunniness instead of
  the observed sunniness.
prediction: >-
  Fixed before any result was read. Part 1: the member-level cell-temperature arm (d) still loses to
  the free network (a) or ties it, by less than experiment 644's explicit network lost (0.37, 0.25,
  and 0.27 points at day 1-2, 3-5, and 6-10); I expect a gain of at most 0.1 points at day 1-2, with
  the largest part of any gain coming from averaging over members (c against c0) and a smaller part
  from the cell temperature (d against c). Part 2: the tracker by forecast sunniness keeps 60% to
  90% of the 644 gain (0.20 to 0.31 points at day 1-2, from the 0.34 gain) and beats the shuffled-error placebo by
  more than the bar.
decision_rule: >-
  Fixed before any result was read. CRPS is in points (percent of effective capacity), pooled over
  the six MW Primaries, the three cutoffs (2025-07, 2026-03, 2026-06), and the sunny half-hours
  (10:00 to 15:00 UTC, observed clear-sky index at least 0.7) of the 60 days after each cutoff,
  averaged over five seeds; a contrast is the control's CRPS minus the arm's, so a positive value
  favours the arm. Intervals are day-block bootstraps at 1 - 0.05/40 (a Bonferroni correction over
  40 contrasts). The bars are experiment 644's: 0.084, 0.093, and 0.123 points at day 1-2, 3-5, and
  6-10. Arm (d) is worth keeping if it beats the free network (a) at day 1-2 by at least 0.084
  with an interval that excludes zero, and loses no more than 0.02 points over all half-hours. The
  tracker by forecast sunniness is worth keeping if it keeps at least half of its 644 gain (0.17,
  0.23, and 0.285 points at day 1-2, 3-5, and 6-10) and beats the shuffled-error placebo by the bar
  at each band. Stop the explicit-PV line if neither holds.
benchmark: >-
  Arm (a), the free network on ensemble inputs, and arm (b), the mean-sky explicit network, both
  re-scored from experiment 644's stored fits on the same rows. The floors (climatology and the
  manual heuristic) are scored on the same rows as in experiment 644.
row_set_id: >-
  Experiment 644's test rows (the 60 days after 2025-07, 2026-03, and 2026-06, cut at 2026-07-01,
  real world only, every ensemble run at lead 0 to 240 hours, six MW Primaries, cleaned power,
  farm-gap rows dropped). The code asserts that the row keys equal those stored by experiment 644.
metric: >-
  Continuous ranked probability score (CRPS) in points from the 11 quantiles of each forecast, on
  sunny middays and on all half-hours, by lead band; mean absolute error of the median forecast in
  points of effective capacity (the PV line's normalisation, the same as experiment 644); coverage
  of the 5% to 95% interval; pinball loss and coverage at the 0.05 and 0.95 quantiles (the stored
  forecasts hold 11 quantiles and no 0.01 or 0.99).
---

# Does a member-level solar term with cell temperature, or a tracker by forecast sunniness, help?

**Planned, no result yet.** [Experiment 644](../00644-pv-explicit-term-with-ens-forecast-weather/result.md)
found that telling the demand network each Primary substation's solar capacity made its forecast
worse than the plain network once the weather was an ensemble forecast. That experiment computed
its solar curve from the ensemble-mean sunshine, with a cell temperature that depended on the output
but not on the wind. Solar output is not a
straight line in sunshine, so the curve at the mean sky is not the mean output over the ensemble's 51
members, and the smooth mean also removes the spread caused by cloud edges. Cell temperature, not
air temperature, sets how much a panel loses to heat, and it can be 20 K above air temperature on
a sunny day. We ask whether a curve built per member, with a cell temperature, changes the verdict.
We also ask whether the 14-day error correction of experiment 644, which that experiment applied
on half-hours observed to be sunny, still works when it is applied on half-hours the ensemble
forecast calls sunny, the only call a live service can make.

## Plan

**Size, one line per question.**

- Stored: fits and quantiles (private folder) and a member-level weather extract (1.5 GB).
- Designs: more than one defensible; the member-level construction, the cell-temperature fit, and
  the tracker's dating are fixed here.
- Callers: imports experiments 48, 87, 397, 405, 428, 510, 554, 634, 638, 639, and 644, editing
  none; the names are in Files.
- Serving path and degradation rule: not touched. Asset graph: not touched.

**Standing practices followed.** Monthly refits (the fine-tune of each month, as experiment 644).
Cleaned power only, with the flagged half-hours dropped, and the D15 and D18 cut not relevant
(the six MW Primaries only). Fitted networks saved as `.pt` in the experiment's data folder. Stored
predictions of experiment 644 reused for arms (a) and (b). Development months only: nothing dated
2026-07-01 or later is read.

**Departures from the brief, all small.**

- Real world only. The brief names the real months' rows and bars; experiment 644's step world is
  not repeated.
- The cell-temperature scale is fitted once per cutoff against the farms, not once per Primary.
  The metered farms (Solar A to F) are not located at the Primaries, so a per-Primary fit would be
  a prior only.
- Arm (c0) is added: the mean-sky curve with the new physics (tilt 20 degrees, air-temperature
  derate). It separates "members" from "new physics" in the contrast of arm (c) with arm (b).
- Experiment 644's curve already had clipping and a cell temperature (output-driven, no wind). The
  new cell temperature differs by being driven by the plane-of-array irradiance and the member wind.

**Arms.**

- (a) free network. (b) experiment 644's explicit network with the mean-sky, air-temperature
  curve (calibrated capacity). Both are re-scored from experiment 644's stored quantiles.
- (c0) explicit network, new physics, mean sky (ensemble-mean irradiance, temperature, and wind),
  air temperature in the derate. (c) as (c0) but the curve is computed for each of the 51 members
  and averaged. (d) as (c) but the derate uses a cell temperature. (e) as (d), plus one network
  input, the standard deviation over members of the per-member curve.
- (pd) capacity-permuted placebo and (sd) shuffled-day placebo, each on arm (d)'s curve, with
  experiment 639's permutation and donor-day rules.
- (f) the 14-day tracker (experiment 644's arm d, observed sunniness), re-scored from the stored
  fits, beside (T-bin), the tracker on half-hours whose ensemble-mean clear-sky index (stored per
  row by experiment 644, the same clear-sky model as the observed index) is at least 0.7;
  (T-cont), weight `clip((kt - 0.5) / 0.3, 0, 1)` on the ensemble-mean index `kt`; and (T-shuf),
  (T-cont-shuf), the same corrections with the correction series shuffled across the days of the
  same Primary and cutoff. The tracker is applied to the free network (a). The history of a row
  with origin date O holds the day-1 forecasts (lead below 24 hours) of the dates O-14 to O-1 on
  half-hours the ensemble calls sunny (weighted for T-cont), so no history uses observed sunniness
  or a date on or after O. Scoring stays on observed-sunny middays; the page also reports each
  tracker's gain on middays where forecast and observed sunniness disagree.

**The physics, fixed here.**

- Plane of array: Erbs split of the shortwave irradiance into direct and diffuse, then the plane of
  array at tilt 20 degrees, azimuth 180 degrees (experiment 474's C1), by `studies.pv_physics`.
- Direct-current to alternating-current ratio 1.15, clipped at the alternating-current capacity.
- Derate: output times `1 - 0.004 (T - 25)`, floored at zero, where T is the air temperature (c0,
  c) or the cell temperature (d, e, pd, sd).
- Cell temperature: `T_air + s * G_poa / (25 + 6.84 * wind)`, with `G_poa` in W/m2, wind the 10 m
  speed in m/s, and the constants of the Faiman model. The scale `s` is 1 in the nominal model.
- The farm fit reads past-weather estimates (the irradiance and 2 m temperature of the frame of
  experiment 39), and its fixed 3 m/s wind differs from the ensemble wind used in the arms. The
  scale `s` is fitted per cutoff on the six farms' sunny-midday output (clear-sky index at
  least 0.7, 10:00 to 15:00 UTC) before the cutoff, against the farm's irradiance and temperature
  from the frame of experiment 39 and a fixed wind of 3 m/s, with a log-normal prior of standard
  deviation 0.25 around 1, and a free amplitude per farm. The Primaries' own demand is never used. The fitted `s` and its posterior width are reported; if
  `s` returns its prior, the page says that arm (d) against arm (c) cannot be resolved.
- Each curve is divided by one constant per cutoff and arm: the 99th percentile, pooled over the
  six Primaries, of the member-mean curve over daylight half-hours of the fine-tuning window. The
  standard deviation over members that arm (e) takes as input is divided by the same constant and
  then standardised on the fine-tuning rows.
- The ensemble members, irradiance, temperature, and wind come from the archive of already
  downloaded ensemble runs (`data/NWP`, 51 members, one 00:00 UTC run per day), cut to each Primary's
  H3 cell and resampled to half-hours by `_upsample_nwp_to_half_hourly`, as experiment 48 did for the
  cached summaries. The extract is checked against the cached summaries (mean over members equals
  the cached mean to float precision) before any fit.
- Fine-tuning starts from experiment 639's pretrained weights, which were pretrained with its
  curve on estimates of past weather. The new curve is substituted at fine-tuning. Arm (b)
  matches its pretraining and arms (c0) to (sd) do not, so contrasts against (a) and (b) mix the
  curve with the mismatch. The contrasts (c) against (c0), (d) against (c0), and (d) against (c)
  are free of it and are reported beside the decision-rule contrast. Re-pretraining is outside this
  plan.

**The 40 contrasts.** Per lead band (3 bands): d-a, d-b, d-c, c-c0, d-c0, e-d, d-pd, d-sd (24);
T-bin-a, T-cont-a, T-bin-T-shuf, T-cont-T-cont-shuf, T-bin-f (15); plus d-a over all half-hours
(1). Each is the control's CRPS minus the arm's.

**Coverage of the ensemble archive.** The archive holds 953 daily 00:00 UTC runs from 2024-04-01 to
2026-11-09 with no missing day. The development months need runs from 2024-04-01 (the start of the
fine-tuning window of the first cutoff) to 2026-06-30, all present.

**Compute.** The extract is CPU only. Fits: 6 new arms (c0, c, d, e, and the two placebos on d)
over 5 months and 5 seeds is 150 fine-tunes of about 11 s each (experiment 644: 300
fits in 0.94 GPU-hours), so about 0.5 GPU-hours, below the 4-hour ceiling.

**Reviews.** A Sonnet plan review, a Sonnet code review before the first fit, a Sonnet review of
the results, and a Sonnet review of the write-up and its prose. Every review is a fresh agent that
starts no sub-agent.

## Method

To be completed with the write-up.

## Files

To be completed with the write-up.
