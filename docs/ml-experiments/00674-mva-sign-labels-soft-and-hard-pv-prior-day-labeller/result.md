---
id: "00674"
slug: mva-sign-labels-soft-and-hard-pv-prior-day-labeller
title: >-
  Can a per-Primary day labeller tell whether an apparent-power Primary was exporting, and does a
  soft or a hard PV prior help?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  medium. Stored: small day tables, frozen thresholds, and label counts in this experiment's private
  folder. Serving path: not touched. Degradation rule: not touched. Designs: the PV prior can enter
  a labeller in more than one way, so the three arms and their numbers are fixed below before any
  score is read. Callers: the scripts import the modules of experiments 506 and 508 (data layer,
  sun-elevation day matrices) and experiment 417's W-shape score, and edit none of them. Asset
  graph: not touched. No model is fitted. CPU only, one process, at most 2 cores.
hypothesis: >-
  A per-Primary day labeller that combines the lowest reading at the edges of the sun's path
  (experiment 508) with the W-shape score of experiment 417 finds the days on which an
  apparent-power Primary exported, up to an export of about three times the night import. A soft PV
  prior (the evidence needed to call a day reverse flow rises as the day gets less sunny) keeps the
  recall of a labeller with no prior and lowers its false reverse flow on cloudy days. A hard
  PV-high mask repeats experiment 417's trade: far fewer false calls and far lower recall. Neither
  prior removes the false calls on low-demand days.
prediction: >-
  Fixed before any score is read. All figures pool the six held-out Primaries (month blocks as the
  resampling unit) over days with a day ratio `r` of 1.5 to 3; recall means flagged reverse-flow
  days over all reverse-flow days, with sign-unknown and not-scorable days counted as misses. P1:
arm
  C (no prior) has recall of at least 0.6 and a false-reverse rate of at most 5% (point estimates).
  P2: arm B (hard mask) has at least 40% lower false-reverse rate and at least 25% lower recall than
  arm C (experiment 417: 12.0% to 3.6%, and 0.77 to 0.37). P3: arm A (soft prior) has pooled recall
  within 0.10 of arm C and, on mixed and cloudy days, at most half of arm C's false-reverse rate.
  P4: on low-demand days no arm has both a false-reverse rate of at most 5% and recall of at least
  0.6, and arm C's false-reverse rate there is at least twice its rate on other days. P5: on the
  eight apparent-power Primaries, in arm A the reverse-flow label is on at most 5% of sunny April to
  August days at each Primary except D12, and on at most 5% at D12 among the days that are not
  sign-unknown. D12's share of sign-unknown days is reported but not predicted, because `r` of 3 or
  more is sign-unknown by definition. Arm C labels more reverse-flow days than arm A at D8 and D22,
  with the excess on mixed and cloudy days.
decision_rule: >-
  Fixed before any score is read. Scoring is leave-one-Primary-out over the six signed-MW Primaries
  with injected solar farms at levels 1, 2, and 3 times the median sunny midday reading, plus the
  uninjected world (real reverse flow at D10, 58 crossing days, and D5, 5 days). The training set
  for a held-out Primary leaves out the Primary that carries the same injected farm (D5 with D11, D9
  with D14, D10 with D16), so no farm day is both fitted and scored. A reverse-flow day is a day
  with at least 2 daylight half-hours of negative signed power and at least one of positive signed
  power. Intervals are 95% intervals from resampling month blocks (Primary and calendar month
  together, 5,000 resamples), which is stricter than resampling days; they do not include threshold
  uncertainty. A bar in P1 to P3 is met when the point estimate meets it; it is met firmly when the
  95% interval also lies on the right side of the bar. Verdict: supported if P1 holds for arm C, P3
  holds for arm A, and P2 holds; partly supported if P1 holds for at least one of arms A and C but
  P2 or P3 fails, or P1 holds for arm A only; not supported if P1 fails for both arms A and C;
  inconclusive instead of any of these if the deciding bar is met by the point estimate but not
  firmly. P4 and P5 do not change the verdict; they are reported as findings, with the take-home
  message saying whether they held. If the held-out false-reverse rate of arm C exceeds 10% in
either
  `r` bin, the thresholds count as uncalibrated and every count on the eight is labelled so. No gain
  in forecast score is claimed, because no forecast is fitted, and the threshold for porting is not
  applicable: nothing here enters production. Counts on the eight carry no claim about truth;
  agreement with NGED's "Monitor Direction" labels is reported as agreement with a hint, and the
  plan expects disagreement with the "Yes" at D8 if D8's low readings come from the gas engine.
benchmark: >-
  Experiment 417's export gate (A: 12.0% false reverse, recall 0.77) and hard high-irradiance gate
  (A+H: 3.6%, recall 0.37); experiment 508's shoulder windows (area 0.85 and 0.88 at export 1.5 to 3
  times); NGED's "Monitor Direction" labels as a hint.
row_set_id: not_applicable_no_forecast
metric: >-
  Day-level recall, false-reverse rate, share labelled sign-unknown, and share not scorable, by arm,
  export-ratio bin, and PV stratum, each with a month-block bootstrap interval; label counts per
  class on the eight Primaries.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/
---

## Plan

**What we are asking.** Eight Primaries report apparent power, which is never negative, so a day on
which a Primary exports to the grid looks like a day of low demand. We test a simple day labeller
on known answers (six Primaries that report signed power, where we add the recorded output of real
solar farms and hide the sign) and then apply it to the eight with no claim about truth. The
labeller reads two features of each day: the lowest reading at the edges of the sun's path
(experiment 508) and how well a V fits the demand-removed reading (the W-shape score of experiment
417). Three arms decide how the solar prior enters. No forecasting model is fitted.

**Reading the plan.** `r` is a ratio of observed magnitudes, not a measured export: on an injected
day it can look like an ordinary high-demand day, so the 1.5 to 3 bin mixes both. A Primary (NGED's
33 kV to 11 kV substation) reports signed real power in
megawatts (six of them, D5, D9, D10, D11, D14, D16, used for the known-answer tests) or apparent
power in megavolt-amperes (eight, D2, D3, D4, D8, D12, D19, D21, D22). Readings are in points of the
Primary's capacity. The night import `I` of a Primary is its median reading when the sun is more
than 6 degrees below the horizon, from April to September, before 2026-07-01. The day export ratio
`r` is the day's median reading from 10:00 to 15:00 UK clock time divided by `I`. A day is
reverse flow when its signed power is negative somewhere in daylight and positive elsewhere.

**Day features (both computed from readings only).**

- **Shoulder minimum `s`.** The lowest reading in the morning and the evening shoulder windows of
  experiment 508 (sun 5.6 to 19.0 degrees above the horizon), divided by `I`, taking the smaller of
  the two windows. A day needs at least 3 finite half-hours in each window.
- **W-shape score `w`.** Experiment 417's limb score (`e00417_day_flip.score_days`), built on a
  `Table` made from the Primary's readings and CAMS arrays. The score is copied into this
experiment's folder, so no earlier script is edited. Lab shortcut:
  the ERA5 temperature term is dropped (the copy has none), so the demand profile is an overcast-day
  median by block, day type, and hour with no temperature slope, and bank holidays are not used. The
  profile is built from the readings of the world being scored, with no labels, so it uses the
  scored days too (hindsight, a lab shortcut).
- **Sign-unknown class.** A day with `r` of 3 or more is labelled sign-unknown, because experiment
  508 found that nothing calibrates there (over 98% of injected days cross zero). A day with neither
  `s` nor `w` computable is not scorable; a day with one feature is scored on that feature alone.
  `w` needs a daily maximum of the clear-sky shape of at least 0.6 (experiment 417's rule, and the
  shape includes cloud), so on most cloudy days the labeller rests on `s` alone. Solar-farm dropouts
  also remove half-hours from the injected worlds (experiment 508 found about 22% of days
  qualify), which raises the not-scorable share there. Sign-unknown and not-scorable days count as
  "not flagged" in recall (the conservative reading) and are reported on their own.

**Arms (all use leave-one-Primary-out thresholds frozen before the held-out Primary is read).** A
day is flagged when `s` is below `tau_s` or `w` is above `tau_w`. Each threshold is the value at
which the rule alone flags `alpha / 2` of the non-reverse training days of the same `r` bin (below
1.5, and 1.5 to 3) and the same stratum. A cell with fewer than 30 training days pools the two `r`
bins, then all strata; the number of fallbacks is reported.

- **C, no PV prior.** One stratum for all days, `alpha = 5%`.
- **B, hard PV-high mask.** Only days in the PV-high stratum can be flagged, with `alpha = 5%`
  calibrated on those days; all other days are labelled no reverse flow. Reproduces the idea of
  experiment 417's A+H arm.
- **A, soft PV prior.** Three strata with `alpha` of 5% (PV-high), 2.5% (mixed), and 0.5% (cloudy):
  the evidence needed to flag a day rises where solar output is less likely, and never becomes
  infinite. The 1 : 0.5 : 0.1 ratio is a pre-registered choice and a sweep (post hoc, labelled) over
  1 : 1 : 1 and 1 : 0.25 : 0.05 is reported.
- **Strata.** PV-high: mean clear-sky index `k` (CAMS) from 10:00 to 15:00 of at least 0.7 and the
  capacity gate holds. Cloudy: `k` below 0.3. Mixed: the rest. Capacity gate: the Primary's
  AC p90 PV capacity in the latest season-year of experiment 474's R4 table (plus, in the injected
  worlds, the injected farm's true size, an oracle lab shortcut) is at least 0.5 times the night
  import in MW. The gate is circular at the eight (experiment 474 fitted that table on the same
  readings at reactive power zero and its placebo failed), so a post hoc sensitivity run drops the
  gate and strata use `k` alone.

**Known-answer test.** The six Primaries rotate as held-out Primary; the other five at levels 0 (no
injection), 1, 2, and 3 calibrate the thresholds. Farms Solar A to C rotate over the six as in
experiment 408, and no generator's name or series appears on the page. April to September days
only for the headline; all months for the counter-example in D12's winter dips. Report, per arm, per
`r` bin: recall (flagged reverse-flow days over all reverse-flow days, and over those not
sign-unknown), false-reverse rate (flagged over all non-reverse days), the share sign-unknown, the
share not scorable. **Low-demand days** are the 20% of each Primary's April to September days with
the
lowest median 10:00 to 15:00 reading before injection, reported separately. D5 and D10's real
crossing days (level 0) give a no-injection recall (63 days, so the interval will be wide).

**Application to the eight.** Thresholds are frozen from all six Primaries and written to disk
before any of the eight is read. Report label counts per class (reverse flow, no reverse flow,
sign-unknown, not scorable) per Primary, arm, and stratum, for April to September and for sunny
April
to August days. Agreement with NGED's hint is reported as agreement with a hint (D8 "Yes", D12 and
D22 "Maybe", the other five "No"): a Primary agrees when the reverse-flow share of sunny April to
August days is above 5% for "Yes" and at most 5% for "No", and "Maybe" makes no claim.

**Counter-examples and how each arm treats them.**

- **A signed-MW series negative on 19,054 of 24,058 rows at all hours (Wrangle T1).** Not scored,
  because experiment 412's CAMS arrays hold only the 16 demand Primaries (checked in the folder). By
construction arm B never flags a day outside the PV-high
  stratum, and arm A needs strong evidence there; arm C has no such protection.
- **D8's gas engine pulling apparent power to zero without sun.** Flagged days at D8 split by
  stratum, per arm; if experiment 508's stored day table holds an engine flag, the overlap is
  reported too.
- **D22's unexplained dips.** The 13 dip days of experiment 508 against each arm's label.
- **D12's 13 midday dips between October and February.** The labeller is calibrated on April to
  September, so winter days are outside its calibrated domain (low sun, few shoulder half-hours).
  Report each arm's label on those 13 days, most of which may be not scorable or sign-unknown, and
  count them as an extrapolation, not a test.

**Standing practices.** Cleaned data throughout (flagged half-hours dropped by `scan_power`),
nothing
dated 2026-07-01 or later is read (the CAMS arrays are cut at that date), the D15 and D18 cut does
not
apply (neither is used). Monthly refits, saved models, stored predictions, tails and curtailment
cost, forecast examples, and load-shift event plots do not apply, because no forecast is made.
Caveat to repeat: NGED's event logs are incomplete, so clean days may hold unlogged load shifts and
counts are lower bounds.

**What the earlier advice implies, and where the plan departs from it.** Follows experiment 508's
advice to use sun-elevation shoulders and not the 10:00 to 15:00 minimum, not to calibrate above
three times the night import (hence the sign-unknown class), and to check a shoulder dip against
cloudy days (the strata). Follows experiment 417's advice that a hard mask costs recall. Departs
from
nothing.

**Reproducing experiment 417.** The write-up says which of 417's two figures arm B reproduces: its
drop in false-reverse rate (12.0% to 3.6%) and its drop in recall (0.77 to 0.37), side by side with
this experiment's day-level figures, which use different rows.

**Plan review (Sonnet, 2026-11-09).** Twenty-two findings. Accepted and applied above: the complete
verdict mapping, the point-versus-firm bar wording, the P5 bars, the farm-leakage fix, the minimum
cell size, the 2-half-hour crossing rule, the W-score cloud limit, the winter domain, the hindsight
and oracle notes. Not applied: threshold uncertainty in the intervals (stated as a limit).

**Reviews.** One Sonnet plan review, one fresh Sonnet code review before any score is read, one
Sonnet results check (re-derives the headline numbers), one Sonnet prose review. The audit triggers
of the `ml-experiment` skill run on the finished numbers.
