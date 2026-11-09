---
id: "00674"
slug: mva-sign-labels-soft-and-hard-pv-prior-day-labeller
title: >-
  Can a per-Primary day labeller tell whether an apparent-power Primary was exporting, and does a
  soft or a hard PV prior help?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: finished
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
  days over all reverse-flow days, with sign-unknown and not-scorable days counted as misses. P1: arm
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
  uninjected world (real reverse flow at D10, 54 crossing days, D5, 3, and D9, 2). The training set
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
  message saying whether they held. If the held-out false-reverse rate of arm C exceeds 10% in either
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
verdict: inconclusive
best_arm: >-
  Arm C (no PV prior), export 1.5 to 3 times night import: recall 0.644 (0.598 to 0.690) at a
  false-reverse rate of 0.030 (0.014 to 0.055), on injected solar output only.
headline: >-
  The day labeller finds about 64% of injected reverse-flow days at a 3% false rate when export is
  1.5 to 3 times the night import, but only 5 of 59 real reverse-flow days at the signed-MW
  Primaries, and on the eight apparent-power Primaries it flags 39% of D8's and 13% of D4's sunny
  days, so it cannot yet be trusted to label them.
take_home_message: >-
  A day labeller built from the lowest reading at the edges of the sun's path and a W-shape score
  finds injected solar export well but misses almost all real reverse-flow days, which are shallow
  and short, so it is not a usable sign label for the eight apparent-power Primaries. A hard PV
  mask removes most false calls at little cost in recall on injected days, but on real data the
  soft prior is no better than no prior. A simple rule on the midday reading (day ratio below 0.3)
  found 52 of the 59 real days, so the next labeller should start from the midday dip and treat
  the shoulder minimum as a second, separate signal.
lessons: >-
  Injected farms make deep, long, sun-shaped exports, so a labeller validated on them can score
  well and still miss real shallow exports. The shoulder minimum is not specific at a Primary whose
  day ratio is 1.5 to 3 and whose own shoulders dip often.
advice:
  - >-
    Do not validate a sign labeller only on injected farm output; it missed 54 of 59 real
    reverse-flow days at D5, D9, and D10 that it caught 64% of when injected (Results, real days).
  - >-
    Treat a labeller trained on injected days with suspicion at any Primary whose day ratio is
    1.5 to 3 on most days (D4: 71% of days); its shoulder threshold of 0.85 times night import
    flags 13% to 44% of D4's sunny days (Results, the eight).
  - >-
    Keep the midday-dip rule (day ratio below 0.3) as the starting point for real shallow exports;
    it found 52 of 59 real days at 3.2% false calls, mostly low-demand days at D5 (Checks, post
    hoc).
  - >-
    Use a hard PV mask where false calls matter more than recall: it cut false reverse by 68% at
    export 1.5 to 3 times for a 14% loss of recall on injected days, but it needs a capacity table
    that is not circular at the eight (Results, arms).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/
---

## Summary

**Take-home.** A day labeller built from the lowest reading at the edges of the sun's path and a
W-shape score finds injected solar export well, but real reverse flow at the signed-MW Primaries is
shallow and short, and the labeller finds 5 of 59 such days. On the eight apparent-power Primaries
it
flags a large share of days at D8 (39% of sunny April to August days) and D4 (13%), where NGED's
hint says "Yes" and "No". So the labeller is not yet a usable sign label for the eight. A simple
midday-dip rule does far better on real days, which narrows the next attempt to that rule. Verdict:
`inconclusive` under the registered rule, because the bar of P1 was met by the point estimates but
not firmly, and P2, P3, and P5 failed. All numbers below are validated on injected days or on 59
real
days only, not on truth at the eight. NGED's event labels are incomplete, so counts are lower
bounds.

**What we asked.** Eight of NGED's primary substations report apparent power, which is never
negative, so a day on which one exported to the grid looks like a day of low demand. We built a
simple per-day labeller (the lowest reading at the edges of the sun's path, and how well a V fits
the demand-removed reading), tested it on six substations that report signed power (adding recorded
output of real solar farms and hiding the sign), and applied it to the eight. Three arms differ in
how they use the expectation that export happens on sunny days: no prior (C), a hard sunny-days
mask (B), and a soft prior (A). The method re-uses experiment 508's shoulder windows and
[experiment 417's W-shape score](../00417-hidden-sign-mva-export-gate-zero-crossing-labels/result.md);
nothing is fitted.

## Figures

*Figure 1: recall and false-reverse rate by arm on the six, leave-one-Primary-out, 95% month-block
intervals.*

![Recall and false-reverse rate by arm](figure_1_recall_and_false_reverse_by_arm.svg)

*Figure 2: share of April to September days in each label, on the eight.*

![Labels on the eight](figure_2_labels_on_the_eight_apparent_power_primaries.svg)

*Figure 3: example days with the reading and each arm's label (the real D10 days first).*

![Example days](figure_3_example_days_and_labels.svg)

## Results

- **Known answers, export 1.5 to 3 times night import (1,004 reverse-flow days, all injected).**
  Recall, false-reverse: C 0.644 (0.598 to 0.690), 0.030 (0.014 to 0.055); B 0.553 (0.503 to 0.602),
  0.010 (0.002 to 0.021); A 0.578 (0.525 to 0.630), 0.038 (0.018 to 0.067). Share not scorable
  0.216;
  no day is sign-unknown on the six.
- **Pre-registered predictions.** P1 holds by the point estimates for C but not firmly (both
  intervals cross the bars), and fails for A (recall 0.578). P2 fails: B cuts false-reverse by 68%
  (bar 40%) but recall by only 14% (bar 25%); experiment 417's drop was larger (0.77 to 0.37). P3
  fails: A is within 0.10 of C on recall, but on mixed and cloudy days A's false-reverse rate is
  0.034 against C's 0.030, not half. P4 holds vacuously, because no arm reaches recall 0.6 on any
  subset; C's false-reverse rate is 0.104 on low-demand days against 0.026 on other days (pooled
  below 3 times). P5 fails (below).
- **All days below 3 times night import, pooled.** Recall C 0.314 (0.290 to 0.337), B 0.191
  (0.174 to 0.208), A 0.277 (0.254 to 0.300); false-reverse 0.036, 0.018, 0.029. Not scorable: 0.237
  (injected farm dropouts remove half-hours).
- **Real reverse-flow days (59: D10 54, D5 3, D9 2).** C flags 5, A and B flag 2, A_flat 4. The
  real days are shallow midday exports (median day ratio 0.15) whose shoulders stay near the night
  import (median shoulder ratio 0.83); the W-shape score is 0 or null on 56 of 59.
- **The eight, A, sunny April to August days.** Reverse-flow label: D2 0.2%, D3 1.1%, D19 0.8%,
  D21 0%, D22 0.9% (all at most 5%); D4 13.3%; D8 38.9%; D12 5.7% of all days and 63.4% of the days
  that are not sign-unknown (91.0% are sign-unknown, day ratio of 3 or more). Without the capacity
  gate D4 rises to 44.3% for arms B and A. Hint agreement: D8 "Yes" agrees; D12 and D22 "Maybe" make
  no claim; D2, D3, D19, and D21 "No" agree; D4 "No" disagrees.
- **Counter-examples.** D8: 42 of the 44 dip days of experiment 508 that lie in April to September
  are labelled reverse by C and A, and 32 of those also have the engine detector flag, so the
  labeller does not separate the gas engine from solar. D22: its 11 dip days in the table are all
  labelled no reverse flow by every arm. D12's winter dips: 27 of 28 days with a midday reading
  below
  6.9 points are labelled reverse by C and A (B: no reverse flow; the thresholds are calibrated on
  April to September, so this is an extrapolation, with non-dip winter days flagged at about 18%
  per the audit). A signed-MW series negative on 19,054 of 24,058 rows was not scored: the CAMS
  arrays hold only the 16 demand Primaries.
- **Soft-prior sweep (post hoc).** A_flat (5%, 5%, 5%) and A_steep (5%, 1.25%, 0.25%) move recall
  by under 0.04 and do not change a conclusion.

## Checks

- **Median minus mean forecast per site.** Not applicable: no forecast is made.
- **Mean-forecast error beside quantile contrasts.** Not applicable.
- **Triggers fired (audit by a fresh Sonnet agent, `briefs/674_audit.md`).** Trigger 1 (real days
  missed), trigger 2 (P2, P3, and the contradiction with experiment 417), and trigger 4 (leaving out
  D10, which holds 453 of the 626 known non-reverse days at the headline bin, makes P3 true on a
  small base and reverses the false-reverse contrast of B against C). The audit reproduced every
  headline number and the leave-one-Primary-out thresholds. Findings accepted and written into this
  page: the headline bin holds only injected days (9% of reverse-flow days), the W-shape threshold
  is
  degenerate (zero in every 1.5 to 3 cell, so the realised training flag rate is about 2.6%, not
  5%), the D4 flag rate follows from its day ratio (71% of D4's days lie in the 1.5 to 3 bin against
  7% of the six's uninjected days), and D8's flags come from the below-1.5 bin (the shoulder
  minimum is bimodal). Not applied: none rejected.
- **Post hoc (audit finding, reproduced in the printed report).** The rule "day ratio below 0.3"
  flags 52 of the 59 real days and 198 of 6,231 other days (3.2%; D5 160, D10 32, D9 6).
- **Lab shortcuts.** ERA5 temperature dropped from the W-shape profile; the capacity gate uses the
  injected farm's true size in the injected worlds; the profile is built on the scored days.
- **Resampling unit.** Month blocks (Primary and calendar month, four levels pooled), which is
  stricter than resampling days; threshold uncertainty is not in the intervals.
- **Data.** Cleaned data, nothing dated 2026-07-01 or later. Thresholds frozen from the six before
  the eight were built (file digest starts 7cf8570b). Models: none fitted. Code and printed report:
  `ml-experiments/00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/`.

## What to try next

- Start from the midday dip (day ratio below 0.3), which found 52 of 59 real days, and test it at
  the eight with the shoulder minimum as a separate second signal.
- Calibrate on real days only where they exist (D10), and score on injected days as a stress test.
- Ask NGED for a few weeks of signed power at D12 and D8, which would settle both without a fit.

## Plan (as registered)

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
crossing days (level 0) give a no-injection recall (59 days, so the interval will be wide).

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
and oracle notes. Also applied after the code review: days with no ratio count as not scorable, the
false-reverse rate is reported over both known-label days and all non-reverse days, and low-demand
membership no longer depends on the injection level. Not applied: threshold uncertainty in the
intervals (stated as a limit).

**Reviews.** One Sonnet plan review, one fresh Sonnet code review before any score is read, one
Sonnet results check (re-derives the headline numbers), one Sonnet prose review. The audit triggers
of the `ml-experiment` skill run on the finished numbers.
