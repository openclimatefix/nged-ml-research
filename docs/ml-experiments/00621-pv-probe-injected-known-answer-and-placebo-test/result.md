---
id: "00621"
slug: pv-probe-injected-known-answer-and-placebo-test
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: done
kind: other
size: >-
  medium. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  injection and probe design is defensible, so the worlds, the probe regression, and every pass bar
  are fixed below before any recovery is read. Callers: this folder's scripts import experiments 405,
  409, 448, 510, and 554 and edit none of them. Asset graph: not touched. One GPU process, about 0.7
  GPU-hours.
hypothesis: >-
  A lag-free neural network pretrained on ERA5 and CAMS estimates of past weather, probed with its
  datetime inputs held fixed and donor-day weather swapped in, returns the AC capacity that was
  injected into its training target at the six MW Primaries (raw recovery 0.85 to 1.15), reads a
  profile-only and a shuffled-sky placebo near zero, and is not fooled by a cloud-following load.
prediction: >-
  Fixed from the Opus design review of 2026-11-07 before any recovery was read. (1) Median recovery
  of a 4 MW real-farm injection is 0.75 to 0.90 and lies in 0.85 to 1.15 at 5 of the 6 Primaries;
  for 2 MW at 4 of 6. (2) The physics-curve world recovers more than the real-farm world. (3) Each
  placebo reads under 0.20 of the injected MW (95th percentile over cutoffs and seeds) at 5 of 6.
  (4) With the cloud template, the cloud-following load fakes under 0.15 of a 2 MW farm. (5) The
  nominal 80% interval covers the injected capacity in at least 70% of injections in the physics-curve
  world and in the linear cloud-load world.
decision_rule: >-
  Fixed before any recovery is read. Each bar passes or fails as written in the prediction; a failed
  bar names its world and stops the line with no rescue tuning. The step world is not run (see the
  deviations). Recovery is the probed capacity of the injected network minus that of the uninjected
  twin with the same seed, divided by the injected megawatts; it is never divided by the recovery at
  other farms. The threshold for acting on a probe capacity in production is not set here.
benchmark: >-
  The uninjected twin (same cutoff and seed, same rows). Experiment 474's rung R4 and experiment
  479's anomaly rung for the real-data capacities, as context only.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, half-hour ends from 10:00 to 15:00 UTC.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day regression; recovery as a share of the
  injected megawatts; day-block bootstrap 80% interval.
production_status: not-for-production
adopted_in: none
verdict: not supported
best_arm: >-
  None. The physics-curve world is recovered best (0.95 to 1.03 of the injected MW at each Primary and
  size); the real-farm world is read at 0.65 to 0.72.
headline: >-
  The probe reads an injected real farm at 0.68 of its megawatts (median over 2 cutoffs, 3 seeds, 2
  rotations; 0.66 to 0.70 at each of the six MW Primaries), against a pass
  bar of 0.85 to 1.15. Both placebos pass (95th percentile 0.08 to 0.12 and 0.09 to 0.16, bar 0.20).
  Interval coverage is 0.68 in the physics-curve world (bar 0.70, fail) and 0.72 and 0.69 with a weak
  and a strong linear cloud-following load.
lessons:
  - >-
    The probe is precise and biased: the median seed standard deviation of the capacity difference is
    0.10 MW and the bootstrap standard deviation 0.02 MW, yet the real farm is read at 0.68 while the
    physics-curve world is read at 0.98. The shortfall is therefore consistent with a mismatch between
    a real farm's output and the probe's unit curve, which this experiment did not test.
  - >-
    The cloud template removes the cloud-following load but removes most of the real PV with it:
    real-farm recovery falls to a median of 0.08, and the cloud-load read-out turns negative (down to
    -0.68 of a 2 MW farm).
  - >-
    A fixed calendar does remove the profile failure of experiments 405 and 477: the profile-only
    placebo reads 0.08 to 0.12 (95th percentile).
take_home_message: >-
  A neural network that is probed with its calendar held fixed reads an injected solar farm at about
  two thirds of its true megawatts (0.68, tightly), reads the two placebos at about 0.1, and is
  fooled by a cloud-following load unless a cloud template is added, which also removes most of the
  real PV. The probe is stable enough to be useful as a scaled estimate, but its intervals do not
  cover the truth and it cannot separate PV from cloud-following demand, so it validates no capacity.
advice:
  - >-
    Do not report a probe capacity as a measurement; report it beside a divided estimate (probe over
    0.68) and say the division rests on one known-answer test at one set of farms.
  - >-
    Do not add a flat cloud template to the donor-day regression: it absorbs the real PV as well as
    the load.
  - >-
    Before trusting an interval from the probe, widen it by a bias term as well as the seed spread:
    the bootstrap and seed spread together cover the truth at 0 of 144 real-farm injections.
---

# Injected known-answer and placebo test of the donor-day PV probe

**Take-home: the donor-day probe reads an injected real solar farm at about 0.68 of its megawatts,
so it fails the 0.85 to 1.15 bar, and the line stops here with no rescue tuning.** The reading is
stable (0.66 to 0.70 at each of the six MW Primaries, and 0.65 to 0.72 when split by cutoff and size), so a
divided estimate is possible. The probe separates a real farm from a profile-only and a shuffled-sky
placebo, which the neural attempts of experiments 405 and 477 could not. It cannot separate PV from
a load that follows the clouds: without a cloud template a strong linear load reads as 0.39 to 0.45
of a 2 MW farm, and with the template the real PV is mostly removed too.

The question: if a lag-free neural network (experiment 554's modular network M0, pretrained on ERA5
and CAMS estimates of past weather) is trained on data with a known PV farm added at the six MW
Primaries, does a probe with its datetime inputs held fixed read the injected capacity back? Each
injected network is compared with an uninjected twin of the same cutoff and seed. The cutoffs are
2026-03 and 2026-06, with 3 seeds each; 15 conditions give 90 pretrainings of about 22 s each on the
GPU (about 0.7 GPU-hours in all).

![Recovery by world and Primary](recovery_by_world.svg)

![Probed against injected capacity](recovered_vs_injected.svg)

![Cloud-following load read as PV](cloud_load_faking.svg)

**Pass or fail, as pre-registered** (a bar passes at 5 of 6 Primaries; 4 of 6 for the 2 MW farm):

| Bar | Result | Pass |
|---|---|---|
| Real farm 4 MW inside 0.85 to 1.15 | 0 of 6 (medians 0.65 to 0.71) | fail |
| Real farm 4 MW median 0.75 to 0.90 (prediction 1) | 0.68 pooled | not met |
| Real farm 2 MW inside 0.85 to 1.15 | 0 of 6 (medians 0.67 to 0.70) | fail |
| Physics-curve world recovers more than the real farm (prediction 2) | 0.98 against 0.68 | met |
| Profile-only placebo, 95th percentile under 0.20 | 6 of 6 (0.08 to 0.12) | pass |
| Shuffled-sky placebo, 95th percentile under 0.20 | 6 of 6 (0.09 to 0.16) | pass |
| Cloud load under 0.15 of 2 MW with the template | 6 of 6, all four loads, but negative, down to -0.68 | pass, uninformative |
| 80% interval covers truth, physics-curve world | 0.68 (bootstrap only 0.10) | fail |
| 80% interval covers truth, linear cloud load (weak, strong) | 0.72, 0.69 | pass, fail |

- **Arms.** Real farm (farms A, B, C, F, two rotations, 2 MW and 4 MW), physics curve driven by the
  Primary's own CAMS shortwave (2 MW and 4 MW), profile-only and shuffled-sky placebos (4 MW,
  two rotations), cloud-following load (linear and squared, averaging 0.10 and 0.30 of 2 MW over
  daylight half-hours), and the uninjected twin.
- **Absolute readings (uninjected twins).** Probed effective AC capacity 1.7 to 2.6 MW per Primary
  (D5 2.0, D9 1.7, D10 1.7, D11 2.4, D14 2.7, D16 2.1), 2.07 MW at the 2026-03 cutoff and 2.10 MW at
  2026-06 on average, with a seed standard deviation of 0.04 to 0.13 MW. Whether these are right is
  untested.
- **Physics-curve world.** Recovery 0.95 to 1.03 at each Primary and size (0.81 to 1.17 for single
  networks); this world uses the probe's own regressor as the injected shape, so it shows what the
  probe loses when the shape is known exactly (about 2%) and is not evidence for the real-farm
  worlds.
- **Cloud load without the template.** A weak linear load reads as 0.11 to 0.16 of 2 MW, a strong
  linear load as 0.39 to 0.45, and a strong squared load as 0.56 to 0.66. The template removes these
  but also cuts the real-farm recovery to a median of 0.08 (0.04 to 0.16 by cutoff and size), and the
  strong squared load reads as -0.38 to -0.68, so the template is not a usable fix.
- **Interval.** The interval is the day-block bootstrap widened by the across-seed spread, as
  pre-registered. It is too narrow because the error is bias, not noise: it covers the injected
  capacity at 0 of 144 real-farm injections.
- **Deviations (fixed before the grid was read).** The network has no time input, so the step world
  is not run. Farms D and E are left out for their daylight gaps, so the six Primaries take farms A,
  B, C, F, A, B. Half-hours with a gap in any of those four farms (15.5% of the six Primaries' rows at
  the 2026-06 cutoff) are dropped from every arm. The cloud load runs at two sizes.
- **Not run.** The contrastive variant (a synthetic +1 MW of PV added to net demand and compared with
  the uninjected rows) was not run here; another session runs contrastive injections on this
  pipeline.
- **Caveats.** The cloud load is a UTC-day mean of the cloud response, which the probe's single
  donor half-hour only partly sees; one injected farm shape per Primary and rotation; the probe is
  tested on one network family (stage `pre` of M0, no lags, no drift input).
- **Reuse.** Code: `ml-experiments/00621-pv-probe-injected-known-answer-and-placebo-test/`; it imports
  experiments 405, 409, 510, 554, and 448 and edits none of them.
- **What to try next.** (1) Decide whether to publish the probe's capacity divided by 0.68, with an
  interval widened for bias. (2) Test whether the 0.68 shortfall comes from the unit curve
  (orientation, tilt, clipping) by injecting the physics curve with a different tilt. (3) Keep
  experiment 474's R4 estimate as the main capacity table, with the probe as a second, independent
  reading.
