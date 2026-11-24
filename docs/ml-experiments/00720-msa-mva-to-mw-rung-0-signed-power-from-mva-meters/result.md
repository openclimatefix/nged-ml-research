---
id: "00720"
slug: msa-mva-to-mw-rung-0-signed-power-from-mva-meters
title: >-
  Can the sign of an apparent-power (MVA) Primary's reading, and the split of that reading into
  demand, solar output, and a generator state, be recovered in hindsight on the CPU with no neural
  network?
theme: solar-disaggregation
also_relevant_to:
  - switching-labels-in-the-forecast
  - generator-forecasting
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-world scores and private posterior tables for six Primaries, two
  injection levels, and four folds; (2) more than one design defensibly tests the hypothesis (two
  sign read-outs and a switching model), so one experiment compares them; (3) it imports code from
  experiments 412, 417, 420, 694, and 716, and the callers of that code cannot be listed without
  searching the repository; (4) it touches no serving path or degradation rule; (5) it changes no
  contract. Full set of reviews: plan, code (Opus, each new script, twice), results, write-up, prose.
hypothesis: >-
  The sign of a Primary's apparent-power reading can be recovered in hindsight from the readings,
  the CAMS sky, and F7's estimate of unmetered solar output, without a neural network, because
  gross demand at one half-hour of the day and day type does not depend on the sky once calendar
  and temperature are given. A gross-demand anchor built from dull-sky rows (rung 0b) therefore
  separates export half-hours from import half-hours better than the same rows with the sky shifted
  by 28 days, and a switching state-space model (rung 0e) labels the sign and a generator state at
  least as well as experiment 694's tracker.
prediction: >-
  Written before any fit. Rung 0a finds negative half-hours on 0.76% of D10's fold half-hours, on
  none at D5, D9, D11, D14, and D16, and on 69.1% of D15's. Rung 0d finds the export gate open at D8
  and D12 and shut at the other six MVA-only Primaries, with D22 closest to the line. Rung 0b finds
  that the anchored read-out beats its shifted-sky repeat by at least 0.1 in area under the receiver
  operating characteristic curve, and calls reverse flow on under 1% of D5's sunny middays with the
  gate forced open. Rung 0e passes V1, the paired V2 contrast has a lower bound above 0, the 1 MW
  schedule scores under 0.3 F1, and 0.02 to 0.05 of running half-hours in the era before the solar
  farm registered (the era table is private) have a largest unit-count posterior under 0.7. The
  prediction fails if any of these misses.
decision_rule: >-
  Rungs 0a to 0d. `supported` when 0a, 0c, and 0d reproduce their expected results and the anchored
  read-out passes the stop rule of rung 0b (area with the real sky at least 0.1 above the shifted
  sky, and false reverse flow under 1% of D5's sunny middays with the gate forced open) and its
  references bar (it beats the reading-blind and the hour-of-day references, which do not read the
  meter, with the lower end of the paired interval above 0 on the pooled area; the bar is in the
  table in the section "Rung 0e" next to the other bars of rung 0b). `partly supported` when the
  anchored read-out fails and the unanchored control passes, or when the anchored read-out passes
  its stop rule and does not beat a reference, because the sign is then no better than the PV
  estimate or the hour of day alone. `not supported`
  when both fail. `invalid` when a known-answer check of 0c fails. `inconclusive` otherwise. Rung
  0e has its own verdict, read from the bar table in the section "Rung 0e" (V1, the paired V2
  contrast, K1, V3, the Brier reliability term, and the signed CRPS). Both fail gives `not
  supported` and steps 4 and 5 of rung 0e do not run. Every verdict states that power factor 1 is
  assumed and that every score is a hindsight ceiling.
benchmark: >-
  The gated solar rule of experiment 506 and the day labeller of experiment 674 on the same injected
  rows (hindsight references); experiment 694's M2 rerun on the same segments (rung 0e); the
  shifted-sky repeat of each read-out (placebo).
row_set_id: >-
  not applicable (no forecast is scored): rung 0 labels past half-hours of the folds 2025-12 and
  2026-04 to 2026-06, and no valid time on or after 2026-07-01 is read.
metric: >-
  Area under the receiver operating characteristic curve and recall at a 5% false-reverse rate (2
  points of capacity dead band) for the sign; F1 for "any unit running" and unit-count accuracy for
  the generator state; the reliability term of the Brier score and the signed CRPS of the two-lobe
  signed distribution; the mean absolute error of the recovered signed power.
---

**We ask whether the sign of an apparent-power reading, which is a magnitude, can be recovered
after the fact without training a neural network.** Eight of the 16 Primaries report apparent power
(MVA), so an export of solar output shows as a rise in the reading and not as a negative value. We
test three things on the CPU. First, we count how often the six Primaries that report real power
(MW) go negative, and which MVA Primaries could export at all. Second, we add a real solar farm's
recorded output to the six, fold the readings into magnitudes, and check whether a hand-built
calculation recovers the sign. Third, we fit a switching state-space model that splits a reading
into demand, solar output, and the number of units of a generator running, and score it against
answers we know by construction. Nothing here forecasts, so every score is a ceiling for what a
live service could do. This page is the plan: only the census and the gate table have run on real
data, and every number below other than the section on those two rungs is a design value or a
figure from an earlier experiment.

**The honest expectation is that the anchored read-out passes on injected worlds and that the
generator branch is the part most likely to fail.** Experiment 694's tracker scored an F1 of 0.678
on injected segments, and rung 0e adds a demand mean, a solar term, neighbours, and a sign to that
tracker. The earlier oracle simulation shows that identifiability is not the limit, so a miss would
mean model misspecification.

## Why this runs

**This experiment is rung 0 of the design for signed megawatts from the MVA-metered Primaries.**
The design (`mva_to_mw_design.md`) and its addendum (`mva_to_mw_addendum_1.md`) in the private
`data/ml-experiments/briefs/msa_architecture/` folder were each reviewed by an Opus agent and
amended after the triage. Rung 0 costs no GPU and fits no network. It decides whether rung 1 (the
hidden-sign test inside experiment 713's network) is worth a GPU hour. The standing approval of the
maintainer names F2 to F9, so rung 1 needs its own recorded approval; rung 0 is the CPU part that
the design's open item 2 says to run now.

**Experiments 358, 362, 412, 417, 420, 424, 427, and 674 and experiment 506 are the signed-MW line
this builds on.** Three fits through the meter invented reverse flow at D5, and four day-labellers
failed their bars. The design explains why: the folded likelihood is unchanged when the sign of
any mixture component flips, so the sign has to come from a restriction.

## What runs

**Rung 0a counts real reverse flow.** Per Primary, the number of negative half-hours in each
fold's training span and scored span, for the six signed-MW Primaries and for D15 (history from
2025-03). The expected result comes from the reverse-flow analysis: D10 0.76%, the other five
Primaries 0%, and D15 69.1%.

**Rung 0b is the question: a no-network posterior for the sign.** The six get a farm's recorded
output added at levels 1.0 and 2.0 (power factor 1, reactive floor `Q` of 2 points of capacity),
and the result is folded. The injected reading is built from a within-half-hour interpolation (6
sub-steps) so that crossings read as D12's do. The sign posterior `ρ` is computed by two read-outs.

- **Anchored read-out.** The gross-demand estimate `G` and its spread come from dull-sky rows
  only (CAMS clear-sky index under 0.3, at the same half-hour and day type, from the training
  window or from ages 2 to 91 days, never the scored span).
- **Unanchored control.** The 14-row window, every row starting at `ρ = 0`, `G` weighted,
  leave-one-out, three passes, prior log-odds `b` of -4, run with start weights `β_day` and with
  weights peaked on rows of the target day's sky.

Each read-out reports the area under the curve, the recall at a 5% false-reverse rate with a
2-point dead band, the same with the sky shifted by 28 days, and the posterior after one, two, and
three passes. The no-injection world reports the false-reverse rate on D5's sunny middays with the
gate forced open. The references are the solar rule of experiment 506 and the labeller of
experiment 674. The solar rule runs on the same injected rows. The labeller is scored on its own
days against its own labels, so it is not a paired comparison on this experiment's truth.

Two further references never read the meter, and their definitions are fixed here before any real
run. The reading-blind reference ranks each half-hour by F7's PV estimate `p`, which is the host's
estimate plus the injected farm's capacity times the clear-sky yield. The hour-of-day reference
ranks each half-hour by the mean clear-sky yield of the reference days at that half-hour of the
day. The anchored read-out must beat both (the bar table below).

The injected capacity that enters `p` is the oracle capacity, which is the injected increment, and
the contrast is repeated with 0.7 and 1.3 times that capacity. The design's capacity estimated from
experiment 662's probe is not used. The day type does not include bank holidays, and the trailing
window of ages 2 to 91 days is not implemented. The injection is rebuilt in code, because no
earlier experiment stored injected readings.

**Rung 0c checks the folded loss against known answers.** On draws from a known 3-component
mixture: the folded negative log-likelihood is minimised at the truth and at each sign-reversed
copy; the folded quantiles cover at their nominal rates; the fold sits at zero megawatts, with the
history clip applied in `v` units; the censored branch `P(|X| ≤ Q)` matches the simulated share;
and a fit from a reversed start returns the reversed mixture at the same loss. A toy also checks
the aliasing of one and two units of a generator (rung 0e), and a synthetic hidden-sign world
holds two exporting and two non-exporting Primaries, with the exporting pair carrying the real
export share of D12's history windows (70% to 93% of days from March to September).

**Rung 0d is the gate table.** Experiment 420's gate statistic (Spearman correlation of the
reading with the yield curve on sunny summer-midday rows, and the 1st percentile) per Primary and
fold cutoff, from the training window only.

## Rungs 0a and 0d on real data

**The census and the gate table matched the prediction.** In the scored folds, 2 of the 8 signed-MW
Primaries have negative readings: D10 on 0.76% of half-hours and D15 on 69.2%. D5, D9, D11, D14, and
D16 have none, and D18 has none in the scored folds. D18's training windows each hold 48 negative
half-hours (0.24% to 0.49% of the window, by fold), which the prediction did not cover. None of the
8 MVA-only Primaries has a negative reading, as expected of a meter that cannot show one.

**The export gate opens when the Spearman correlation exceeds 0.3 or the 1st percentile of the
reading is under 3 points.** The statistic comes from the sunny summer-midday rows of the training
window. D12 opens on the correlation (+0.53 to +0.56 across the 8 folds), and D8 opens on the
percentile (2.29 to 2.47 points, with a correlation between -0.04 and +0.04). Both are open in all
8 folds, which is the prediction. D10 and D15 are the other two Primaries with an open gate, on a
negative percentile. The other 12 Primaries are shut in all 8 folds. D22 is the nearest shut
MVA-only Primary, on the percentile (6.4 points against the 3-point line), and D8 opens by under 1
point. D2 and D18 have strongly negative correlations (about -0.58 and -0.55) and are shut, because
the correlation test needs a value above +0.3. This page publishes no per-generator value for D8
and D12 beyond the open gate.

## Rung 0e

**Rung 0e fits a hidden Markov model with a gridded demand residual to the past readings of one
Primary and returns gross demand, unmetered solar output, and a generator's unit count.** The
reading is `sqrt(X² + Q²)` with Student-t noise (4 degrees of freedom), `X = D − c p − n g`, the
demand mean is a regression on calendar and temperature, the residual is a first-order
autoregression on a grid of residual values (41 values in the design and 21 in the V2, K1, and V3
harness; 81 when the coefficient would exceed 0.98), and the emission sits on the transition so that
the within-half-hour averaging of a crossing is exact. The output per half-hour is a two-lobe signed
distribution, smoothed and filtered, stored as 11 signed quantiles and `P(X < 0)`.

**Bars, written before any fit.**

| Test | Bar |
|---|---|
| V1 (injected farms at the six) | Area with the real sky at least 0.1 above the shifted sky; false reverse flow under 1% of D5's sunny middays with the gate forced open; paired contrast against the anchored read-out with a lower bound above 0, chosen on rotations r2 and r3 |
| V2 (injected generator schedules) | Paired contrast against experiment 694's M2 on the same segments: lower bound above 0 and a gain of at least 0.05 F1; at most 3% of un-injected half-hours flagged and at most one active day a month |
| K1 | F1 of at least 0.769 |
| V3 (real negative controls at D8) | At most 3% flagged and at most one active day a month in each period, and the forced `K` = 3 control |
| Sign probability | Reliability term of the Brier score at most 0.01, and no bin of at least 200 half-hours off the diagonal by more than 0.1 |
| Signed CRPS | Below the hard-sign version's, with a paired interval above zero. The hard-sign version uses the observed magnitude, never the true one. |
| Rung 0b references | At injection levels 1 and 2 (oracle capacity, host scale 1), the anchored read-out's area exceeds each of the two references that do not read the meter, with the lower end of a 95% day-block bootstrap interval of the paired difference above 0, on the pooled gated half-hours. Each world has its own interval, and a level's interval is the mean of its worlds' interval ends. The midday area (10:00 to 14:00 UTC) is judged only when at least 6 worlds hold at least 30 import and 30 reverse half-hours there, and then the anchored read-out must not lose (the upper end of the interval stays above 0). A pooled or midday area with fewer than 30 half-hours of either class is `not evaluable`, and it never counts as a pass. |
| Rung 0b non-exporter check | The stop rule above (false reverse flow under 1% on D5's sunny middays, gate forced open) is read from the rows with host scale 1.0. A synthetic world shows the check can fail: with the PV estimate 2.5 times too high the rate measured 0.19, against 0.00 at 1.0 and 1.3 times. The real run builds no 2.5-times world, because its host scales are 0.7, 1.0, and 1.3. |

**Stop rules.** If V1 fails, the hindsight model is not used at D12. If V2 fails its paired bar, K1,
or its false-alarm bar, the D8 branch stops. The slow-trend term of the hindsight demand mean is a
change made before any real V2 run, in answer to two reviews. A first trend is the 0.8 quantile of
the daily-mean residual over a 21-day centred window, smoothed over 7 days, limited to 0.03 MVA per
day, and calibrated on the pre-roll. The rate limit stops a spell of engine output, which looks like
a step, from pulling the trend down. After a first scan of the unit size, the trend is refitted from
the half-hours with a smoothed `P(n = 0)` above 0.9, on days with at least 12 such half-hours and
bridged across the other days, and the unit size is searched again near its first value. The unit
size is never searched below 1 MVA, so a step of demand smaller than a unit is absorbed by the
demand mean. The synthetic tests are: drift of -0.5 and -1 MVA over 120 days with no engine (nothing
flagged), an engine with no drift, a spell of two units for 21 days with and without drift (F1 above
0.95, mean error of unit size times unit count under 0.35 MVA), and a step of -0.7 MVA (under 3%
flagged). Two limits remain. A persistent step of 1 MVA or more in a Primary's demand cannot be told
from an engine that stays on, so V3 would flag it. A spell of two units is returned as one unit of
twice the size, which leaves F1 and the output unchanged and lowers the count accuracy. The
injected-segment bootstrap resamples the schedule-weeks of experiment 694 (5 primary schedules over
7 weeks). The number of generator states `K` is chosen by the Bayesian information criterion. A V2
run stops when the rerun `M2` disagrees with the stored `inject.pkl` on any segment, and zero
matched segments is a failure. V3's bar applies to each period separately. The run stops at 15
core-hours, and the first fit measures the time per pass and stops the run when the projected cost
exceeds the cap. D8 analyses are restricted to the era before the solar farm registered, unless the
variant with farm and generator together passes the paired bar.

## Anonymisation and disclosures

**Any inferred generator state is written under the private data folder only.** This page, its
figures, its pull request, and its commit messages carry no generator time series, no dated
generator value, and no unit size. D8 and D12 get aggregate statistics only: scores, pooled shares,
and the bands of the design. Every figure script writes to the private folder unless it shows only
pooled or banded aggregates, and each such script refuses to write a dated generator value. No
existing figure or page of another experiment is edited. D8's metadata record is not quoted. The
maintainer has not yet decided what to do about earlier pages that may breach the rule, so no
page about D8 is published from this experiment until the maintainer says so.

- **Power factor 1 is assumed at every MVA Primary,** so MW equals MVA away from zero.
- **Every score is a hindsight ceiling.** The smoothed posterior reads the half-hour it labels and
  later half-hours, and CAMS arrives about a day late. The filtered posterior is reported beside it.
- **F7's code is imported, not copied,** from experiment 716's folder on `PYTHONPATH`, which sits on
  the unmerged branch `msa-f7-pv-part-1`. Nothing is merged from that branch, and the module that
  uses it states so.
- **Experiment 713's known-answer gate was waived,** which does not touch this experiment because it
  trains no network.

## Compute

**The design estimates about 3 core-hours for rungs 0a to 0d and 5 to 15 core-hours for rung 0e,
with a hard cap of 15 core-hours for rung 0e and a projected-cost gate.** One job runs at a time in
a `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0` scope with 1 BLAS thread and the
virtual environment's Python called directly. No GPU is used.

## Measured cost of the hidden Markov model

**One forward-backward pass at 164 states over a 122,000-half-hour series took 158.5 s on 1 core,
about 93 times the 1.7 s the addendum estimated.** The peak resident memory was 540 MB. The pair
emission, which integrates the path between the midpoint of the previous and the current signed
power over 4 sub-steps, causes most of the difference. A fit of one unit size needs about 50 passes
(a profile over unit size, then the Baum-Welch rounds), so one whole-span era costs about 2.2
core-hours at 164 states. Rung 0e on experiment 694's 44 segments (V2) at 164 states would cost
about 14.6 core-hours, which sits against the cap of 15.

**The reduced grid brings V2 to about 7 core-hours.** The harness runs V2, K1, and V3 on a grid of
21 residual bins (84 states), a coarse-then-fine profile of the unit size with a trend refit, and 2
Baum-Welch rounds, which is 17 passes per fit (6 coarse, 2 fine, 6 after the trend refit, 2
Baum-Welch, 1 final). A synthetic 148-day segment with `K` = 2 took about 39 s per fit. At `K` = 3
the measured 2.78 s per pass projects to about 47 s per fit, and one rerun of experiment 694's `M2`
on the same segment took 7.7 s. One injected run (hindsight fit plus `M2` rerun) therefore takes
about 55 s. The 44 segments and 9 schedules make 396 runs, about 6 core-hours, and the un-injected
fits, K1, and V3 add about 0.6 core-hours. The fit loop also stops when the elapsed time passes the
cap. `CostGate` times the first run and raises when the projected total exceeds 15 core-hours, so a
run whose projection is above the cap does not start its second run. The per-pass time at 84 states
is measured on synthetic data only, so the real projection waits for the first real run's own
timing.

## What the synthetic worlds say about the references bar

**On synthetic exporter worlds, the anchored read-out loses to the reading-blind reference.** The
worlds are the self-test's: gross demand is a smooth function of the half-hour, the day type, and
the temperature, the farm is 40% or 50% of the demand's capacity scale, and 900 days give both
classes at midday. The anchored area is 0.86 against 0.96 to 0.98 for the reading-blind reference,
and the lower end of the paired interval is about -0.22 in both worlds. The hour-of-day reference
is also ahead. On the self-test's all-export world the anchored area is 0.91 and the reading-blind
reference is higher still, but that world has 2 import half-hours, so no area from it is evaluable.
The worlds were not tuned to change this result.

**This is a finding about the design, and a change is proposed here, not adopted.** A likely reason,
not tested here, is that in these worlds demand varies little around its half-hour profile, so the
PV estimate alone orders the signs almost perfectly and the anchor adds noise from the dull-sky
rows. On real data demand varies more, which may favour the anchored read-out. The rung 0b run
decides that. If the anchored read-out still loses to the reading-blind reference there, the
maintainer could consider an anchored read-out that takes the PV estimate as its prior and uses the
anchored gross-demand residual only to move the sign near the zero crossing. That change would need
its own pre-registration and its own review before any run, and nothing in this experiment depends
on it. By the decision rule, losing to the reading-blind reference caps the rung 0b verdict at
`partly supported`.

**The anchored half of the signed-CRPS bar stays pending.** The run gives no seed-noise estimate,
so the second half of the bar is reported and not judged.

## Departures from the first report, and status of every rung

**The code departs from the design in these places.** Rung 0b injected worlds use the oracle
capacity for the injected farm, which is the injected increment, with 0.7 and 1.3 times as a
sensitivity. The host's own PV estimate is F7's, with the capacity as of the origin. Experiment
694's injected segments are not about a farm, so they use no capacity. The unanchored control uses a
symmetric window of 14 neighbours. The anchored read-out relaxes its anchor cut from an unreduced
share of 0.3 to 0.6 when too few anchor rows exist. The day type does not include bank holidays. The
pair emission runs from the midpoint of the previous and current signed power. On experiment 694's
series no weather exists, so the demand mean of the hindsight model is calendar only and is fitted
on the 28-day pre-roll. The panel loader is built on the torch-free functions of experiment 506, and
it cuts the 126,624-step `dense_raw.npy` of experiment 713 to the 121,248 steps of the grid that
experiment 412's CAMS arrays share. V3 runs the fit with `K` chosen by the Bayesian information
criterion and the forced `K` = 3 control. The history class of the `M1` re-cut is the class held by
most of the UTC days 2 to 15 days before the origin's date, as in `reverse_flow_analysis.md` section
4, and the measure is `M1` minus the Candidate. The second control period of V3 ends on a date that
is read from a private file, because the date is an era date. Experiment 716's folder comes in
through `EXTRA_PYTHONPATH` at commit 126577d1d, and the code stops at the start when the checkout is
at any other commit. Nothing is merged from that branch.

**Status of every rung.** Only the census and the gate table (rungs 0a and 0d) have run on real
data. Each line says which of three states the
rung is in: done on synthetic data, wired (the real-data branch is written and was exercised against
stubs), or not run on real data.

- Rungs 0a and 0d (census, gate table): done on synthetic data and run on real data once (the
  section above). The read-only loader is wired, and its grid-length rule is checked against the
  shapes of the stored arrays.
- Rung 0b (sign posterior): done on synthetic data. The `--run` branch is wired: farm series with a
  union gap mask, the F7 PV estimate and as-of capacity, the references of experiments 506 and 674
  and the two references that do not read the meter, capacity scales of 1, 0.7, and 1.3, the stored
  per-row posterior, the Brier and signed-CRPS bars of the hindsight read-out (smoothed and
  filtered), and a cost gate. A fake panel exercises the whole world loop. Not run on real data.
- Rung 0c (known-answer checks of the folded loss): done; six checks pass.
- Rung 0e V2, K1, and V3: wired against experiment 694's assets. A stub of those assets exercises
  every line, including the comparison of the rerun `M2` with the stored `inject.pkl` and `k1.pkl`,
  the paired cluster bootstrap, and the private outputs. Not run on real data. The variants "unit
  size scaled to the host" and "farm and engine injected together" are not wired.
- Rung 0e V4 and the section 3 analyses: the pure functions (matched running and idle profiles,
  running share by group, price fifths, export-day bands, class re-cut of stored forecasts) are done
  on synthetic data, and `run_section3` and `run_m1_recut` are wired but were never run, not even
  against stubs. `run_section3` refuses to run unless V2 and K1 both passed. The system-price fifths
  and the cleaner series in the MSA network are not wired.
- Figure and report scripts: done on synthetic tables. Every chart goes to the private folder unless
  its table passes the public-aggregate guard, and the guard refuses a profile table and a dated
  generator table.

## Method

The experiment is planned and has no result yet, so these bullets describe the design.

- **Architecture:** none for rung 0. Rung 0b is a closed-form posterior over two candidate values of
  gross demand per history token. Rung 0e is a hidden Markov model with a gridded demand residual
  and a unit count, fitted by expectation-maximisation, and it forecasts nothing.
- **Training regime:** no neural network is fitted and no GPU is used. Parameters of the hidden
  Markov model are fitted per Primary and era on the CPU. Injected worlds add a real solar farm's
  recorded output (experiment 39's frame) to the six signed-MW Primaries.
- **Measurement:** the folds 2025-12 and 2026-04 to 2026-06, on cleaned data, with D15 and D18
  cut before 2025-03-01; no valid time on or after 2026-07-01 is read; intervals come from a
  cluster bootstrap where the bars name one.
- **Standing practices followed:** cleaned data and the short-history cut. Not applicable: monthly
  refit of a network, saved network states, forecast plots (nothing is forecast).

## Reviews

An Opus agent reviewed the design and its addendum, and both were revised after the triage. Still
to come: an Opus code review of each new script before the first real-data run, a second review
of the fixed code, a results review, a numbers review, and a prose review.

## Files

- `data/ml-experiments/briefs/msa_architecture/mva_to_mw_design.md`, `f_mva_design_review.md`,
  `f_mva_design_triage.md`, `mva_to_mw_addendum_1.md`, `mva_to_mw_addendum_1_review.md`, and
  `mva_to_mw_addendum_1_triage.md` (read, private): the design that this page condenses.
- `ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/`: the code, written and
  unit-checked on the CPU on synthetic worlds. Only the census and the gate table have run on real
  data. Each bullet names a script and its step.
    - `e00720_fold.py` and `e00720_check_loss.py` (rung 0c): the folded mixture loss, the censored
      branch, the folded and signed quantiles, the fold point in `v` units, and the six known-answer
      checks.
    - `e00720_sign.py`, `e00720_worlds.py`, and `e00720_check_sign.py` (rung 0b): the anchored and
      unanchored sign read-outs, the within-half-hour injected reading, the synthetic hidden-sign
      world, the metrics, and their checks.
    - `e00720_hmm.py`, `e00720_fit.py`, and `e00720_check_hmm.py` (rung 0e): the hidden Markov model
      with pair emissions, Baum-Welch for the unit-count transitions, the two-lobe signed
      distribution, the Brier reliability and signed CRPS, the cost gate, the demand-mean fit, and
      the checks (brute force, aliasing toy, oracle simulation, hidden-sign world).
    - `e00720_rung0b.py`: the driver that scores four read-outs, each with the real and the shifted
      sky, on one injected world. Its `--run` branch is wired and not run on real data.
    - `e00720_census.py` (rungs 0a and 0d) and `e00720_real.py`: the census, the gate table, the
      fold windows, and the read-only loader of the stored panel.
    - `e00720_guard.py` and `e00720_check_guard.py`: the anonymisation guard.
    - `e00720_v2.py` and `e00720_check_bars.py` (rung 0e): the hindsight fit on one 694 segment, the
      paired cluster bootstrap against `M2`, the false-alarm statistics, and the bar functions of
      `e00720_bars.py`.
    - `e00720_references.py`: experiment 506's gated rule and experiment 674's day labels on the
      same worlds. `e00720_analyses.py`: V4 and the section 3 functions. `e00720_figures.py` and
      `e00720_report.py`: the charts and the report, private by default.
    - `run.sh`: the slot wrapper (8 GB cap, no swap, one thread, 25 GB free check).
- **Not wired:** the price fifths, the cleaner series in the MSA network, and the V2 variants with
  the unit size scaled to the host and with a farm and an engine together. The first real run of
  each script waits for an Opus code review.
- Read from earlier experiments: experiments 412, 417, 420, and 405 (injection and gate), 694
  (generator tracker, to rerun on the same segments), and 716 (the PV estimate and the as-of
  capacity table, imported through `PYTHONPATH`).
- Written by this experiment (to be created): tables under
  `data/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/`, never committed.
