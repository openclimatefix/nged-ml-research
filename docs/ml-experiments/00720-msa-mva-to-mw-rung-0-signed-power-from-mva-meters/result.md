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
  sky, and false reverse flow under 1% of D5's sunny middays with the gate forced open). `partly
  supported` when the anchored read-out fails and the unanchored control passes. `not supported`
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
live service could do. This page is the plan: no script has run on real data, and every number
below is a design value or a figure from an earlier experiment.

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
experiment 674, run on the same injected rows. The injected capacity that enters `p` is
estimated as the design specifies, and the contrast is repeated with that capacity scaled by 0.7
and 1.3. The injection is rebuilt in code, because no earlier experiment stored injected readings.

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

## Rung 0e

**Rung 0e fits a hidden Markov model with a gridded demand residual to the past readings of one
Primary and returns gross demand, unmetered solar output, and a generator's unit count.** The
reading is `sqrt(X² + Q²)` with Student-t noise (4 degrees of freedom), `X = D − c p − n g`, the
demand mean is a regression on calendar and temperature, the residual is a first-order
autoregression on a 41-value grid (81 when the coefficient would exceed 0.98), and the emission
sits on the transition so that the within-half-hour averaging of a crossing is exact. The output
per half-hour is a two-lobe signed distribution, smoothed and filtered, stored as 11 signed
quantiles and `P(X < 0)`.

**Bars, written before any fit.**

| Test | Bar |
|---|---|
| V1 (injected farms at the six) | Area with the real sky at least 0.1 above the shifted sky; false reverse flow under 1% of D5's sunny middays with the gate forced open; paired contrast against the anchored read-out with a lower bound above 0, chosen on rotations r2 and r3 |
| V2 (injected generator schedules) | Paired contrast against experiment 694's M2 on the same segments: lower bound above 0 and a gain of at least 0.05 F1; at most 3% of un-injected half-hours flagged and at most one active day a month |
| K1 | F1 of at least 0.769 |
| V3 (real negative controls at D8) | At most 3% flagged and at most one active day a month in each period, and the forced `K` = 3 control |
| Sign probability | Reliability term of the Brier score at most 0.01, and no bin of at least 200 half-hours off the diagonal by more than 0.1 |
| Signed CRPS | Below the hard-sign version's, with a paired interval above zero |

**Stop rules.** If V1 fails, the hindsight model is not used at D12. If V2 fails its paired bar, K1,
or its false-alarm bar, the D8 branch stops. The run stops at 15 core-hours, and the first fit
measures the time per pass and stops the run when the projected cost exceeds the cap. D8 analyses
are restricted to the era before the solar farm registered, unless the variant with farm and
generator together passes the paired bar.

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

**The design estimates about 3 core-hours for rungs 0a to 0d and 5 to 15 core-hours for rung 0e, with
a hard cap of 15 core-hours for rung 0e and a projected-cost gate.** One job runs at a time in a
`systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0` scope with 1 BLAS thread and the
virtual environment's Python called directly. No GPU is used.

## Measured cost of the hidden Markov model

**One forward-backward pass at 164 states over a 122,000-half-hour series took 158.5 s on 1 core,
about 93 times the 1.7 s the addendum estimated.** The peak resident memory was 540 MB. The pair
emission, which integrates the path between the midpoint of the previous and the current signed power
over 4 sub-steps, causes most of the difference. A fit of one unit size needs about 50 passes
(a profile over unit size, then the Baum-Welch rounds), so one whole-span era costs about 2.2
core-hours at 164 states. Rung 0e on experiment 694's segments (V2) at the full 164 states would
cost about 14.6 core-hours, which sits against the cap of 15. The harness therefore runs V2 and K1
on a grid of 21 residual bins, a coarse-then-fine profile of the unit size (9 passes), and 2
Baum-Welch rounds, and the cost gate measures the first pass and stops the run when the projected
total exceeds the cap.

## Departures from the first report, and status of every rung

**The code departs from the design in these places.** Injected worlds use the oracle capacity, with
0.7 and 1.3 times as a sensitivity, instead of a capacity estimated from a probe. The unanchored
control uses a symmetric window of 14 neighbours. The anchored read-out relaxes its anchor cut from
an unreduced share of 0.3 to 0.6 when too few anchor rows exist. The day type may include bank
holidays. The pair emission runs from the midpoint of the previous and current signed power. On
experiment 694's series no weather exists, so the demand mean of the hindsight model is calendar
only and is fitted on the 28-day pre-roll. The panel loader is built on the torch-free functions of
experiment 506, and it cuts the 126,624-step `dense_raw.npy` of experiment 713 to the 121,248 steps
of the grid that experiment 412's CAMS arrays share.

**Status, with no script yet run on real data:**

- Rungs 0a and 0d (census, gate table): done on synthetic data; the read-only loader is wired.
- Rung 0b (sign posterior): the posterior, the metrics, and the injection plumbing are done on
  synthetic data. The `--run` branch, which needs the farm series, the F7 PV estimate, and the
  references of experiments 506 and 674, is not wired.
- Rung 0c (known-answer checks of the folded loss): done; six checks pass.
- Rung 0e (hindsight model): the model, the Brier and signed-CRPS functions, the bar functions, and
  the V2 and K1 harness against experiment 694's `M2` (paired cluster bootstrap, cost gate) are done
  on synthetic data. The `--run` branch of V2 is written but has never run. V3, V4, the addendum's
  section 3 analyses, and the filtered and smoothed bar tables over real series are not written.
- Figure and report scripts: not written.

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
  unit-checked on the CPU on synthetic worlds. No script has run on real data. Each bullet names a
  script and its step.
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
      sky, on one injected world. Its `--run` branch is not wired.
    - `e00720_census.py` (rungs 0a and 0d) and `e00720_real.py`: the census, the gate table, the
      fold windows, and the read-only loader of the stored panel.
    - `e00720_guard.py` and `e00720_check_guard.py`: the anonymisation guard.
    - `e00720_v2.py` and `e00720_check_bars.py` (rung 0e): the hindsight fit on one 694 segment, the
      paired cluster bootstrap against `M2`, the false-alarm statistics, and the bar functions of
      `e00720_bars.py`.
    - `run.sh`: the slot wrapper (8 GB cap, no swap, one thread, 25 GB free check).
- **Not written yet:** the references of experiments 506 and 674 on the injected rows; the farm and
  F7 wiring of the real run of rung 0b; V3, V4, and the analyses of the addendum's section 3; the
  figure scripts; the report script. The first real run of each script waits for an Opus code review.
- Read from earlier experiments: experiments 412, 417, 420, and 405 (injection and gate), 694
  (generator tracker, to rerun on the same segments), and 716 (the PV estimate and the as-of
  capacity table, imported through `PYTHONPATH`).
- Written by this experiment (to be created): tables under
  `data/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/`, never committed.
