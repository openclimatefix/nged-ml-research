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
status: finished
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
verdict: inconclusive
best_arm: >-
  No arm of rung 0e clears its bar. Experiment 694's M2 tracker, the comparison arm, scores a pooled
  F1 of 0.819 for "an engine unit is running" on 44 injected segments, and the hindsight hidden
  Markov model scores 0.696, a gap of -0.123 (95% interval -0.213 to -0.054) against a required
  gain of at least +0.05.
headline: >-
  A hand-built hidden Markov model fitted in hindsight to the past readings of one substation does
  not find a generator's running units as well as experiment 694's existing tracker. Its F1 is
  0.696 against 0.819 on 44 injected segments (difference -0.123, interval -0.213 to -0.054), and
  on stretches with no generator it flags 21.6% to 59.2% of half-hours at the eight hosts, against
  0% for the tracker and a bar of 3%. The sign recovery of rung 0b did not run on real data, so
  rungs 0a to 0d give no verdict on whether the sign can be recovered.
lessons: >-
  The model's demand mean has no weather input on experiment 694's series, so a demand swing may be
  read as units of the generator (a hypothesis; the temperature diagnostic that would test it is
  parked by the maintainer). The stop rule worked as designed: after the failed paired contrast,
  K1 and V3 were not fitted, which saved the remaining compute. The census and the gate table
  matched the pre-registered prediction on real data.
take_home_message: >-
  We could not show that a new model, fitted after the fact to a substation's readings, finds a
  generator's running units better than the existing tracker, and it flagged a generator running
  during stretches with none far more often than the tracker did. The follow-up that would have
  applied the model to a real generator is closed, and the hindsight model is not used there. The
  question of whether the sign of a reading from a substation that reports apparent power can be
  recovered without a neural network is still open, because that rung was never run on real data. A
  next experiment would have to give the model's demand mean weather inputs before testing it
  again (a hypothesis).
advice:
  - >-
    Do not build on experiment 720's hindsight hidden Markov model to count a generator's running
    units until its demand mean has weather inputs, because its F1 trailed experiment 694's M2 by
    0.123 on 44 injected segments (V2 contrast).
  - >-
    Score the false-alarm rate on stretches with no generator at every host, not only the pooled
    rate, because the hindsight model flagged 21.6% to 59.2% of half-hours across eight hosts
    while M2 flagged none (false-alarm table).
  - >-
    Read the export gate table before injecting a farm at a Primary: the gate is open at 4 of the 16
    Primaries and shut at the other 12 in all 8 folds (census and gate table).
  - >-
    Rerun a reference tracker on the same segments and stop the run when its output disagrees with
    the stored one: the 396 reruns of M2 matched the stored file on every segment (V2 checks).
production_status: not-for-production
adopted_in: none
---

**We asked whether the sign of an apparent-power reading can be recovered after the fact without
training a neural network, and the part that ran on real data answers only a smaller question.**
Eight of the 16 Primaries (primary substations) report apparent power in megavolt-amperes (MVA), a
magnitude, so an export of solar output shows as a rise in the reading and not as a negative value.
The plan had three parts, all run on the CPU alone. First, count how often the Primaries that
report real power in megawatts (MW) go negative, and find which MVA Primaries could export at all.
Second, add a real solar farm's recorded output to six Primaries, fold the readings into
magnitudes, and check whether a hand-built calculation recovers the sign. Third, fit a hidden
Markov model that splits a reading into demand, solar output, and the number of units of a
generator running, and score it against answers known by construction. The first part ran and
matched its prediction. The second part never ran on real data. The third part ran on its first
test and failed it, so the stop rule ended the rest.

**Verdict: inconclusive, as a judgement under the plan's last row, with the generator-state model
failing its pre-registered bar.** The plan's verdict table gives `supported`, `partly supported`,
and `not supported` only through the real-data result of rung 0b (the sign read-out), and that
result does not exist, because rung 0b was held and never run on real data. The known-answer checks
of rung 0c did not fail, so `invalid` does not apply. The row that is left is `inconclusive`.
Rung 0e has its own bars. Its first bar (V2, the paired contrast against experiment 694's tracker)
failed, and the stop rule then closed the generator branch. Every verdict here assumes power factor
1 and is a hindsight ceiling, because the smoothed posterior reads half-hours after the one it
labels.

**What the result means for the forecast: the hindsight model is not a better way to find a
generator than the tracker that already exists, and the sign question is still open.** The
substation forecast needs the sign of a reading only where the Primary can export. The experiment
does not change any forecast, and the model is not for production.

## Results

### Rung 0e: the hindsight model lost the pre-registered contrast

**The pre-registered test (V2) failed both of its bars.** The test injected the schedules of a
generator's units into 44 segments of readings that experiment 694 had stored, and scored "an
engine unit is running" by F1. F1 is the harmonic mean of the share of running half-hours found and
the share of flagged half-hours that were running. The model and experiment 694's M2 tracker were
fitted to the same 396 runs (44 segments and 9 schedules), and the contrast is the model's F1 minus M2's.
The bootstrap resamples 35 clusters of segments.

| Test | Bar | Result | Passed |
|---|---|---|---|
| V2 paired contrast | Lower end of the interval above 0, and a gain of at least +0.05 | -0.123 (interval -0.213 to -0.054) | No |
| V2 false alarms | At most 3% of half-hours flagged and at most 1 active day a month, with no generator injected | Worst host: 59.2% flagged and 20.75 active days a month | No |
| K1 | F1 of at least 0.769 | Not run | Not run |
| V3 | At most 3% flagged and 1 active day a month in each control period | Not run | Not run |

**The model's pooled F1 is 0.696 and M2's is 0.819 on the same segments.** Both numbers pool the
half-hours of all 396 runs. The interval excludes zero on the negative side, so M2 is better, and
the pre-registered prediction that the contrast's lower end would sit above 0 is wrong.

**Across the eight hosts of the un-injected segments, the hindsight model flagged 21.6% to 59.2% of
half-hours (31.5% on average) and 9.8 to 20.8 active days a month, and M2 flagged none.** A host is
one of the eight Primaries that supply the segments with no generator injected. An active day
is a day with at least 4 flagged half-hours. The bar is 3% and 1 active day a month, so the
model fails at every one of the eight hosts. The per-host values are stored in the private folder
and are not shown here.

**The stop rule fired as written.** The plan says that a failed paired bar, K1, or false-alarm bar
stops the generator branch, so K1 and V3 were not fitted, the section 3 analyses (which refuse to
run unless V2 and K1 both pass) did not run, and the hindsight model is not used on a real
generator's history. Experiment 724 imports this experiment's hidden Markov code and its V2, K1,
and V3 harness. The maintainer has closed the plan of experiment 724 as parked. The stop rule
closed only the hand-built hindsight approach, not the neural untangler of experiment 729.

**One reading is a hypothesis, not a finding: the model's demand mean may absorb a demand swing as
generator units.** On experiment 694's segments no weather exists, so the model's demand mean uses
the calendar only and is fitted on a 28-day pre-roll. A change of demand that a temperature input
would explain could then read as units running, which fits the high false-alarm share on segments
with no generator. A diagnostic that adds temperature would test the reading. The maintainer has
parked that diagnostic, and this page does not plan it.

### Rungs 0a and 0d: the census and the gate table matched the prediction

**On real data, 2 of the 8 signed-MW Primaries have negative readings in the scored months, as
predicted.** Over the 8 scored months from 2025-11 to 2026-06, D10 has 88 negative half-hours of
11,614 (0.76%) and D15 has 7,967 of 11,511 (69.2%, against a prediction of 69.1%). D5, D9, D11,
D14, D16, and D18 have none. The training windows of D18 each hold 48 negative half-hours (0.24% to
0.49% of the window, by fold), which the prediction did not cover. None of the 8 MVA-only
Primaries has a negative reading, as expected of a meter that cannot show one.

**The export gate opens when the Spearman correlation of the reading with the clear-sky solar yield
exceeds 0.3 or the 1st percentile of the reading is under 3 points of capacity.** The statistic
comes from the sunny summer-midday rows of each fold's training window. The gate is open in all 8
folds at D8 and D12, the two MVA-only Primaries the prediction named, and at D10 and D15, which
have negative readings. The other 12 Primaries are shut in all 8 folds. D22 is the nearest shut
MVA-only Primary, with a 1st percentile of 6.4 points against the 3-point line. D2 and D18 have
strongly negative correlations (-0.58 to -0.56 and -0.55 to -0.53) and are shut, because the
correlation test needs a value above +0.3. This page publishes no per-generator value for D8 and
D12 beyond the open gate.

### Rungs 0b and 0c: not run on real data, and no stored output

**Rung 0b, the read-out of the sign, was not run on real data.** The anchored read-out builds an
estimate of gross demand from dull-sky rows and asks which half-hours of an injected farm's output
are exports. The plan's stop rule for rung 0b and its references bar (the read-out must beat a
reading-blind reference and an hour-of-day reference, neither of which looks at the meter) were
therefore never judged. During code review, synthetic worlds suggested the anchored read-out would
lose to the reading-blind reference, and the plan page proposed an alternative read-out as a second
arm that needs its own pre-registration. The maintainer holds the choice between running rung 0b as
written and adding that arm, and no number from those synthetic worlds is stored with this result.

**Rung 0c checks the folded loss against known answers, and this page has no stored output of it.**
The page for the plan recorded six checks passing on draws from a known mixture and on synthetic
worlds. The check scripts are committed, and the outputs are not in the stored folder, so this page
does not restate the checks' numbers.

## Checks

- **Median forecast minus mean forecast, per site:** not applicable, because nothing is forecast.
- **Error of the mean or median forecast beside the quantile-score contrasts:** not applicable,
  because the scores are F1 and the area under a curve.
- **Audit triggers:** trigger 2 fired, because the result contradicts the prediction that the paired
  contrast's lower bound sits above 0, and the one explanation offered (the demand mean has no
  weather input) is untested. No audit has run. Trigger 7 may also apply, because the plan of
  experiment 724, which the maintainer has closed as parked, imports this code. Triggers 1 and 3
  to 6 did not fire, since no control ran and no quantile score or leave-one-site-out exists.
- **Consistency gate:** the rerun of M2 agreed with experiment 694's stored output on all 396
  reruns, and none disagreed.
- **Reviews:** an Opus agent reviewed the design and its addendum, and four Opus code reviews
  (private folder, `mva_code_review_1.md` to `mva_code_review_4.md`) came before the first real
  run. The numbers check follows in the next commit, and a prose review by a fresh agent has not
  run.

## Method

**The experiment fits no neural network and no GPU was used.** The bullets give the design of the
arm that carries the headline (the hindsight model on V2) and then the differences.

- **Architecture:** the hindsight model is a hidden Markov model for the readings of one Primary.
  The reading is `sqrt(X² + Q²)` with Student-t noise (4 degrees of freedom), where `X` is the
  signed power, `D − c p − n g` (demand `D`, the F7 estimate `p` of unmetered solar output scaled by
  its capacity `c`, and `n` units of size `g` running), and `Q` is a reactive floor (0.14 MVA in
  the V2 harness, and 2 points of capacity in rung 0b). The demand mean is a regression on the
  calendar, and on temperature where a series has weather. The residual is a first-order
  autoregression on a grid of residual values (21 values in the V2 harness, so 84 states), and
  the emission sits on the transition so that the averaging within a half-hour is exact. The output
  per half-hour is a two-lobe signed distribution, smoothed and filtered, stored as 11 signed
  quantiles and the probability that `X` is negative.
- **Architecture, trend:** a slow trend term keeps a drift of demand from reading as units. It is
  the 0.8 quantile of the daily-mean residual over a 21-day centred window, smoothed over 7 days and
  limited to 0.03 MVA a day. After a first scan of the unit size, the trend is refitted from the
  half-hours with a smoothed probability of no unit running above 0.9, on days with at least 12 such
  half-hours, and the unit size is searched again. The unit size is never searched below 1 MVA. The
  rate limit, the probability threshold, and the day minimum were set on synthetic worlds after the
  first trend failed on them, before any real V2 run.
- **Reference (M2):** experiment 694's tracker, rerun on the same segments from its code, and
  compared with its stored `inject.pkl` before any score is read.
- **Training regime:** the parameters are fitted per segment by expectation-maximisation with 2
  Baum-Welch rounds, after a coarse-then-fine profile of the unit size (19 to 20 passes per fit),
  and the number of units `K` is chosen by the Bayesian information criterion. The 396 fits took
  5.9 core-hours in total on one core each, and the 396 reruns of M2 took 1.28 hours. The run was
  capped at 15 core-hours, with a cost gate that times the first run.
- **Training data for the injection:** the segments, schedules, and un-injected hosts are experiment
  694's stored assets (44 segments, 9 schedules, 8 hosts). Rungs 0a and 0d read the stored panel of
  readings read-only.
- **Measurement:** F1 of "any unit running" pooled over the 396 runs, the paired difference from M2
  with a 95% cluster bootstrap over 35 clusters of segments, and the flagged share and active days a
  month on un-injected half-hours, per host. The bars are the table in the section "Results". Rungs
  0a and 0d count negative half-hours and compute the gate statistic from the 8 monthly folds
  2025-11 to 2026-06 on cleaned data, with D15 and D18 cut before 2025-03-01, and no valid time on
  or after 2026-07-01 is read.
- **Standing practices followed:** cleaned data and the short-history cut. Not applicable: monthly
  refit of a neural network, saved network states, and forecast plots, because nothing is forecast.
  No figure is published, because every chart of the generator-state results would show a host's
  flagged series and is kept in the private folder.

## Files

- `data/ml-experiments/briefs/msa_architecture/mva_to_mw_design.md`, `f_mva_design_review.md`,
  `f_mva_design_triage.md`, `mva_to_mw_addendum_1.md`, `mva_to_mw_addendum_1_review.md`, and
  `mva_to_mw_addendum_1_triage.md` (read, private): the design that this experiment condenses.
- `ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/`: the code. Each bullet
  names a script and the step it runs.
    - `e00720_census.py` and `e00720_real.py` (rungs 0a and 0d): the census, the gate table, the
      fold windows, and the read-only loader of the stored panel. They wrote the two stored tables
      below.
    - `e00720_v2.py`, `e00720_bars.py`, and `e00720_check_bars.py` (rung 0e): the hindsight fit on
      one segment of experiment 694, the paired cluster bootstrap against M2, the false-alarm
      statistics, and the bar functions. They wrote the V2 tables below.
    - `e00720_hmm.py`, `e00720_fit.py`, and `e00720_check_hmm.py`: the hidden Markov model with
      pair emissions, Baum-Welch for the unit-count transitions, the signed distribution, the cost
      gate, and the demand-mean fit.
    - `e00720_fold.py` and `e00720_check_loss.py` (rung 0c): the folded mixture loss and its
      known-answer checks.
    - `e00720_sign.py`, `e00720_worlds.py`, `e00720_check_sign.py`, `e00720_rung0b.py`, and
      `e00720_references.py` (rung 0b, not run on real data): the sign read-outs, the injected
      worlds, and the references of experiments 506 and 674.
    - `e00720_analyses.py`, `e00720_figures.py`, `e00720_report.py`, `e00720_guard.py`, and
      `e00720_check_guard.py`: the section 3 analyses that did not run, the private charts and
      report, and the anonymisation guard.
    - `run.sh`: the slot wrapper (8 GB cap, no swap, one thread, 25 GB free check).
- Read from earlier experiments: experiments
  [412](../00412-mdn-signed-mw-from-mva-meters-known-answer/result.md),
  [417](../00417-hidden-sign-mva-export-gate-zero-crossing-labels/result.md),
  [420](../00420-neural-mva-export-gated-signed-final/result.md), and
  [405](../00405-mdn-explicit-pv-term-learned-capacity-known-answer/result.md) (injection and gate),
  [506](../00506-mva-sign-evidence-dip-depth-and-gated-solar-rule/result.md) and
  [674](../00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/result.md) (references),
  [694](../00694-gas-engine-state-tracker-part-1/result.md) (the tracker, its injected segments, and
  `inject.pkl`), and experiment 716 (the PV estimate and the as-of capacity table, imported through
  `PYTHONPATH` from the unmerged branch `msa-f7-pv-part-1` at commit 126577d1d).
- Written by this experiment, under `data/ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/private/`
  and never committed: `rung0a_census.csv` (negative half-hours per Primary, fold, and span),
  `rung0d_gate_table.csv` (the gate statistic per Primary and fold), `v2_bar.csv` (the paired
  contrast and the bar result), `v2_fits.csv` (one row per fit: F1 of both arms, seconds, and
  recall by unit count), and `v2_false_alarm_hindsight_by_host.csv` and
  `v2_false_alarm_m2_by_host.csv` (flagged share and active days a month per host).

**Docstring-only edits after the run.** The docstrings of the scripts below were edited after the
experiment ran, to satisfy the docstring linter. No executable code changed: the syntax trees of
every edited file, compared with the docstrings removed, are identical before and after. The list
gives each file's git blob hash (first 10 hexadecimal digits) before and after the edit.

- `e00720_analyses.py`: `dea70ecd2e` to `fb3c35a6c5`
- `e00720_bars.py`: `3031a6f91d` to `b29daa4a1f`
- `e00720_figures.py`: `f6fa9a68dc` to `0f33280d3d`
- `e00720_fit.py`: `1de281c8c6` to `bba39ed666`
- `e00720_hmm.py`: `90c825aef7` to `c61499c517`
- `e00720_real.py`: `e7ea5ddb21` to `c0d0530c82`
- `e00720_references.py`: `9541b70fc6` to `0f18ee08e3`
- `e00720_rung0b.py`: `2799d4851b` to `15cb990439`
- `e00720_sign.py`: `426038548e` to `df1acfed64`
- `e00720_v2.py`: `df116b4580` to `be5e5a2575`
