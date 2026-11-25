---
id: "00725"
slug: msa-live-buildable-history-weather
title: >-
  Does the axial-attention demand network still forecast better than the Candidate when every
  history token reads only weather that a live service holds at the forecast origin?
theme: lagged-power-and-error-correction
also_relevant_to:
  - weather-products-and-blending
status: finished
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It planned about 61 fits, including the phase 0 smoke fit, and ran 48 scored fits
  plus the smoke fit (49 in all), storing per-row forecasts and neural network weights for each;
  (2) more than one rule for the history weather defensibly tests the hypothesis, and the page fixes
  one; (3) it subclasses the data source of experiment 713, and the callers of
  that class cannot be listed without searching the repository; (4) it touches no serving path or
  degradation rule; (5) it changes no contract. Full set of reviews: plan, code (Opus, each new
  script, twice), results, write-up, prose.
hypothesis: >-
  The gain of the axial-attention demand network of experiment 713 over the Candidate at the six
  signed-MW Primaries does not depend on analysis weather in the newest history rows, because the
  across-day attention reads the power of the recent days and needs the weather only to explain that
  power.
prediction: >-
  Written before any fit. L1 (`L` minus `C`) lies between -0.22 and -0.08 points of capacity at the
  six signed-MW Primaries and between -0.14 and +0.19 at the eight MVA-only Primaries. L2 (`L` minus
  `M*`) lies inside +/-0.054 at the six signed-MW Primaries and inside +/-0.142 at the eight
  MVA-only Primaries. The predicted verdict is `small gain` or `inconclusive`. A prediction fails if
  a point estimate falls outside its range.
decision_rule: >-
  Two counted contrasts, L1 and L2, each at the six signed-MW Primaries and at the eight MVA-only
  Primaries, so four counted readings, each read at 98.75% from a month-block bootstrap, with
  experiment 713's three guards, noise floors, and verdict tables. The verdict is read from L1
  alone. The smallest gain worth porting is 0.05 points of capacity, experiment 713's value. Seeds 2
  to 4 run only if L1 has a point estimate below zero in at least one group after the 8-fold, 2-seed
  reading. Every verdict, headline, and take-home message states that experiment 713's known-answer
  gate was waived and that the target row of every arm still reads a CAMS day summary that a live
  service may not hold.
benchmark: >-
  `C` (the Candidate of experiment 667, stored seeds 0 to 4) and `M*` (experiment 713's selected
  arm, `M1`, stored seeds 0 to 4); neither is refitted.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  CRPS in points of capacity on leads from 24 h up to but not including 240 h after the 12:00 UTC
  origin, with the mean absolute error of the median forecast normalised by capacity beside it; q01
  and q99 pinball loss; the stopping pass of every fit; the read-out weights by age.
verdict: small gain
best_arm: >-
  `L` (experiment 713's axial-attention demand network refitted on live-buildable history weather,
  seeds 0 to 4). `L` minus the Candidate `C` is -0.152 points of CRPS (98.75% interval -0.269 to
  -0.024) at the six signed-MW Primaries, with all 5 of 5 seeds negative. `L` minus `M*` is -0.005
  (-0.010 to +0.001) there.
headline: >-
  L1 (`L` minus `C`) is -0.1522 points of CRPS (98.75% interval -0.2694 to -0.0241, 5 of 5 seeds
  negative) at the six signed-MW Primaries, which the pre-registered rule reads as `small gain`, and
  +0.0287 (-0.0882 to +0.1130) at the eight MVA-only Primaries, which reads as no detectable
  difference. L2 (`L` minus `M*`) is -0.0050 (-0.0102 to +0.0008) at the six signed-MW Primaries and
  +0.0066 (-0.0095 to +0.0350) at the eight MVA-only Primaries, both no detectable difference.
  Experiment 713's known-answer gate was waived, and the target row of every arm still reads a CAMS
  day summary that a live service may not hold.
lessons: >-
  Replacing the analysis weather in the newest history rows with weather that a live service holds
  at the forecast origin changed the CRPS of the axial-attention network by less than its
  pre-registered margin at both groups (L2). Arm `Z`, the refit with no weather change, reproduced
  the stored `M*` bit for bit, so the refit procedure adds no noise of its own. The gain at the six
  signed-MW Primaries is negative in 7 of 8 months, and the sign at the eight MVA-only Primaries
  flips when one Primary is left out. The experiment was parked after phase 6, so the budget probe
  (not triggered), arm `E`, and the `T3` read-out of how much of L1 rests on the target row's CAMS
  day summary were not run, and no `T3` number is quoted.
take_home_message: >-
  Experiment 713's axial-attention demand network still forecasts better than the Candidate at the
  six signed-MW Primaries (the 6 primary substations that report signed megawatts) when its history
  rows read only weather that a live service holds at the forecast origin. The gain is small, 0.152
  points of CRPS, and it matches the gain of the laboratory version of the network that reads
  analysis weather, to within 0.005. At the eight MVA-only Primaries (the 8 that report unsigned
  apparent power in megavolt-amperes), neither network differs detectably from the Candidate.
  Experiment 713's known-answer gate was waived, and the target row still reads a CAMS day summary
  that a live service may not hold. The read-out that would bound that effect was not run, because
  the experiment was parked.
advice:
  - >-
    Treat live-buildable weather history as free for experiment 713's network: refitting on it
    moved CRPS by -0.005 (-0.010 to +0.001) at the six signed-MW Primaries and +0.007 (-0.010 to
    +0.035) at the eight MVA-only Primaries (L2, five seeds).
  - >-
    Do not expect the gain over the Candidate at the eight MVA-only Primaries, because `L` minus
    `C` is +0.029 (-0.088 to +0.113) there, and the sign flips when one Primary is left out (L1,
    supported_guards in `score/five_seed.json`).
  - >-
    Refit stages 2 and 3 from a stored stage 1 state with the same seed to get a control that
    matches the stored arm exactly, because `Z` minus `M*` was 0.0 on every quick-look fold and
    seed (phase 3, audit trigger 1).
  - >-
    Read a failing bit-exact check at the last training origin of a fold against the fold's cutoff
    before suspecting the model code, because the `T3` failure came from target-row half-hours cut
    by the cutoff (Fix A section).
  - >-
    Run the `T3` read-out before relying on the gain in a live service, because every arm still
    reads the CAMS day summary of D - 1 in the target row and this experiment did not measure how
    much of L1 rests on it (What was not run).
production_status: not-for-production
adopted_in: none
---
**Weather history that can be built live gave the demand forecast about the same skill as laboratory
weather history, and a small gain over the Candidate at the six signed-MW Primaries.** The
experiment asks whether experiment 713's neural network for substation demand, the axial-attention
network ("the MSA"), keeps its lead when its inputs hold only data that a live service has at the
forecast origin, which is the time at which the forecast is made. The network reads 14 days of
history rows, and each history row holds the weather and power of one day. Experiment 713's selected
arm `M*` fills the weather in those rows with analysis weather, which is weather reconstructed after
the event from ERA5 (the European Centre for Medium-Range Weather Forecasts' reanalysis) and CAMS
(the Copernicus Atmosphere Monitoring Service's satellite irradiance estimate). A live service does
not yet hold that analysis weather for the newest rows. We refitted the network as arm `L`, whose
history rows hold only weather that a live service holds at the forecast origin. The Candidate `C`
is the stored model of experiment 667. L1 is `L` minus `C`, and L2 is `L` minus `M*`, both in CRPS
(continuous ranked probability score) points of capacity, where lower is better. Each contrast uses
5 seeds and 8 monthly folds. Each interval is a 98.75% month-block bootstrap interval, which
resamples whole months of forecasts 100,000 times and reports the middle 98.75% of the resampled
differences.

**The two groups of Primary substations are read separately.** "The six signed-MW Primaries" are the
6 Primary substations whose telemetry is signed megawatts, and "the eight MVA-only Primaries" are
the 8 whose telemetry is unsigned apparent power in megavolt-amperes (MVA). At the six signed-MW
Primaries, `L` gave a small gain over `C` and no detectable difference from `M*`. At the eight
MVA-only Primaries, `L` showed no detectable difference from `C` or from `M*`.

**The pre-registered rule gives the verdict `small gain`, read from L1 alone.**

| Contrast (five seeds, 8 folds) | Six signed-MW Primaries | Eight MVA-only Primaries |
|---|---|---|
| L1: `L` minus `C` | -0.1522 [-0.2694, -0.0241], 5 of 5 seeds negative: small gain | +0.0287 [-0.0882, +0.1130]: no detectable difference |
| L2: `L` minus `M*` | -0.0050 [-0.0102, +0.0008], 3 of 5 seeds negative: no detectable difference | +0.0066 [-0.0095, +0.0350]: no detectable difference |

**The reading is "small gain" and not "gain", because the t guard fails.** The t interval over the
8 monthly differences is -0.3316 to +0.0397 at the six signed-MW Primaries, so it includes zero. The
contrast is negative in 7 of 8 months (2026-06 is +0.1307). The guards for `supported` also fail:
the 95% upper bound of the q01 pinball loss of `L` minus `C` at the six signed-MW Primaries is
+0.0505, above the limit of 0.05, and at the eight MVA-only Primaries leaving out one Primary flips
the sign of L1. All four point estimates fell inside the ranges that the plan predicted before any
fit, and the predicted verdict was `small gain` or `inconclusive`.

**Two limits apply to every number on this page.** Experiment 713's known-answer gate failed two of
its checks and the maintainer waived it, so no arm here makes a statement about following level
shifts or recognising an era. The target row of every arm (`C`, `M*`, and `L`) still reads the CAMS
mean and clear-sky ratio of UTC day D - 1, which a live service may not hold at 09:00 UTC on day D.
L1 is therefore like for like in the history weather, and neither score is a live score.

**The maintainer parked the experiment after phase 6, because the question was already answered and
the remaining arms were not worth the compute and quota.** Phases 0 to 3 and phase 6 ran. Phase 5
(the budget probe) did not run, because the probe flag did not fire: L2 was below the two-seed noise
floor in both groups, and 0 of 16 fits of `L` were flagged. Arm `E` (phase 7) did not run. Phase 4,
the `T3` read-out, started and was stopped (see "What was not run"). No attention map and no
forecast time-series plot was produced, because the experiment was parked, and the page has no
figure.

## What was run

- **Phase 0 (wiring checks and reproduction):** all checks KL1 to KL10 passed under `checks_hash`
  `9b4b6cddf6122323`. Arm `Z`, the refit with no weather change, was fitted on fold 2025-11, seed 0.
  Its best validation loss matched the stored `M*` fit to 0.0 nats in stage 2 and in stage 3, and
  its CRPS equalled the stored `M*` CRPS (2.8113 at the six signed-MW Primaries, 3.4218 at the eight
  MVA-only Primaries). The reproduction check passed.
- **Phases 1 and 2:** `L` on 8 folds, seeds 0 and 1, and `S` predictions on the same folds. `S` is
  the stored stage 3 state of `M*` predicting on the live rule with no retraining.
- **Phase 1 quick look (interim: 4 folds, 2 seeds, 95% intervals):** `L` minus `C` was -0.0944
  [-0.1976, +0.0229] at the six signed-MW Primaries and -0.0651 [-0.1476, +0.0360] at the eight
  MVA-only Primaries. `L` minus `M*` was -0.0066 [-0.0133, +0.0028] and +0.0082 [-0.0083, +0.0312].
  `S` minus `C` was -0.0914 [-0.1864, +0.0356] and -0.0741 [-0.1495, +0.0327], and `S` minus `M*`
  was -0.0036 [-0.0249, +0.0123] and -0.0008 [-0.0056, +0.0061]. The audit rule for a large `L`
  minus `M*` did not fire.
- **Phase 3 (arm `Z` on the quick-look folds, then the counted two-seed reading):** `Z` minus `M*`
  was exactly 0.0 on every fold and seed in both groups, so audit trigger 1 did not fire. The
  two-seed reading (8 folds, 98.75%) gave L1 -0.1612 [-0.3140, -0.0251] at the six signed-MW
  Primaries (2 of 2 seeds negative) and +0.0283 [-0.0801, +0.1337] at the eight MVA-only Primaries.
  L2 was -0.0108 [-0.0274, +0.0008] and +0.0008 [-0.0124, +0.0185]. The verdict word was the same as
  the five-seed reading, and the futility rule did not fire, because L1 was below zero at the six
  signed-MW Primaries.
- **Phase 6:** `L` on 8 folds, seeds 2 to 4, giving the five-seed reading in the table above.

## What was not run

- **Phase 5 (budget probe), arm `E`, and phase 7:** not run, for the reasons above. The page
  therefore has no reading of `L` minus `E`, and no statement about dropping the ERA5T or
  half-hourly CAMS feeds.
- **`T3` (the Candidate, `M*`, and `L` predicting with the target row's CAMS day summary moved from
  D - 1 to D - 3):** phase 4 started and was stopped when the experiment was parked. These files
  exist.
    - `T3-M`: 4 prediction files (folds 2025-11 and 2025-12, seeds 0 and 1) under `work/fits/T3-M/`,
      and 4 logs.
    - `T3-C`: one one-off run for fold 2025-11, seed 0, under `t3_candidate/2025-11/`, with its
      patched trailing-power table, its prediction file, and `log_s0.json`.
    - `T3-L`: no file.

  **No `T3` reading is made on this page and none is quoted,** because a `T3` contrast needs all
  three arms on the same folds and seeds. The share of L1 that rests on the target row's CAMS day
  summary is unknown.
- **Plots of attention maps and of forecast time series:** not produced, because the experiment was
  parked.
- **The median minus the mean per site:** not computed, because the stored scoring files hold no
  per-site mean difference.

## Compute

**The experiment spent 3.54 GPU-process-hours, as recorded in `work/gpu_hours.json`, against a
reporting ceiling of 12.0 and projections of 6.04 (phase 0) and 5.34 (after phase 1).** The
projection of 5.34 is in `work/projection.json`, and the 6.04 comes from the manager's brief and is
not in a stored file. The spend is lower than the projections because phases 5 and 7 did not run and
phase 4 stopped early.

**The two tallies differ because `e00725_phases.py hours` skips the wiring-check jobs.** The raw sum
of `work/gpu_hours.json` is 12,728 s, or 3.5356 h: 2.0246 h of fits, 1.1589 h of predictions, and
0.3521 h of 16 wiring-check jobs (one train and one predict job for each of the 8 folds), which ran
with `--device cuda`. The `hours` command sums every key except those that contain " CHECK ", which
gives 3.1835 h. Phase 6 (1.44 h) and the smoke fit (149.6 s) are in both tallies. The exact figure
is 3.54 h, the file's sum, because the check jobs held a GPU. Both tallies miss work run outside the
runner or after its last write: the `T3-M` prediction of fold 2025-12, seed 1 (52.8 s in
`work/logs/T3-M/2025-12_s1_predict.json`, whose output exists), the one-off `T3-C` prediction
(49.6 s in `t3_candidate/2025-11/log_s0.json`; its fit time is not in that file), and the read-only
comparison runs of Fix A. We did not time the comparison runs, and estimate them at well under 0.2
h.

## Fix A: the `T3` known-answer check

**The `T3` known-answer check failed in 5 of 8 folds because of a fault in what the check
expected.** A read-only Opus audit found that the failing pick in each fold was the last training
origin, 12:00 UTC on the day before the fold's cutoff. The cutoff cuts that origin's target row in
half. The parent source holds zero weather at the unfilled half-hours, and `LiveSource` wrote the
moved CAMS summary into them. No arm, stored fit, or stored prediction was wrong, and `T3-L`,
`T3-M`, and `T3-C` predict only on fully filled rows.

**Fix A writes the moved summary only where the ensemble row exists, and the stored states are read
with the old code hash.**

| Hash | Before Fix A | After Fix A |
|---|---|---|
| `own_files_hash` (fit modules) | `f2c4e81c5a521927` | `2bd6286cad125ae6` |
| `checks_hash` (check and fit modules) | `f78faf11b97f94d5` | `9b4b6cddf6122323` |
| `code_hash` (stored states, gate) | `dad2fdbd70a1b0fb` | `d8a361ef74c85258` |

**Two read-only comparisons of the code before and after Fix A found no difference on a filled
half-hour.** On the predict phase of folds 2025-11, 2026-04, 2026-01, and 2026-02, every sample is
bit-identical, and the maximum absolute difference over every tensor of the 2,832 samples of folds
2026-01 and 2026-02 is 0.0. On the train phase of folds 2026-04, 2026-01, and 2025-12, 40 of 42
samples are bit-identical. The other 2 are the fold's last origin, and their 78 differing half-hours
are all unfilled. A second audit round confirmed the diff. The stored `L`, `Z`, and `E` states are
read with `--accept-code-change dad2fdbd70a1b0fb`, because the changed branch never runs in a fit.
The wiring checks ran again under the new `checks_hash` and passed.

## Docstring lint fix after the experiment was parked

**A docstring-only edit of four modules changed the three hashes, and nothing stored was
regenerated.** The lint step `pydoclint` reported 12 errors in `e00725_batch.py`,
`e00725_poison.py`, `e00725_sources.py`, and `e00725_world.py`. The edit added the missing `Args:`
and `Returns:` sections. The syntax trees of all 14 modules, compared with the docstrings stripped,
are identical before and after the edit.

| Hash | Before the edit (head 188eac42e) | After the edit |
|---|---|---|
| `own_files_hash` (fit modules) | `2bd6286cad125ae6` | `a93840a6a1ed7358` |
| `checks_hash` (check and fit modules) | `9b4b6cddf6122323` | `986904f8269b02c2` |
| `code_hash` (stored states, gate) | `d8a361ef74c85258` | `82cb4025ac6912e2` |

**No stored fit, prediction, or check file was regenerated, and the stored files keep the hashes
of the table before the edit.** The line is parked, so no later `predict` needs
`--accept-code-change`. A later session that resumes the line would pass `--accept-code-change
d8a361ef74c85258` and rerun the wiring checks under the new `checks_hash`.

## Checks

- **Audit triggers:** trigger 1 (`Z` minus `M*` above the two-seed noise floor) did not fire. The
  quick-look audit rule (a large `L` minus `M*`) did not fire. The probe flag did not fire (L2
  above the floor in neither group, 0 of 16 fits flagged). The `T3` known-answer failure was an
  implementation-check fault, audited by a read-only Opus audit, fixed with Fix A, and re-checked.
- **Reproduction:** `Z` equals `M*` to 0.0 on best validation loss and on CRPS, and the check
  passed. The across-seed standard deviation of fold 2025-11's CRPS is 0.0759 at the six signed-MW
  Primaries and 0.0627 at the eight MVA-only Primaries.
- **Look-ahead statement (KL8):** `L` and the inputs of `E` show no negative margin, and `Z` shows
  negative margins of up to 111 h by design. The read of `E` is the wiring-check read of the `E`
  inputs, because `E` was never fitted.
- **Experiment 713's work folder:** its hash was unchanged before and after every phase.
- **Guards for `supported`:** not met. The q01 pinball guard fails at the six signed-MW Primaries,
  and the leave-one-Primary-out guard fails at the eight MVA-only Primaries. The t guard fails the
  "gain" reading at the six signed-MW Primaries. None of the three changes the verdict word, because
  the verdict is `small gain`.
- **Not produced or not run:** the attention-map plots, the forecast time-series plots, `T3`, and
  `L` minus `E`. "What was not run" says why.
- **Host memory:** the largest peak in `work/memory_peak/` is 7,993,483,264 B (7.99 GB), from the
  prediction of `Z` for fold 2026-06, seed 1 (`e00725_predict.py_Z_2026-06_1.txt`). The peak of
  6,639,824,896 B is the `T3-M` prediction for fold 2025-12, seed 1, and is not the largest. Fits,
  predictions, and per-fold checks ran under `run.sh`'s 9 GB scope (`MemoryMax=9G`).
- **Median minus mean per site:** not computed.

## Method

- **Architecture and training regime:** experiment 713's `MsaNet`, stages 2 and 3, warm-started from
  the stored stage 1 state of `M*` of the same fold and seed, with frozen encoders and no new
  parameter. Each arm differs from the others in the history weather alone.
- **History weather of arm `L`:** each history token takes the ENS-mean weather of the stored run
  that a live service holds at the 09:00 UTC native origin (a run issued at least 9 h earlier) at
  ages 2 to 6, and keeps analyses only where the lags allow them. CAMS analyses are used up to UTC
  day D - 3 (a lag of 36 h after the end of the day), and ERA5 analyses from age 7 (a lag of 120 h).
  The rule changes 3,763 of the 14,112 weather values of a history matrix (27%). Training cells with
  no stored ENS row fall back to the analysis, and scored cells with no row read zero.
- **Measurement:** CRPS in points of capacity on leads from 24 h up to but not including 240 h after
  the 12:00 UTC origin, on experiment 713's rows and 8 monthly folds, month-block bootstrap at
  98.75% with 100,000 resamples, no valid time on or after 2026-07-01 scored. Seed `s` of one arm is
  paired with seed `s` of the other. The noise floors and `δ` are experiment 713's pre-registered
  values.
- **Differences for the other arms:** `C` and `M*` are stored forecasts, never refitted. `S` is the
  stored `M*` state predicting on the live rule with no retraining. `Z` is `M*`'s inputs refitted
  through this experiment's code. `E` (ENS at every age) and `T3` were planned and not run.
- **Standing practices followed:** monthly refit, cleaned data, fitted states saved outside git,
  stored forecasts reused as controls, and a screen on 4 folds first. Plots and the budget probe
  were not produced, as above.

## Files

**This page names experiment 713 and experiment 667 in plain text, because their pages are not yet
on `main`.**

- `ml-experiments/00725-msa-live-buildable-history-weather/`: `run.sh`, the live rule
  (`e00725_rule.py`), the sources and batch (`e00725_sources.py`, `e00725_batch.py`), the world and
  fit (`e00725_world.py`, `e00725_fit.py`), the prediction (`e00725_predict.py`), the checks
  (`e00725_checks.py`, `e00725_poison.py`, `e00725_weather_lookahead.py`), the frozen checkout and
  code hash (`e00725_frozen.py`), the `T3-C` rebuild (`e00725_t3_candidate.py`), the phase runner
  (`e00725_phases.py`), and the scorer (`e00725_score.py`, whose `five` command wrote
  `score/five_seed.json`).
- `data/ml-experiments/00725-msa-live-buildable-history-weather/` (private, never committed):
  `score/` (`quick.json`, `two_seed.json`, `five_seed.json`, `zrule.json`, `repro.json`,
  `probe_flag.json`), `decisions/`, `work/fits/` (the forecasts of `L`, `S`, `Z`, and `T3-M`),
  `work/models/`, `work/checks/`, `work/logs/`, `work/gpu_hours.json`, and `t3_candidate/`.
- `data/ml-experiments/briefs/msa_architecture/f8_design.md`, `f8_t3_code_review.md`, and
  `f8_t3_known_answer_audit.md` (private): the design, the `T3` code review, and the audit of the
  `T3` known-answer failure.
