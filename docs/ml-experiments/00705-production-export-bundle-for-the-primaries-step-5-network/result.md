---
id: "00705"
slug: production-export-bundle-for-the-primaries-step-5-network
title: >-
  Can production rebuild the step-5 demand network's forecasts for the Primaries from a plain
  directory of weights, standardisers, a config, and golden fixtures, without importing research
  code?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
kind: other
size: >-
  Complex. (1) It stores one export bundle per refit and arm (weights, standardisers, a config, and
  golden fixtures); (2) the bundle format is a design choice with several defensible layouts; (3) it
  imports code from experiments 48, 87, 397, 428, 437, 510, 558, 560, 562, 572, 578, 677, and 704;
  (4) it touches no serving path and no degradation rule; (5) it changes no contract. Reviews:
  code, write-up, and prose.
hypothesis: >-
  A directory holding the step-5 network's weights per seed, its standardisers, its column order,
  its expected-weather rows, and a config that names every constant is enough for a script that
  imports only torch, numpy, and polars to reproduce the stored quantile forecasts of the Primaries.
prediction: >-
  The standalone reference check reproduces, at five fixture origins and three seeds, the input
  matrix to 1e-5 and each seed's quantiles and the seed mean to 2e-5 of capacity, and the exporter's
  own rebuild reproduces the stored per-seed contract forecasts of experiment 677 exactly.
decision_rule: >-
  Supported if every tolerance in the prediction holds at every fixture origin and seed; otherwise
  the bundle is not fit to hand to production.
benchmark: >-
  The stored per-seed forecasts of experiment 677's arm `k1_5w_sp1` (refit 2026-04) and the input
  matrices that experiment 677's prediction pass builds.
row_set_id: none
metric: >-
  Largest absolute difference between the reference check's values and the stored values, per
  stage (input matrix, raw network outputs, per-seed quantiles, seed-mean quantiles, contract
  megawatts over capacity).
verdict: supported
best_arm: >-
  Format version 1; largest per-seed quantile error 1.1e-5 of capacity against a tolerance of 2e-5.
headline: >-
  A 102 MB bundle reproduces the step-5 network's quantiles from torch, numpy, and polars alone, to
  1.1e-5 of capacity.
lessons: >-
  A port must keep the research order of interpolating each ENS member before averaging, must change
  production's analysis proxy to the ensemble mean of each run's first day, and, if ERA5 arrives 5
  days late, cannot run a refit with a 2026-12-01 cut-off before about 2026-12-06.
take_home_message: >-
  Production can now rebuild the first demand forecast for the Primaries from a folder of files and
  check its own feature code against reference values for every input column. The port must
  average the 51 ensemble members only after interpolating them, and production's recent-weather
  history needs a small change to use the ensemble mean of each run's first day. If ERA5 arrives 5
  days late, the December refit can first run on about 6 December 2026.
advice:
  - >-
    Hand production a model as a bundle of plain files with golden fixtures per input column, not as
    research code, and keep a standalone reference check in the bundle (Checks).
  - >-
    Interpolate each ENS member to half-hours before taking the member mean and standard deviation,
    and derive wind speed per member; averaging first gives up to 4.2 m/s of error (Finding 1).
  - >-
    Build any recent-weather column in production from the ensemble mean at ENS leads of 9.5 to 33
    hours, as research does, not from member 0 of the freshest run (Finding 2).
  - >-
    If ERA5 arrives 5 days late, plan a monthly refit for about 5 days after its cut-off, because
    pre-training reads ERA5 up to 3.5 hours before the cut-off (Finding 3).
production_status: candidate
adopted_in: none
---

**Status: finished.** This experiment is a tool for the port rather than a test of a change. Its
code ran before this page existed, so the page has no planned commit of its own. The hypothesis
above is the acceptance test the code was written against.

## The question

**The first production demand model for the Primaries is the step-5 network, and production
re-implements its inference rather than importing research code.** The step-5 network is step 5 of
[experiment 677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md)
at monthly refits (arm `k1_5w_sp1`, seeds 0, 1, and 2). Production code never imports this
repository, so the network must leave research as files that a reviewer can read, plus reference
values that show where a production feature disagrees with research. The experiment builds that
export, proves the files are complete with a standalone check, and records three findings that the
port depends on.

## The bundle

**One bundle is one plain directory per refit month and arm, with a versioned format.** The format
is in `bundle_readme.md`, which the exporter copies into each bundle as `README.md`:

- `manifest.json`: format version, arm, refit time, training cut-off, research commit, the pinned
  production-package commit, library versions, the fixture origins, the export checks, and the
  SHA-256 of every file.
- `config.json`: every constant that inference reads, including the ENS summary recipe, the lag
  rule, the column slices, the 11 quantile levels, the bisection range and steps, the
  standard-deviation clamp, and the expected-weather rule, plus the training provenance.
- `primaries.json`: the 16 Primaries in embedding order, with H3 cell, latitude, longitude,
  `effective_capacity_mw`, and `capacity_mw`.
- `components/step5/`: the 60 input columns in order, the standardisers, the offset-knot dates, the
  expected-weather rows, and one `state_dict` per seed.
- `fixtures/step5/origin=<date>/`: for each fixture origin, the raw 51-member ENS values before and
  after interpolation, the ENS summary, every input column raw and standardised, the power the lags
  read, the raw network outputs, and the quantiles per seed and seed-averaged, at the 09:00 native
  origin and at the 12:00 contract origin in megawatts.
- `reference_check.py`: the standalone forward pass and check.

**A later bundle can carry the K1 network and the Candidate without a format change.** Each network
gets its own `components/<name>/`, its own `fixtures/<name>/`, and its own sections in
`manifest.json` and `config.json`. The Candidate's table T, neighbour list, and K1 step-2 weights
become further files in its component folder.

**The fixture origins cover the cases most likely to break a port.** The 2026-04 test bundle has
five origins: 2026-04-01 (week lag missing at some Primaries), 2026-04-03 (Good Friday), 2026-04-04
(a Saturday), 2026-04-17, and 2026-04-30 (week lag missing at 140 rows). No fixture origin has a
missing day lag, so the bundle also stores the quantiles of 2026-04-30 with both lags set to the
missing encoding, which the check reproduces.

## Checks

| Stage | Largest absolute error | Tolerance |
|---|---|---|
| Input matrix (60 standardised columns) | 0.0 | 1e-5 |
| Raw network outputs | 3.3e-6 | 1e-5 |
| Quantiles per seed (fraction of capacity) | 1.1e-5 | 2e-5 |
| Seed-mean quantiles | 3.7e-6 | 2e-5 |
| Contract megawatts over capacity | 3.7e-6 | 2e-5 |
| Quantiles with both lags missing | 2.6e-6 | 2e-5 |

**The reference check passes at all five origins and three seeds, in 7 seconds on a CPU.** The
check runs as `python -I reference_check.py <bundle>`, which keeps the research folders off the
import path, and it imports only the standard library, torch, numpy, and polars. The differences
are consistent with float32 rounding in the stored forecasts' GPU pass against the check's CPU
pass, which the quantile bisection amplifies. The raw-output tolerance of 1e-5 is an extra
tolerance of `reference_check.py`, beyond the prediction above.

**The exporter reproduces the stored forecasts exactly before it writes the fixtures.** Re-running
the prediction pass of the stored weights gives per-seed contract forecasts equal to experiment
677's stored backtest (largest difference 0.0 MW at every origin and seed). The exporter rebuilds
the lag, knot-position, and B1 columns from the raw columns and the standardisers, and the rebuilt
columns equal the network's input exactly. The five fixture ENS runs, rebuilt from the archive with
the pinned production function, equal the stored summary files exactly.

**The check rebuilds the standardised matrix, not the raw columns.** The ENS summary, the solar and
calendar columns, and the B1 columns are checked only against the fixture files, so production must
compare its own values of those columns with `input_matrix.parquet`.

## Findings for the port

**Finding 1: the research ENS summaries interpolate each member before averaging.**
`_half_hourly_rows` of experiment 48 calls the production function
`ml_core.features._nwp._upsample_nwp_to_half_hourly` (pinned commit `2c1bb51`) on each member's
native 3-hourly and 6-hourly steps, derives wind speed from each member's interpolated u and v, and
only then takes the member mean and the population standard deviation. On run 2026-04-01, this
order reproduces the stored summary exactly. Averaging the 51 members first and interpolating once
differs by up to 5.75 W/m2 in shortwave, 2.56 W/m2 in longwave, and 4.24 m/s in wind speed.
Production already has the interpolation function, so a port has to apply the function to each
member and average afterwards.

**Finding 2: the research analysis proxy differs from production's `select_analysis_proxy` in three
ways.** Research takes the ensemble mean, where production takes member 0. Research takes each
half-hour from the run whose first day covers it (ENS leads of 9.5 to 33 hours), where production
takes the freshest run, at leads of 0 to 9 hours for the last 9 hours. Research leaves a missing
value missing, where production falls back to an older run. On 12 days before the origin
2026-04-01 at Primary D2, the installed production function differs from the research history by up
to 0.63 °C in temperature and 57.2 W/m2 in shortwave. Two optional arguments, `member=None` for the
ensemble mean and `min_lead=9 h` with the existing `max_lead=33.5 h`, make a prototype of the
production function equal the research history exactly. Only temperature and shortwave enter the
14 B1 columns.

**Finding 3: a refit with a 2026-12-01 cut-off can first run on about 2026-12-06.** Pre-training
and the standardisers read every row with a valid time up to 3.5 hours before the cut-off, and the
ERA5 rows of pre-training arrive about 5 days late, so ERA5 up to 2026-11-30 20:30 UTC arrives on
about 2026-12-05 evening. The other inputs are ready earlier: NGED power to 2026-11-30 20:30 by
about 2026-12-01 02:30, assuming the 6-hour delay of the production contract, and the 00Z ENS run of
2026-11-30 on that day. Forecasts for origins 2026-12-01 to 2026-12-05 would therefore use the
November weights, whose offset is held flat past its newest knot. The 5-day ERA5 lag comes from the
port brief and the 6-hour NGED delay from the production contract's port notes; neither was
measured here, no production constant records the NGED delay, and the run time of the chain is not
included.

## Method

- `run_export.sh ARM MONTH` runs the three steps for one refit: `e00705_capture.py` reruns the
  stored models' prediction pass on the GPU in a scratch root and dumps the input matrix and
  outputs; `e00705_export.py` writes the bundle on the CPU and asserts the reproductions above; and
  `e00705_check.py` reloads the bundle and runs `reference_check.py` in a clean interpreter.
- `ARM` is `replay` (experiment 677's own chain) or one of experiment 704's arms, `proxy` or `drop`.
- `e00705_findings.py` computes the evidence for findings 1 and 2.
- The test bundle is `replay`, refit 2026-04, exported at commit `8e797d8` with a clean tree.

## Files

- Code: `ml-experiments/00705-production-export-bundle-for-the-primaries-step-5-network/`. The
  scripts import experiments 48, 87, 397, 428, 437, 510, 558, 560, 562, 572, 578, 677, and 704,
  and load experiment 574's fit module through experiment 677's stored chain.
- Bundles: `data/ml-experiments/00705-production-export-bundle-for-the-primaries-step-5-network/bundles/<arm>/<month>/`.
- Findings: `findings.txt` and `findings_evidence.json` in the same data folder.
