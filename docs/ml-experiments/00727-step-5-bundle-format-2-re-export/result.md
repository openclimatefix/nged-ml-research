---
id: "00727"
slug: step-5-bundle-format-2-re-export
title: >-
  Can the step-5 demand network's export bundle be re-exported as bundle format 2, with numpy
  weights and NGED's 13 delivery quantile levels, without changing any forecast?
theme: production-reference-and-best-guess
also_relevant_to: []
status: finished
kind: other
size: >-
  Simple. (1) It stores one re-exported bundle per source bundle; (2) one design is defensible,
  because the brief fixes the three changes; (3) it imports the export and check code of
  experiment 705 only; (4) it touches no serving path and no degradation rule; (5) it changes no
  contract. Reviews: code and prose.
hypothesis: >-
  The step-5 network's mixture-density head gives any quantile level by bisection of its
  cumulative distribution function, so the format 1 bundle can be re-exported with numpy weights
  and NGED's 13 delivery levels, and the quantiles at the 7 levels that format 1 also holds do not
  change.
prediction: >-
  At every fixture origin and seed, each of the 7 shared quantile levels differs from the format 1
  fixture by at most 2e-5 of capacity, every numpy array equals its torch tensor exactly, and the
  format 2 reference check passes at the format 1 tolerances.
decision_rule: >-
  Supported if every tolerance in the prediction holds. Stop and report to the manager if any
  shared level differs by more than its tolerance.
benchmark: >-
  The format 1 bundles of experiment 705 (`bundles/replay/2026-04` and `bundles/drop/2026-10`).
row_set_id: none
metric: >-
  Largest absolute difference, as a fraction of capacity, between the format 2 fixtures and the
  format 1 fixtures at the 7 shared levels, and the largest error of the format 2 reference check
  per stage.
verdict: supported
best_arm: >-
  Format version 2; largest difference to format 1 at the 7 shared levels 1.1e-5 of capacity
  (replay bundle) against a tolerance of 2e-5.
headline: >-
  Both step-5 bundles are re-exported as format 2 with numpy weights and NGED's 13 quantile
  levels, and the 7 levels that format 1 also held move by at most 1.1e-5 of capacity.
lessons: >-
  The mixture head gives any quantile level, so a change of levels needs no retraining. The format
  2 reference check cannot test the regenerated quantiles, because the same read-out wrote them.
take_home_message: >-
  Production can now load the first demand forecast for the Primaries without torch, and gets the
  13 quantile levels that NGED asked for. The forecasts did not change: at the 7 levels that the
  old bundle also held, the new values differ by at most 1.1e-5 of a substation's capacity.
advice:
  - >-
    Change a mixture-density network's output levels in the bundle's config and regenerate the
    quantile fixtures, without retraining, and compare the levels the old and new bundles share
    (Checks).
  - >-
    Do not read a pass of the bundle's own reference check as evidence for regenerated quantile
    fixtures, because the same code wrote them; rest the evidence on the comparison with the
    previous bundle (Checks).
production_status: candidate
adopted_in: none
---

**Status: finished.** The experiment re-exports two bundles of experiment 705 and tests no change
to a forecast. The hypothesis above was the acceptance test.

## The question

**Production pull request 1707 needs the step-5 demand network's bundle without `torch`, and with
the 13 quantile levels that NGED asked for.** The step-5 network is the first demand forecaster for
the 16 primary substations (Primaries) of the Flexpectation study. [Experiment
705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md) wrote its
weights, standardisers, configuration, and test fixtures as a plain directory (bundle format 1).
This experiment re-writes that directory as bundle format 2 and changes three things. The weights
are also stored as numpy arrays. The manifest gains `model_release`, set to 0 because neither
bundle is a model release. The quantile levels become 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50,
0.65, 0.80, 0.90, 0.95, 0.98, and 0.99. No network is retrained, and nothing under `src/` or any
production package changes.

**The answer is yes: the 13 levels need no retraining, because the network outputs a mixture of
three Gaussian distributions.** The read-out bisects the mixture's cumulative distribution function
at each level, so a level is a number in the configuration. The new bundles reproduce the old
quantiles at the 7 levels that both formats hold (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, and 0.99) to
within 1.1e-5 of capacity, against the 2e-5 tolerance of experiment 705's reference check. The
levels 0.25, 0.40, 0.60, and 0.75 leave the bundle, and the levels 0.02, 0.20, 0.35, 0.65, 0.80,
and 0.98 enter it.

**Production pull request 1707 is not blocked on anything else here.** The bundles are:

- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/replay/2026-04` (experiment
  677's arm `k1_5w_sp1` refitted at 2026-04-01, 102 MB; the arm that the request named).
- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/drop/2026-10` (experiment
  704's arm B refitted at 2026-10-01, 97 MB; the bundle that experiment 705 called the production
  bundle).

## The bundle

**Each bundle is one directory, and `README.md` inside it describes the format for a reader who has
not seen experiment 705.** The directory holds these files:

- `manifest.json`: format version "2", `model_release` 0, the arm, the refit time, the training
  cut-off, the research commit, library versions, the fixture origins, the export checks, and the
  SHA-256 of every other file.
- `config.json`: every constant that inference reads, including the 13 quantile levels.
- `primaries.json`: the 16 Primaries with their H3 cell, coordinates, and capacity.
- `components/step5/`: the 60 input columns, the standardisers, the offset-knot dates, the
  expected-weather rows, and each seed's weights as `weights/seed<N>.npz` (numpy arrays keyed by
  the `state_dict` names) and `weights/seed<N>.pt` (the same tensors, for research code).
- `fixtures/step5/origin=<date>/`: for each of five fixture origins, the raw and interpolated
  ensemble weather, every input column, the raw network outputs, and the quantiles at the 13
  levels per seed and as the seed mean, at the 09:00 native origin and the 12:00 contract origin.
- `reference_check.py`: a standalone forward pass and check that needs only `torch`, `numpy`, and
  `polars`.

## Checks

| Quantity | Replay 2026-04 | Drop 2026-10 | Tolerance |
|---|---|---|---|
| Raw network outputs against format 1 | 3.3e-6 | 2.6e-6 | 1e-5 |
| Per-seed quantiles at the 7 shared levels, fraction of capacity | 1.1e-5 | 2.3e-6 | 2e-5 |
| Seed-mean quantiles at the 7 shared levels | 3.7e-6 | 7.2e-7 | 2e-5 |
| Contract megawatts over capacity at the 7 shared levels | 1.1e-5 | 2.3e-6 | 2e-5 |
| Quantiles with both lags missing at the 7 shared levels | 2.1e-6 | 3.0e-6 | 2e-5 |
| `.npz` against `.pt` (names, shapes, dtypes, bytes) | 96 of 96 arrays equal | 96 of 96 arrays equal | exact |
| Format 2 reference check, all stages | passed in 9 s | passed in 8 s | 1e-5 and 2e-5 |

- **The comparison with format 1 is the evidence for the new quantiles.** The format 2 reference
  check reports 0 for every quantile stage, because the same read-out wrote the regenerated
  fixtures. The check still tests the input matrix (exact) and the raw outputs (3.3e-6 and 2.6e-6).
  The numbers in the first five rows are stored in each bundle's `manifest.json`, under
  `export_checks.format_2_reexport`.
- **The largest difference, 1.1e-5, is at the 2026-04-30 origin of the replay bundle.** Format 1's
  stored quantiles came from a GPU pass, and the CPU pass differs by float32 rounding that the
  bisection amplifies. The same size of difference appears in experiment 705's own check of this
  bundle.
- **The median forecast minus the mean forecast, and the error of the mean forecast beside each
  quantile score, do not apply.** The experiment scores no forecast. The seed-mean quantiles, which
  include the median, differ from format 1 by at most 3.7e-6 of capacity in the replay bundle and
  7.2e-7 in the drop bundle.
- **No audit trigger fired.** No control failed, no result contradicted its prediction, and no
  contrast exists to flip.

## Method

**Architecture.** The architecture is unchanged from experiment 705 and
[experiment 677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md)
step 5.

- The network reads 60 input columns: ensemble weather summaries and spreads, calendar columns,
  the lead time, a day lag and a week lag of recent power, an offset-knot position, and 14 more
  columns.
- Its outputs are three mixture weights (logits), three means, and three log standard deviations,
  clamped to the range -3 to 3. The means include an offset interpolated between monthly knots.
- A quantile is the bisection of the mixture's cumulative distribution function between
  -40 and 40 in standardised units, in 40 steps, in float64.
- The forecast uses only information from before the origin, as in experiment 705.

**Training regime.** No training. Each bundle holds the stored weights of three seeds (0, 1, and
2) from experiment 677 (replay) and experiment 704 arm B (drop).

**Export and measurement.**

- The exporter copies the format 1 bundle, writes each seed's `.npz` from its `.pt`, sets the 13
  levels in `config.json`, and re-runs each seed's forward pass on the CPU from the stored
  standardised input matrix, for all five origins and for the variant with both lags missing.
- It averages the three seeds in float32 and sorts, as format 1 did, and multiplies by `capacity_mw`
  in float64 before rounding to float32 for the contract rows.
- It compares the seven shared levels with the format 1 fixtures and stops with an error if any
  difference exceeds the format 1 tolerance (1e-5 raw, 2e-5 of capacity for quantiles).
- A code review by an Opus agent before the first run found no bug that changes a number, and
  its fixes were applied: the exporter builds in a `.partial` folder and renames it on success, the
  old export checks are filed under `format_1_export`, level names are asserted against level
  values, and the contract-row order is asserted.

## Files

- `ml-experiments/00727-step-5-bundle-format-2-re-export/e00727_reexport.py`: builds a format 2
  bundle from a format 1 bundle and runs the comparison.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/e00727_verify.py`: checks that every
  `.npz` array equals its `.pt` tensor bit for bit, that the file hashes in `manifest.json` match,
  and that the format 2 fields are set.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/reference_check.py`: the format 2 copy of
  experiment 705's reference check (13 level names, weights from `.npz`, the same tolerances),
  copied into each bundle.
- `ml-experiments/00727-step-5-bundle-format-2-re-export/bundle_readme.md`: the bundle's `README.md`.
- `data/ml-experiments/00705-production-export-bundle-for-the-primaries-step-5-network/bundles/`:
  the format 1 bundles that the exporter read (`replay/2026-04` and `drop/2026-10`), from
  [experiment 705](../00705-production-export-bundle-for-the-primaries-step-5-network/result.md).
- `data/ml-experiments/00727-step-5-bundle-format-2-re-export/bundles/`: the two format 2 bundles
  written by this experiment (`replay/2026-04` and `drop/2026-10`).
