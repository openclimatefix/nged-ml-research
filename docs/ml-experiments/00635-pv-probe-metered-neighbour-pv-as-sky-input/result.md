---
id: "00635"
slug: pv-probe-metered-neighbour-pv-as-sky-input
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights, probe results and a 30 km census in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  way to feed a metered farm into the network is defensible, so the input column, the held-out rule,
  the probe rescaling, and every bar are fixed below before any recovery is read. Callers: this
  folder imports experiments 405, 409, 448, 510, 554, and 621 and edits none of them. Asset graph:
  not touched. One GPU process at a time, at most 3 GPU-hours (about 1.2 expected).
hypothesis: >-
  Giving a lag-free neural network the normalised output of a metered solar farm near the Primary, as
  a second sky input beside the CAMS shortwave pixel, raises the share of an injected real farm that
  the donor-day probe reads back (median 0.68 per injected AC MW in experiment 621), because a
  measured local sky closes the part of the gap that comes from a CAMS pixel sitting 5 to 17 km from
  the farm.
prediction: >-
  Fixed before any recovery is read. (1) Arm b (CAMS plus the nearest usable metered farm) reads the
  4 MW real farm at a median at least 0.04 above arm a (0.68), that is 0.72 or more, and no more than
  0.94 per injected energy. (2) Arm c (metered farm only) reads below arm b. (3) The profile placebo
  with the metered column shuffled across days reads under 0.25 of the arm's real-farm recovery (95th
  percentile over seeds and Primaries). (4) Development-month validation CRPS of arm b is no more
  than 2% above arm a, and of arm c no more than 10% above.
decision_rule: >-
  Fixed before any recovery is read. Arm b counts as a gain if its median 4 MW real-farm recovery
  per injected AC MW exceeds arm a's by at least 0.04 and the paired difference over the 6 Primaries
  and 2 rotations has a 90% bootstrap interval that excludes zero, with the profile placebo (metered
  column shuffled across days) under 0.25 of the real recovery. The 0.04 threshold is the gap
  between the monotone arm of experiment 623 and the plain arm (0.03) rounded up, because adding a
  metered farm adds a data source to the pipeline. A placebo above 0.25 makes arm b invalid
  whatever its recovery. No rescue tuning.
benchmark: >-
  Arm a is the plain probe of experiment 621 at the 2026-06 cutoff (stored networks and probe
  results, same rows, same seeds). Each metered-input network is compared with its own uninjected
  twin that has the same input column.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-06
  cutoff, as in experiment 621 (nothing dated 2026-07-01 or later). Probe dates are 12 Wednesdays
  from 2025-04-09 to 2025-09-10.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day regression; recovery per injected AC MW
  and per injected energy (MWh over the probed days); development-month validation CRPS.
production_status: not-for-production
adopted_in: none
protocol: none
---

# Metered neighbour PV as a sky input to the donor-day probe (planned)

This page is the plan; the write-up replaces it when the fits have run. Read
[experiment 621](../00621-pv-probe-injected-known-answer-and-placebo-test/result.md) first.

- **Question.** A satellite pixel (CAMS) may sit 5 to 17 km from the solar farm that matters, and it
  cannot see the cloud edge over the farm. Does a measured output of a nearby farm, normalised by
  its own 99th percentile, give the network a better sky and lift the share of an injected farm that
  the probe reads back?
- **Arms.** (a) CAMS only: experiment 621's stored plain networks. (b) CAMS plus the metered column.
  (c) The metered column replaces the CAMS shortwave (column 0 is zeroed); the other six weather
  columns stay.
- **Input column.** For each of the six MW Primaries, the nearest of farms A, B, C, and F within
  30 km of the Primary's centroid, other than the farm injected at that Primary. Farms D and E are never an
  input because 3% to 36% of their half-hours are gaps, which the row mask does not cover. The ten other Primaries get 0.
- **Public Elexon units.** The solar Balancing Mechanism Unit census gives 12 single-site units, and
  Sutton Bridge Solar Farm is the only one within 60 km of a MW Primary (21 to 57 km). It has only 12
  months of settled output, too short for pretraining, so it enters the census and no network.
- **Worlds.** Those of experiment 621 at the 2026-06 cutoff and 3 seeds: real farm 2 and 4 MW (two
  rotations), physics curve 2 and 4 MW, profile and shuffled-sky placebos, four cloud-following
  loads, plus a profile placebo, with its own uninjected twin, whose metered column is shuffled across days within the month in training and in the probe. CRPS is scored on all validation rows and on the six Primaries' alone.
- **Standing practices.** Cleaned data and the D15/D18 cut are those of experiment 554's rows; the
  network is not refit monthly because it is a pretraining-only known-answer test; the networks are
  saved in the private data folder; no calendar column is added.
