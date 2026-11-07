---
id: "00623"
slug: pv-probe-monotone-arm-known-answer
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  way to make net demand non-increasing in shortwave irradiance is defensible, so the construction
  and every pass bar are fixed below before any recovery is read. Callers: this folder's scripts
  are copies of experiment 621's, and import experiments 397, 405, 409, 428, 448, 510, and 554 and
  edit none of them. Asset graph: not touched. One GPU process, about 0.7 GPU-hours.
hypothesis: >-
  A lag-free neural network whose net demand is constrained to be non-increasing in the shortwave
  irradiance input, with the datetime inputs held fixed, returns the AC capacity injected into its
  training target at the six MW Primaries at least as well as the unconstrained network of
  experiment 621, and with a smaller spread between seeds.
prediction: >-
  Fixed before any monotone-arm recovery was read. (1) The median recovery of the 4 MW real-farm
  injection lies in 0.80 to 1.20 at 4 of the 6 Primaries in the monotone arm. (2) The 95th percentile
  of each placebo is under 0.25 of the injected megawatts at 4 of 6. (3) With the cloud template, the
  cloud-following load fakes under 0.20 of a 2 MW farm. (4) The nominal 80% interval covers the
  injected capacity in at least 60% of injections. (5) The across-seed standard deviation of the
  probed capacity is lower in the monotone arm than in the plain arm at 4 of 6 Primaries. (6) The
  monotone constraint costs under 0.01 nats of validation negative log-likelihood.
decision_rule: >-
  Fixed before any recovery is read. Each bar passes or fails as written in the prediction. Stop and
  report when the median recovery is under 0.70 in both arms, or when a placebo reads over 0.30; no
  rescue tuning. The monotone arm counts as better than the plain arm only if it passes a bar the
  plain arm fails, or its across-seed spread is lower at 4 of 6 Primaries. The threshold for adopting
  the constraint in a production network is not set here, because the constraint adds a network
  component and would need a threshold that pays for it.
benchmark: >-
  The plain arm of experiment 621 (same rows, same injections, same probe code, same cutoffs and
  seeds), and the uninjected twin of each network.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, half-hour ends from 10:00 to 15:00 UTC.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day regression; recovery as a share of the
  injected megawatts; day-block bootstrap 80% interval; validation negative log-likelihood.
production_status: not-for-production
adopted_in: none
---

# Monotone arm of the injected known-answer test of the donor-day PV probe

*Planned. The write-up replaces this page when the fits and probes have run.*

**Arm.** The network is experiment 554's modular lag-free network M0 (stage `pre`) with one
addition: the shortwave input (column 0) enters only through a dedicated path that is non-decreasing
in shortwave irradiance, and that path is subtracted from the mean of every mixture component. The
construction is documented in `e00623_model.py`: the weather encoder sees the other six weather
columns with shortwave zeroed, and a context network reads the datetime encoder output and the
Primary embedding and returns the amplitude, slope, and offset of 16 softplus units.

**Everything else is experiment 621's.** The worlds, injected megawatts, farm rotations, cutoffs,
seeds, uninjected twins, probe dates, donor-day regression, and day-block bootstrap are those of
experiment 621. The plain arm is not repeated: its probe results are read from experiment 621's
private folder, and its code is the code this folder copied.

**Relaxed pass bars** (fixed before any read-out). Median recovery of the 4 MW injection between
0.80 and 1.20 at 4 of 6 Primaries (raw, never divided by other farms); 95th percentile of each
placebo under 0.25 of the real recovery; with the cloud template, the cloud load fakes under 0.20 of
a 2 MW farm; the 80% interval covers the injected capacity in at least 60% of injections. Each
Primary's value is the median (the 95th percentile for placebos) over the 2 cutoffs, the 3 seeds,
and the 2 farm rotations, as in experiment 621.
