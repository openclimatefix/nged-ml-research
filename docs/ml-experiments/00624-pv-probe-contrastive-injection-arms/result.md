---
id: "00624"
slug: pv-probe-contrastive-injection-arms
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights and probe results in this experiment's private folder.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one way to build a
  contrastive injection loss is defensible, so the construction, the arms, and every bar are fixed
  below before any recovery is read. Callers: this folder's scripts import experiments 397, 428,
  448, 510, 554, 621, and 623 and edit none of them. Asset graph: not touched. One GPU process, about
  1 to 2 GPU-hours.
hypothesis: >-
  Teaching a lag-free neural network, during pretraining, to reproduce a known synthetic +1 MW PV
  injection in paired rows (a contrastive loss) makes the donor-day probe recover injected capacity
  in worlds the loss never saw (a real farm, the physics curve) closer to 1 than the plain network
  does, without raising the placebo reads or lowering forecast skill.
prediction: >-
  Fixed before any recovery was read. (1) Arm (a) recovers a 4 MW real-farm injection with a median
  between 0.80 and 1.20 at 4 of 6 Primaries (the plain network's median is 0.68). (2) The profile
  twin reads under 0.25 of the physics twin's response, and the 95th percentile of each placebo world
  is under 0.25 at 4 of 6. (3) With the cloud template, a cloud-following load fakes under 0.20 of a
  2 MW farm. (4) The 80% interval covers the injected capacity in at least 60% of held-out
  injections. (5) Validation CRPS of arm (a) is within 1% of the plain network's. (6) Arm (b), the
  post-hoc check, reads the physics twin near 1 MW and so cannot detect the real-farm shortfall.
decision_rule: >-
  Fixed before any recovery is read. Each bar passes or fails as written. An arm counts as better
  than the plain network only if it passes the real-farm recovery bar that the plain network fails,
  and its placebo and cloud bars pass. Stop and report if the real-farm median recovery of arm (a)
  is under 0.70 (no rescue tuning). The threshold for adding a contrastive loss to a production
  network is not set here, because the loss adds a training component.
benchmark: >-
  The plain network of experiment 621 (same rows, worlds, cutoffs, seeds, and probe), and the
  monotone arm of experiment 623 as context.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the 2026-03 and
  2026-06 cutoffs (development data only; nothing dated 2026-07-01 or later). Probe dates are 12
  Wednesdays from 2025-04-09 to 2025-09-10, half-hour ends from 10:00 to 15:00 UTC.
metric: >-
  Effective AC capacity in MW per Primary from the donor-day regression; recovery as a share of the
  injected megawatts; day-block bootstrap 80% interval; validation CRPS.
production_status: not-for-production
adopted_in: none
---

# Contrastive injection arms for the donor-day PV probe

*Planned. The write-up replaces this page when the fits and probes have run.*

**Construction (how "the loss asks whether the network sees the +1 MW against the same rows without
it").** Every real training row of the six MW Primaries gets two twin rows that share its inputs.
The physics twin has a synthetic +1 MW of PV subtracted from its target, with the shape of
experiment 269's pooled plant table (the unit curve) driven by the row's own CAMS shortwave. The
profile twin has +1 MW of a metered farm's calendar-mean shape subtracted, which has no sky
dependence. Each twin trains its own embedding vector (12 added vectors, no other parameter), so the
pairing of an injected and an uninjected row is carried by the embedding. The probe is then run
through the real vector and through each twin vector at the same Primary; the difference is the
network's response to the synthetic +1 MW.

**Arms.** (a) The twin rows enter the negative log-likelihood, and a paired-difference term
`1.0 * mean(((mean_twin - mean_real - delta) / 0.05)^2)` forces the twin's mixture mean to differ
from the real row's by the known change `delta` in standardised units, on the same inputs. (b) A
post-hoc check on experiment 621's plain networks: only the twin vectors are fitted (4 epochs at
learning rate 0.01, every other parameter frozen) on the twin rows. (c) Arm (a) followed by the same
post-hoc refit. Every arm runs with the profile twin as the placebo.

**Held-out worlds.** The synthetic +1 MW is the only injection that the loss sees. Each network is
also trained in one world injected into every row's target at the six Primaries, as in experiment
621: a real farm (4 MW, two rotations), the physics curve (4 MW), a profile-only placebo (4 MW), a
shuffled-sky placebo (4 MW), a strong linear cloud-following load, and no injection (the control).
Recovery of that world is scored against the control of the same arm, cutoff, and seed.

**Bars** are experiment 623's relaxed bars: median 4 MW real-farm recovery between 0.80 and 1.20 at
4 of 6 Primaries; 95th percentile of each placebo under 0.25 at 4 of 6; spurious capacity from a
cloud load under 0.20 of a 2 MW farm with the cloud template; 80% interval coverage of at least 60%.
Also reported: the twin's synthetic recovery (in sample), validation CRPS against the plain network,
and the arm (b) check.
