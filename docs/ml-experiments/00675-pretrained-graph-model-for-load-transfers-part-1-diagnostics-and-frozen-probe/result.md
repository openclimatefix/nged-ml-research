---
id: "00675"
slug: pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe
title: >-
  Why does the search of experiment 670 leave most recipient megawatts unpaired, and does a
  pretrained encoder find the donor of an injected load transfer better than the event table's own
  features?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-09
kind: other
size: >-
  complex. (1) Stored: private residual panels, injected-world banks, probe tables, and figures
  under data/ml-experiments/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/
  only. (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one
  is defensible for the pretraining objective, the residual panel, and the probe, so each is fixed
  below before any score is read. (5) Callers: this folder's scripts, which import the folders of
  experiments 653, 658, 660, 665, 670, and 672; nothing imports them. Step 0 is CPU plus under 1
  GPU-hour; the probe adds at most 3 GPU-hours, behind the maintainer's one-year backtests.
hypothesis: >-
  Experiment 672 trained a graph neural network on 120 injected worlds and 6-hour blocks. The
  network found donors about as often as E1-J (0.47 of the long and deep donor megawatts against
  0.44) and then paired few of them (0.09). Three changes have not been tested: a network that
  first learns each Primary's ordinary variability by reconstructing masked windows of real
  history, an input set that includes the same step statistics E1-J uses, and a residual that is
  not distorted by the convex demand model's error. This experiment first asks where E1-J loses
  the recipient megawatts it finds (0.73 found, 0.22 paired), then asks whether a pretrained
  encoder, frozen and read by a logistic probe, finds donors and recipients better than a random
  encoder of the same shape and better than the event table's features alone. It reopens the graph
  line that experiment 672's stop rule closed at 15 Primaries, at the maintainer's request.
prediction: >-
  Written before any score is read. Step 0a (no prediction beyond a ranking): the reason that
  holds the largest share of the unpaired recipient megawatts in the long and deep and the medium
  and deep cells will be named, among six reasons fixed in the method. Step 0c: E1-J on the new
  residual panel pairs less than 0.60 in at least one of the long and deep and the medium and deep
  cells (so stop rule 0 does not fire). Probe (each a mean over 3 seeds; no seed more than 0.05
  below its bar): masked pretraining (P-M) minus the random encoder (R) in donor megawatts found
  at E1-J's false-donor rate is at least +0.05 in the long and deep, medium and deep, and long and
  shallow cells and at least +0.03 in the medium and shallow cell; P-M minus the logistic
  regression on the node inputs alone (F) is at least +0.03 in the two deep cells and 0 or more in
  the long and shallow cell; P-M minus R in recipient AUROC is at least +0.05 in the two deep cells
  and +0.03 in the long and shallow cell; P-M minus the causal arm (P-C) is 0 or more in the two
  deep cells.
decision_rule: >-
  Fixed before any score is read. Worlds come from experiment 660's generator, unchanged, injected
  into the power of experiment 660's panel; seeds follow experiment 672: training from 100000,
  validation from 105000, scored worlds 900 to 939, cross-injector worlds 1000 to 1009. The
  final-test window from 2026-07-01 is never read, and pretraining sees no row on or after
  2024-01-01. Stop rule 0 (step 0c): if E1-J on the new residual panel pairs 0.60 or more in both
  the long and deep and the medium and deep cells, adopt E1-J on that panel as the event table and
  do not run experiment 676. Probe stop rule: if P-M fails both its R bar and its F bar in both
  deep cells, drop pretraining and run experiment 676 with the randomly initialised arm as its main
  arm; if P-C beats P-M by 0.03 or more in a deep cell, experiment 676 uses P-C's weights. Leak
  tests: both leak tests of the method (masked-window and scored-target) pass in the code review
  before any GPU fit, or no fit runs. The porting threshold is zero, because the result is a
  research table that changes no production code. Verdict. `supported` if the P-M prediction
  holds in both deep cells for donor megawatts found against both R and F; `partly supported` if it
  holds against R only, or in one deep cell only; `not supported` otherwise; `inconclusive` if a
  deep cell has fewer than 20 events or 10 edge-transfer ends.
benchmark: >-
  E1-J on the new residual panel and on experiment 672's panel, from experiment 670's stored rows
  for the latter; the random-initialised encoder R and the no-encoder arm F for the probe.
row_set_id: >-
  experiment 660's injected worlds (default configuration, scored seeds 900 to 939, scored years
  2024-01-01 to 2026-06-30) and experiment 653's real record of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted share of unpaired recipient megawatts by reason; E1-J paired, found, donor-found, and
  recipient shares per cell on each panel; probe: share of donor megawatts found at E1-J's
  false-donor rate, and recipient AUROC given the donor, per cell, with 95% intervals from
  resampling whole worlds. No forecast is scored, so the four load-shift scenarios, tail
  calibration, and the curtailment-cost metric do not apply.
code: ml-experiments/00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/
---

**This is the plan, committed before any score is read.** It reopens the graph line that
[experiment 672](../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md) closed at
15 Primaries under its pre-registered stop rule. The maintainer asked for the second test, and the
page says so because the stop rule is not overturned by new evidence. It is a deliberate test of
components that experiment 672 did not build, not a repeat of it. Experiment 675 is part 1; the
edge-state network is part 2 (experiment 676), run only if part 1 does not adopt the event table.

## The question

[Experiment 670](../00670-event-table-recipient-search-on-six-cell-grid/result.md) found that the
event table with a recipient search (E1-J) pairs 0.29 of the injected megawatts in the long and
deep cell and 0.25 in the medium and deep cell. It finds 0.73 of the recipient megawatts in the
long and deep cell and pairs 0.22, and no experiment has said where the other 0.51 is lost. Experiment
672's graph network found donors as well as E1-J and then failed to link them (0.47 of donor
megawatts found, 0.09 paired). This part answers two questions in order. First, why does E1-J leave
found recipients unpaired, and does a residual panel built by the lag-free modular network (instead
of experiment 327's convex residual) change what E1-J finds? Second, does an encoder pretrained on
masked windows of real history carry information about donors and recipients that a random encoder
and the event table's own features do not?

## Method

- **Step 0a, E1-J's unpaired recipients (CPU).** From experiment 670's stored rows for the long
  and deep and the medium and deep cells, each found but unpaired recipient end is sorted into the
  first reason that applies: the donor was not found; the recipient start lies outside the search
  window; the megawatt ratio lies outside the share window; the recipient was linked to the wrong
  donor; the transfer was split across two or three recipients; other. The page reports the
  megawatt share of each reason and says which end the next step should aim at.
- **Step 0b, the residual panel.** In order, the first that succeeds is used: (1) the lag-free
  modular network (weather encoder, date-time encoder, mixer, no lags) on ERA5 and CAMS from
  2019-09, cross-fitted by year so each year is scored by the copy trained on the other years, for
  the 15 Primaries with experiment 591's early-calibration cuts; (2) the same network from 2022 if
  the 2019 to 2021 cleaned power or weather is not on disk; (3) experiment 327's convex residual.
  The all-Primaries median is removed in all three. At the maintainer's instruction (relayed by the
  manager), the lag-free model is made as good as the tricks that worked allow. Before the panel is
  built, the plan's first task is an inventory from the merged pages of the lag-free demand recipes
  and tricks that helped: the best-guess and K1/K3 lag-free bodies, the explicit solar term of
  experiments 639 and 643, temperature-trend inputs such as experiment 622, the D5 offset question,
  holiday and calendar features, and whatever else the pages show. The expected-power model then
  takes (1) an explicit solar term from CAMS and ERA5 known weather (experiment 649's capacity table
  as the centre with its interval, a physics curve from the pooled plant table with a learned fleet
  blur, calibrated rather than raw capacity, never scored on forecast weather; traps: a sunny spring
  can look like a capacity step, and a cloud-following load looks like solar), (2) temperature-trend
  inputs, and (3) the other proven tricks. It uses no lagged power and no free slow offset that
  could absorb a long event (E1's masked frozen seasonal slow level instead), is cross-fitted by year
  so a year's events are not in the copy that scores it, and cuts ERA5 and CAMS at 2026-07-01. The
  page reports its out-of-fold residual noise per Primary against experiment 327's arm C residual
  (the one E1 and E1-J used), uses the better residual for E1, E1-J, and the network alike, and
  re-runs E1 and E1-J on it so that a better residual cannot pass for a better detector. If this
  needs more than 1 GPU-hour in step 0, the plan pull request says so. The page names the panel
  used and whether its weather exists at forecast time (it does not: ERA5 arrives about 5 days
  late, so the panel is a laboratory residual, as in experiments 670 and 672).
- **Step 0c, equal treatment.** E1 and E1-J are re-run on the new panel with experiment 660's
  default generator, unchanged, and E1-J's penalties are retuned on validation worlds with
  experiment 670's grid. These re-runs are the baselines for the probe and for experiment 676.
- **Step 0d.** The verified D5 and D16 transfer is classified into its cell with experiment 653's
  noise units.
- **Probe encoder.** A shared bidirectional temporal convolutional network per Primary on two
  streams: a 1-hour stream of 6 dilated layers with 64 channels, and a daily stream of 4 layers
  with 32 channels, fused by adding the upsampled daily features. Node inputs are the residual in
  noise units (clipped at 30), a present flag, the expected-power q05 to q95 width in noise units,
  the mean-residual gaps over the past and the future 1, 3, and 7 days, step statistics at 6-hour
  and daily scales in both directions of time, the noise unit as log megawatts, the routine-step
  rate, a megawatt-or-megavolt-ampere flag, and hour and weekday as sine and cosine. Each past
  statistic has a future twin stored as a separate channel, and every window carries a coverage
  channel.
- **Pretraining.** The masked arm (P-M) reconstructs the residual in masked windows with a
  Student-t distribution per hour (1-hour stream, windows of 3 to 24 hours, 1 to 3 per Primary) and
  a daily mean (daily stream, windows of 1 to 7 days). In 30% of samples the same window is also
  masked at one or two neighbours, and 40% of Primaries carry a mask. The loss counts masked hours
  only. The causal arm (P-C) predicts the mean and spread of the next 3 to 6 hours with causal
  convolutions. Rows before 2024-01-01 only; AdamW at learning rate 1e-3 with cosine decay, about
  20,000 steps, timed over the first 500 steps against the GPU budget.
- **Leak tests (both run in the code review, before the first fit).** (1) Masked-window leak:
  for 200 seeded samples, replace the masked residual (at the masked Primary and every co-masked
  neighbour) with Gaussian noise of 10 noise units, recompute every input, and assert the inputs are
  bit-identical (`torch.equal`, float64, CPU) and the reconstruction predictions agree to 1e-12;
  repeat with a step of +20 noise units. Every statistic whose window overlaps the mask is
  recomputed from the readings outside the mask, per sample, after the mask is drawn. (2) Scored
  target leak: replacing the injected-truth table by a shuffled copy must leave every input
  unchanged.
- **Probe arms and training data.** P-M, P-C, R (random initialisation), F (logistic regression on
  the node inputs, no encoder), and P-M-noemb (P-M without the optional weather and date-time
  embeddings), each with 3 seeds. The probes train on 200 injected worlds and are tested on the 40
  scored worlds. The donor probe asks whether an hour lies inside the first 24 hours of a donor
  step; the recipient probe asks, for each true donor onset and each linked neighbour, whether the
  neighbour is the recipient.
- **Measurement.** The share of donor megawatts found at a false-donor rate equal to E1-J's on the
  scored worlds, and the recipient AUROC, per cell, with intervals that resample whole worlds. Cells
  are experiment 665's six cells; shares are MW-weighted; found and paired follow experiment 670.
- **Standing practices.** Cleaned data and the D15 and D18 cut before 2025-03-01 as in the
  injected panel of experiment 660 (no refit cadence applies; no forecast is scored). One GPU process
  at a time, behind the maintainer's one-year backtests, checkpoint every 1,000 steps. CPU work runs
  under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`. The ceiling is 9
  GPU-hours for experiments 675 and 676 together: step 0 under 1, the probe at most 3.
- **Reviews.** A Sonnet plan review before the code; a fresh Sonnet code review (including both leak
  tests) before the first fit; a Sonnet results review; a Sonnet write-up review; and a Sonnet prose
  review. The review findings are triaged in the pull request.
- **Unlogged-event caveat.** NGED's logs may miss load shifts, so matched logged shifts on the real
  record are lower bounds. The long and deep cell is nearly empty on the real record (experiment
  665 found none of the 77 logged shifts longer than 1 day in it), so success there can be shown
  only on injected worlds.
- **Inherited leaks.** The noise units (experiment 653) saw the validation months, and the
  injected start hours come from NGED's logged onsets up to 2026-07, as in experiments 670 and 672.
