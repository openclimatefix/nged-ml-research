---
id: "00676"
slug: pretrained-graph-model-for-load-transfers-part-2-edge-state-network
title: >-
  Does a neural network with an explicit state for every link between two Primaries, and a recipient
  lag, pair more injected load-transfer megawatts than the donor-probe hybrids of experiment 678?
theme: lsc-detectors
also_relevant_to:
  - differentiable-physics-network
status: planned
date: 2026-11-10
kind: other
size: >-
  complex. (1) Stored: private injected-world scores, network weights, and figures under
  data/ml-experiments/00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network/
  only. (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one
  is defensible for the edge-state network, so the construction is fixed below before any score is
  read. (5) Callers: this folder's scripts, which import the folders of experiments 653, 660, 665,
  670, 672, 675, and 678; nothing imports them. The network trains on the GPU (ceiling 2 GPU-hours
  for the first look, 5 GPU-hours in total); the post hoc variant of experiment 678 runs on at most
  2 CPU cores.
hypothesis: >-
  Experiments 672 and 678 closed the graph line at 15 Primaries under their stop rules, and the
  maintainer asked to continue it. Experiment 678 bolted a donor posterior onto E1-J's recipient
  search; the explicit per-link state with a recipient lag, decoded jointly with the Primaries'
  states, is the one core of the design no experiment has tested. A network that learns the state of
  every link (off, first Primary donates, second Primary donates) hour by hour, and the lag at which
  the recipient follows, pairs more injected transfer megawatts in the long and deep and the medium
  and deep cells than the hybrids of experiment 678 do. A post hoc variant of experiment 678, whose
  added donors may end in under 2 days, pairs more short-event megawatts than experiment 678 did,
  at a false-transfer rate that stays within 1.5 times E1-J's.
prediction: >-
  Written before any score is read. "Paired" is the MW-weighted share of injected edge-transfer
  ends paired correctly, as in experiment 670. Cells are those of experiment 665. Network
  predictions (Part B) are for the first look, one seed per arm; the three-seed predictions apply
  only if the continuation gate is met. G is the edge-state network on the real graph. G1: G pairs at
  least the mean paired share of R-hyb (experiment 678, three seeds) plus 0.05 in the long and deep
  cell (R-hyb 0.345, bar 0.395) and in the medium and deep cell (R-hyb 0.297, bar 0.347). G2: G
  pairs at least E1-J plus 0.10 in both cells (E1-J 0.286 and 0.245, bars 0.386 and 0.345). G3: the
  rewired-graph placebo G-R pairs at most a third of what G pairs in the two deep cells. G4: the
  false transfers of G (decoded transfer events that are not a correct pair, per Primary-year) are
  at most 1.5 times E1-J's (0.379, bar 0.57). G5: G's median recipient lag error on correctly paired
  recipients is at most 4 half-hours. Post hoc variant (Part A): H1: F-hyb and R-hyb with donors
  that may end after 6 hours or more pair at least 0.03 more than E1-J in the medium and shallow
  cell and at least 0.02 more in each short cell (E1-J 0.131, 0.080, 0.027), with false transfers at
  most 1.5 times E1-J's.
decision_rule: >-
  Fixed before any score is read. Worlds: experiment 678's 40 scored worlds (experiment 660's
  default generator, seeds 900 to 939, scored years 2024-01-01 to 2026-06-30). Residual panel:
  experiment 327's arm C, as in experiments 670, 675, and 678. The final-test window from
  2026-07-01 is never read. Training worlds: the 200 worlds of experiment 675 (seeds from 100000,
  events starting before 2023-07-01, every later reading hidden). Validation worlds, for the decoder
  grid only: 20 worlds from seed 105000, readings kept to 2024-01-01, events scored only when they
  start from 2023-07-01. Reference rows: E1-J and the hybrids of experiment 678, read from its stored
  rows (same worlds, same panel); E1-J and R-hyb are not rerun. Branch rule for Part B: if G meets G1
  in both deep cells, train two more seeds of G (total ceiling 5 GPU-hours) and report the three-seed
  means; otherwise stop and close the learnt-graph line at 15 Primaries on the page, naming the
  branch taken and the numbers. Verdict. `supported` if, after the continuation, the three-seed mean
  of G meets G1 and G2 in both deep cells with no seed more than 0.05 below its bar, and G3 and G4
  hold; `partly supported` if the continuation ran and G1 holds in both deep cells but G2, G3, or G4
  fails, or if G1 holds in one deep cell only; `not supported` otherwise. G5, Part A, and the short
  and shallow cells are reported as held or failed and do not enter the verdict. A deep cell with
  fewer than 20 events or 10 edge-transfer ends is unreadable. Adoption: if G's mean paired share is
  0.60 or more in both deep cells the page recommends adopting G as the event table. The porting
  threshold is zero, because the result is a research table that changes no production code.
benchmark: >-
  E1-J and the hybrids F-hyb and R-hyb of experiment 678 on the same 40 worlds, with
  world-resampled 95% intervals
row_set_id: >-
  experiment 660's injected worlds (default configuration, scored seeds 900 to 939, scored years
  2024-01-01 to 2026-06-30) with experiment 653's real graph of 15 Primaries; not comparable with
  the leaderboard
metric: >-
  MW-weighted paired share, found share, and donor-found share per cell, with 95% intervals from
  resampling whole worlds; decoded transfer events that are not a correct pair, per Primary-year (an
  upper bound on false transfers); recipient lag error in half-hours; start and end timing; events
  per Primary-year on the real record. No forecast is scored, so the four load-shift scenarios, tail
  calibration, and the curtailment-cost metric do not apply.
code: ml-experiments/00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network/
---

**This is the plan, committed before any score is read.** The graph line was closed by earlier stop
rules ([experiment
672](../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md) at 15 Primaries, and
[experiment 678](../00678-donor-posterior-hybrid-for-load-transfer-pairing/result.md), whose stop
rule closed it again). It is continued here at the maintainer's request, because only 1.8 of the 9
GPU-hours budgeted for the line were used and the explicit link state of the original design has
never been built. Experiment 676 is a time-boxed final try, and the page says so again in its
results.

## The question

Experiment 678 added the donor alarms of a probe to E1-J's recipient search and gained +0.08 in the
two deep cells (the stop rule asked for +0.10). The hybrids only improve where E1-J looks for a
donor. The remaining untested part of the design is a network that carries an explicit state for
every link between two connected Primaries, with a lag of 0 to 36 half-hours before the recipient
follows, so that the donor and the recipient are found together instead of one after the other. This
experiment builds that network once, trains it on injected worlds from a random initialisation, and
reads whether it pairs more transfer megawatts than the hybrids do.

## Part A: a post hoc variant of experiment 678 (CPU, at most 2 cores)

- **Question.** Experiment 678's added donors could not end sooner than 2 days after their start, so
  the short cells equalled E1-J exactly. Do donors that may end after 6 hours or more help the short
  cells?
- **Construction.** Everything of experiment 678's donor construction is kept except the earliest end
  (6 hours after the start instead of 2 days) and the running-median window of the end search (6
  hours instead of 1 day). The probe scores, thresholds, duplicate rule, and recipient search are
  experiment 678's, so the arms are F-hyb and R-hyb (three seeds), each with the changed end rule.
- **Reporting.** Post hoc; every cell, with the false-transfer rate beside each result, and the
  deep-cell numbers beside experiment 678's to show what the changed end rule does there.

## Part B: the edge-state network (GPU)

A network that reads the node inputs of experiment 675 and outputs, per link and hour, the state of
the link and the recipient lag. The construction below is fixed before any run, and it departs from
the Opus specification of the line in the ways listed under "Departures from the specification".

- **Architecture, node side.** The bidirectional encoder of experiment 675 (an hourly stream of 6
  dilated convolutions with 64 channels, a daily stream of 4 dilated convolutions with 32 channels,
  one round of neighbour averaging), from a random initialisation, on the 35 node-input channels of
  experiment 675.
- **Architecture, link side.** For each of the 25 links of the real graph and each hour, the
  features are the two Primaries' encoder outputs, the cross-correlation of the Primaries' 6-hour
  forward steps at 19 recipient lags (0 to 18 hours at 1-hour spacing), the log ratio of the two
  steps' amplitudes, and the two Primaries' megawatt-or-megavolt-ampere flags. The link head is
  antisymmetric by construction: it scores "first donates" on the features in one orientation and
  "second donates" on the same features swapped, and scores "off" on the symmetrised features. A
  second head scores the 19 recipient lags given the donating orientation.
- **Node side head.** Each Primary-hour also gets a three-class state (no step, a downward step, an
  upward step). Its evidence is added to the link decode, so the two are decoded together.
- **Decoder.** A three-state chain per link with a fixed stay probability, run forward and backward
  over the hours of a world. The onset cost and the link cut-off are chosen on the validation worlds
  from a grid frozen in code (at least 4 log-spaced values each), by the highest paired share in the
  two deep cells together, subject to 3 to 9 decoded edge events per Primary-year on the real record
  in 2023-07 to 2023-12. A greedy pass accepts the link segment with the highest posterior, removes
  its hours from the donor Primary, and repeats. Each accepted segment becomes a donor event, with
  the size measured on the residual by `step_size`, and recipient events at the decoded lag, as the
  recipient search of experiment 670 emits them. There is no post hoc pairing window.
- **Training worlds and labels.** 200 injected worlds (seeds from 100000, experiment 660's default
  generator, unchanged). Link labels come from the generator's truth table: the link is "donates" in
  the donor's hours, the lag is the recipient's start minus the donor's, and all other time is "off"
  at full weight, except in the hours where E1 decodes an event on the un-injected real record
  (weight 0). That mask uses no label. The weight of the transfer states is 5.
- **Loss and optimiser.** Cross-entropy on the link state (weight 1.0), on the lag (0.3), and on the
  node state (0.5). AdamW with a learning rate of 1e-3 and cosine decay, windows of 2,880 hours, a
  batch of 4, half the windows centred on an injected edge transfer. The number of steps follows
  from timing the first 500 steps against the 2 GPU-hour ceiling; a checkpoint is saved every 1,000
  steps and a rerun resumes.
- **Arms.** G (the network on the real graph), G-R (the same network trained and scored on the
  degree-preserving rewired graph that keeps the fewest real links, from 300 draws, with the share
  kept reported), and G-0 (the link features zeroed, so the link head sees only each Primary's own
  encoder output and cannot tell the true neighbour from any other). References, read from
  experiment 678's stored rows: E1-J, F-hyb, and R-hyb.
- **Departures from the specification.** The following are not built, to keep the first look
  inside 2 GPU-hours: masked pretraining (experiment 675 found it did not help), the daily-scale
  pretraining masks, 1,000 or more training worlds (200 are used), the lag-free modular network as
  the residual panel (experiment 327's arm C is used, as in experiments 670 to 678), the amplitude
  and recipient-share head, the conservation and wobble terms, the donor head (the node state
  replaces it), the 16-wide weather embedding, the cross-injector arms, and the semi-Markov duration
  prior (a fixed stay probability replaces it). Each departure makes the network weaker than the
  specification, so a failure here is a failure of this smaller network.
- **Leak tests.** Both run in a fresh Sonnet code review, before any GPU fit, on CPU with a fixed
  seed. First, replacing the residual inside a hidden window (Gaussian noise of 10 noise units, then
  a step of +20 noise units) at one Primary and at its neighbours leaves every node input and every
  link feature bit-identical in float64, including the cross-correlations; a positive control that
  reads the unmasked panel must fail the same test. Second, replacing a scored world's truth table
  with a shuffled copy leaves the network's outputs bit-identical, because labels enter only the
  loss.

## Reviews and running

- **Order.** Sonnet writes the plan and the code. A fresh Sonnet agent reviews the plan, then
  another reviews the code (including both leak tests passing) before the first fit. A fresh Sonnet
  agent reviews the results and the write-up, and a second reviews the prose. At most one Opus review
  is added, and only if a verdict could change.
- **Running.** CPU work runs under `systemd-run --user --scope -p MemoryMax=16G -p
  OOMPolicy=continue`, with `OMP_NUM_THREADS=1`, at most 2 cores. One GPU process runs at a time,
  after `nvidia-smi` and `uptime` show that the maintainer's backtests have the GPU free; a fit
  pauses between networks if they need it.
- **Inherited caveats.** NGED's event labels are an incomplete dump, so matched logged shifts on the
  real record are lower bounds. The injected start hours come from NGED's logged onsets up to
  2026-07, as in experiments 670, 672, 675, and 678.
- **Summary page.** Whatever the outcome, the PR also adds a plain-words summary page of the whole
  event-table line (experiments 653, 665, 670, 672, 674, 675, 678, and this one), written after the
  results.
