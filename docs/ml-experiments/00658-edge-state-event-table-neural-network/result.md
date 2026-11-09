---
id: "00658"
slug: edge-state-event-table-neural-network
title: >-
  Does a neural network that reads all 15 Primaries' demand residuals at once find and pair
  load-shift events better than the penalised segmentation of experiment 653?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
date: 2026-11-09
kind: other
size: >-
  complex. (1) Stored: private injected-world banks, fitted network weights, event tables, and score
  tables under data/ml-experiments/00658-edge-state-event-table-neural-network/ only. (2) Serving
  path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for
  the network, the decoder, and the edge head, so each is fixed below before any score is read. (5)
  Callers: this folder's scripts, which import the folder of experiment 653 (residual panel,
  injector, decoder, scorer) and nothing imports them. GPU, shared today with experiments 654 and
  655; ceiling 6 GPU-hours.
hypothesis: >-
  A bidirectional recurrent neural network that reads every Primary's residual and the electricity
  network's neighbour links together, and is trained on synthetic events injected into the real
  residual, finds more injected load-shift megawatts than the penalised segmentation of experiment
  653 at the same rate of false events, and pairs the two ends of an injected transfer far more often
  than the 0.18 that experiment 653 reached, because the network learns from many thousand injected
  transfers what a donor and a recipient look like together.
prediction: >-
  Written before any score is read. P1: on held-out injected events of at least 4 noise units that
  last at least 1 day (experiment 653's scored years and worlds), the MW-weighted share found is at
  least 0.08 above experiment 653's 0.64 (so 0.72 or more), with no more real-record events per
  Primary-year than experiment 653's 6.57. P2: the MW-weighted share of injected edge transfers paired
  correctly is 0.50 or more (experiment 653: 0.18). P3: a network trained on one injector and tested
  on the other (constant megawatts against shares of the donor's own power, square steps against wide
  edges and loose shares) finds at most 0.05 less than a network trained on the test injector. P4: a
  classifier that tells injected events from the verified real events has an area under the curve
  below 0.65. P5: on the real record in the scored years, the table matches at least 11 of the 77
  logged shifts of more than 1 day under the strict rule (experiment 653: 8) at an alarm rate no
  higher than experiment 653's 6.6 events per Primary-year. P6: the end is within 6 hours of the true
  end for 65% of injected events (experiment 653: 0.48). P7: inference on a degree-preserving rewired
  graph classes a third or fewer of the real edge events as edge events (experiment 653: 2 against 4).
decision_rule: >-
  Fixed before any score is read. Harness: experiment 653's residual panel (15 Primaries, half-hourly,
  2022-01-01 to 2026-06-30, lag-free demand model of experiment 130, common mode removed), its
  injector with its seeds, its truth table, and its scorer; the scored years are 2024-01-01 to
  2026-06-30 and the final-test window from 2026-07-01 is never read. The network trains on injected
  worlds built from 2022-01-01 to 2023-06-30 only, the decoder's costs and the stopping epoch are
  chosen on injected worlds built from 2023-07-01 to 2023-12-31 only (largest recall with at most 6
  decoded events per Primary-year on the real residual of those months), and nothing changes after the
  first scored number is read. Three network seeds; every interval resamples whole worlds and the
  seeds are reported side by side. The comparison with experiment 653 is read at equal real-record
  event rate, and a rewired-graph placebo and equal-width scaling are not applicable (the table has no
  bands) but the rewired-graph run is printed beside every pairing result. Support: P1 and P2 hold, and
  P3 and P4 hold. Not supported and the neural line closes: recall does not beat experiment 653's by
  0.08, or P3 fails, or P4 fails. If the failure is a bug found by the code review of the fitted
  results, it is fixed and the experiment is run once more before closing. The porting threshold is
  zero, because the table is a research output and changes no production code.
benchmark: >-
  Experiment 653 (default arm: masked frozen seasonal slow level with penalised segmentation), re-run
  on the same worlds in this experiment's script; RO-NC and LR_b20 as in experiment 254 on the real
  record; the network with experiment 653's pairing rule (to separate detection from pairing).
row_set_id: >-
  experiment 258's grid of 15 Primaries, half-hourly, 2022-01-01 to 2026-06-30, cleaned power; same
  rows as experiment 653; not comparable with the leaderboard
metric: >-
  MW-weighted share of injected events found, by duration band; MW-weighted share of injected edge
  transfers paired correctly; start and end error in hours; area under the curve of the
  injected-against-real classifier; events per Primary-year on the real record; logged shifts matched
  (strict and loose); edge events on the rewired graph; normalised MAE of the median forecast in the
  four load-shift scenarios.
code: ml-experiments/00658-edge-state-event-table-neural-network/
---
**Planned. Written before any code runs.** Experiment 653 built a table of load-shift events with a
penalised segmentation. It found about two thirds of injected megawatts but paired only 18% of
injected transfers, and a shuffled neighbour graph gave half as many edge events as the real graph.
This experiment asks whether a neural network trained on many injected events does better on the same
residuals, the same injected worlds, and the same scorer.

## Deviations from the brief, stated before any score is read

- **Data.** The brief asks for the 2019-onward history at 16 Primaries with the expected-power model
  cross-fitted by year. That residual does not exist and building it needs about 1 GPU-hour of
  fitting, so this experiment reads experiment 653's residual panel (2022 onward, 15 Primaries) as
  the brief's own like-for-like instruction requires. The residual's demand model trained on all
  rows, so earlier real events leak into its baseline, exactly as in experiment 653.
- **Persistence prior.** The answer file trains through a semi-Markov chain. This experiment trains
  with per-block cross-entropy and applies a fixed-cost Viterbi decode afterwards (switching costs
  chosen on validation worlds), which is the same persistence structure without backpropagation
  through the chain.
- **Not built:** the causal variant and its distillation loss, the conservation loss, the boundary
  loss, the reconstruction loss, the 2019 to 2021 history, and recall-by-size on real events.

## Method (planned)

**Architecture.**

- **Input.** Per Primary and 6-hour block: the residual with experiment 653's masked frozen slow
  level removed, in noise units (clipped at 30); a present flag; the 1-day and 3-day step statistics
  in noise units; the share of blocks in a centred 60-day window whose 1-day step exceeds 4 noise
  units; and the block's position in the day (sine and cosine).
- **Temporal encoder.** A two-layer bidirectional gated recurrent unit with 48 hidden units per
  direction, shared by all Primaries, followed by a graph layer: each Primary's state is joined with
  the mean state of its neighbours and passed through a second bidirectional recurrent layer.
- **Primary head.** Per block: four class logits (none, shift, ramp, routine), a sign logit, and a
  signed amplitude in noise units.
- **Edge head.** For each link between two Primaries and each block: three logits (off, i to j, j to
  i), made antisymmetric in direction by construction, from the two Primaries' states.
- **Decoder.** Per Primary, a four-state Viterbi path (none, shift up, shift down, ramp or routine)
  with fixed switching costs; start and end refined to the half-hour by experiment 653's refinement;
  event amplitude taken from the observed residual as in experiment 653. Pairing: opposite-sign
  events at neighbouring Primaries whose spans overlap and whose mean edge posterior is 0.5 or more.

**Training regime.**

- **Data.** Injected worlds built by experiment 653's injector on 2022-01-01 to 2023-06-30 (a bank of
  200 worlds, mixed shapes), real residual with no injection in 20% of windows. Crops of 240 days.
- **Targets.** Class and edge state per block from the injected truth. Time with no injected event
  has a target of none with weight 0.1, never 1.
- **Loss.** Weighted cross-entropy for class, sign, and edge state, Huber loss on amplitude, and a
  small penalty on the number of state changes per day. Adam, learning rate 0.003, early stopping on
  the validation worlds' recall at the validation false rate. Three seeds, GPU, one process at a time.
- **Cross-injector networks:** one trained on constant-megawatt events with square edges and default
  durations, one trained on events proportional to the donor's power with wide edges, loose shares,
  and staggered starts; each tested on both injectors (seed 0 only).

**Measurement.**

- **Injected scoring:** experiment 653's worlds (same seeds), match rule, and recall definitions;
  experiment 653's default arm re-run beside it to confirm it reproduces the stored table.
- **Real record:** events per Primary-year by class, the D8 rate, logged shifts matched strictly and
  loosely against LR_b20, the rewired-graph placebo (100 draws), the verified D5 and D16 pair, and
  the injected-against-real classifier on the verified events and the experiment 653 edge events.
- **Four scenarios:** normalised MAE of the median forecast (the lag-free demand model, as in
  experiment 653) before and after the table, for all rows, rows outside events, rows inside events,
  and the 7 days after an event ends. Event time series: D5 and D16 from 14 days before to 7 days
  after, with the observed power and the K1 baseline, plus the event-free rows as their own stratum.

## Which practices and earlier advice this plan follows

- **Follows:** experiment 653's advice to fit the slow level with the candidate windows masked and
  then freeze it; to report the share of megawatts found and the share paired correctly separately;
  and to check pairing on injected transfers before reading anything into real pairs. Unlogged-event
  caveat: NGED's logs miss events, so every real-event count is a lower bound.
- **Departs from:** the standing practices on monthly refits, tails, and the curtailment cost, which
  do not apply because the table produces no forecast.
- **Compute:** about 1 to 2 GPU-hours (three seeds, two cross-injector networks, one possible re-run);
  one seed is timed first; ceiling 6; at most 4 CPU cores.
