---
id: "00734"
slug: switching-offset-model-part-1
title: >-
  Does an unsupervised Bayesian model of the switched power find more of the long, shallow injected
  load transfers than the event table and a plain changepoint search?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: finished
date: 2026-11-27
kind: other
size: >-
  Medium. (1) Stored: private posterior summaries, event tables, tuning tables, and score tables
  under data/ml-experiments/00734-switching-offset-model-part-1/ only; nothing is fitted by
  gradient descent and no neural network is trained. (2) Designs: more than one design defensibly
  tests the hypothesis (the state grid, the transition priors, the shared term, and the neighbour
  evidence), so every choice is fixed below before any world is decoded, and two ablation arms
  and one reference row are read without being counted. (3) Callers: this folder's scripts, which
  import the folders of experiments 653, 660, 665, 670, and 680 read-only; nothing imports them.
  (4) Serving path and degradation rule: not touched. (5) Contracts: not touched. CPU only, 0
  GPU-process-hours, about 32 CPU-hours or more, with a stop at 40 CPU-hours that may be reached.
  Reviews: plan, code (and a second code review if the first finds many problems), numbers, and
  prose.
hypothesis: >-
  A hidden Markov model of each Primary's demand residual, whose hidden state is the total switched
  offset on a grid of 41 levels, which prefers to return to zero, which treats two or three steps
  in a row in one direction as nearly as probable as one step, which carries separate slow states
  for a permanent level and a demand-shaped gain, and which reads a shared term across Primaries
  and opposite-sign steps at neighbouring Primaries, finds more of the megawatts of long, shallow
  injected load transfers than the event table E1-J-sym of experiment 680, at no more called events
  per Primary-year on the real record. The gain comes from averaging over the whole of a long event
  (forward-backward inference over the whole record) instead of a fixed penalty per block.
prediction: >-
  Written before any world is decoded. All shares are MW-weighted shares of injected events found,
  on the 40 scored default worlds of generator 660 (seeds 900 to 939), in the long and shallow cell
  of experiment 665 (more than 7 days or no end; 2 to under 8 noise units), where E1-J-sym finds
  0.393 (stored rows of experiment 680). P1 (counted, C1): H-nbr minus E1-J-sym is +0.10 or more
  (H-nbr about 0.50). P2 (counted, C2): H-nbr minus PELT is +0.15 or more (PELT about 0.33). P3:
  H-own minus E1-J-sym is +0.03 to +0.08. P4: H-nbr minus H-own is +0.02 or more (the generator
  draws recipients from the same neighbour graph that H-nbr reads, and its size law is the H arms'
  jump prior, so both flatter the H arms). P5: in the long and deep cell H-nbr minus E1-J-sym is
  +0.05 or more (E1-J-sym 0.515). P6: in both short cells H-nbr is within 0.03 of E1-J-sym. P7:
  the 80% interval of the predicted switched megawatts covers the injected megawatts for 0.70 to
  0.90 of found injected edge-transfer and solo events in each long and medium cell. P8: on the 20
  cascade worlds H-nbr finds at least as much of the cascaded events' megawatts as E1-J-sym, and
  H-nbr-noramp, which gives a second step in the same direction the probability of a fresh step,
  finds 0.05 or
  more less of the cascaded events' megawatts than H-nbr. P9: on the 20 cold-spell worlds H-nbr
  calls at most half as many events inside cold spells as H-nbr without the shared term. P10:
  H-nbr's called rates (all calls, and calls of 2 noise units or more) in the 27.6 Primary-years of
  event-free time are each within 10% of E1-J-sym's. Maintainer-side estimate that P1 holds: about
  30%.
decision_rule: >-
  Fixed before any world is decoded. Two counted contrasts, C1 (H-nbr minus E1-J-sym) and C2 (H-nbr
  minus PELT), both in the long and shallow cell of the 40 scored default worlds, so each interval
  is 97.5% (Bonferroni over two). Each interval is computed twice, once resampling whole worlds and
  once resampling the 30 calendar months in which the injected events start, 10,000 draws each,
  and a contrast counts only when both intervals exclude zero. Vetoes, each applied to two called
  rates per Primary-year (all called events, and called events of 2 noise units or more, where
  the threshold is in megawatts from experiment 653's noise unit for every method): V1, each of
  H-nbr's two rates on the real record from 2024-01-01 to 2026-06-30 (36.0 Primary-years) is no
  more than 1.10 times E1-J-sym's; V2, the same on the 27.6 Primary-years of event-free time.
  Verdict: `supported` if C1's point estimate is +0.10 or more, C1 counts, and V1 and V2 pass;
  `small gain` if C1 counts with a point estimate below +0.10 and V1 and V2 pass; `partly
  supported` if C1 counts and V1 or V2 fails; `refuted` if both C1 intervals lie wholly below
  zero; `not supported` otherwise. C2 is reported as its own counted claim ("H-nbr beats a plain
  changepoint search") and does not change the verdict word. Every prior, penalty, threshold, and
  on/off switch is chosen
  once, on the 30 tuning worlds (seeds 734000 to 734029, power from 2024-01-01 hidden) and on the
  real record before 2024-01-01, before any scored world is decoded, and none changes after. Gates
  (G1 to G5 under "Gates") must pass first, or the verdict is `invalid`. The porting threshold of
  C1 is +0.10 rather than zero, because H-nbr replaces a working event table with a larger hidden
  Markov model whose priors need tuning, and a smaller gain in one cell does not justify the
  switch. The
  final-test window from 2026-07-01 is never read.
benchmark: >-
  E1-J-sym, the event table of experiment 680 (experiment 653's penalised segmentation with the
  recipient search of experiment 670 and the mirror search of experiment 680), read from experiment
  680's stored rows on the 40 default worlds and re-run with experiment 680's code on the new
  worlds and on the real record; PELT, a plain per-Primary penalised changepoint search written in
  this folder; E1 (experiment 653's original table) as a reference row from experiment 670's
  stored rows.
row_set_id: >-
  experiment 660's injected worlds (seeds 900 to 939, default configuration, events starting
  2024-01-01 to 2026-06-30), plus 20 cascade worlds, 20 cold-spell worlds, and the stored
  sensitivity worlds of experiment 680, and experiment 653's real record of 15 Primaries; not
  comparable with the leaderboard
metric: >-
  MW-weighted share of injected events found, per cell of experiment 665's six-cell grid, with
  world-resampled and month-resampled 97.5% intervals; called events per Primary-year, all and of
  2 noise units or more, on the real record and in event-free time; pinball loss and 80% interval
  coverage of the predicted switched megawatts on injected events; start and end timing error; calls
  on
  hard negatives, cold spells, and solar growth; logged shifts matched under experiment 254's
  strict and loose rules.
verdict: not supported
best_arm: >-
  E1-J-sym in the long and shallow cell (0.393 of the injected megawatts found, against PELT 0.384,
  H-nbr 0.355, and H-own 0.352). H-nbr is best in the medium cells (0.449 against 0.352 for
  E1-J-sym in the medium and shallow cell). On the real record H-nbr calls 0.34 times as many
  events as E1-J-sym, but 1.09 times as many events of 2 noise units or more. All numbers are
  interim.
headline: >-
  The probabilistic offset model with neighbour evidence (H-nbr) found a smaller share of the
  megawatts of long, shallow injected load transfers than the event table E1-J-sym (0.355 against
  0.393; C1 -0.039, 97.5% world interval -0.090 to +0.014), so the verdict is `not supported`. Of
  the 10 predictions, only P6 held. H-nbr did find more in the medium cells, and in event-free time
  it called about a quarter as many events as E1-J-sym when all called events are counted, and 0.97
  times as many when only events of 2 noise units or more are counted. Neither probabilistic model
  beat the plain changepoint search (C2 -0.029, -0.078 to +0.016).
code: ml-experiments/00734-switching-offset-model-part-1/
---

**We ask whether a probabilistic model that reads a Primary substation's whole history can find
more of the long, shallow load transfers than the best of the event tables that experiments 653 to
682 built.** A Primary substation steps electricity down to a local distribution network, and the
project forecasts the demand at each one. Operators sometimes switch part of a circuit from one
Primary to a neighbour for days or months, which moves demand between the two. This page calls
that switching event a load transfer. The [switching events](../background/switching-events.md) page
says what a transfer looks like in the data and why the project wants to find transfers.

**The probabilistic model needs no labels and no training.** For each Primary the probabilistic
model takes the residual, which is the observed power minus a demand forecast that uses no lagged
power, and asks, block by block, how much of the residual is switched power. The answer is a
probability distribution over the switched megawatts in every 6-hour block, plus a table of events
with a grade that says how sure the probabilistic model is that each event is a switch. To test
the probabilistic model, we add synthetic transfers of known
size, start, and end to the real history of 15 Primaries, and measure how many of the added
megawatts each method finds. This page holds the plan, which was written before any world was
decoded, and the result.

**The headroom is larger than the earlier pages suggest.** The figure of 0.67 of long, deep
megawatts found, quoted on the event-table pages, comes from the six worlds of experiment 653's own
injector. On the 40 default worlds of the shared generator of experiment 660, which this experiment
uses, the event table E1-J-sym finds 0.515 of the long and deep megawatts and 0.393 of the long and
shallow megawatts (stored rows of experiment 680, recomputed for this plan).

## Result

*Every number below comes from the first and only scoring of the scored worlds, and is interim until
the reviews are complete. The numbers come from the console output of the report stage, stored as
`report_output.txt`, and from the tables it reads: `scored/<set>/<arm>_<seed>.parquet` and
`scored/<set>/<arm>_<seed>_calls.parquet` (each world), `real/summary.json` and
`real/events_<arm>.parquet` (the real record), `params.json`, `edge_check.json`, and `g3b.json`, all
under `data/ml-experiments/00734-switching-offset-model-part-1/`. No figure on this page is a chart
of real substation demand. The event figures made from the real record and from the synthetic worlds
show demand of D8-region Primaries and stay in the private folder.*

### The result in plain terms

**The offset model did not find more of the long, shallow transfers than the event table.** The
question was whether a hidden Markov model that reads a Primary's whole record (H-nbr: the model
with the shared term and the neighbour mirror evidence) finds more of the megawatts of injected
transfers that last more than 7 days (or never end) and are 2 to under 8 noise units deep. The score
is the MW-weighted share found: the injected megawatts of the transfers a method found, over all the
injected megawatts in the cell (higher is better; 0.355 means that 35.5% of the injected megawatts
were found). On the 40 scored default worlds H-nbr found 0.355, E1-J-sym (the event table of
experiment 680, a detector that segments each Primary's record with a fixed penalty per block) found
0.393, and PELT (a plain per-Primary penalised changepoint search) found 0.384.

**C1, the counted contrast, is -0.039, and its interval includes zero.** C1 is H-nbr minus E1-J-sym
in the long and shallow cell. The 97.5% interval resampling whole worlds is -0.090 to +0.014, and
the interval resampling the calendar months of the events is -0.094 to +0.019. The bar was +0.10, so
the verdict is `not supported`. Both vetoes passed, so they do not change the verdict. H-nbr's
called rates on the real record, as multiples of E1-J-sym's, are 0.34 for all called events and 1.09
for events of 2 noise units or more (V1, limit 1.10). In event-free time they are 0.26 and 0.97 (V2,
limit 1.10).

**C2 also fails: H-nbr minus PELT is -0.029 (world interval -0.078 to +0.016).** The two
probabilistic models, H-own (the model without neighbour evidence) and H-nbr, are no better than a
plain penalised search in the cell H-own and H-nbr were built for. H-nbr minus H-own (P4) is +0.003
(-0.031 to +0.036), so the neighbour evidence added nothing in the long and shallow cell.

**All of this is on synthetic transfers injected into the real history of 15 Primaries.** The real
record has no labels that could confirm the result. The 77 logged shifts in the real record are
matched by H-nbr 13 times under the strict rule and 19 under the loose rule, and by H-own 14 and 22
times.

### Found share by cell, 40 scored default worlds

| Cell | Injected events | H-own | H-nbr | PELT | E1-J-sym |
|---|---|---|---|---|---|
| long, deep | 220 | 0.511 | 0.465 | 0.522 | 0.515 |
| long, shallow (counted) | 840 | 0.352 | 0.355 | 0.384 | 0.393 |
| medium, deep | 127 | 0.583 | 0.571 | 0.468 | 0.485 |
| medium, shallow | 458 | 0.406 | 0.449 | 0.286 | 0.352 |
| short, deep | 163 | 0.098 | 0.110 | 0.162 | 0.119 |
| short, shallow | 561 | 0.019 | 0.023 | 0.021 | 0.047 |

**H-nbr finds more than E1-J-sym in the two medium cells and less in the long cells, and every
method finds little in the short cells.** The gain is +0.096 in the medium and shallow cell and
+0.086 in the medium and deep cell. In the long and deep cell H-nbr is 0.050 below E1-J-sym (P5,
whose bar was +0.05 or more), so H-own and H-nbr do best on events of 1 to 7 days. E1-J-sym finds
those events less well. The short and shallow cell is close to zero for every method (0.019 to
0.047), and the short and deep cell is low (0.098 to 0.162). H-nbr is within 0.03 of E1-J-sym in
both short cells, as the plan predicted (P6 held).

### Predictions

| Id | Claim | Value | Held |
|---|---|---|---|
| P1 | C1, H-nbr minus E1-J-sym, long and shallow: +0.10 or more | -0.039 | no |
| P2 | C2, H-nbr minus PELT: +0.15 or more | -0.029 | no |
| P3 | H-own minus E1-J-sym: +0.03 to +0.08 | -0.042 | no |
| P4 | H-nbr minus H-own: +0.02 or more | +0.003 | no |
| P5 | H-nbr minus E1-J-sym, long and deep: +0.05 or more | -0.050 | no |
| P6 | H-nbr within 0.03 of E1-J-sym in both short cells | -0.009 (deep) and -0.024 (shallow) | yes |
| P7 | 80% interval covers the injected megawatts in 0.70 to 0.90 of the half-hour blocks inside found temporary events, in each long and medium cell | 0.03 to 0.36 | no |
| P8 | cascade worlds: H-nbr at least E1-J-sym, and H-nbr at least 0.05 above H-nbr-noramp | -0.006 and +0.025 | no |
| P9 | cold-spell worlds: H-nbr calls at most half as many events in spells as H-nbr-noshared | 0.82 (441 against 535) | no |
| P10 | H-nbr's called rates each within 10% of E1-J-sym's in event-free time (ratio 0.90 to 1.10) | 0.26 and 0.97 | no |

**P10 failed on the all-calls rate, which was 0.26 times E1-J-sym's, below the band of 0.90 to 1.10,
while the veto V2 only limits the ratio from above.**

**The 80% intervals are far too narrow (P7).** The interval of the predicted switched megawatts
covered the injected megawatts in only 0.03 to 0.36 of the half-hour blocks inside found temporary
events in each long and medium cell, where 0.80 was the target. A user who reads the interval as a
probability statement would be misled. The output of the offset model therefore cannot yet serve as
the distribution over switched power that the use "interpreting lagged power features" on the
[switching events](../background/switching-events.md) page needs.

**Removing the shared term lowers the long and shallow share by 0.028, and removing the ramp rule
changes it by less than 0.001.** Without the shared term (H-nbr-noshared) the share falls from 0.355
to 0.327 and the calls inside cold spells rise from 441 to 535. Without the ramp rule (H-nbr-noramp)
the long and shallow share stays at 0.355, and the share found in the cascade worlds falls from
0.392 (H-nbr) to 0.367.

### Other reads

- **The real record, scored years:** over 36.3 Primary-years H-nbr calls 84 events (2.32 per
  Primary-year, 1.93 for events of 2 noise units or more), H-own 99, PELT 89, and E1-J-sym 6.87 and
  1.76 per Primary-year. Of H-nbr's 84 events, 10 are mirrored and returned, 64 returned without a
  visible partner, and 10 neither.
- **Hard negatives (steps, ramps, pulses, and placebo pairs that are not switching events):** H-nbr
  makes 0.86 calls per hard negative, H-own 1.15, PELT 0.97, and E1-J-sym 1.90.
- **Placebo worlds (the same events moved to random times):** C1 on the placebo worlds is -0.100
  (world interval -0.164 to -0.038), so H-nbr loses to E1-J-sym there too.
- **Un-injected worlds:** H-own calls 3 false events, H-nbr 2, and PELT 73.
- **Solar growth (24 worlds, 72 grown Primaries) and seasonal bias (2 worlds, 10 grown Primaries):**
  H-nbr makes 0.48 false events per grown Primary-year under solar growth and 0.96 under seasonal
  bias; H-own makes 0.60 and 1.04.
- **Tuning:** H-own's best value of the return-to-zero rate (`rho_off`) sat on the edge of its
  extended search grid. The other tuned values were interior or limited by a called rate (see
  Amendment 1).

### What the result means

**In the long and shallow cell, H-nbr finds fewer injected megawatts than E1-J-sym, so the result
gives no reason to replace E1-J-sym there.** H-nbr does find more megawatts of events that last 1 to
7 days, and it calls fewer events in total on the real record (0.34 times E1-J-sym's rate), though
about as many events of 2 noise units or more (1.09 times). The claim the plan set out to test, a
gain of +0.10 in the long and shallow cell, did not hold. The plan's later step, a restricted
version of the offset model (part 2), is for the maintainer to decide; this page makes no
commitment.

## Terms

- **Residual:** observed cleaned power minus the forecast of the lag-free convex demand model of
  [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), as stored by
  [experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md) (arm C, refitted
  monthly, driven by ERA5 and CAMS estimates of past weather). Every output here is therefore a
  hindsight product.
- **Block:** a 6-hour mean of the residual, on four fixed UTC blocks a day (00:00 to 06:00, 06:00 to
  12:00, 12:00 to 18:00, and 18:00 to 24:00). The record from 2022-01-01 to 2026-06-30 has 6,568
  blocks per Primary.
- **Noise unit:** 1.4826 times the median absolute deviation of a Primary's block means about its
  slow level, as [experiment 653](../00653-edge-state-event-table-penalised-segmentation/result.md)
  defines and stores it (0.16 to 1.02 MW across the 15 Primaries, median 0.22 MW). The 6-hour
  noise unit is a different unit from the 2-hour step unit of [experiment
  732](../00732-msa-d8-untangling-network-synthetic-injections/result.md).
- **Cell:** one of the six groups of injected events of [experiment
  665](../00665-event-table-recall-by-length-and-depth/result.md). *Long* is more than 7 days or no
  end, *medium* more than 1 day up to 7 days, *short* 1 day or less; *deep* is 8 noise units or
  more, *shallow* 2 to under 8. The size and the duration are the injected ones.
- **Found:** an injected event is found when a method has an event at the same Primary with the
  same sign, a start within 24 half-hours (12 hours), and an intersection over union of the two
  intervals of at least 0.5 (the `match` function of experiment 653), among the method's called
  events. The **MW-weighted share found** is the injected megawatts of found
  events over all injected megawatts in the cell.
- **World:** one copy of the real power panel with synthetic events added by the generator of
  [experiment 660](../00660-shared-generator-of-injected-load-transfer-events/result.md). Every
  world sits on the same real history, so worlds share all real-record noise and real unlabelled
  events.
- **Called event:** an event in a method's output table at the method's tuned threshold. The two
  **called rates** are the number of called events per Primary-year, and the number of called
  events of 2 noise units or more per Primary-year. The 2-unit threshold is set in megawatts from
  experiment 653's stored noise unit of each Primary, for every method, so that the methods'
  different internal units cannot move it.
- **Event-free time:** the 27.6 Primary-years from 2024-01-01 to 2026-06-30 that lie more than 14
  days from every logged or verified event of [experiment
  580](../00580-verified-load-shift-event-set-and-label-audit/result.md). NGED's logs miss events,
  so the called rate in event-free time is an upper bound on the false-call rate.
- **Primary-year:** one Primary observed for one year. **D3, D5, D16, and the like:** anonymised
  Primary names. **MVA Primary:** one of the 8 of the 15 Primaries whose meter reports apparent
  power, which has no sign.

## The arms

**Four arms are counted or benchmarked, and two ablations and one reference row are read without
being counted.**

| Arm | What it is | Input |
|---|---|---|
| E1-J-sym | [Experiment 680](../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md)'s event table: experiment 653's Viterbi segmentation with the two-way neighbour search | residual minus experiment 653's common mode |
| PELT | a plain penalised changepoint search per Primary ([Killick, Fearnhead, and Eckley, 2012](https://doi.org/10.1080/01621459.2012.737745)), with a Gaussian cost on block means | as E1-J-sym |
| H-own | the offset model, own series only, no shared term and no neighbour evidence | as E1-J-sym |
| H-nbr | the offset model with the fitted shared term and one round of neighbour evidence | residual, with the shared term in place of experiment 653's common mode |
| H-nbr-noramp | H-nbr with a second step in the same direction given the probability of a fresh step (ablation) | as H-nbr |
| H-nbr-noshared | H-nbr with experiment 653's common mode in place of the fitted shared term (ablation) | as E1-J-sym |
| E1 | experiment 653's table without any search (reference row) | as E1-J-sym |

**The arms are ordered so that each step changes one part.** H-own against E1-J-sym changes only
the inference, because the two read the same input. H-nbr against H-own adds the shared term and
the neighbour evidence, and the two ablations split that difference.

**E1-J-sym is the benchmark rather than experiment 653's original table, because E1-J-sym is the
current event table.** Experiments 685 and 692 ran E1-J-sym, and experiment 680's advice is to
compare any new method with E1-J-sym. E1-J-sym's stored rows exist for the 40 scored worlds, and
its real-record table (249 events in 2024 to mid-2026) is stored. E1-J-sym finds slightly more than
E1 in every cell (0.393 against 0.365 in the long and shallow cell), so it is also the harder
benchmark.

**PELT is written by hand in this folder, because `ruptures` is not a dependency in
`pyproject.toml`.** PELT is an exact optimal partitioning with pruning; on 6,568 blocks per Primary
it runs in about a second.

## The offset model

**The offset model writes each Primary's block residual as a slow level, plus a shared term, plus
three hidden components, plus noise.** For Primary *s* and block *t*:

y(s, t) = slow(s, t) + β(s) · σ(s) · c(s, t) + o(t) + ℓ(t) + g(t) · F(s, t) + noise,

where the components are these:

- **slow(s, t)** is experiment 653's seasonal slow level (3 day-of-year harmonics plus a linear
  spline with knots every 365 days, fitted with a Huber loss with candidate event windows given
  zero weight, then frozen). The slow level is refitted in every world, as experiment 653 does.
- **c(s, t)** is the shared term: the median, over the other Primaries with a reading, of each
  Primary's block residual minus its slow level, in its own noise units. Leaving Primary *s* out of
  its own median stops a large event at *s* from entering its own shared term. A transfer moves
  two Primaries in opposite directions, so the median of 14 barely moves with it. The loading β(s)
  is a Huber fit on 2022 and 2023 with candidate event windows masked.
- **o(t)** is the total switched offset, on 41 levels from -20 to +20 noise units. The forecast of
  the normal running arrangement removes o(t).
- **ℓ(t)** is a permanent level, on 21 levels from -20 to +20 noise units in steps of 2. The
  forecast of the normal running arrangement follows ℓ(t).
- **g(t)** is a slow gain on the demand forecast F(s, t), on 5 levels from -0.06 to +0.06 (a share
  of the forecast demand). The forecast of the normal running arrangement follows g(t).
- **The noise** is a Student-t density with 4 degrees of freedom, as in experiment 653, with one
  scale per Primary and per block of the day, fitted on 2022 and 2023.

**The transitions encode the cue that a switch is temporary, and allow a switch to arrive in two or
three steps.**

- **Temporary.** From o = 0, the offset jumps with probability ρ_on per block, to a level drawn
  with weight 1/|k| (log-uniform, the same law as the generator's sizes, which flatters the offset
  model on injected events). From a nonzero level the offset returns to zero with probability
  ρ_off, which is far larger than ρ_on, and moves to another nonzero level j with probability
  ρ_move = ρ_on, split with weight 1/|j − i|. The permanent level ℓ jumps with probability ρ_ℓ =
  ρ_on to another level j, split with the same weight 1/|j − i|, with no preference for returning.
  A step that later returns therefore has a higher prior probability as a switch than as two
  permanent jumps. A step that never returns lowers the probability of the switch path by a factor
  of about 1 − ρ_off for every block spent away from zero, so a step that never returns is more
  probable as a permanent jump.
- **Steps in a row.** A flag records whether the offset moved in the previous block and in which
  direction. A further move in the same direction in the next block has probability ρ_cont, which
  is far larger than ρ_on but well below 1. The rule makes two cases nearly as probable as a single
  step: a cascade of two or three switches within a day, and a step that starts part-way through a
  6-hour block, whose first block holds only part of the step. The rule does not separate a switch
  from a slow ramp: a ramp over several days moves less than one grid level per block, so the
  offset model explains it with the gain, the shared term, or the noise whatever ρ_cont is, and a
  ramp that the grid does resolve is treated by ρ_cont like a cascade. The ramp check under "Gates"
  is therefore a printed diagnostic, not a gate.
- **Slow gain.** The gain moves by at most one level per block, with probability ρ_g = 0.02. A
  sudden step of any shape is therefore more probable as o or ℓ than as g, and g follows slow,
  demand-shaped drift.

**The four blocks of the day separate a flat shift from a demand-shaped change and from embedded
solar.** A switched offset moves all four blocks by the same amount. A gain change moves them in
proportion to the forecast demand, which differs between night and day. A change in embedded solar
moves only the two daylight blocks, which carry a larger noise scale, so no component of the
offset model explains a daylight-only change well and the offset model leaves the change in the
noise. The night block (00:00
to 06:00 UTC) holds a little solar output on summer mornings after about 04:00 UTC.

**Inference is exact.** The joint state of the offset, its move flag, the permanent level, and the
gain has 41 × 3 × 21 × 5 = 12,915 values. The transition matrix factorises over the three
components, so one forward-backward pass ([Rabiner,
1989](https://doi.org/10.1109/5.18626)) over a Primary's 6,568 blocks takes about 7 seconds on one
core (a timing of the forward step alone gave 0.21 milliseconds per block; the backward pass and
the jump probabilities have not been timed). The pass gives the posterior distribution of o(t) in
every block. A Viterbi pass ([Viterbi, 1967](https://doi.org/10.1109/TIT.1967.1054010)), in log
space, gives the single most likely path, from which the event table is read; the Viterbi pass is
timed first in stage 1, because its run time is not yet measured.

**Neighbour evidence is one round of message passing, without feedback.** Pass 1 runs the offset
model on each Primary alone and records the probability of an upward and of a downward jump in o
at every block. Pass 2 runs each Primary again with the probability of a fresh jump in each
direction (a jump from a settled block, not a continuation of a move) multiplied by 1 + κ · m(t),
where m(t) is the largest probability, over the Primary's neighbours and over the
blocks within 3 of t (18 hours either side), of a jump in the opposite direction at the neighbour
in pass 1. The window of 3 blocks covers the generator's recipient start offset of up to 36
half-hours. A Primary's pass 2 never reads its own pass 1 through a neighbour, so no evidence is
counted twice. Each Primary keeps an unseen-partner sink: the unboosted jump probability, so a
transfer whose partner is unmetered keeps its own-series odds. 21 of the 66 hand-filtered real
transfers have no metered receiver. The graph is experiment 383's list of 25 links between the 15
Primaries, first-order neighbours only.

**An MVA reading near zero is reported as unknown.** At an MVA Primary the reading is the magnitude
of the power, so a step near zero can fold. Blocks where the observed MVA block mean is below 3
noise units get no emission term (the offset model treats them as missing), and every output
there carries the flag `unknown`. A block flagged `unknown` gives no neighbour evidence.

**The output has three parts.** Per Primary and block: the posterior quantiles q05, q10, q25, q50,
q75, q90, and q95 of the switched megawatts, read off the 41-level grid without interpolation and
converted to megawatts with the Primary's noise unit, the probability that the switched megawatts
are zero, and the `unknown` flag. Per event: start and end (refined to the half-hour by experiment
653's step
search within 12 hours of the block edge), amplitude, a confidence, and one of four grades:

- **mirrored and returned:** a neighbour has an opposite-sign called event whose start lies within
  3 blocks, and the event ends inside the record;
- **mirrored, not yet returned:** the same, with no end inside the record;
- **returned, no visible partner:** the event ends inside the record, with no mirror;
- **neither:** no end and no mirror, which includes every jump of the permanent level. Here the
  posterior of the switched megawatts is two-peaked (zero, or the full step), and the forecast
  follows the change by default, because one wrongly masked 60-day interval raised the tail error
  by 57% in [experiment 38](../00038-switching-mask-value/result.md).

**The event table is read from the Viterbi path.** An o-event is a maximal run of blocks with a
nonzero offset of one sign on that path, split where the level changes by 2 noise units or more
after at least one settled block, so a run of consecutive moves in one direction (a cascade, or the
partial first block of a step that starts mid-block) stays one event. The o-event's confidence is
the mean posterior probability that o is nonzero over the run. An ℓ-event starts at each jump of ℓ
on the path and ends at the next jump that brings ℓ back within one level of its earlier value, or
stays open; its confidence is the posterior probability of an ℓ jump
within one block of the start. Both kinds enter the found share, because the cells of experiment
665 include permanent shifts. An event is called when its confidence reaches the threshold τ.

## The worlds

**The bank is split into tuning worlds and scoring worlds, and the tuning worlds hide the scored
years.**

| Set | Seeds | Worlds | Configuration | Read for |
|---|---|---|---|---|
| Tuning | 734000 to 734029 | 30 | default, record cut at 2024-01-01 (2,920 blocks decoded) | every prior, κ, τ, and the PELT penalty |
| Scored default | 900 to 939 | 40 | default; E1-J-sym rows stored by experiment 680 | C1, C2, P1 to P7, P10 |
| Cascade | 734100 to 734119 | 20 | default, then 15% of positive events given cascaded edges | P8 |
| Cold spell | 734200 to 734219 | 20 | default, plus 3 synthetic cold spells per winter | P9 |
| Sensitivity | 1000 to 1009 each | 30 | `long_heavy`, `short_heavy`, `constant_mw_square`; E1-J-sym rows stored | robustness, read-only |
| Solar growth | experiment 686's 24 worlds | 24 | experiment 653's injector, growth at 3 of 15 Primaries | false calls from solar growth |
| Seasonal bias | experiment 653's 2 worlds | 2 | experiment 653's injector | false calls from a repeating seasonal bias |

**The scored default worlds stay exactly as experiment 680 scored them**, so E1-J-sym's stored rows
remain the benchmark and gate G1 can check them. The cascade and cold-spell layers go on new worlds,
and E1-J-sym is re-run on those worlds with experiment 680's code.

**The cascade layer tests the rule for steps in a row.** In each cascade world, 15% of the positive
events (edge transfers and solo shifts, chosen at random) have their start replaced by 2 or 3 steps
(equal
chance) within 24 hours, at random times on the half-hour, with the event's change split into
random shares that sum to the whole. In half of the chosen events the return is cascaded the same
way. A donor and its recipients are cascaded at the same times. A cascaded event is one truth event
for scoring, with the first step as its start and the whole change as its size.

**The cold-spell layer tests the shared term.** Each cold-spell world has 3 cold spells in each
November to February, each lasting 4 to 14 days with a raised-cosine shape. At each Primary the
spell adds a share *a* · λ(s) of the forecast demand, with *a* drawn from 0.03 to 0.10 for each
spell and λ(s) drawn from 0.5 to 1.5 for each Primary. Two Primaries per world, drawn at random,
get λ = 0, which mimics an industrial Primary that does not follow the weather. Cold spells are hard
negatives: a call that overlaps a spell by half its length is a false call.

**The intervals resample whole worlds and are therefore optimistic.** The 40 worlds share one real
history, so the world-resampled interval ignores the variation that a second real history would
add. Experiment 653 had only 6 worlds. The month-resampled interval resamples the 30 calendar
months in which events start, pooled over worlds, which keeps the real history's month-to-month
variation. A contrast counts only when both intervals exclude zero.

**The cells hold enough events to read.** The long and shallow cell holds 840 injected events over
the 40 default worlds (11 to 45 per world), and every cell holds at least 127 events. A cell with
fewer than 20 events or 10 edge-transfer ends is too small to read, as in experiments 665 to 682.

## Tuning, once, before any scored world is decoded

**Tuning chooses five numbers for the offset model and one for PELT, and fixes everything else in
advance.**

- **Fixed:** the grids above, ν = 4, ρ_move = ρ_ℓ = ρ_on, ρ_g = 0.02, the 3-block mirror window,
  and the MVA fold rule (3 noise units).
- **Tuned for H-own on the tuning worlds:** ρ_on in {1e-4, 3e-4, 1e-3}, ρ_off in {1/60, 1/240}
  (a mean switched duration of 15 or 60 days; the second extension adds 1/20 and 1/8, about 5 and 2
  days), and ρ_cont in {0.05, 0.2}, 12 settings.
- **Tuned for H-nbr:** κ in {0, 5, 20}, with H-own's ρ values. H-nbr-noramp and H-nbr-noshared use
  H-nbr's values.
- **Objective:** the MW-weighted share found of the tuning worlds' events in the four long and
  medium cells (not the counted cell alone), at equal called rates.
- **Equal called rates:** for each setting, τ is the lowest value in {0.5, 0.6, 0.7, 0.8, 0.9,
  0.95, 0.99} at which both called rates on the real record of 2022 and 2023 (all calls, and calls
  of 2 noise units or more) are no higher than E1-J-sym's rates there (E1-J-sym re-run on those
  years). Holding the rate of all calls stops a method from passing by calling many events just
  under 2 noise units. PELT's penalty, from a grid of 12 values spaced geometrically from 2 to 200
  times the noise variance, follows the same rule.
- **When no threshold meets the rates:** a setting for which no τ in the grid (no penalty, for
  PELT) meets both rates is ineligible. If every setting is ineligible, the setting with the lowest
  rate of all calls at τ = 0.99 is used (the highest penalty, for PELT), the page says so, and
  vetoes V1 and V2 are read as they fall.
- **Shared term on or off, per Primary:** on when the shared term lowers the Primary's noise unit
  on the real record of 2022 and 2023 by 5% or more, and off otherwise. The pre-check under "Stage
  0" prints the share of each Primary's block variance that the shared term explains.

**The tuned values are written to `params.json` with a hash at the end of stage 2, before the
scored worlds are generated, and the stage 3 script refuses to run without that file and the
review marker `review_stage3.ok`, which is written once the review of `params.json` has passed.**

## Amendment 1 to the plan of experiment 734 (2026-11-27)

**Date:** 2026-11-27. Written after tuning finished, after the Opus review of the tuned values,
and after the first extension of the H-own grid. No scored world has been read.

### Reason

**The tuned H-own value of rho_on sat on the lowest value of its grid, and the counted cell was
still rising towards it.** The first tuning chose rho_on 1e-4, rho_off 1/60, rho_cont 0.2. On the
30 tuning worlds the four-cell objective was 0.349, 0.364, and 0.371 at rho_on 1e-3, 3e-4, and
1e-4, and the share of long and shallow events found at tau 0.8 was 0.189, 0.217, and 0.243.

**The first extension did not take the best setting off the corner.** After two lower values of
rho_on (3e-5 and 1e-5) joined the grid, the chosen setting was rho_on 1e-5 (lowest of 5), rho_off
1/60 (highest of 2), rho_cont 0.2 (highest of 2), and tau 0.6, and the edge check reported hits.
A parameter with two grid values cannot show an interior best, so the second extension gives every
H-own parameter four or more values.

**Gate G3b tested only H-own's tau.** H-nbr calls at a lower tau, so G3b now runs at both.

### What changes

1. **First extension (done before this text).** Two new rho_on values, 3e-5 and 1e-5, crossed with
   the same two rho_off values (1/240, 1/60) and two rho_cont values (0.05, 0.2): 8 settings
   (`own12` to `own19`).
2. **Second and last extension: 14 new H-own settings (`own20` to `own33`).**
    - Six settings move one parameter at a time from the best setting of the first extension
      (rho_on 1e-5, rho_off 1/60, rho_cont 0.2): rho_on 3e-6 and 1e-6, rho_off 1/20 and 1/8,
      rho_cont 0.5 and 0.8.
    - Eight settings cross the six new values: rho_on in (3e-6, 1e-6), rho_off in (1/20, 1/8),
      rho_cont in (0.5, 0.8).
    - After both extensions the H-own grid has rho_on in 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
      rho_off in 1/240, 1/60, 1/20, 1/8, and rho_cont in 0.05, 0.2, 0.5, 0.8, so the best setting of
      the first extension is an interior value of all three parameters. The one-at-a-time rows put
      it in the interior of each parameter's profile, and the crossed rows test interactions.
    - The natural limits are excluded: a return rate rho_off of 1 per block (the offset would
      return at once) and a continuation probability rho_cont of 1.0 (a ramp would never end).
3. **Gate G3b runs at both taus.** It decodes the same 2 years of synthetic Student-t noise with
   the priors of H-own at H-own's tau, and with the priors of H-nbr at H-nbr's tau, and each rate
   must be under 0.5 calls per Primary-year. Both are written to `g3b.json`. No scored world is read.
4. **H-nbr is tuned again when the chosen H-own values change.** The outputs of H-nbr made under an
   earlier choice are reused only if the chosen rho_on, rho_off, and rho_cont equal that choice.
5. **An edge check runs at the end of every `choose`, for every arm,** and writes
   `edge_check.json`. A chosen value on the lowest or highest grid value is an edge hit when the
   four-cell objective there exceeds the objective at the next value inward by 0.005 or more. The
   threshold of 0.005 is a code choice made before the second extension; the first tuning's
   plateaus differed by 0.002 and 0.0006. A tau on its grid floor is a hit unless a called rate of
   E1-J-sym binds there, because a lower tau would be ineligible. A PELT penalty or a kappa next
   to an ineligible value is reported as rate-binding.

### Stopping rule

**The second extension is the last.** `tune` refuses a third extension of the H-own grid, and
`extension_state.json` counts the extensions so that a re-run cannot add one. After the second
extension `choose` reports edge hits and never extends a grid. A hit then is reported in this
amendment and in the write-up as "tuned to the edge of the extended grid".

### Equal effort for the other arms

**PELT and H-nbr are re-checked by the same rule.** `choose` reports for PELT's penalty (12 values
from 2 to 200), H-nbr's kappa (0, 5, 20), and H-nbr's tau whether the chosen value is on an edge,
next to an ineligible value (rate-binding), or interior. In the first tuning PELT's penalty was
the ninth of 12 and rate-binding, H-nbr's kappa was interior, and H-nbr's tau of 0.5 was the grid
floor with the call rate binding exactly (42 of 42 calls of 2 noise units or more).
If any of these is a true edge hit with the objective still improving towards the edge, `choose`
prints "EQUAL EFFORT: PELT/H-nbr edge hit, needs a decision" and this amendment states it. Their
grids are never extended silently: `tune --extend-pelt` and `tune --extend-kappa` each add up to
3 values beyond the edge, once, only when passed explicitly.

**H-nbr's kappa and tau get the same equal effort.** `tune --extend-kappa` adds up to 3 kappa
values above 20, and `tune --extend-tau` adds up to 3 tau
values below the floor of 0.5, from 0.4, 0.3, and 0.2. The tau flag first reads the called rates at
each lower tau from the stored events of the real record, which needs no decoding, and refuses if
the chosen tau is not the floor or if a call rate at the floor already binds, because then every
lower tau is ineligible. It then decodes H-nbr once more over the 30 tuning worlds at the new taus
only (about 2.3 CPU-hours for 3 kappa values), and records the values in `extension_state.json`.
A second tau extension with different values is refused.

**No extension of the PELT penalty, kappa, or tau was needed.** The final edge check found PELT's
penalty rate-binding (ninth of 12), H-nbr's kappa on the highest value but on a plateau (objective
0.445 at kappa 20 against 0.4424 at kappa 5, a gain of 0.0026, under the 0.005 threshold), and
H-nbr's tau interior (0.6, second of 7). `tune --extend-pelt`, `--extend-kappa`, and `--extend-tau`
were not run.

### Budget

**The manager extended the CPU-hour stop of the plan from 40 to 55 CPU-hours.** The code enforces
the stop: `require_budget` refuses a task when the spent hours in `time_log.csv`, or the projection
of the timed world, pass 55, and it refuses to run at all without `cpu_budget_extended.ok`. The
first extension cost 6.0 CPU-hours (estimated 4.1), and the second cost 10.5 CPU-hours (estimated
7.1). The base tuning ran about 13% over its projection and the second extension 47% over, so
stages 3 and 4 may exceed their projection of 19.6 CPU-hours. The run stops and asks the manager
at 55.

### History the amendment must disclose

**After the first extension, H-nbr's kappa and tau also looked like edge hits, and neither was
extended then.** H-nbr's kappa 20 was on the highest value of its grid with a gain of 0.018 over
kappa 5, and its tau sat on the grid floor with no called rate binding. Re-tuning H-nbr on the new
H-own values removed both: kappa 20 is now a plateau (gain 0.0026) and tau 0.6 is interior.

**The mean switched durations of rho_off are 60 days at 1/240, 15 days at 1/60, about 5 days at
1/20, and about 2 days at 1/8, all counted in 6-hour blocks.** The optimum of H-own moved towards
short switches as the grid grew, from 1/60 to 1/8. The final tuned H-own therefore prefers
switched offsets that return within about 2 days.

**The search sizes differ between arms for stated reasons.** H-own has 34 settings, PELT has 12
penalties, and H-nbr has 3 kappa values. PELT's penalty is bracketed by ineligible penalties on
the low side and a plateau on the high side, and H-nbr inherits H-own's rho values, so its search
is only over kappa and tau.

**The C Viterbi kernel was written after the stage 1 review.** The gates match before and after
the kernel, the tests compare it with the numpy Viterbi path for path, and all tuning used the C
kernel.

**The read-only filter check was run after the tuning and is post hoc.** It reads only tuning
worlds and the real record before 2024-01-01, and it counts for no contrast.

### What stays

- The 30 tuning worlds (seeds 734000 to 734029) and the real record before 2024-01-01.
- Every other rule of the plan: the objective, the eligibility rule, the vetoes, the predictions,
  and the verdict rule. The grids of kappa, tau, and the PELT penalty stay as they were.
- All first outputs. They are reused, and the files that `choose` overwrites are copied first to
  `*_v1` (before the first extension) and `*_v2` (before the second), never replacing a copy.
- No scored world is read, and the scored data stay untouched.

### Which edges each arm hit

| Arm | Parameter | Grid | Chosen | Position | Edge hit? | Objective at chosen / next inward |
|---|---|---|---|---|---|---|
| H-own | rho_on | 1e-6 to 1e-3, 7 values | 1e-5 | 3 of 7 | no, interior | 0.4543 / not needed |
| H-own | rho_off | 1/240, 1/60, 1/20, 1/8 | 1/8 (0.125) | 4 of 4 | **yes, highest value** | 0.4543 / 0.4405 (at 1/20) |
| H-own | rho_cont | 0.05, 0.2, 0.5, 0.8 | 0.2 | 2 of 4 | no, interior | 0.4543 / not needed |
| H-own | tau | 0.5 to 0.99 | 0.5 | 1 of 7 | no, rate-binding floor | a called rate binds, so a lower tau is ineligible |
| PELT | penalty | 12 values from 2 to 200 | 56.96 | 9 of 12 | no, rate-binding | 0.4043 / next lower value ineligible |
| H-nbr | kappa | 0, 5, 20 | 20 | 3 of 3 | no, plateau | 0.445 / 0.4424 (at 5) |
| H-nbr | tau | 0.5 to 0.99 | 0.6 | 2 of 7 | no, interior | not needed |

**H-own's rho_off is tuned to the edge of the extended grid.** The objective rose by 0.0139 from
rho_off 1/20 to 1/8, and the plan allows no third extension. The write-up states this wherever it
quotes an H-own result.

### New tuned values

| Arm | Value | First tuning | First extension | Second extension (final) |
|---|---|---|---|---|
| H-own | rho_on, rho_off, rho_cont, tau | 1e-4, 1/60, 0.2, 0.8 | 1e-5, 1/60, 0.2, 0.6 | 1e-5, 1/8, 0.2, 0.5 |
| PELT | penalty | 56.96 | 56.96 | 56.96 |
| H-nbr | kappa, tau | 5, 0.5 | 20, 0.6 | 20, 0.6 |
| G3b | calls per Primary-year, H-own and H-nbr | 0.0 (H-own only) | 0.0 (H-own only) | 0.0 and 0.0 |

The ramp diagnostic calls every ramp as one event of 4 noise units, so the ramp rule stays a
diagnostic and is not counted.

No arm was tuned harder than another: every arm's grid was checked by the same edge rule, and a
grid was extended only where the check found the best value on an edge with the objective still
rising.

## Pre-registered predictions

| Id | Claim | Number | Counted |
|---|---|---|---|
| P1 | H-nbr minus E1-J-sym, long and shallow, found share | +0.10 or more | C1 |
| P2 | H-nbr minus PELT, long and shallow, found share | +0.15 or more | C2 |
| P3 | H-own minus E1-J-sym, long and shallow | +0.03 to +0.08 | no |
| P4 | H-nbr minus H-own, long and shallow (flattered: the generator draws recipients from the neighbour graph H-nbr reads, and its size law is the H arms' jump prior) | +0.02 or more | no |
| P5 | H-nbr minus E1-J-sym, long and deep (E1-J-sym 0.515) | +0.05 or more | no |
| P6 | H-nbr minus E1-J-sym, each short cell | within ±0.03 | no |
| P7 | 80% interval of the predicted switched megawatts covers the injected megawatts, found edge-transfer and solo events, each long and medium cell | 0.70 to 0.90 | no |
| P8 | cascade worlds, found share of cascaded events: H-nbr minus E1-J-sym; H-nbr minus H-nbr-noramp (the value of pricing a second step in the same direction above a fresh step) | 0 or more; +0.05 or more | no |
| P9 | cold-spell worlds: calls inside cold spells, H-nbr over H-nbr-noshared | 0.5 or less | no |
| P10 | H-nbr's two called rates in event-free time over E1-J-sym's | each 0.90 to 1.10 | V2 |

## Measured without being counted

- **Every cell for every arm**, with both intervals, and the share paired correctly for E1-J-sym and
  H-nbr (H-nbr pairs by its mirror grade).
- **Calibration of the predicted shift.** For each injected edge-transfer and solo event of 2 noise
  units or more, at every block inside the event, the posterior quantiles of the switched megawatts
  in the injected world minus the posterior median in the uninjected world at the same block,
  scored against the injected megawatts: pinball loss at the seven quantiles and coverage of the
  80% interval (q10 to q90), by cell. Permanent shifts are scored apart on the posterior of o + ℓ,
  because the offset model is meant to put a permanent shift in ℓ.
- **Placebo shift on hard negatives.** On generator 660's routine steps, ramps, pulses, placebo
  pairs, and common-mode swings, the called events per hard negative and the posterior mean shift
  over the hard negative's window, for every arm, and the same on `shuffled_placebo` tables.
- **Conservation residual on mirrored events.** For H-nbr events graded "mirrored" between two
  signed-MW Primaries, the sum of the two posterior mean shifts over the event, as a share of the
  donor's shift.
- **Real transfers, a weak check only.** The 77 logged shifts of more than 1 day that start in 2024
  to mid-2026 and, if the coder can rebuild it on this grid, the set of 35 real transfers over 1
  day, matched under experiment 254's strict rule (start within 4 half-hours) and loose rule (within
  48
  half-hours), beside E1-J-sym's stored counts (9 and 19 of 77). Never a target.
- **The two verified transfers** (D5 to D16 in December 2025, and D3 and D22 in October 2024):
  start, end, amplitude, grade, and the posterior quantiles, drawn from 14 days before to 7 days
  after.
- **False calls from embedded solar growth** on experiment 686's 24 worlds, beside the stored
  figure for experiment 653's masked arm (333 false events), and from a repeating seasonal bias on
  experiment 653's 2 worlds (2.2 per injected Primary-year for experiment 653's masked arm).
- **Calls in the real record's Christmas fortnights** (20 December to 6 January) against the rest
  of event-free time, and calls that start within one block of a monthly refit of the residual's
  demand model.
- **Start and end timing error** (median and 90th percentile in hours; a miss counts as a failure),
  by cell.
- **The robustness worlds:** H-own and H-nbr against E1-J-sym's stored rows in each cell.
- **Calls at D8,** whose multi-day drops are events of unknown cause rather than transfers (the
  [data-processing page](../data-processing.md)), reported apart.
- **Four scenarios,** as experiment 653 reported them: the normalised MAE of the lag-free demand
  model's median, before and after subtracting the posterior mean of the switched offset in
  hindsight, on all rows, quiet rows, rows inside called events, and the 7 days after, with the
  note that the inside-event row falls by construction.

## Gates

- **G1, benchmark reproduction:** E1-J-sym re-run with experiment 680's code on worlds 900 to 903
  reproduces the stored `found` flags exactly.
- **G2, real-record reproduction:** E1-J-sym re-run on the real record reproduces the 249 stored
  events.
- **G3, known answers:** on synthetic Student-t noise, a single 30-day step of 5 noise units is
  found with the posterior median within 0.5 noise units of the truth inside the step and a
  probability of nonzero offset above 0.9; forward-backward agrees with brute-force enumeration on
  a problem of 3 levels and 6 blocks; a cascaded step is called as one event. The check that needs
  tuned values (a series with no step gives a called rate below 0.5 per Primary-year at the tuned
  τ) runs as gate G3b after `params.json` is written and before its review, on synthetic series
  only. At the same point the script prints, as a diagnostic and not a gate, what H-nbr and
  H-nbr-noramp call on synthetic linear ramps of 4 noise units over 3, 12, and 24 blocks.
- **G4, world smoke check:** every cascade and cold-spell layer is nonzero exactly where it should
  be, on the real files, before any world is decoded.
- **G5, data limits:** no reading on or after 2026-07-01 is loaded (the grid holds 78,816
  half-hours), and the tuning worlds are cut at 2024-01-01, so no tuning pass reads a later block.

## Synthetic world gallery

**The gallery is drawn in stage 1, before any tuning, and is added to this page before the tuning
starts.** The gallery shows each injected component alone, then the offset model's posterior on it.

- **Injected events alone:** 6 examples of an edge transfer (donor and recipient, both shown), 2 of
  a solo shift, 2 of a permanent shift, 2 of a staged or partial return, and 2 of overlapping events
  at one Primary, each as the injected change in megawatts over 60 days.
- **The same events on the real residual,** as block means with the injected change marked.
- **Cascades:** 4 cascaded starts and 2 cascaded returns, at the half-hour, against a single step
  of the same size.
- **Cold spells:** 3 spells across all 15 Primaries in one panel per spell, with the two λ = 0
  Primaries marked, and the shared term c(s, t) beside them.
- **Solar growth:** 2 of experiment 686's worlds, daylight blocks against the night block over 2
  years.
- **Hard negatives:** one each of a routine step, a ramp, a pulse, a placebo pair, and a common-mode
  swing.
- **The posterior on injected worlds:** for 8 injected events across the cells, drawn from tuning
  worlds only (seeds 734000 to 734029, with the cascade and cold-spell layers built on a tuning
  seed for this figure alone, and one solar-growth world), the residual, the posterior median and
  the 80% interval of the switched megawatts, the probability of zero, the permanent level, and the
  called events with their grades.
- **Real-data aggregates:** the distribution of noise units, the shared term's explained variance
  per Primary, and the share of MVA blocks flagged `unknown`. No metered generator appears.

**The write-up carries many event figures, each a time series of every Primary involved.** Each
figure shows the demand of the donor, the recipient, and any mirror partner that the offset model
used, with the decoded switched power and its 80% band, over the whole event plus a margin of 25%
of its duration (at least 2 days) on each side. Injected worlds shade the true window, and real
data shade the called window. The figure set has 4 parts:

- **Injected events, 6 per grade** ("mirrored and returned", "mirrored, not yet returned",
  "returned, no visible partner", "neither"), with hits, misses, and false calls, in the long-and-deep
  and long-and-shallow cells, and the cascades. Tuning seeds are drawn in stage 1. Scored seeds are
  drawn only after the scoring stage has run.
- **Real events, 12 in the stage-5 table of real events**, including logged shifts that the offset
  model finds and logged shifts that it misses. Primaries appear by D code, and no generator series
  appears.
- **The same event under E1-J-sym and PELT,** as paired panels or thin overlays, so the reader sees
  what each arm called on identical data.
- **Selection rule.** The first events of each category by a fixed seeded draw, not the best
  examples, so the set shows the misses as well as the hits.

## Stages and compute

**Every stage is CPU only, and the experiment uses 0 GPU-process-hours.** Each command runs inside
`systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0 -p OOMPolicy=continue`, with
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` set to 1 before any worker pool
starts. Before stage 2 the session sends the ML experiment manager a "fits starting" line with the
worker count, the projected CPU-hours, and the memory cap, and stays inside the slot the manager
allocates.

| Stage | Work | CPU-hours | Wall time at 6 workers |
|---|---|---|---|
| 0 | pre-checks: shared-term explained variance, noise units with and without the shared term, and the baseline comparison | under 0.2 | minutes |
| 1 | gallery and smoke check G4; gates G1 to G3; Viterbi timed first, then one timed tuning world (seed 734000, never a scored world) | about 0.5 | under 30 minutes |
| 2 | tuning on the 30 tuning worlds, each cut at 2024-01-01, and the real record of 2022 and 2023 | about 10.5 | about 2 hours |
| 3 | scored, cascade, and cold-spell worlds, all arms | about 16 or more | about 3 hours or more |
| 4 | sensitivity, solar-growth, and seasonal-bias worlds (H-own and H-nbr only), and the real record | about 5 | about 1 hour |
| 5 | report and figures | under 0.2 | minutes |

**The total is about 32 CPU-hours or more, so the stop at 40 CPU-hours may be reached.** The
estimate rests on one measured number: the forward step over the 12,915 joint states took 0.21
milliseconds per block on one core. With the emission, the backward pass, and the jump
probabilities, one Primary is expected to take about 7 seconds per forward-backward pass, and the
Viterbi pass adds an untimed amount, which is why stage 1 times it first. Two savings are built in:
the tuning worlds are decoded only up to 2024-01-01 (2,920 blocks instead of 6,568), and pass 1 of
H-nbr-noshared is H-own's pass, reused rather than run again. If the timed world projects more than
40 CPU-hours in all, the session stops and asks the manager, who may cut the cascade or cold-spell
worlds to 10 each or drop the sensitivity worlds before any scored world is decoded. Peak memory is
about 1.5 GB per worker (the forward messages of one Primary in 32-bit floats take 340 MB), so the
scope cap is 12 GB for 6 workers. Each worker writes one file per world
and arm, atomically, so a crashed worker is re-run without touching the others.

**Each stage checks the review marker of the stage before it.** The code review comes before stage
1's gates (marker `review_stage1.ok`); the tuned `params.json` from stage 2 is reviewed before stage
3 (marker `review_stage3.ok`).

**Stage 0 includes the baseline comparison and decides nothing in stage 1 apart from the shared
term's on/off switch.** On event-free blocks of 2022 and 2023 it compares, per Primary, the noise
unit and the 30-day drift of three baselines: the stored lag-free demand forecast (the baseline of
every arm), a centred median weekly profile of the observed power over the 4 weeks either side, and
the Primary's share of its neighbourhood's total. A baseline that lowers the noise unit by 10% or
more at a Primary is recorded as a candidate for part 2, and stage 1 keeps the stored forecast at
every Primary so that the arms differ only in their inference.

## Reading the result

**A null result says that whole-record inference does not find more long, shallow megawatts than a
segmentation with a fixed penalty per block on this bank.** If C1's intervals include zero while
P7 (calibration) and P9 (cold spells) hold, the offset model's value lies in its calibrated
distribution and its grades, which
serve use-cases 2 and 3, and not in events found; the next experiment would then test the grades on
NGED's answers rather than chase recall. If H-own already matches H-nbr, the neighbour evidence
adds nothing at 15 Primaries, which agrees with experiment 682 (an all-pairs graph paired 0.32
against 0.33 for the real links).

**A gain that holds only in the world-resampled interval is reported as not counted.** The 40
worlds share one real history, so a gain that the month-resampled interval does not support may
come from a few real months.

**What would change our mind.** The design would be wrong, rather than unlucky, in four cases:

- the known-answer test G3 passes but H-own finds less than E1-J-sym in the long and deep cell,
  which would mean the return-to-zero prior hands long events to the permanent level;
- H-nbr-noramp finds as much of the cascaded events' megawatts as H-nbr on the cascade worlds,
  which would mean the higher probability of a second step in the same direction does no work for
  cascades or for steps that start part-way through a block;
- H-nbr-noshared matches H-nbr on the cold-spell worlds, which would mean experiment 653's fixed
  common mode is already enough;
- the share of events graded "neither" exceeds half of all found injected temporary events, which
  would mean the offset model cannot tell a temporary switch from a permanent change at this noise
  level.

## Risks and failure modes

**The main risk is a baseline error that looks like an unpaired temporary transfer.** A holiday
fortnight, a cold spell at an industrial Primary that does not follow the shared term, or one of
D8's multi-day drops shifts one Primary's residual for days to weeks and then returns. The offset
model grades that shift "returned, no visible partner", and with 21 of 66 real transfers lacking a
metered receiver that grade cannot be dropped. The Christmas, cold-spell, and D8 rows above measure
how often the risk occurs, and only NGED's answers can measure the grade's real precision.

**The second risk is headroom.** On this bank E1-J-sym already finds 0.515 of the long and deep
megawatts and 0.393 of the long and shallow megawatts, and experiment 653's 25-state decoder already
has a forward-backward pass, so the gain may be smaller than the +0.10 bar.

- **Proportional shapes.** The generator's events follow the donor's daily profile for a share of
  0.3 to 1.0, and the two verified transfers measured 0.31 to 0.96, so the "flat" cue is weak in the
  bank and in reality. The gain moves at most one level per block, so a sudden proportional step
  should still go to o or ℓ; the found share by the generator's `prop_share` (below 0.5 against 0.5
  or more) checks this.
- **Permanent shifts in the counted cell.** 130 of the 840 long and shallow events are permanent
  shifts, of which E1-J-sym finds 0.09 of the megawatts. ℓ-events enter the found share, so the
  model is not penalised for calling a permanent shift permanent.
- **The size prior matches the generator.** The log-uniform jump prior is the generator's own size
  law, which flatters every arm of the offset model on injected events.
- **The geometric duration.** A constant ρ_off gives a geometric duration, while the generator's
  durations are log-uniform from 3 hours to 112 days, so the prior probability of a very long
  switch is lower than the generator's, and the tuning of ρ_off can only partly correct that.
- **The shared term at few Primaries.** Where fewer than 8 other Primaries have a reading (before
  D15 and D18 start, or in gaps), the shared term is set to zero.
- **Monthly refits.** The residual's demand model refits monthly, which can step the residual at a
  month boundary; the refit-boundary row measures this.
- **Unlabelled real events.** The worlds sit on the real record, so a real unlogged transfer can
  coincide with an injected event. Following the data-processing page, a called event that recurs
  at the same Primary and day in 10 or more of the 40 worlds is treated as a real-record event, not
  a false call, wherever false calls on injected worlds are counted.

## Audit triggers

**The audit triggers of the `ml-experiment` skill apply, adapted to a detector.** Trigger 1 fires if
a gate or the hard-negative placebo fails; trigger 2 if a counted prediction fails without a tested
explanation; trigger 4 if leaving out one Primary or one world flips C1's verdict (the report prints
the leave-one-out table); trigger 5 if H-own beats H-nbr in the counted cell by more than C1's
half-width; trigger 6 if C1's world-resampled interval is more than twice or less than half as wide
as experiment 680's interval on the same worlds; trigger 7 if a later experiment, such as a
long-window neural network taught by this model, is planned on this result before it is replicated.
Trigger 3 does not apply, because no forecast quantile score is contrasted. An audit runs on a fresh
agent that re-derives C1 from the stored event tables with its own code.

## Standing practices and earlier advice

- **Followed:** cleaned data with the `substation_zero` half-hours dropped (the count is reported);
  D15 and D18 rows before 2025-03-01 dropped; stored rows reused as the benchmark; a synthetic world
  gallery and a smoke check; a per-phase time log; no stage past its review gate; the hash of every
  edited reviewed script recorded; mid-day onsets in the known-answer tests; the four scenarios.
- **Not applicable:** monthly refits, model saving, training-loss plots, and screening folds,
  because nothing is trained; tail coverage of a forecast, because no forecast is scored; weather
  at the forecast origin, because the residual uses estimates of past weather and the output is a
  hindsight product, which the page states.
- **Advice followed:** compare with E1-J-sym (experiment 680); report shares found and shares paired
  separately (experiment 653); report short events apart (experiment 665); read a cell only with 20
  events or more (experiment 665); import generator 660 and name its configuration, report a
  `shuffled_placebo` beside every contrast, and treat un-injected time as unlabelled (experiment
  660); allow a start tolerance of at least 4 half-hours between the two ends of a transfer
  (experiment 686); drop candidates that recur in many worlds (experiment 692); first-order
  neighbours only, because an all-pairs graph paired as well as the real one (experiment 682).
- **Advice departed from:** experiment 653's advice to build an event table without partners next.
  This experiment keeps partners as evidence, never as a requirement, because the unseen-partner
  sink gives an unpaired event the same prior probability as in H-own.

## Changes from the design memo

**Six changes to the design that the maintainer approved, each made for a reason found while
planning.**

- **Benchmark.** The memo names experiment 653's table; this plan uses E1-J-sym, the current event
  table, with E1 as a reference row.
- **Headroom.** The memo quotes 0.67 of long, deep megawatts from experiment 653's own worlds. On
  generator 660's worlds the figures are 0.515 and, in the counted cell, 0.393.
- **Observation.** A 6-hour block cannot hold both a night mean and a day mean, and a daily step
  would blur the edges to a day. The offset model keeps 6-hour blocks and lets each block of the day
  carry its own demand loading and noise scale.
- **Permanent level.** The memo's two-peaked "neither" distribution needs an explanation that
  competes with a switch, so the offset model adds the permanent level ℓ beside the gain.
- **Lag-free demand estimate.** The stored residual already subtracts a lag-free demand forecast,
  so every arm keeps that baseline, and stage 0 compares it with a weekly profile and a
  neighbourhood share for part 2 only, so that stage 1's arms differ only in their inference.
- **Mirror window.** The memo says "within one step"; the plan uses 3 blocks, because the generator
  starts a recipient up to 36 half-hours after its donor.

## Method

**Architecture.**

- **Input:** the residual of the Terms section on the 15 Primaries, 2022-01-01 to 2026-06-30, as
  6-hour block means; experiment 653's common mode removed for every arm except H-nbr, which
  removes the fitted shared term instead.
- **Model:** the hidden Markov model of "The offset model", with 12,915 joint states, exact
  forward-backward and Viterbi passes, and one round of neighbour evidence for H-nbr.
- **Output:** block quantiles of the switched megawatts, the probability of zero, the `unknown`
  flag, and a graded event table refined to the half-hour.
- **Baselines:** E1-J-sym and E1 from experiments 680 and 670 (stored rows, re-run where needed);
  PELT written here.

**Training regime.**

- **No training:** priors, κ, τ, and the PELT penalty are tuned once on 30 tuning worlds with the
  scored years hidden and on the real record of 2022 and 2023; NGED's labels enter nothing.
- **Compute:** CPU only, about 68 CPU-hours at 1 thread per worker (stop at 90 after two
  extensions), 0 GPU-process-hours.

**Measurement.**

- **Rows:** injected events of 2 noise units or more, of kind edge, solo, permanent transfer, or
  permanent shift, starting 2024-01-01 to 2026-06-30, weighted by absolute megawatts, in the cells
  of experiment 665.
- **Matching:** experiment 653's `match` function for every arm.
- **Intervals:** paired world-resampled and month-resampled bootstraps, 10,000 draws, 97.5%.
- **Real record:** called rates in 2024 to mid-2026 and in event-free time, logged shifts matched,
  and the verified transfers.

## Files

- `ml-experiments/00734-switching-offset-model-part-1/e00734_panel.py`: loads the residual panel,
  the forecast, the noise units, and the neighbour graph, and builds the block means and the shared
  term.
- `.../e00734_hmm.py`: the state grid, the factorised transitions, the emission, and the forward,
  backward, Viterbi, and jump-probability passes.
- `.../e00734_events.py`: event extraction, grading, and half-hour refinement.
- `.../e00734_pelt.py`: the hand-written PELT search.
- `.../e00734_layers.py`: the cascade and cold-spell layers and their truth tables.
- `.../e00734_run.py`: stages 0 to 4, one world and arm per job, and the `freeze-code` stage that
  hashes the scoring code.
- `.../e00734_score.py` and `.../e00734_figures.py`: scoring, intervals, and every figure.
- `.../test_e00734_hmm.py`: the known-answer tests of gate G3.
- Read only: the folders of experiments 653, 660, 665, 670, and 680; experiment 327's stored power
  and forecasts; experiment 653's `noise_units.json`; experiment 680's `injected_default_*.parquet`,
  `injected_*_1000.parquet`, and `real_events_E1-J-sym.parquet`; experiment 670's
  `injected_default_*.parquet` for E1; experiment 580's event tables; experiment 383's neighbour
  graph; experiment 686's stored false events.
- Written: `data/ml-experiments/00734-switching-offset-model-part-1/` (never committed).
