---
id: "00660"
slug: shared-generator-of-injected-load-transfer-events
title: >-
  Can one shared generator of injected load-transfer events look real enough to train and test
  detectors on?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
date: 2026-11-09
kind: other
size: >-
  medium. (1) Stored: private injected panels, event tables, and score tables under
  data/ml-experiments/00660-shared-generator-of-injected-load-transfer-events/ only. (2) Serving
  path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible
  for the shape of a step, the duration law, and the hard negatives, so each is a configurable
  range with a default and a sensitivity arm. (5) Callers: later experiments import the module from
  this folder by path; no package or production code imports it. The maintainer's brief limits the
  work to CPU, 2 cores, and no GPU. Moving the module to a shared package would need the
  maintainer's review.
hypothesis: >-
  A generator that adds load-transfer events to real power history, with every assumption set from
  a measured number or marked unmeasured, produces windows that a classifier cannot tell from real
  events (area under the curve under 0.65), and a detector tuned on one configuration of the
  generator loses at most 0.05 of its recall on another.
prediction: >-
  Written before any score is read. P1 (realness): the area under the curve of a classifier that
  separates injected from real event windows, on the features a detector reads and with injected
  events matched to the real events' size, is under 0.65 for the default configuration. P2
  (cross-injector): the penalised-segmentation detector of experiment 653, tuned on constant-MW
  square steps and tested on shaped transfers with a different duration law (and the reverse),
  loses 0.05 or less of its recall against the same detector tuned on the test configuration. P3
  (recall by size): the detector's recall on injected events falls inside the interval of its
  recall on real logged events in every size band that holds at least 5 real events. P4 (edge
  width): the real transition at each end of the two verified transfers takes 4 half-hours or
  fewer, the generator's default range.
decision_rule: >-
  Fixed before any score is read. Supported if P1 and P2 both hold. Partly supported if exactly one
  holds. Not supported if neither holds. P3 and P4 are reported without entering the verdict,
  because the real events are too few to give them power: the page says so. If P1 fails, the page
  lists the features that give the generator away and the generator is changed once, with the
  change recorded as post hoc. The porting threshold is zero, because the module changes no
  production code. All data stops before 2026-07-01, and the final-test window is never read.
benchmark: >-
  Chance (area under the curve 0.5) for the realness classifier; the in-configuration recall of
  the same detector for the cross-injector test; the detector's recall on the logged shifts of
  experiment 653 for the recall-by-size check.
row_set_id: >-
  experiment 258's grid of 15 Primaries, half-hourly, 2022-01-01 to 2026-06-30, cleaned power;
  not comparable with the leaderboard
metric: >-
  Area under the curve of the realness classifier; MW-weighted recall of injected events and its
  drop between configurations; recall by size in noise units; transition width in half-hours and
  the constant share of the step shape, each with an interval.
code: ml-experiments/00660-shared-generator-of-injected-load-transfer-events/
---
**This page is the plan; the results are written after the runs.** The question: can one shared,
tested module add load-shift events (power moved between neighbouring Primary substations) to real
power history in a way that later experiments can trust? The plan is in the brief of the Discuss
session of 2026-11-09. The code lives in this experiment's folder; moving it to a shared package
would need the maintainer's review.
